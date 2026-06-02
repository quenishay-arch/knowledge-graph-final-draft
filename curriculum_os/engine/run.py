"""
Phase 4: Run curriculum extraction across P1–S6 PDFs and export knowledge graph.
"""

import argparse
import base64
import json
import sys
from pathlib import Path
import re

if __package__ is None or __package__ == "":
    # Enables running as: `python pipeline/run.py`
    # (adds project root to sys.path so `import curriculum_os.engine...` works)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curriculum_os.engine.chunking import (
    classify_page_type,
    detect_toc_map,
    detect_toc_units,
)
from curriculum_os.engine.hybrid_segmenter import hybrid_segment_units
from curriculum_os.engine.config import (
    BY_GRADE_DIR,
    KNOWLEDGE_GRAPH_JSON,
    KNOWLEDGE_GRAPH_XLSX,
    OUTPUT_DIR,
    RAW_PDF_DIR,
    discover_pdfs,
)
from curriculum_os.engine.export import build_knowledge_graph, export_to_excel, export_to_json
from curriculum_os.engine.canonical import (
    ParsedDocument,
    enforce_ntp_grade,
    ntp_locked_grade,
    parse_canonical_document,
    resolve_locked_grade,
)
from curriculum_os.engine.extract import (
    classify_document_type,
    detect_unit_boundaries_from_images,
    extract_entities,
    extract_entities_from_images,
    extract_unit_number,
    extract_global_metadata,
    infer_grade_from_images,
    infer_grade_from_content,
    get_llm_client,
    is_ntp_source,
    normalize_grade_label,
    parse_filename_hints,
)
from curriculum_os.engine.ingest import ingest_document_pages_structured
from curriculum_os.engine.grade_signals import (
    GRAMMAR_TOPIC_BANDS,
    GradeDecision,
    GradeSignal,
    choose_grade,
    curriculum_topic_signals,
    explicit_grade_signals,
    filename_grade_signal,
    lexical_complexity_signal,
    metadata_grade_signal,
)
from curriculum_os.engine.models import DocumentAnalysis, GradeInference, LevelEvidence, UnitBoundaryPlan
from curriculum_os.engine.unit_signals import detect_units, parse_filename_unit_numbers
NUMBER_WORD_GRADES = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
}


def _text_quality(pages: list[str]) -> dict:
    cleaned = [re.sub(r"\s+", " ", p or "").strip() for p in pages]
    nonempty = [p for p in cleaned if p]
    total_chars = sum(len(p) for p in nonempty)
    unique_pages = len(set(nonempty))
    repeated_ratio = 1.0 - (unique_pages / len(nonempty)) if nonempty else 1.0
    curriculum_terms = sum(
        len(re.findall(r"\b(unit|module|grammar|vocabulary|reading|writing|speaking|listening|contents)\b", p, re.I))
        for p in nonempty
    )
    return {
        "total_chars": total_chars,
        "chars_per_page": total_chars / max(1, len(pages)),
        "unique_pages": unique_pages,
        "repeated_ratio": repeated_ratio,
        "curriculum_terms": curriculum_terms,
        "likely_insufficient": total_chars < 1200
        or (repeated_ratio > 0.65 and curriculum_terms < 8)
        or curriculum_terms == 0,
    }


def _render_pages_as_data_urls(
    pdf_path: Path,
    *,
    start_page: int = 1,
    end_page: int | None = None,
    max_pages: int = 8,
    scale: float = 1.2,
) -> list[str]:
    try:
        import fitz
    except ImportError:
        return []

    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        doc.close()
        return []
    end_page = end_page or doc.page_count
    start_idx = max(0, start_page - 1)
    end_idx = min(doc.page_count, end_page)
    candidates = list(range(start_idx, end_idx))
    if len(candidates) > max_pages:
        step = (len(candidates) - 1) / max(1, max_pages - 1)
        candidates = [candidates[round(i * step)] for i in range(max_pages)]

    seen: set[int] = set()
    urls: list[str] = []
    for page_idx in candidates:
        if page_idx in seen or not 0 <= page_idx < doc.page_count:
            continue
        seen.add(page_idx)
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        data = pix.tobytes("jpeg", jpg_quality=75)
        urls.append("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
        if len(urls) >= max_pages:
            break
    doc.close()
    return urls


def _visual_chunks_from_plan(
    pages: list[str],
    plan: UnitBoundaryPlan,
) -> list[dict]:
    chunks: list[dict] = []
    page_count = len(pages)
    for unit in plan.units:
        start = max(1, min(page_count, unit.start_page))
        end = max(start, min(page_count, unit.end_page))
        title = (unit.unit_title or f"Unit {len(chunks) + 1}").strip()
        chunks.append(
            {
                "unit_title": title,
                "content": "\n\n".join(pages[start - 1 : end]).strip(),
                "start_page": start,
                "end_page": end,
                "boundary_source": "visual_unit_boundary",
                "boundary_evidence": unit.evidence,
            }
        )
    return chunks


def _unit_quality(
    chunks: list[dict],
    toc_map: list[dict],
    pages: list[str],
    unit_decision,
    *,
    filename_unit_numbers: list[int] | None = None,
) -> dict:
    warnings: list[str] = list(unit_decision.warnings)
    sources = {c.get("boundary_source", "") for c in chunks}
    confidence = unit_decision.confidence

    if toc_map and len(chunks) != len(toc_map):
        warnings.append("Segmenter chunk count differs from table-of-contents entries.")
        confidence = min(confidence, 0.72)

    if not chunks:
        warnings.append("No unit chunks produced.")
        confidence = 0.0
    elif len(chunks) == 1 and len(pages) > 20 and not toc_map and "visual_unit_boundary" not in sources:
        warnings.append("Long document produced one unit; human check may be needed.")
        confidence = min(confidence, 0.55)

    if not any(s in sources for s in ("toc", "structural_marker", "numeric_header", "heading", "module_heading", "regex_flat")):
        if unit_decision.count == 0:
            warnings.append("No reliable unit/chapter/module boundary markers found.")

    detected_titles = [u.display_title for u in unit_decision.units]
    return {
        "unit_count": len(chunks),
        "detected_unit_count": unit_decision.count,
        "detected_units": detected_titles,
        "confidence": round(confidence, 3),
        "boundary_sources": sorted(s for s in sources if s),
        "structural_source": unit_decision.source,
        "cross_check": unit_decision.cross_check,
        "warnings": list(dict.fromkeys(warnings)),
        "needs_human_review": unit_decision.needs_human_review or confidence < 0.68 or bool(warnings),
        "filename_unit_hint": filename_unit_numbers or [],
    }


def _unit_number_from_title(title: str) -> str:
    m = re.search(r"\b(?:Unit|U)\s*([0-9]{1,2})\b", title or "", re.IGNORECASE)
    return m.group(1) if m else ""


def _front_matter_text(pages: list[str]) -> str:
    snippets = pages[:5]
    for page in pages[:8]:
        lower = page.lower()
        if "contents" in lower or "table of contents" in lower:
            snippets.append(page)
    return "\n\n".join(snippets)[:20000]


def _infer_grade_from_front_matter(front_matter: str) -> tuple[str, list[str]]:
    """Deterministic grade from cover/front-matter labels, independent of LLM output."""
    text = front_matter or ""
    compact = re.sub(r"\s+", " ", text)
    lower = compact.lower()
    evidence: list[str] = []

    for word, number in NUMBER_WORD_GRADES.items():
        if re.search(rf"\bprimary\s+{word}\b", lower):
            evidence.append(f"cover label: Primary {word.title()}")
            return f"P{number}", evidence
        if re.search(rf"\bsecondary\s+{word}\b", lower):
            evidence.append(f"cover label: Secondary {word.title()}")
            return f"S{number}", evidence

    primary_token = re.search(r"\b([1-6])\s*([AB])\b", compact, re.IGNORECASE)
    if primary_token and re.search(r"\bprimary\b", lower):
        grade = f"P{primary_token.group(1)}"
        evidence.append(f"cover token with primary label: {primary_token.group(0)}")
        return grade, evidence

    secondary_token = re.search(r"\b([1-6])\s*([AB])\b", compact, re.IGNORECASE)
    if secondary_token and re.search(r"\bsecondary\b", lower):
        grade = f"S{secondary_token.group(1)}"
        evidence.append(f"cover token with secondary label: {secondary_token.group(0)}")
        return grade, evidence

    return "", evidence


def _infer_school_level(front_matter: str, source_code: str) -> tuple[str, list[str]]:
    text = (front_matter or "").lower()
    evidence: list[str] = []
    primary_hits = 0
    secondary_hits = 0

    primary_terms = ["primary", "phonics", "my family", "five senses", "cvc"]
    secondary_terms = [
        "secondary",
        "teenage",
        "argumentative",
        "discussion forum",
        "shop reviews",
        "hkdse",
    ]
    for t in primary_terms:
        if t in text:
            primary_hits += 1
            evidence.append(f"front matter term: {t}")
    for t in secondary_terms:
        if t in text:
            secondary_hits += 1
            evidence.append(f"front matter term: {t}")

    if source_code.upper().startswith("NTP"):
        secondary_hits += 1
        evidence.append(f"source code pattern suggests secondary: {source_code}")

    if secondary_hits > primary_hits:
        return "secondary", evidence
    if primary_hits > secondary_hits:
        return "primary", evidence
    return "unknown", evidence


def _infer_level_from_content(
    extractions: list[dict],
    *,
    school_level: str = "unknown",
    raw_text: str = "",
) -> tuple[str, float, list[LevelEvidence]]:
    topic_signals = curriculum_topic_signals(extractions, raw_text=raw_text)
    if school_level == "secondary":
        topic_signals = [s for s in topic_signals if s.grade.startswith("S")]
    if not topic_signals:
        return "Unknown", 0.0, []
    best = max(topic_signals, key=lambda s: s.confidence)
    evidence = [
        LevelEvidence(evidence=", ".join(s.evidence), suggested_level=s.grade, confidence=s.confidence)
        for s in topic_signals[:6]
    ]
    return best.grade, best.confidence, evidence


def _consistency_checks(extractions: list[dict], inferred_level: str) -> list[str]:
    issues: list[str] = []
    if not extractions:
        issues.append("No unit-level extractions were produced.")
        return issues
    if any(not x.get("unit_title") for x in extractions):
        issues.append("Some segments are missing unit titles.")

    hinted_levels = []
    for item in extractions:
        text = " ".join(
            item.get("grammar_points", [])
            + item.get("vocabulary_themes", [])
            + item.get("language_skills", [])
            + [item.get("unit_title", "")]
        ).lower()
        for level, hints in GRAMMAR_TOPIC_BANDS.items():
            if any(h in text for h in hints):
                hinted_levels.append(level)
                break
    if hinted_levels:
        unique = sorted(set(hinted_levels))
        if len(unique) >= 3:
            issues.append(
                f"Cross-unit level spread is wide ({', '.join(unique)}), check segmentation."
            )
        if inferred_level != "Unknown" and inferred_level not in unique:
            issues.append("Inferred level does not align with cross-unit grammar signals.")
    return issues


def _run_inference_engine(
    metadata,
    extractions: list[dict],
    *,
    allow_primary_hints: bool,
) -> tuple[str, float, list[LevelEvidence]]:
    """
    Soft prediction only — never used for NTP when canonical form is locked.
    """
    explicit = normalize_grade_label(metadata.explicit_grade_label, metadata.school_level)
    if explicit != "Unknown":
        return explicit, 0.78, []

    school = "secondary" if not allow_primary_hints else metadata.school_level
    return _infer_level_from_content(extractions, school_level=school)


def _valid_grade_label(label: str) -> str:
    label = (label or "").strip().upper()
    if re.fullmatch(r"[PS][1-6]", label):
        return label
    return "Unknown"


def _grade_decision_from_ranked_signals(
    parsed: ParsedDocument,
    metadata,
    extractions: list[dict],
    *,
    front_matter: str,
    all_text: str,
    content_grade: GradeInference | None = None,
) -> GradeDecision:
    """Parser + classifier pipeline with explicit priority ranking."""
    signals: list[GradeSignal] = []

    # Priority 0 — explicit grade anchors in document text
    signals.extend(explicit_grade_signals(front_matter, source="explicit_text", priority=0))
    if all_text and all_text != front_matter:
        signals.extend(
            explicit_grade_signals(all_text[:12000], source="explicit_text", priority=0)
        )

    explicit_metadata = normalize_grade_label(metadata.explicit_grade_label, metadata.school_level)
    if explicit_metadata != "Unknown":
        signals.append(
            GradeSignal(
                source="explicit_metadata",
                grade=explicit_metadata,
                confidence=0.94,
                priority=0,
                evidence=metadata.evidence[:5] or [metadata.explicit_grade_label],
            )
        )

    # Priority 1 — filename shorthand (P4.pdf, p3b.pdf, NTP form token)
    signals.extend(filename_grade_signal(parsed.raw_filename))

    # Priority 2 — semantic curriculum classification
    if content_grade:
        inferred = _valid_grade_label(content_grade.predicted_grade)
        if inferred != "Unknown":
            signals.append(
                GradeSignal(
                    source="semantic_content",
                    grade=inferred,
                    confidence=content_grade.confidence,
                    priority=2,
                    evidence=content_grade.evidence[:5],
                )
            )
    signals.extend(curriculum_topic_signals(extractions, raw_text=all_text))

    # Priority 3 — lexical complexity fallback
    lexical_signal = lexical_complexity_signal(all_text)
    if lexical_signal:
        signals.append(lexical_signal)

    # Priority 4 — publisher metadata (explicit label only, never series ranges)
    meta_signal = metadata_grade_signal(
        metadata.explicit_grade_label or "",
        school_level=metadata.school_level,
        evidence=metadata.evidence[:5],
    )
    if meta_signal:
        signals.append(meta_signal)

    return choose_grade(signals)


def _grade_inference_payload(
    front_matter: str,
    chunks: list[dict],
    extractions: list[dict],
) -> str:
    unit_summaries = []
    for idx, chunk in enumerate(chunks[:8], start=1):
        content = re.sub(r"\s+", " ", chunk.get("content", ""))[:1200]
        unit_summaries.append(
            f"Unit candidate {idx}: {chunk.get('unit_title', '')}\n{content}"
        )
    extraction_summaries = []
    for idx, item in enumerate(extractions[:8], start=1):
        extraction_summaries.append(
            "\n".join(
                [
                    f"Extraction {idx}: {item.get('unit_title', '')}",
                    f"Grammar: {item.get('grammar_points', [])}",
                    f"Vocabulary themes: {item.get('vocabulary_themes', [])}",
                    f"Skills: {item.get('language_skills', [])}",
                ]
            )
        )
    return "\n\n".join(
        [
            "Front matter:",
            front_matter[:6000],
            "Representative unit text:",
            "\n\n".join(unit_summaries),
            "Extracted curriculum signals:",
            "\n\n".join(extraction_summaries),
        ]
    )


def _decide_final_grade(
    parsed: ParsedDocument,
    metadata,
    extractions: list[dict],
    *,
    front_matter_grade: str = "",
    front_matter_evidence: list[str] | None = None,
    content_grade: GradeInference | None = None,
    front_matter: str = "",
    all_text: str = "",
) -> tuple[str, float, list[str], dict, bool, list[LevelEvidence], bool]:
    """Grade resolution by ranked evidence, with filename as one signal only."""
    if front_matter_grade and front_matter_evidence:
        front_matter = "\n".join([front_matter, *front_matter_evidence])
    decision = _grade_decision_from_ranked_signals(
        parsed,
        metadata,
        extractions,
        front_matter=front_matter,
        all_text=all_text,
        content_grade=content_grade,
    )
    provenance = {
        "canonical": parsed.model_dump(),
        "content_inference": content_grade.model_dump() if content_grade else None,
        "grade_decision": decision.model_dump(),
        "final": {
            "predicted_grade": decision.grade,
            "confidence": decision.confidence,
            "signal": decision.source,
            "source_of_truth": decision.source,
            "needs_human_review": decision.needs_human_review,
            "confidence_factors": decision.confidence_factors or {},
        },
    }
    trace = [
        f"{signal.source}: {signal.grade} ({signal.confidence:.2f})"
        for signal in sorted(decision.signals, key=lambda s: (s.priority, -s.confidence))
    ]
    trace.extend(decision.warnings)
    inference_used = decision.source not in {"filename", "filename_metadata", "none"}
    level_evidence = [
        LevelEvidence(evidence=", ".join(signal.evidence), suggested_level=signal.grade, confidence=signal.confidence)
        for signal in decision.signals
        if signal.source in {"curriculum_topics", "lexical_complexity", "semantic_content"}
    ]
    return (
        decision.grade,
        decision.confidence,
        trace,
        provenance,
        decision.needs_human_review,
        level_evidence,
        inference_used,
    )


def process_document(
    client,
    pdf_path: Path,
    *,
    max_chunks: int | None = None,
    frozen_grade: str | None = None,
) -> dict:
    print(f"\n=== Processing: {pdf_path.name} ===")
    parsed_doc = parse_canonical_document(pdf_path)
    locked_ntp = ntp_locked_grade(parsed_doc)
    if locked_ntp:
        print(f"  Canonical grade: {locked_ntp} (from filename form {parsed_doc.form_number})")
    elif parsed_doc.inferred_grade:
        print(
            f"  Canonical: {parsed_doc.publisher} form={parsed_doc.form_number} "
            f"-> {parsed_doc.inferred_grade}"
        )

    page_records = ingest_document_pages_structured(pdf_path)
    pages = [p["text"] for p in page_records]
    page_types = [
        {"page_number": p["page_number"], "page_type": classify_page_type(p["text"])}
        for p in page_records
    ]
    text_quality = _text_quality(pages)
    front_matter = _front_matter_text(pages)
    front_matter_grade, front_matter_grade_evidence = _infer_grade_from_front_matter(front_matter)
    filename_hints = parse_filename_hints(pdf_path)
    doc_type = classify_document_type(client, front_matter)
    metadata = extract_global_metadata(client, front_matter)
    school_level_trace: list[str] = []
    if not metadata.source_code and filename_hints["source_code"]:
        metadata.source_code = filename_hints["source_code"]
        metadata.evidence.append(f"source code from filename: {filename_hints['source_code']}")
    if parsed_doc.publisher == "NTP":
        metadata.school_level = "secondary"
    elif metadata.school_level == "unknown":
        inferred_school_level, school_level_trace = _infer_school_level(front_matter, metadata.source_code)
        if inferred_school_level != "unknown":
            metadata.school_level = inferred_school_level
            metadata.evidence.extend(school_level_trace)
    metadata.filename_grade_hint = filename_hints["grade_token"]
    metadata.filename_unit_hint = filename_hints["unit_number"]
    if parsed_doc.unit:
        metadata.unit_number = parsed_doc.unit
    elif not metadata.unit_number:
        metadata.unit_number = extract_unit_number(front_matter) or filename_hints["unit_number"]
    if filename_hints.get("ntp_grade"):
        metadata.evidence.extend(filename_hints["evidence"])

    toc_titles = detect_toc_units(pages)
    toc_map = detect_toc_map(pages)
    chunks = hybrid_segment_units(
        pages, toc_map=toc_map, source_file=pdf_path.name
    )
    visual_boundary_plan = None
    visual_boundary_error = ""
    if text_quality["likely_insufficient"] and len(pages) <= 40:
        boundary_images = _render_pages_as_data_urls(
            pdf_path,
            max_pages=min(24, len(pages)),
            scale=1.0,
        )
        if boundary_images:
            try:
                visual_boundary_plan = detect_unit_boundaries_from_images(
                    client,
                    boundary_images,
                    text_hint=front_matter,
                )
                visual_chunks = _visual_chunks_from_plan(pages, visual_boundary_plan)
                if visual_chunks and visual_boundary_plan.confidence >= 0.55:
                    chunks = visual_chunks
            except Exception as e:
                visual_boundary_error = str(e)
    print(f"  Chunks: {len(chunks)}")

    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    unit_decision = detect_units(
        pages,
        chunk_count=len(chunks),
        source_file=pdf_path.name,
    )
    filename_unit_nums = parse_filename_unit_numbers(pdf_path.name)
    unit_quality = _unit_quality(
        chunks,
        toc_map,
        pages,
        unit_decision,
        filename_unit_numbers=filename_unit_nums,
    )

    extractions = []
    for i, chunk in enumerate(chunks):
        print(f"  Extracting {i + 1}/{len(chunks)} …")
        try:
            if text_quality["likely_insufficient"]:
                image_urls = _render_pages_as_data_urls(
                    pdf_path,
                    start_page=int(chunk.get("start_page") or 1),
                    end_page=int(chunk.get("end_page") or len(pages)),
                    max_pages=10,
                    scale=1.2,
                )
                if image_urls:
                    ext = extract_entities_from_images(
                        client,
                        chunk,
                        image_urls,
                        text_hint=chunk.get("content", ""),
                    )
                else:
                    ext = extract_entities(client, chunk)
            else:
                ext = extract_entities(client, chunk)
            row = ext.model_dump()
            row["start_page"] = chunk.get("start_page")
            row["end_page"] = chunk.get("end_page")
            row["unit_number"] = _unit_number_from_title(
                row.get("unit_title", "")
            ) or _unit_number_from_title(chunk.get("unit_title", ""))
            extractions.append(row)
        except Exception as e:
            print(f"  Warning chunk {i + 1}: {e}")
            extractions.append(
                {
                    "unit_title": chunk.get("unit_title", ""),
                    "grammar_points": [],
                    "vocabulary_themes": [],
                    "language_skills": [],
                    "start_page": chunk.get("start_page"),
                    "end_page": chunk.get("end_page"),
                    "_error": str(e),
                }
            )

    content_grade = None
    explicit_metadata_grade = normalize_grade_label(
        metadata.explicit_grade_label, metadata.school_level
    )
    if not front_matter_grade and explicit_metadata_grade == "Unknown":
        try:
            content_grade = infer_grade_from_content(
                client,
                _grade_inference_payload(front_matter, chunks, extractions),
            )
        except Exception as e:
            content_grade = GradeInference(
                predicted_grade="Unknown",
                confidence=0.0,
                evidence=[f"content grade inference failed: {e}"],
            )
        if (
            _valid_grade_label(content_grade.predicted_grade) == "Unknown"
            and text_quality["likely_insufficient"]
        ):
            image_urls = _render_pages_as_data_urls(pdf_path, max_pages=6)
            if image_urls:
                try:
                    visual_grade = infer_grade_from_images(
                        client,
                        image_urls,
                        text_hint=front_matter,
                    )
                    if visual_grade.confidence > content_grade.confidence:
                        content_grade = visual_grade
                        content_grade.evidence.insert(0, "visual PDF page inference")
                except Exception as e:
                    content_grade.evidence.append(f"visual grade inference failed: {e}")

    if frozen_grade and frozen_grade not in ("", "UNKNOWN", "Unknown"):
        output_grade = frozen_grade
        level_confidence = 1.0
        decision_trace = [f"frozen_grade from orchestrator: {frozen_grade}"]
        grade_provenance = {
            "canonical": parsed_doc.model_dump(),
            "inference": {"predicted_grade": "SKIPPED", "skipped_reason": "frozen_grade"},
            "final": {
                "predicted_grade": frozen_grade,
                "confidence": 1.0,
                "signal": "canonical_parser",
                "source_of_truth": "canonical_parser",
            },
        }
        low_confidence = False
        level_evidence = []
        inference_used = False
        enforce_ntp_grade(parsed_doc, output_grade)
    elif locked_ntp:
        output_grade = locked_ntp
        level_confidence = 0.98
        decision_trace = [f"NTP form lock: {locked_ntp} (form {parsed_doc.form_number})"]
        grade_provenance = {
            "canonical": parsed_doc.model_dump(),
            "grade_decision": {"grade": locked_ntp, "source": "filename_ntp", "warnings": []},
            "final": {
                "predicted_grade": locked_ntp,
                "confidence": 0.98,
                "signal": "filename_ntp",
                "source_of_truth": "canonical_parser",
                "needs_human_review": False,
            },
        }
        low_confidence = False
        level_evidence = []
        inference_used = False
    else:
        (
            output_grade,
            level_confidence,
            decision_trace,
            grade_provenance,
            low_confidence,
            level_evidence,
            inference_used,
        ) = _decide_final_grade(
            parsed_doc,
            metadata,
            extractions,
            front_matter_grade=front_matter_grade,
            front_matter_evidence=front_matter_grade_evidence,
            content_grade=content_grade,
            front_matter=front_matter,
            all_text="\n\n".join(pages),
        )

    source_of_truth = grade_provenance.get("final", {}).get("source_of_truth", "none")
    canonical_log = {
        "file": pdf_path.name,
        "parsed_form": parsed_doc.form_number,
        "parsed": {
            "publisher": parsed_doc.publisher,
            "series": parsed_doc.series,
            "form_number": parsed_doc.form_number,
            "track": parsed_doc.track,
            "unit": parsed_doc.unit,
        },
        "final_grade": output_grade,
        "source": source_of_truth,
        "source_of_truth": source_of_truth,
        "inference_used": inference_used,
    }
    inferred_level = output_grade if output_grade != "UNKNOWN" else "Unknown"
    claimed_level = normalize_grade_label(metadata.explicit_grade_label, metadata.school_level)
    consistency_issues = _consistency_checks(extractions, inferred_level)
    consistency_issues.extend(unit_quality["warnings"])
    if text_quality["likely_insufficient"]:
        consistency_issues.append(
            "Extractable PDF text is sparse or repetitive; OCR/vision evidence may be required for reliable grade and unit detection."
        )
    if visual_boundary_error:
        consistency_issues.append(f"Visual unit-boundary detection failed: {visual_boundary_error}")
    if content_grade and content_grade.evidence:
        metadata.evidence.extend(
            f"content_grade_inference: {e}" for e in content_grade.evidence[:5]
        )

    if grade_provenance.get("final", {}).get("needs_human_review"):
        consistency_issues.append("Grade classification flagged for human review.")
    if unit_quality["needs_human_review"]:
        consistency_issues.append("Unit segmentation flagged for human review.")
    if consistency_issues:
        decision_trace.extend(consistency_issues)

    analysis = DocumentAnalysis(
        document_type=doc_type,
        metadata=metadata,
        claimed_level=claimed_level,
        inferred_level=output_grade if output_grade != "UNKNOWN" else "Unknown",
        level_confidence=level_confidence,
        level_evidence=level_evidence,
        consistency_issues=consistency_issues,
        level_decision_trace=decision_trace,
        total_units_detected=len(chunks),
        document_unit_number=metadata.unit_number or filename_hints["unit_number"],
    )

    return {
        "grade": output_grade,
        "predicted_grade": output_grade,
        "grade_confidence": level_confidence,
        "grade_provenance": grade_provenance,
        "canonical": canonical_log,
        "low_confidence": low_confidence,
        "source_file": pdf_path.name,
        "document_unit_number": analysis.document_unit_number,
        "total_units_detected": analysis.total_units_detected,
        "toc_unit_count": len(toc_map),
        "structural_unit_decision": unit_decision.model_dump(),
        "page_types": page_types,
        "text_quality": text_quality,
        "unit_quality": unit_quality,
        "human_review_flags": {
            "grade": low_confidence,
            "units": unit_quality["needs_human_review"],
            "text_quality": text_quality["likely_insufficient"],
            "overall": low_confidence or unit_quality["needs_human_review"] or text_quality["likely_insufficient"],
        },
        "visual_unit_boundary_plan": visual_boundary_plan.model_dump() if visual_boundary_plan else None,
        "unit_hierarchy": [
            {
                "unit_title": c.get("unit_title", ""),
                "start_page": c.get("start_page"),
                "end_page": c.get("end_page"),
                "boundary_source": c.get("boundary_source", "heading"),
            }
            for c in chunks
        ],
        "document_analysis": analysis.model_dump(),
        "extractions": extractions,
    }


def run_pipeline(
    *,
    max_chunks: int | None = None,
    grades: list[str] | None = None,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BY_GRADE_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs()

    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {RAW_PDF_DIR}. "
            "Place PDFs in data/input_pdfs/"
        )

    client = get_llm_client()
    documents = []

    for path in pdfs:
        doc = process_document(client, path, max_chunks=max_chunks)
        if grades and doc["grade"] not in set(grades):
            continue
        documents.append(doc)
        safe_grade = re.sub(r"[^A-Za-z0-9_-]+", "_", doc["grade"]) or "Unknown"
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(doc["source_file"]).stem)
        for stale in BY_GRADE_DIR.glob(f"*_{stem}.json"):
            if stale.name != f"{safe_grade}_{stem}.json":
                stale.unlink()
        grade_path = BY_GRADE_DIR / f"{safe_grade}_{stem}.json"
        with grade_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"  Saved {grade_path}")

    graph = build_knowledge_graph(documents)
    export_to_json(graph, KNOWLEDGE_GRAPH_JSON)
    export_to_excel(graph, KNOWLEDGE_GRAPH_XLSX)

    print(f"\nKnowledge graph JSON: {KNOWLEDGE_GRAPH_JSON}")
    print(f"Knowledge graph Excel: {KNOWLEDGE_GRAPH_XLSX}")
    return graph


def main():
    parser = argparse.ArgumentParser(description="P1–S6 curriculum knowledge graph export")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit chunks per PDF (for testing)",
    )
    parser.add_argument(
        "--grades",
        nargs="*",
        help="Only run these grades (e.g. P1 P3)",
    )
    args = parser.parse_args()
    run_pipeline(max_chunks=args.max_chunks, grades=args.grades)


if __name__ == "__main__":
    main()
