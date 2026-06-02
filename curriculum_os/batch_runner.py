"""
Curriculum OS batch runner — canonical parse → route → pipeline → merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from curriculum_os.canonical import parse_canonical_document, resolve_locked_grade
from curriculum_os.config import (
    BY_DOCUMENT_DIR,
    KNOWLEDGE_GRAPH_JSON,
    KNOWLEDGE_GRAPH_XLSX,
    OUTPUT_DIR,
    ROUTING_JSON,
)
from curriculum_os.export import build_knowledge_graph, export_to_excel, export_to_json
from curriculum_os.pipelines.unified import process_pdf
from curriculum_os.router import route_batch
from curriculum_os.schemas import ProcessedDocument


def _frozen_grade_for_pdf(pdf_path: Path) -> tuple[str, str, dict]:
    parsed = parse_canonical_document(pdf_path)
    locked, source = resolve_locked_grade(parsed)
    hint = locked or ""
    canonical_log = {
        "file": pdf_path.name,
        "parsed_form": parsed.form_number,
        "parsed": parsed.model_dump(),
        "filename_grade_hint": hint,
        "final_grade": "UNKNOWN",
        "source": source or "none",
        "source_of_truth": source or "none",
        "inference_used": False,
    }
    return locked or "UNKNOWN", source or "none", canonical_log


def _save_document(doc: ProcessedDocument) -> Path:
    BY_DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(doc.source_file).stem
    safe_grade = doc.grade.replace("/", "_")
    out = BY_DOCUMENT_DIR / f"{safe_grade}_{stem}.json"
    payload = {
        "grade": doc.grade,
        "predicted_grade": doc.grade,
        "source_file": doc.source_file,
        "extractions": doc.extractions,
        "route": doc.route,
        "frozen_grade": doc.frozen_grade,
        "grade_source": doc.grade_source,
        "canonical": doc.canonical,
        "routing": doc.routing,
    }
    if doc.document_analysis:
        payload["document_analysis"] = doc.document_analysis
    if doc.unit_hierarchy:
        payload["unit_hierarchy"] = doc.unit_hierarchy
    if doc.page_types:
        payload["page_types"] = doc.page_types
    if doc.extra:
        payload.update(doc.extra)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_batch(
    pdf_dir: Path,
    *,
    max_chunks: int | None = None,
    route_only: bool = False,
    no_merge: bool = False,
) -> dict:
    pdf_dir = pdf_dir.resolve()
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    decisions = route_batch(pdf_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROUTING_JSON.write_text(json.dumps(decisions, indent=2), encoding="utf-8")

    for d in decisions:
        print(f"  {d['file']}: {d['route']} — {', '.join(d['signals'][:3])}")

    if route_only:
        return {"routing": decisions, "documents": []}

    documents: list[ProcessedDocument] = []

    for d in decisions:
        pdf_path = pdf_dir / d["file"]
        frozen_grade, grade_source, canonical_log = _frozen_grade_for_pdf(pdf_path)
        routing_meta = {
            "route": d["route"],
            "signals": d["signals"],
            "structure": d.get("structure"),
        }

        doc = process_pdf(
            pdf_path,
            frozen_grade,
            max_chunks=max_chunks,
            routing_meta=routing_meta,
            canonical_log=canonical_log,
        )

        path = _save_document(doc)
        print(f"  Saved {path}")
        documents.append(doc)

    graph = {}
    if not no_merge and documents:
        graph = build_knowledge_graph([d.to_graph_document() for d in documents])
        export_to_json(graph, KNOWLEDGE_GRAPH_JSON)
        export_to_excel(graph, KNOWLEDGE_GRAPH_XLSX)
        print(f"\nKnowledge graph: {KNOWLEDGE_GRAPH_JSON}")

    return {"routing": decisions, "documents": documents, "graph": graph}
