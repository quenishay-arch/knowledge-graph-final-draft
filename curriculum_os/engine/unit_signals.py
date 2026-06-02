"""
Unit detection pipeline — structural markers, TOC, numeric headers, cross-validation.

Priority:
  1. Explicit labels (Unit 1, Chapter 3, Module)
  2. Table of contents entries with titles
  3. Body headers (including bare numbers like "5 My toys")
  4. Module-with-N-texts pattern
  5. Cross-check all signals; filename unit hints validate (never drive segmentation alone)
  If input is a single unit pdf then return only 1 unit with the title of the unit
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from curriculum_os.engine.chunking import detect_toc_map

STRUCTURAL_LABEL = re.compile(
    r"^\s*(Unit|Chapter|Module)\s*(\d+)\s*[:\.]?\s*(.*)$",
    re.IGNORECASE,
)
STRUCTURAL_INLINE = re.compile(
    r"\b(Unit|Chapter|Module)\s*(\d+)\b",
    re.IGNORECASE,
)
NUMERIC_HEADER = re.compile(
    r"^(\d{1,2})\s+([A-Z][A-Za-z][\w\s'\-]{3,})$",
    re.IGNORECASE
)
PAGE_NOISE = re.compile(
    r"^(?:page\s+\d+|\d{1,3}\s*$|copyright|©|pearson|all rights reserved)$",
    re.IGNORECASE,
)
EXERCISE_NOISE = re.compile(
    r"^(?:question|activity|exercise|q\.?\s*\d|answer\s+the\s+questions)",
    re.IGNORECASE,
)
MODULE_WITH_TEXTS = re.compile(
    r"\b(\d+)\s+texts?\b|\b(\d+)\s+reading\s+passages?\b",
    re.IGNORECASE,
)
TEXT_SUBUNIT = re.compile(
    r"^\s*Text\s+(\d+)\s*[:\.]?\s*(.+)$",
    re.IGNORECASE,
)
UNIT_COMPONENTS = (
    "vocabulary",
    "reading",
    "language focus",
    "language structures",
    "listening",
    "speaking",
    "writing",
    "phonics",
    "pre-reading",
    "main task",
    "review",
    "unit contents",
    "grammar",
    "practice",
)
HEADING_SCAN_LINES = 30


@dataclass
class UnitCandidate:
    unit_number: int | None
    title: str
    source: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    start_page: int | None = None

    def model_dump(self) -> dict:
        return asdict(self)

    @property
    def display_title(self) -> str:
        if self.unit_number is not None and not re.match(r"^(Unit|Chapter|Module)\s", self.title, re.I):
            prefix = self.title.split()[0].lower() if self.title else ""
            if prefix not in ("unit", "chapter", "module"):
                return f"Unit {self.unit_number}: {self.title}"
        return self.title


@dataclass
class UnitCountSignal:
    source: str
    count: int
    confidence: float
    evidence: list[str]
    units: list[UnitCandidate] = field(default_factory=list)

    def model_dump(self) -> dict:
        d = asdict(self)
        d["units"] = [u.model_dump() for u in self.units]
        return d


@dataclass
class UnitDecision:
    count: int
    confidence: float
    source: str
    units: list[UnitCandidate]
    signals: list[UnitCountSignal]
    warnings: list[str]
    needs_human_review: bool = False
    cross_check: dict = field(default_factory=dict)

    def model_dump(self) -> dict:
        return {
            "count": self.count,
            "confidence": self.confidence,
            "source": self.source,
            "units": [u.model_dump() for u in self.units],
            "signals": [s.model_dump() for s in self.signals],
            "warnings": self.warnings,
            "needs_human_review": self.needs_human_review,
            "cross_check": self.cross_check,
        }


def parse_filename_unit_numbers(source_file: str | None) -> list[int]:
    """Extract expected unit numbers from filename tokens (U5U6, c5_c6, NTP U1)."""
    if not source_file:
        return []
    stem = Path(source_file).stem

    m = re.search(r"U(\d{1,2})U(\d{1,2})\b", stem, re.IGNORECASE)
    if m:
        return sorted({int(m.group(1)), int(m.group(2))})

    m = re.search(r"c(\d{1,2})[_\-]c(\d{1,2})\b", stem, re.IGNORECASE)
    if m:
        return sorted({int(m.group(1)), int(m.group(2))})

    m = re.search(r"(\d{1,2})[A-B]?U(\d{1,2})\b", stem, re.IGNORECASE)
    if m:
        return [int(m.group(2))]

    m = re.search(r"\bU(\d{1,2})\b", stem, re.IGNORECASE)
    if m:
        return [int(m.group(1))]

    return []


def _page_has_unit_components(page_text: str, *, scan_lines: int = 40) -> bool:
    lower = page_text.lower()
    lines = page_text.splitlines()[:scan_lines]
    hits = sum(1 for comp in UNIT_COMPONENTS if comp in lower)
    if hits >= 2:
        return True
    for line in lines:
        if EXERCISE_NOISE.match(line.strip()):
            return False
    return hits >= 1


def _is_sequential(numbers: list[int]) -> bool:
    if len(numbers) < 2:
        return True
    ordered = sorted(numbers)
    return all(b - a == 1 for a, b in zip(ordered, ordered[1:]))


def detect_structural_markers(pages: list[str]) -> UnitCountSignal:
    """Step 1: explicit Unit/Chapter/Module labels with optional titles."""
    seen: dict[int, UnitCandidate] = {}
    evidence: list[str] = []

    for page_idx, page in enumerate(pages, start=1):
        for line in page.splitlines()[:HEADING_SCAN_LINES]:
            line = line.strip()
            if not line or PAGE_NOISE.match(line) or EXERCISE_NOISE.match(line):
                continue
            m = STRUCTURAL_LABEL.match(line)
            if m:
                num = int(m.group(2))
                title_tail = (m.group(3) or "").strip()
                title = line if title_tail else f"{m.group(1)} {num}"
                if num not in seen:
                    seen[num] = UnitCandidate(
                        unit_number=num,
                        title=title,
                        source="structural_marker",
                        confidence=0.92,
                        evidence=[line[:80]],
                        start_page=page_idx,
                    )
                    evidence.append(line[:80])
                continue
            inline = STRUCTURAL_INLINE.search(line)
            if inline and inline.group(0).lower() == line.lower()[: len(inline.group(0))]:
                num = int(inline.group(2))
                if num not in seen:
                    seen[num] = UnitCandidate(
                        unit_number=num,
                        title=line[:80],
                        source="structural_marker",
                        confidence=0.88,
                        evidence=[line[:80]],
                        start_page=page_idx,
                    )
                    evidence.append(line[:80])

    units = sorted(seen.values(), key=lambda u: u.unit_number or 0)
    count = len(units)
    conf = 0.0
    if count >= 2:
        conf = 0.9
    elif count == 1:
        conf = 0.75
    return UnitCountSignal("structural_marker", count, conf, evidence[:12], units)


def detect_toc_units(pages: list[str], *, max_pages: int = 14) -> UnitCountSignal:
    """Step 2: parse Contents table for unit titles and numbers."""
    toc_map = detect_toc_map(pages, max_pages=max_pages)
    if not toc_map:
        text = "\n".join(pages[:max_pages]).lower()
        if "contents" not in text and "table of contents" not in text:
            return UnitCountSignal("toc_parse", 0, 0.0, [], [])
        return UnitCountSignal("toc_parse", 0, 0.0, ["contents page found but no unit rows parsed"], [])

    units: list[UnitCandidate] = []
    evidence: list[str] = []
    for item in toc_map:
        title = item["unit_title"]
        num_m = re.search(r"\b(?:Unit|Chapter|Module)\s*(\d+)", title, re.I)
        num = int(num_m.group(1)) if num_m else None
        units.append(
            UnitCandidate(
                unit_number=num,
                title=title,
                source="toc_parse",
                confidence=0.94,
                evidence=[item.get("evidence", title)[:80]],
                start_page=item.get("start_page_label"),
            )
        )
        evidence.append(title[:80])

    return UnitCountSignal("toc_parse", len(units), 0.93 if units else 0.0, evidence[:12], units)


def detect_numeric_headers(pages: list[str]) -> UnitCountSignal:
    """Bare numeric section headers (e.g. '5 My toys') validated by unit-level content."""
    candidates: dict[int, UnitCandidate] = {}
    
    # First check if this is a module document - if so, skip numeric header detection
    full_text = "\n\n".join(pages).lower()
    if "module:" in full_text:
        return UnitCountSignal("numeric_header", 0, 0.0, ["Skipped numeric header detection due to module presence"], [])

    for page_idx, page in enumerate(pages, start=1):
        lines = [ln.strip() for ln in page.splitlines()[:HEADING_SCAN_LINES] if ln.strip()]
        for i, line in enumerate(lines):
            if PAGE_NOISE.match(line) or EXERCISE_NOISE.match(line):
                continue
            m = NUMERIC_HEADER.match(line)
            if not m:
                continue
            num = int(m.group(1))
            title_text = m.group(2).strip()
            
            # Skip if title is too short or doesn't look like a proper unit title
            if len(title_text) < 4:
                continue
                
            # Unit titles should start with capital letter and be meaningful phrases
            if not title_text[0].isalpha() or title_text[0].islower():
                continue
            
            # Check if this looks like a comprehension question or exercise
            # Questions often start with question words or are incomplete sentences
            question_indicators = ["why", "what", "how", "when", "where", "who", "which", "complete", "fill", "match", "choose", "true/false"]
            first_word = title_text.split()[0].lower() if title_text.split() else ""
            if first_word in question_indicators:
                continue  # Skip comprehension questions
                
            # Skip common subsection headers and exercise types
            SUBSECTION_KEYWORDS = [
                "vocabulary", "grammar", "practice", "phonics",
                "post-reading", "now i can", "reading", "text",
                "photo journal", "personal description", "blog entry", "listening", "speaking", "writing",
                "language focus", "language structures", "language practice", "language review", "language project",
                "task", "appendix", "revision notes", "contents", "summary", "review", "main task", "unit contents",
                "exercise", "worksheet", "answer", "question", "activity", "comprehension", "interview", 
                "job advertisement", "letter of application", "feature article", "film review", "tv programme",
                "description", "discussion", "role play", "presentation", "project", "portfolio",
            ]

            if any(word in title_text.lower() for word in SUBSECTION_KEYWORDS):
                continue  # skip subsection headers

            # Check if this looks like a real unit by checking surrounding context
            # Real units should have multiple unit components
            context = "\n".join(lines[max(0, i-2):i+8])
            if not _page_has_unit_components(context):
                continue
                
            # Check if this line appears to be a proper heading (not inline text)
            # Real unit headings are usually standalone lines or at the beginning of content
            if i > 0 and len(lines[i-1]) > 2:  # Previous line has content
                prev_line_lower = lines[i-1].lower()
                # Check if previous line is a section end or page marker
                if not any(term in prev_line_lower for term in ["page", "unit", "chapter", "module", "end", "summary"]):
                    # Might not be a real heading, could be inline text
                    continue
                    
            # Check if this looks like a text within a module
            # Texts within modules are subsections, not units
            if "text" in line.lower() or "reading" in line.lower():
                # Check surrounding context for module indicators
                surrounding = "\n".join(lines[max(0, i-3):min(len(lines), i+3)]).lower()
                if "module" in surrounding:
                    continue  # This is a text within a module, not a unit
                    
            if num not in candidates:
                candidates[num] = UnitCandidate(
                    unit_number=num,
                    title=title_text,
                    source="numeric_header",
                    confidence=0.72,
                    evidence=[line[:80]],
                    start_page=page_idx,
                )

    numbers = sorted(candidates)
    
    # Validate that numbers are reasonable (1-20 for units)
    numbers = [n for n in numbers if 1 <= n <= 20]
    
    if not numbers:
        return UnitCountSignal("numeric_header", 0, 0.0, ["No valid numeric headers found"], [])
    
    # Check if numbers are sequential - units should be sequential
    if len(numbers) >= 2 and not _is_sequential(numbers):
        # Non-sequential numbers suggest these might not be actual units
        # Could be exercise numbers or other non-unit numbering
        return UnitCountSignal(
            "numeric_header", 
            len(numbers), 
            0.2,  # Very low confidence for non-sequential
            [f"Non-sequential numbers detected, likely not units: {numbers}"],
            []
        )
    
    # For single number, check if it's likely a real unit
    # Single units should have strong evidence of being a complete unit
    if len(numbers) == 1:
        single_num = numbers[0]
        candidate = candidates[single_num]
        # Check if single unit has strong evidence
        context_around = "\n".join(pages[max(1, candidate.start_page or 1)-1:min(len(pages), (candidate.start_page or 1)+2)])
        if not _page_has_unit_components(context_around):
            # Not enough unit components for a standalone unit
            return UnitCountSignal("numeric_header", 0, 0.0, ["Single numeric header lacks unit components"], [])
    
    # Boost confidence for sequential units
    if len(numbers) >= 2 and _is_sequential(numbers):
        for n in numbers:
            candidates[n].confidence = min(0.88, candidates[n].confidence + 0.12)

    units = [candidates[n] for n in numbers]
    count = len(units)
    # Higher confidence for multiple sequential units
    conf = 0.85 if count >= 2 and _is_sequential(numbers) else (0.65 if count == 1 else 0.0)
    return UnitCountSignal(
        "numeric_header",
        count,
        conf,
        [u.evidence[0] for u in units][:12],
        units,
    )


def detect_body_headers(pages: list[str]) -> UnitCountSignal:
    """Step 3: repeated section headers at page tops (explicit labels only)."""
    seen: dict[str, UnitCandidate] = {}
    for page_idx, page in enumerate(pages, start=1):
        for line in page.splitlines()[:HEADING_SCAN_LINES]:
            line = line.strip()
            m = STRUCTURAL_LABEL.match(line)
            if not m:
                continue
            num = int(m.group(2))
            key = f"{m.group(1).lower()}_{num}"
            if key not in seen:
                seen[key] = UnitCandidate(
                    unit_number=num,
                    title=line[:80],
                    source="header_grouping",
                    confidence=0.8,
                    evidence=[line[:80]],
                    start_page=page_idx,
                )
            break

    units = sorted(seen.values(), key=lambda u: u.unit_number or 0)
    count = len(units)
    conf = 0.82 if count >= 2 else (0.65 if count == 1 else 0.0)
    return UnitCountSignal(
        "header_grouping",
        count,
        conf,
        [u.evidence[0] for u in units][:12],
        units,
    )


def detect_module_subunits(pages: list[str]) -> UnitCountSignal | None:
    """Detect modules - either 'Module: <title>' pattern or 'module with N texts' pattern."""
    module_title = None
    module_evidence = []
    contains_multiple_units = False
    
    # Look for "Module: <title>" pattern in first few pages
    for page in pages[:8]:
        for line in page.splitlines()[:HEADING_SCAN_LINES]:
            line = line.strip()
            if "module:" in line.lower():
                # Extract the title after "Module:"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    module_title = parts[1].strip()
                    module_evidence.append(f"Found module title: {module_title}")
                    
                    # Check if this module mentions multiple units (e.g., "Units 1–3")
                    # If so, this might be a full textbook, not a single-unit PDF
                    if re.search(r"units?\s*\d+[–-]\d+", line.lower()) or re.search(r"units?\s*\d+\s*(?:and|&)\s*\d+", line.lower()):
                        contains_multiple_units = True
                        module_evidence.append(f"Module mentions multiple units, treating as full textbook")
                    break
        if module_title:
            break
    
    # If module mentions multiple units, don't treat it as a single-unit module
    # This is likely a full textbook with modules grouping units
    if contains_multiple_units:
        return None
    
    # If no explicit module title, check for "module with N texts" pattern
    if not module_title:
        full_text = "\n\n".join(pages)
        m = MODULE_WITH_TEXTS.search(full_text)
        if not m:
            return None
        n = int(m.group(1) or m.group(2))
        subunits: list[str] = []
        for page in pages[:12]:
            for line in page.splitlines():
                tm = TEXT_SUBUNIT.match(line.strip())
                if tm:
                    subunits.append(f"Text {tm.group(1)}: {tm.group(2).strip()[:50]}")

        module_evidence = [f"1 module with {n} sub-units/texts"]
        module_evidence.extend(subunits[:n])
        module_title = "Module (multi-text)"
    
    # If we found a module (and it doesn't contain multiple units), return it as a single unit
    if module_title:
        units = [
            UnitCandidate(
                unit_number=1,
                title=module_title,
                source="module_subunits",
                confidence=0.85,  # Higher confidence for clear module patterns
                evidence=module_evidence,
                start_page=1,
            )
        ]
        return UnitCountSignal("module_subunits", 1, 0.85, module_evidence[:12], units)
    
    return None


def _merge_unit_lists(signals: list[UnitCountSignal]) -> list[UnitCandidate]:
    """Prefer TOC titles, enrich with structural/numeric sources."""
    merged: dict[int, UnitCandidate] = {}
    priority = {"toc_parse": 4, "structural_marker": 3, "numeric_header": 2, "header_grouping": 1}

    for signal in sorted(signals, key=lambda s: priority.get(s.source, 0), reverse=True):
        for unit in signal.units:
            num = unit.unit_number
            if num is None:
                continue
            if num not in merged or unit.confidence > merged[num].confidence:
                merged[num] = unit
            elif num in merged and len(unit.title) > len(merged[num].title):
                merged[num].title = unit.title
                merged[num].evidence.extend(unit.evidence[:2])

    return sorted(merged.values(), key=lambda u: u.unit_number or 0)


def _cross_check_counts(signals: list[UnitCountSignal]) -> tuple[dict, list[str]]:
    active = {s.source: s.count for s in signals if s.count > 0}
    warnings: list[str] = []
    counts = list(active.values())
    if len(set(counts)) > 1 and len(counts) >= 2:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(active.items()))
        warnings.append(f"Unit count mismatch across signals: {detail}.")
    return {"counts_by_source": active, "agreement": len(set(counts)) <= 1}, warnings


def _check_filename_units(
    filename_units: list[int],
    detected_units: list[UnitCandidate],
) -> tuple[float, list[str]]:
    """Validate detected units against filename hints; never override content."""
    if not filename_units:
        return 0.0, []
    warnings: list[str] = []
    detected_nums = {u.unit_number for u in detected_units if u.unit_number is not None}
    expected = set(filename_units)

    if detected_nums and expected == detected_nums:
        return 0.08, []
    if detected_nums and expected.isdisjoint(detected_nums):
        warnings.append(
            f"Filename unit hint {sorted(expected)} disagrees with detected units {sorted(detected_nums)}."
        )
        return -0.15, warnings
    if len(expected) != len(detected_units) and detected_units:
        warnings.append(
            f"Filename suggests {len(expected)} unit(s) {sorted(expected)}; "
            f"content detected {len(detected_units)}."
        )
        return -0.1, warnings
    return 0.0, warnings


def detect_units(
    pages: list[str],
    *,
    chunk_count: int | None = None,
    source_file: str | None = None,
) -> UnitDecision:
    """Full pipeline with cross-validation and confidence scoring."""
    signals: list[UnitCountSignal] = [
        detect_structural_markers(pages),
        detect_toc_units(pages),
        detect_numeric_headers(pages),
        detect_body_headers(pages),
    ]
    module_signal = detect_module_subunits(pages)
    if module_signal:
        signals.append(module_signal)

    if chunk_count is not None and chunk_count > 0:
        signals.append(
            UnitCountSignal(
                source="segmentation",
                count=chunk_count,
                confidence=0.72,
                evidence=[f"segmenter produced {chunk_count} chunks"],
                units=[],
            )
        )

    cross_check, cross_warnings = _cross_check_counts(signals)
    warnings = list(cross_warnings)
    filename_units = parse_filename_unit_numbers(source_file)

    active = [s for s in signals if s.count > 0]
    if not active:
        fn_warn: list[str] = []
        if filename_units:
            fn_warn.append(
                f"Filename hints units {filename_units} but no structural markers found in text."
            )
        return UnitDecision(
            0, 0.0, "none", [], signals, fn_warn or ["No structural unit markers found."], True, cross_check
        )

    if module_signal and module_signal.count > 0:
        fn_delta, fn_warn = _check_filename_units(filename_units, module_signal.units)
        warnings.extend(fn_warn)
        conf = max(0.0, min(0.95, module_signal.confidence + fn_delta))
        return UnitDecision(
            count=1,
            confidence=round(conf, 3),
            source="module_subunits",
            units=module_signal.units,
            signals=[module_signal],  # ✅ only keep module signal
            warnings=warnings,
            needs_human_review=bool(warnings),
            cross_check=cross_check,
        )

    by_source = {s.source: s for s in active}
    toc = by_source.get("toc_parse")
    structural = by_source.get("structural_marker")
    numeric = by_source.get("numeric_header")
    headers = by_source.get("header_grouping")

    merge_sources = [s for s in (toc, structural, numeric, headers) if s and s.units]
    merged = _merge_unit_lists(merge_sources)

    if merged:
        units = merged
        count = len(units)
        source = merge_sources[0].source if merge_sources else "merged"
        base_conf = max(s.confidence for s in merge_sources)
    else:
        ranked = sorted(active, key=lambda s: (s.confidence, s.count), reverse=True)
        winner = ranked[0]
        units = winner.units
        count = winner.count
        source = winner.source
        base_conf = winner.confidence

    if toc and toc.count > 0 and count == toc.count:
        source = "toc_parse"
        base_conf = max(base_conf, 0.92)
    elif structural and structural.count == count and count > 0:
        base_conf = max(base_conf, 0.88)

    seg_n = by_source.get("segmentation")
    if seg_n and seg_n.count != count:
        warnings.append(f"Segmenter chunk count ({seg_n.count}) differs from detected units ({count}).")

    fn_delta, fn_warn = _check_filename_units(filename_units, units)
    warnings.extend(fn_warn)
    confidence = max(0.0, min(0.98, base_conf + fn_delta))

    if len(cross_check.get("counts_by_source", {})) >= 2 and not cross_check.get("agreement"):
        confidence = min(confidence, 0.72)

    needs_review = confidence < 0.68 or bool(warnings) or count == 0

    return UnitDecision(
        count=count,
        confidence=round(confidence, 3),
        source=source,
        units=units,
        signals=signals,
        warnings=warnings,
        needs_human_review=needs_review,
        cross_check=cross_check,
    )


# Backward-compatible thin wrappers
def count_units_regex(text: str) -> UnitCountSignal:
    pages = text.split("\n\n") if text else []
    if len(pages) <= 1:
        pages = [text]
    return detect_structural_markers(pages)


def count_units_toc(pages: list[str], *, max_pages: int = 14) -> UnitCountSignal:
    return detect_toc_units(pages, max_pages=max_pages)


def count_units_headers(pages: list[str], *, scan_lines: int = 25) -> UnitCountSignal:
    return detect_body_headers(pages)


def count_module_subunits(text: str) -> UnitCountSignal | None:
    pages = text.split("\n\n") if text else [text]
    return detect_module_subunits(pages)