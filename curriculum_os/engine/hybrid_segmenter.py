"""
Hybrid unit segmentation: TOC → structural/numeric headers → regex → single-unit fallback.

Chunk boundaries come from content signals only — never from filename unit hints.
"""

from __future__ import annotations

import re

from curriculum_os.engine.chunking import (
    MODULE_TITLE_PATTERN,
    UNIT_PATTERN,
    build_unit_hierarchy,
    chunk_document,
    chunk_pages_by_module,
    detect_toc_map,
    detect_toc_units,
)
from curriculum_os.engine.unit_signals import (
    NUMERIC_HEADER,
    STRUCTURAL_LABEL,
    UnitCandidate,
    detect_numeric_headers,
    detect_structural_markers,
)

HEADING_SCAN_LINES = 30
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
)


def _find_module_title(pages: list[str], max_pages: int = 6) -> str | None:
    for page in pages[:max_pages]:
        for line in page.splitlines()[:HEADING_SCAN_LINES]:
            m = MODULE_TITLE_PATTERN.match(line.strip())
            if m:
                return f"Module: {m.group(1).strip()}"
    return None


def _find_first_unit_heading(pages: list[str]) -> str | None:
    for page in pages:
        for line in page.splitlines()[:HEADING_SCAN_LINES]:
            line = line.strip()
            if UNIT_PATTERN.match(line) or STRUCTURAL_LABEL.match(line):
                return line
            m = NUMERIC_HEADER.match(line)
            if m:
                return f"Unit {m.group(1)}: {m.group(2).strip()}"
    return None


def _find_document_title(pages: list[str], max_pages: int = 3) -> str | None:
    skip = re.compile(r"^(?:page\s+\d+|\d+)$", re.IGNORECASE)
    skip_substrings = ("copyright", "all rights reserved", "permission", "pearson education", "©")
    skip_exact = {"pearson"}
    candidates: list[str] = []
    for page in pages[:max_pages]:
        for line in page.splitlines()[:HEADING_SCAN_LINES]:
            line = re.sub(r"\s+", " ", line).strip()
            if not line or skip.search(line):
                continue
            lower = line.lower()
            if lower in skip_exact or any(t in lower for t in skip_substrings):
                continue
            if len(line) < 3:
                continue
            candidates.append(line)
            if len(candidates) >= 2:
                break
        if candidates:
            break
    return " ".join(candidates[:2])[:120] if candidates else None


def _heading_on_page(page: str, toc_lookup: dict[str, str]) -> str | None:
    lines = page.splitlines()[:HEADING_SCAN_LINES]
    page_lower = page.lower()
    has_components = sum(1 for c in UNIT_COMPONENTS if c in page_lower) >= 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if UNIT_PATTERN.match(stripped) or STRUCTURAL_LABEL.match(stripped):
            return stripped
        if stripped.lower() in toc_lookup:
            return toc_lookup[stripped.lower()]
        m = NUMERIC_HEADER.match(stripped)
        if m and has_components:
            return f"Unit {m.group(1)}: {m.group(2).strip()}"
        m = MODULE_TITLE_PATTERN.match(stripped)
        if m:
            return f"Module: {m.group(1).strip()}"
    return None


def _count_distinct_headings(pages: list[str]) -> int:
    structural = detect_structural_markers(pages)
    numeric = detect_numeric_headers(pages)
    nums = {u.unit_number for u in structural.units + numeric.units if u.unit_number}
    return max(len(nums), structural.count, numeric.count)


def _page_boundary_chunks(
    pages: list[str],
    toc_titles: list[str],
    *,
    boundary_source: str = "heading",
) -> list[dict]:
    chunks: list[dict] = []
    current_title = ""
    current_start = 1
    buffer: list[str] = []
    toc_lookup = {t.lower(): t for t in toc_titles}

    for page_idx, page in enumerate(pages, start=1):
        heading = _heading_on_page(page, toc_lookup)
        if heading:
            if buffer:
                chunks.append(
                    {
                        "unit_title": current_title or f"Segment {len(chunks) + 1}",
                        "content": "\n\n".join(buffer).strip(),
                        "start_page": current_start,
                        "end_page": page_idx - 1,
                        "boundary_source": boundary_source,
                    }
                )
                buffer = []
            current_title = heading
            current_start = page_idx
        buffer.append(page)

    if buffer:
        chunks.append(
            {
                "unit_title": current_title or "Document",
                "content": "\n\n".join(buffer).strip(),
                "start_page": current_start,
                "end_page": len(pages),
                "boundary_source": boundary_source if current_title else "fallback",
            }
        )
    return [c for c in chunks if c.get("content")]


def _regex_multi_unit_chunks(pages: list[str]) -> list[dict] | None:
    text = "\n\n".join(pages)
    raw = chunk_document(text)
    named = [c for c in raw if c.get("unit_title") not in ("", "Document")]
    if len(named) < 2:
        return None
    out: list[dict] = []
    for c in raw:
        if c.get("unit_title") in ("", "Document"):
            continue
        row = dict(c)
        row["boundary_source"] = "regex_flat"
        out.append(row)
    return out or None


def _chunks_from_unit_candidates(pages: list[str], units: list[UnitCandidate]) -> list[dict] | None:
    """Split pages when detected units have start_page anchors."""
    anchored = [u for u in units if u.start_page and u.start_page >= 1]
    if len(anchored) < 2:
        return None
    anchored = sorted(anchored, key=lambda u: u.start_page or 0)
    chunks: list[dict] = []
    for i, unit in enumerate(anchored):
        start = unit.start_page or 1
        end = (anchored[i + 1].start_page or len(pages) + 1) - 1 if i + 1 < len(anchored) else len(pages)
        end = max(start, min(len(pages), end))
        chunks.append(
            {
                "unit_title": unit.display_title,
                "content": "\n\n".join(pages[start - 1 : end]).strip(),
                "start_page": start,
                "end_page": end,
                "boundary_source": unit.source,
                "unit_number": unit.unit_number,
            }
        )
    return chunks if len(chunks) >= 2 else None


def _is_weak_segmentation(chunks: list[dict], pages: list[str]) -> bool:
    if not chunks:
        return True
    heading_count = _count_distinct_headings(pages)
    if len(chunks) == 1:
        title = (chunks[0].get("unit_title") or "").strip()
        if title in ("", "Document") and len(pages) > 2:
            return True
        if heading_count >= 2 and len(pages) >= 2:
            return True
        return False
    return False


def _single_fragment_unit(pages: list[str]) -> list[dict]:
    module_title = _find_module_title(pages)
    unit_heading = _find_first_unit_heading(pages)
    document_title = _find_document_title(pages)

    if module_title:
        title, boundary = module_title, "module_heading"
    elif unit_heading:
        title, boundary = unit_heading, "heading"
    elif document_title:
        title, boundary = document_title, "document_title"
    else:
        title, boundary = "Document", "fallback"

    return [
        {
            "unit_title": title,
            "content": "\n\n".join(pages).strip(),
            "start_page": 1,
            "end_page": len(pages),
            "boundary_source": boundary,
        }
    ]


def _annotate_boundary(chunks: list[dict], default: str = "heading") -> list[dict]:
    for c in chunks:
        c.setdefault("boundary_source", default)
    return chunks


def hybrid_segment_units(
    pages: list[str],
    *,
    toc_map: list[dict] | None = None,
    source_file: str | None = None,
) -> list[dict]:
    """
    Layered segmentation with intelligent module handling:
      1. Check for single-unit modules (modules without multiple units mentioned)
      2. TOC hierarchy (build_unit_hierarchy)
      3. Unit candidates with page anchors
      4. Deep heading scan (explicit + numeric headers)
      5. Flat regex on joined text
      6. Single-document fallback with best-effort title
    """
    if not pages:
        return []

    # Early regex check for single-page documents
    regex_early = _regex_multi_unit_chunks(pages)
    if regex_early and len(pages) == 1:
        return regex_early

    # Check for modules using the updated logic that distinguishes
    # between single-unit modules and full textbooks with modules
    from curriculum_os.engine.unit_signals import detect_module_subunits
    module_signal = detect_module_subunits(pages)
    if module_signal and module_signal.count == 1:
        # Found a single-unit module (not a full textbook with multiple units)
        module_unit = module_signal.units[0] if module_signal.units else None
        if module_unit:
            return [
                {
                    "unit_title": module_unit.title,
                    "content": "\n\n".join(pages).strip(),
                    "start_page": 1,
                    "end_page": len(pages),
                    "boundary_source": "module_subunits",
                }
            ]

    toc_map = toc_map if toc_map is not None else detect_toc_map(pages)
    toc_titles = detect_toc_units(pages)

    chunks = build_unit_hierarchy(pages, toc_map)
    if chunks and not _is_weak_segmentation(chunks, pages):
        return _annotate_boundary(chunks)

    structural = detect_structural_markers(pages)
    numeric = detect_numeric_headers(pages)
    merged_units = sorted(
        {u.unit_number: u for u in structural.units + numeric.units if u.unit_number}.values(),
        key=lambda u: u.unit_number or 0,
    )
    candidate_chunks = _chunks_from_unit_candidates(pages, merged_units)
    if candidate_chunks and not _is_weak_segmentation(candidate_chunks, pages):
        return candidate_chunks

    deep = _page_boundary_chunks(pages, toc_titles, boundary_source="heading")
    if deep and not _is_weak_segmentation(deep, pages):
        return deep

    # Fallback module check (original logic kept for backward compatibility)
    module_one = chunk_pages_by_module(pages)
    if len(module_one) == 1 and module_one[0].get("unit_title") not in ("", "Document"):
        return _annotate_boundary(module_one, "module_heading")

    regex_multi = _regex_multi_unit_chunks(pages)
    if regex_multi:
        return regex_multi

    if len(pages) <= 80:
        return _single_fragment_unit(pages)

    return [
        {
            "unit_title": "Document",
            "content": "\n\n".join(pages).strip(),
            "start_page": 1,
            "end_page": len(pages),
            "boundary_source": "fallback",
        }
    ]
