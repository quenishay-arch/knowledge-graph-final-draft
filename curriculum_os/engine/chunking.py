import re


UNIT_PATTERN = re.compile(
    r"(Unit\s*\d+[^\n]*|UNIT\s*\d+[^\n]*|Module\s*\d+[^\n]*|Chapter\s*\d+[^\n]*)",
    re.IGNORECASE,
)
MODULE_TITLE_PATTERN = re.compile(r"^\s*Module\s*:\s*(.+)$", re.IGNORECASE)
TOC_LINE_PATTERN = re.compile(
    r"^\s*((?:Unit|Module|Chapter)\s*\d+[A-Za-z]?[^\n]*?)(?:\s+\.{2,}\s*|\s+)(\d{1,3})\s*$",
    re.IGNORECASE,
)
TOC_PAGE_RANGE_PATTERN = re.compile(
    r"^\(?\s*pages?\s+(\d{1,3})(?:\s*[–-]\s*(\d{1,3}))?\s*\)?$",
    re.IGNORECASE,
)
MODULE_UNIT_RANGE_PATTERN = re.compile(
    r"\bUnits?\s*(\d{1,2})(?:\s*[–-]\s*(\d{1,2}))?\b",
    re.IGNORECASE,
)
PAGE_TYPE_KEYWORDS = {
    "cover_page": ["primary", "secondary", "book", "semester", "student's book"],
    "toc": ["contents", "table of contents"],
    "unit_divider": ["unit ", "module ", "chapter "],
    "grammar_page": ["grammar", "language focus", "sentence pattern"],
    "vocabulary_page": ["vocabulary", "word bank", "lexis"],
    "exercise_page": ["activity", "exercise", "answer the questions", "worksheet"],
}


def chunk_document(text: str) -> list[dict]:
    """Split textbook text into unit-level chunks with titles."""
    parts = re.split(UNIT_PATTERN, text)
    chunks: list[dict] = []
    current_title: str | None = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if UNIT_PATTERN.match(part):
            current_title = part
            continue

        if current_title:
            chunks.append({"unit_title": current_title, "content": part})

    if not chunks and text.strip():
        chunks.append({"unit_title": "Document", "content": text.strip()})

    return chunks


def detect_toc_units(pages: list[str], max_pages: int = 8) -> list[str]:
    """Extract unit titles from likely TOC pages in the front matter."""
    titles: list[str] = []
    toc_map = detect_toc_map(pages, max_pages=max_pages)
    if toc_map:
        return [item["unit_title"] for item in toc_map]
    for page in pages[:max_pages]:
        lower = page.lower()
        if "contents" not in lower and "table of contents" not in lower:
            continue
        for line in page.splitlines():
            line = line.strip()
            if not line:
                continue
            if TOC_LINE_PATTERN.match(line):
                titles.append(re.sub(r"\s+\d{1,3}$", "", line).strip())
    return titles


def classify_page_type(page_text: str) -> str:
    text = (page_text or "").lower()
    if not text.strip():
        return "unknown"
    scored: dict[str, int] = {}
    for page_type, keywords in PAGE_TYPE_KEYWORDS.items():
        score = sum(1 for k in keywords if k in text)
        if score > 0:
            scored[page_type] = score
    if not scored:
        return "content_page"
    return max(scored.items(), key=lambda kv: kv[1])[0]


def detect_toc_map(pages: list[str], max_pages: int = 14) -> list[dict]:
    """Extract TOC unit map: unit title + printed start page."""
    toc_map: list[dict] = []
    for page_idx, page in enumerate(pages[:max_pages], start=1):
        if classify_page_type(page) != "toc":
            continue
        module_unit_start = 1
        for line in page.splitlines():
            line = line.strip()
            if not line:
                continue
            module_match = MODULE_UNIT_RANGE_PATTERN.search(line)
            if module_match:
                module_unit_start = int(module_match.group(1))
            m = TOC_LINE_PATTERN.match(line)
            if not m:
                continue
            title = m.group(1).strip()
            start_page = int(m.group(2))
            toc_map.append(
                {
                    "unit_title": title,
                    "start_page_label": start_page,
                    "toc_page_number": page_idx,
                    "evidence": line,
                }
            )
        if toc_map:
            continue

        lines = [line.strip() for line in page.splitlines() if line.strip()]
        title_parts: list[str] = []
        unit_index = module_unit_start
        skip_tokens = {
            "contents",
            "unit",
            "vocabulary",
            "language structures",
            "spiral learning",
            "text types",
            "main task",
            "phonics",
            "appendix",
        }
        for line in lines:
            normalized = line.lower().strip(":")
            if normalized in skip_tokens or normalized.startswith("module:"):
                title_parts = []
                continue
            page_match = TOC_PAGE_RANGE_PATTERN.match(line)
            if page_match and title_parts:
                title = " ".join(title_parts).strip()
                title = re.sub(r"\s+", " ", title)
                toc_map.append(
                    {
                        "unit_title": f"Unit {unit_index}: {title}",
                        "start_page_label": int(page_match.group(1)),
                        "toc_page_number": page_idx,
                        "evidence": f"{title} {line}",
                    }
                )
                unit_index += 1
                title_parts = []
                continue
            if line.startswith("\uf0ab") or line.startswith("•") or "\t" in line:
                continue
            if re.match(r"^(?:page|pages)\b", line, re.IGNORECASE):
                continue
            if re.match(r"^\d+$", line):
                continue
            if line[:1].islower() and not title_parts:
                continue
            if len(title_parts) < 3 and not any(ch in line for ch in "•"):
                title_parts.append(line)
    return toc_map


def build_unit_hierarchy(pages: list[str], toc_map: list[dict]) -> list[dict]:
    """
    Build unit hierarchy using TOC first, then heading fallback.
    Returns units with page ranges and aggregated text.
    """
    if not pages:
        return []

    # Heuristic mapping from printed book pages to physical PDF pages.
    # We align using first TOC unit heading appearance.
    offset = 0
    if toc_map:
        first = toc_map[0]
        title = re.sub(r"^\s*Unit\s+\d+\s*:\s*", "", first["unit_title"], flags=re.I).lower()
        search_start = int(first.get("toc_page_number") or 1)
        for idx, page in enumerate(pages[search_start:], start=search_start + 1):
            if title and title in page.lower():
                offset = idx - first["start_page_label"]
                break

    segments: list[dict] = []
    if toc_map:
        sorted_map = sorted(toc_map, key=lambda x: x["start_page_label"])
        for i, item in enumerate(sorted_map):
            start = max(1, item["start_page_label"] + offset)
            if i + 1 < len(sorted_map):
                end = max(start, sorted_map[i + 1]["start_page_label"] + offset - 1)
            else:
                end = len(pages)
            seg_text = "\n\n".join(pages[start - 1 : end]).strip()
            if seg_text:
                segments.append(
                    {
                        "unit_title": item["unit_title"],
                        "start_page": start,
                        "end_page": end,
                        "content": seg_text,
                        "boundary_source": "toc",
                    }
                )
    if segments:
        return segments

    module_chunks = chunk_pages_by_module(pages)
    if len(module_chunks) >= 2:
        return module_chunks

    return chunk_pages_by_unit(pages)


def chunk_pages_by_unit(pages: list[str], toc_titles: list[str] | None = None) -> list[dict]:
    """Segment pages into units using heading starts, with TOC hints when available."""
    chunks: list[dict] = []
    current_title = ""
    current_start = 1
    buffer: list[str] = []
    toc_titles = toc_titles or []
    toc_lookup = {title.lower(): title for title in toc_titles}

    for page_idx, page in enumerate(pages, start=1):
        heading = None
        for line in page.splitlines()[:12]:
            line = line.strip()
            if UNIT_PATTERN.match(line):
                heading = line
                break
            if line.lower() in toc_lookup:
                heading = toc_lookup[line.lower()]
                break

        if heading:
            if buffer:
                chunks.append(
                    {
                        "unit_title": current_title or f"Segment {len(chunks) + 1}",
                        "content": "\n\n".join(buffer).strip(),
                        "start_page": current_start,
                        "end_page": page_idx - 1,
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
            }
        )

    return [chunk for chunk in chunks if chunk.get("content")]


def chunk_pages_by_module(pages: list[str]) -> list[dict]:
    """Segment full textbooks by pages that start with 'Module: ...'."""
    chunks: list[dict] = []
    current_title = ""
    current_start = 1
    buffer: list[str] = []

    for page_idx, page in enumerate(pages, start=1):
        heading = None
        for line in page.splitlines()[:20]:
            m = MODULE_TITLE_PATTERN.match(line.strip())
            if m:
                heading = f"Module: {m.group(1).strip()}"
                break

        if heading:
            if buffer:
                chunks.append(
                    {
                        "unit_title": current_title or f"Segment {len(chunks) + 1}",
                        "content": "\n\n".join(buffer).strip(),
                        "start_page": current_start,
                        "end_page": page_idx - 1,
                        "boundary_source": "module_heading",
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
                "boundary_source": "module_heading" if current_title else "fallback_document",
            }
        )

    return [chunk for chunk in chunks if chunk.get("content")]
