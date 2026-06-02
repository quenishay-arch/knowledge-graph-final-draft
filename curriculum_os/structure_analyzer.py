"""Document structure analysis — signals only, no routing decision."""

from __future__ import annotations

import re
from pathlib import Path

from curriculum_os.canonical import (
    ParsedDocument,
    filename_grade_hint,
    parse_canonical_document,
    resolve_locked_grade,
)

try:
    import fitz
except ImportError:
    fitz = None  # type: ignore

TOC_LINE = re.compile(
    r"^\s*((?:Unit|Module|Chapter)\s*\d+[^\n]*?)(?:\s+\.{2,}\s*|\s+)(\d{1,3})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def page_count(pdf_path: Path) -> int:
    if fitz is None:
        return 0
    try:
        doc = fitz.open(pdf_path)
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return 0


def toc_entry_count(pdf_path: Path, max_pages: int = 12) -> int:
    if fitz is None:
        return 0
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i in range(min(max_pages, doc.page_count)):
            text += doc.load_page(i).get_text("text") + "\n"
        doc.close()
        if "contents" not in text.lower() and "table of contents" not in text.lower():
            return 0
        return len(TOC_LINE.findall(text))
    except Exception:
        return 0


def analyze_structure(pdf_path: Path) -> dict:
    parsed: ParsedDocument = parse_canonical_document(pdf_path)
    locked, grade_src = resolve_locked_grade(parsed)
    filename_hint = filename_grade_hint(parsed)
    pages = page_count(pdf_path)
    toc_n = toc_entry_count(pdf_path)

    signals = [
        f"publisher:{parsed.publisher}",
        f"page_count:{pages}",
        f"toc_entries:{toc_n}",
    ]
    if parsed.form_number is not None:
        signals.append(f"form_number:{parsed.form_number}")
    if filename_hint:
        signals.append(f"filename_grade_hint:{filename_hint}")

    doc_type = "fragment"
    if parsed.publisher in ("P", "S"):
        doc_type = "structured_textbook"
    elif parsed.publisher == "NTP":
        doc_type = "ntp_fragment"
    elif pages >= 60 and toc_n >= 4:
        doc_type = "structured_textbook"

    return {
        "source_file": pdf_path.name,
        "document_type_hint": doc_type,
        "parsed": parsed.model_dump(),
        "locked_grade": locked or "UNKNOWN",
        "filename_grade_hint": filename_hint or "",
        "grade_source": grade_src or ("filename_hint" if filename_hint else "none"),
        "page_count": pages,
        "toc_unit_count": toc_n,
        "signals": signals,
    }
