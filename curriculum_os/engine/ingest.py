from pathlib import Path

import fitz


def ingest_document(pdf_path: str | Path) -> str:
    """Load a PDF and return plain text."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    return "\n".join(pages)


def ingest_document_pages(pdf_path: str | Path) -> list[str]:
    """Load a PDF and return text per page."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    return [page.get_text("text") for page in doc]


def ingest_document_pages_structured(pdf_path: str | Path) -> list[dict]:
    """
    Load a PDF and return page-level structured text.

    Uses text blocks sorted top-to-bottom, left-to-right to preserve
    heading hierarchy better than flat text extraction.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    pages: list[dict] = []
    for idx, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        rows: list[tuple[float, float, str]] = []
        for block in blocks:
            # PyMuPDF blocks: (x0, y0, x1, y1, text, block_no, block_type, ...)
            x0, y0, _x1, _y1, text = block[:5]
            cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if cleaned:
                rows.append((float(y0), float(x0), cleaned))
        rows.sort(key=lambda r: (r[0], r[1]))
        ordered_text = "\n".join(r[2] for r in rows)
        pages.append(
            {
                "page_number": idx,
                "text": ordered_text,
                "line_count": len(ordered_text.splitlines()),
            }
        )
    return pages
