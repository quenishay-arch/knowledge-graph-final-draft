"""Document profiler for the unified pipeline."""

from __future__ import annotations

import re
from pathlib import Path

from curriculum_os.structure_analyzer import analyze_structure, page_count


def route_pdf(pdf_path: Path) -> dict:
    """Return document signals while keeping all PDFs on the same engine path."""
    stem = pdf_path.stem
    signals: list[str] = []
    pages = page_count(pdf_path)

    if re.match(r"^p\d", stem, re.IGNORECASE):
        signals.append("rule:filename_p_series")
    elif re.search(r"NTP", stem, re.IGNORECASE):
        signals.append("rule:filename_ntp")
    elif pages > 60:
        signals.append(f"rule:page_count_gt_60:{pages}")
    else:
        signals.append("rule:default_unified")

    profile = analyze_structure(pdf_path)
    signals.extend(profile.get("signals", []))

    return {
        "file": pdf_path.name,
        "route": "unified",
        "confidence": 1.0,
        "signals": signals,
        "structure": profile,
    }


def route_batch(pdf_dir: Path) -> list[dict]:
    return [route_pdf(p) for p in sorted(pdf_dir.glob("*.pdf"))]
