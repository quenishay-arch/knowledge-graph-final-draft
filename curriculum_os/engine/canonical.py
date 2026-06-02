"""
Canonical document normalization from filenames.

Strict publisher filename patterns only — ambiguous tokens (1A, 3B-book1) are
NOT treated as grade locks; they become content-inference candidates instead.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ParsedDocument:
    raw_filename: str
    publisher: str = "UNKNOWN"
    series: str | None = None
    form_number: int | None = None
    track: str | None = None
    unit: str | None = None
    inferred_grade: str | None = None

    def model_dump(self) -> dict:
        return asdict(self)


NTP_FILENAME = re.compile(
    r"^NTP(?P<series>\d+[A-Z]*)_(?P<form>\d)(?P<track>[AB])U(?P<unit>\d+)$",
    re.IGNORECASE,
)
P_FILENAME = re.compile(r"^p(?P<form>\d+)(?P<suffix>[a-z]?)$", re.IGNORECASE)
S_FILENAME = re.compile(r"^s(\d+)[a-z]?$", re.IGNORECASE)


def parse_canonical_document(pdf_path: str | Path) -> ParsedDocument:
    """Deterministic filename → structured schema for known shorthand patterns."""
    path = Path(pdf_path)
    raw = path.name
    stem = path.stem

    p_match = P_FILENAME.match(stem)
    if p_match:
        form = int(p_match.group("form"))
        grade = f"P{form}" if 1 <= form <= 6 else None
        return ParsedDocument(
            raw_filename=raw,
            publisher="P",
            series=(p_match.group("suffix") or "").lower() or None,
            form_number=form if 1 <= form <= 6 else None,
            inferred_grade=grade,
        )

    s_match = S_FILENAME.match(stem)
    if s_match:
        form = int(s_match.group(1))
        grade = f"S{form}" if 1 <= form <= 6 else None
        return ParsedDocument(
            raw_filename=raw,
            publisher="S",
            form_number=form if 1 <= form <= 6 else None,
            inferred_grade=grade,
        )

    ntp_match = NTP_FILENAME.match(stem)
    if ntp_match:
        form = int(ntp_match.group("form"))
        grade = f"S{form}" if 1 <= form <= 6 else None
        return ParsedDocument(
            raw_filename=raw,
            publisher="NTP",
            series=ntp_match.group("series").upper(),
            form_number=form if 1 <= form <= 6 else None,
            track=ntp_match.group("track").upper(),
            unit=ntp_match.group("unit"),
            inferred_grade=grade,
        )

    if re.match(r"^NTP", stem, re.IGNORECASE):
        loose = re.search(r"(\d)([AB])U(\d+)", stem, re.IGNORECASE)
        series_m = re.match(r"^NTP([^_]+)", stem, re.IGNORECASE)
        if loose:
            form = int(loose.group(1))
            grade = f"S{form}" if 1 <= form <= 6 else None
            return ParsedDocument(
                raw_filename=raw,
                publisher="NTP",
                series=series_m.group(1).upper() if series_m else None,
                form_number=form if 1 <= form <= 6 else None,
                track=loose.group(2).upper(),
                unit=loose.group(3),
                inferred_grade=grade,
            )

    return ParsedDocument(raw_filename=raw, publisher="UNKNOWN")


def ntp_locked_grade(parsed: ParsedDocument) -> str | None:
    """Hard NTP rule: S{form_number} from book token (1A → S1)."""
    if parsed.publisher != "NTP" or parsed.form_number is None:
        return None
    if not 1 <= parsed.form_number <= 6:
        return None
    return f"S{parsed.form_number}"


def enforce_ntp_grade(parsed: ParsedDocument, final_grade: str) -> None:
    expected = ntp_locked_grade(parsed)
    if expected is None:
        return
    assert final_grade == expected, (
        f"NTP grade override blocked: expected {expected} from form_number={parsed.form_number}, "
        f"got {final_grade} (file={parsed.raw_filename})"
    )


def resolve_locked_grade(parsed: ParsedDocument) -> tuple[str | None, str]:
    """Only NTP filenames produce a hard lock; all other grades use the ranked pipeline."""
    if parsed.publisher == "NTP":
        grade = ntp_locked_grade(parsed)
        if grade:
            return grade, "canonical_parser"
    return None, ""


def filename_grade_hint(parsed: ParsedDocument) -> str | None:
    """Non-locking filename grade hint for P/S shorthand (p4 → P4)."""
    if parsed.inferred_grade and parsed.publisher in ("P", "S", "NTP"):
        return parsed.inferred_grade
    return None
