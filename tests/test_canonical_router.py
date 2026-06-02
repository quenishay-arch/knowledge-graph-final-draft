"""Phase 0–2 tests: canonical grades and deterministic routing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from curriculum_os.canonical import parse_canonical_document, resolve_locked_grade
from curriculum_os.engine.models import GlobalMetadata, GradeInference
from curriculum_os.engine.run import _decide_final_grade, _infer_grade_from_front_matter
from curriculum_os.router import route_pdf


def test_ntp_form_grades():
    cases = [
        ("NTP3E_1AU1.pdf", "S1"),
        ("NTP3E_2BU5.pdf", "S2"),
        ("NTP3E_3AU1.pdf", "S3"),
    ]
    for name, exp in cases:
        p = parse_canonical_document(name)
        g, _ = resolve_locked_grade(p)
        assert g == exp, f"{name}: got {g}"


def test_p_series():
    p = parse_canonical_document("p3b.pdf")
    g, src = resolve_locked_grade(p)
    assert g is None and src == ""
    from curriculum_os.engine.grade_signals import filename_grade_signal

    signals = filename_grade_signal("p3b.pdf")
    assert signals[0].grade == "P3"


def test_ambiguous_book_label_is_not_filename_grade_lock():
    p = parse_canonical_document("3B-book1_20230120-student-2.pdf")
    g, src = resolve_locked_grade(p)
    assert g is None and src == ""


def test_ambiguous_form_tokens_are_not_filename_grade_locks():
    for name in [
        "1A_U5U6.pdf",
        "02_plet2e_1a_c5_c6.pdf",
    ]:
        p = parse_canonical_document(name)
        g, src = resolve_locked_grade(p)
        assert g is None and src == ""


def test_cover_grade_from_front_matter_for_bad_filename():
    grade, evidence = _infer_grade_from_front_matter("5B\nPRIMARY FIVE\nPlaces and activities")
    assert grade == "P5"
    assert evidence


def test_filename_content_mismatch_gets_human_review_flag():
    parsed = parse_canonical_document("p2.pdf")
    grade, confidence, trace, provenance, *_ = _decide_final_grade(
        parsed,
        GlobalMetadata(),
        [],
        front_matter_grade="",
        content_grade=GradeInference(
            predicted_grade="P4",
            confidence=0.82,
            evidence=["comparative adjectives and book report task"],
        ),
    )
    assert grade == "P2"
    assert provenance["final"]["needs_human_review"] is True
    assert any("disagrees" in warning for warning in provenance["grade_decision"]["warnings"])


def test_filename_only_used_as_fallback():
    parsed = parse_canonical_document("p2.pdf")
    grade, _confidence, _trace, provenance, *_ = _decide_final_grade(
        parsed,
        GlobalMetadata(),
        [],
        front_matter_grade="",
        content_grade=GradeInference(predicted_grade="Unknown", confidence=0.1),
    )
    assert grade == "P2"
    assert provenance["final"]["source_of_truth"] == "filename"


def test_routing_rules():
    assert route_pdf(Path("p3b.pdf"))["route"] == "unified"
    assert route_pdf(Path("NTP3E_1AU1.pdf"))["route"] == "unified"
