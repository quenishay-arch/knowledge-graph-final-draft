"""Anti-hardcoding and confidence scoring tests for grade classification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from curriculum_os.engine.grade_signals import (
    choose_grade,
    curriculum_topic_signals,
    explicit_grade_signals,
    filename_grade_signal,
    normalize_grade,
)
from curriculum_os.engine.models import GlobalMetadata, GradeInference
from curriculum_os.engine.run import _decide_final_grade
from curriculum_os.canonical import parse_canonical_document


def test_primary_one_to_six_series_is_not_p1():
    text = "Open English, Primary 1–6 was developed by Hong Kong Metropolitan University"
    signals = explicit_grade_signals(text, source="explicit_text", priority=0)
    assert not any(s.grade == "P1" for s in signals)


def test_unit_one_is_not_grade_p1():
    text = "Unit 1 International food fair\nVocabulary\nReading"
    signals = explicit_grade_signals(text, source="explicit_text", priority=0)
    assert not signals


def test_files_pdf_cover_resolves_p5_not_p1():
    front = "5B\nPRIMARY FIVE\nPlaces and activities"
    boilerplate = "Open English, Primary 1–6 was developed by Hong Kong Metropolitan University"
    signals = explicit_grade_signals(front, source="explicit_text", priority=0)
    signals.extend(explicit_grade_signals(boilerplate, source="explicit_text", priority=0))
    signals.extend(
        curriculum_topic_signals(
            [
                {
                    "grammar_points": ["passive voice", "present perfect tense", "ever and never"],
                    "vocabulary_themes": [],
                    "unit_title": "Unit 2",
                }
            ]
        )
    )
    decision = choose_grade(signals)
    assert decision.grade == "P5"
    assert decision.confidence >= 0.85


def test_p1_with_advanced_grammar_gets_flagged():
    signals = explicit_grade_signals("Primary 1", source="explicit_text", priority=0)
    signals.extend(
        curriculum_topic_signals(
            [
                {
                    "grammar_points": ["passive voice", "present perfect tense"],
                    "vocabulary_themes": [],
                    "unit_title": "",
                }
            ]
        )
    )
    decision = choose_grade(signals)
    assert decision.grade == "P1"
    assert decision.needs_human_review is True
    assert any("advanced grammar" in w.lower() for w in decision.warnings)


def test_filename_and_content_agreement_high_confidence():
    signals = filename_grade_signal("p5b.pdf")
    signals.extend(
        curriculum_topic_signals(
            [
                {
                    "grammar_points": ["passive voice", "present perfect"],
                    "vocabulary_themes": [],
                    "unit_title": "",
                }
            ]
        )
    )
    decision = choose_grade(signals)
    assert decision.grade == "P5"
    assert decision.confidence >= 0.88
    assert decision.confidence_factors.get("filename_content_match") is True


def test_content_only_lower_confidence_and_review():
    signals = curriculum_topic_signals(
        [
            {
                "grammar_points": ["demonstrative pronoun", "connectives", "prepositions of time"],
                "vocabulary_themes": [],
                "unit_title": "",
            }
        ]
    )
    decision = choose_grade(signals)
    assert decision.grade == "S1"
    assert decision.confidence <= 0.72
    assert decision.needs_human_review is True


def test_no_signals_returns_unknown_not_p1():
    decision = choose_grade([])
    assert decision.grade == "UNKNOWN"
    assert decision.confidence == 0.0


def test_ambiguous_filename_no_false_grade():
    parsed = parse_canonical_document("files.pdf")
    signals = filename_grade_signal(parsed.raw_filename)
    assert signals == []


def test_normalize_grade_rejects_series_range():
    assert normalize_grade("Primary 1-6") == "Unknown"
    assert normalize_grade("Primary 1–6 series") == "Unknown"
    assert normalize_grade("PRIMARY FIVE") == "P5"


def test_explicit_conflict_resolves_via_content():
    parsed = parse_canonical_document("files.pdf")
    grade, _conf, _trace, provenance, review, *_ = _decide_final_grade(
        parsed,
        GlobalMetadata(explicit_grade_label="Primary 5", school_level="primary"),
        [
            {
                "grammar_points": ["passive voice", "present perfect tense"],
                "vocabulary_themes": [],
                "language_skills": [],
                "unit_title": "International food fair",
            }
        ],
        front_matter="5B\nPRIMARY FIVE\nOpen English, Primary 1–6 series",
        all_text="PRIMARY FIVE 5B passive voice present perfect",
        content_grade=GradeInference(predicted_grade="P5", confidence=0.85, evidence=["passive voice"]),
    )
    assert grade == "P5"
    assert provenance["final"]["confidence"] >= 0.85
