"""Tests for priority-ranked grade and unit classification pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from curriculum_os.canonical import parse_canonical_document, resolve_locked_grade
from curriculum_os.engine.grade_signals import (
    choose_grade,
    curriculum_topic_signals,
    explicit_grade_signals,
    filename_grade_signal,
)
from curriculum_os.engine.models import GlobalMetadata, GradeInference
from curriculum_os.engine.run import _decide_final_grade, _infer_grade_from_front_matter
from curriculum_os.engine.unit_signals import detect_structural_markers, detect_units


def test_explicit_primary_five_in_content():
    signals = explicit_grade_signals("5B\nPRIMARY FIVE\nPlaces and activities", source="explicit_text", priority=0)
    decision = choose_grade(signals)
    assert decision.grade == "P5"
    assert decision.source == "explicit_text"


def test_filename_p4_shorthand():
    signals = filename_grade_signal("P4.pdf")
    decision = choose_grade(signals)
    assert decision.grade == "P4"
    assert decision.source == "filename"


def test_semantic_secondary_from_grammar_topics():
    extractions = [
        {
            "grammar_points": ["demonstrative pronouns", "connectives", "prepositions of time"],
            "vocabulary_themes": [],
            "language_skills": [],
            "unit_title": "Module 1",
        }
    ]
    signals = curriculum_topic_signals(extractions)
    decision = choose_grade(signals)
    assert decision.grade == "S1"


def test_semantic_primary_two_from_vocabulary():
    extractions = [
        {
            "grammar_points": ["simple present tense"],
            "vocabulary_themes": ["toys", "colours", "clothes"],
            "language_skills": [],
            "unit_title": "Unit 5 My toys",
        }
    ]
    signals = curriculum_topic_signals(extractions)
    decision = choose_grade(signals)
    assert decision.grade == "P2"


def test_semantic_primary_five_from_grammar():
    extractions = [
        {
            "grammar_points": ["passive voice", "present perfect tense"],
            "vocabulary_themes": ["countries and nationalities"],
            "language_skills": [],
            "unit_title": "Unit 2: More places to see",
        }
    ]
    signals = curriculum_topic_signals(extractions)
    decision = choose_grade(signals)
    assert decision.grade == "P5"


def test_explicit_text_beats_filename():
    signals = explicit_grade_signals("PRIMARY FIVE 5B", source="explicit_text", priority=0)
    signals.extend(filename_grade_signal("ok.pdf"))
    decision = choose_grade(signals)
    assert decision.grade == "P5"


def test_filename_beats_semantic_when_no_explicit():
    signals = filename_grade_signal("p2.pdf")
    signals.extend(
        curriculum_topic_signals(
            [{"grammar_points": ["passive voice"], "vocabulary_themes": [], "language_skills": [], "unit_title": ""}]
        )
    )
    decision = choose_grade(signals)
    assert decision.grade == "P2"
    assert decision.needs_human_review is True


def test_unit_regex_count():
    text = "Unit 1 International food fair\n...\nUnit 2: More places to see in Hong Kong"
    pages = text.split("\n")
    signal = detect_structural_markers(pages)
    assert signal.count == 2


def test_unit_module_subunits():
    pages = [
        "Module 1\n3 texts\nText 1 social media post\nText 2 short story\nText 3 news article",
        "Grammar focus",
    ]
    decision = detect_units(pages)
    assert decision.count == 1
    assert decision.source == "module_subunits"


def test_ntp_only_filename_lock():
    p = parse_canonical_document("NTP3E_1AU1.pdf")
    g, src = resolve_locked_grade(p)
    assert g == "S1" and src == "canonical_parser"


def test_p_series_is_hint_not_lock():
    p = parse_canonical_document("p3b.pdf")
    g, src = resolve_locked_grade(p)
    assert g is None and src == ""
    signals = filename_grade_signal("p3b.pdf")
    assert signals[0].grade == "P3"


def test_cover_grade_from_front_matter_for_bad_filename():
    grade, evidence = _infer_grade_from_front_matter("5B\nPRIMARY FIVE\nPlaces and activities")
    assert grade == "P5"
    assert evidence


def test_filename_content_mismatch_gets_human_review_flag():
    parsed = parse_canonical_document("p2.pdf")
    grade, _confidence, _trace, provenance, *_ = _decide_final_grade(
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


def test_filename_used_when_content_unknown():
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
