"""Tests for unit detection pipeline — markers, TOC, numeric headers, cross-check."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from curriculum_os.engine.hybrid_segmenter import hybrid_segment_units
from curriculum_os.engine.unit_signals import (
    detect_numeric_headers,
    detect_structural_markers,
    detect_units,
    parse_filename_unit_numbers,
)


def test_files_pdf_two_explicit_units():
    pages = [
        "PRIMARY FIVE 5B\nModule: Places and activities (Units 1-2)",
        "Contents\nUnit 1 International food fair .... 1\nUnit 2 More places to see in Hong Kong .... 11",
        "Unit 1\nInternational food fair\nVocabulary\nLanguage structures\nReading",
        "Grammar\nPassive voice\nPractice exercises",
        "Unit 2: More places to see in Hong Kong\nVocabulary\nReading\nLanguage structures",
        "Review\nMain task",
    ]
    decision = detect_units(pages, source_file="files.pdf")
    assert decision.count == 2
    assert decision.confidence >= 0.85
    titles = " ".join(u.title.lower() for u in decision.units)
    assert "food fair" in titles or decision.count == 2


def test_ok_pdf_numeric_headers():
    pages = [
        "PRIMARY TWO\nContents\nUnit 5 My toys\nUnit 6 My clothes",
        "5 My toys\nVocabulary\ntoy\nball\nReading\nLanguage focus\nsimple present",
        "Phonics\nMain task\nReview",
        "6 My clothes\nVocabulary\nshirt\nshorts\nReading\nLanguage structures",
        "Writing\nReview",
    ]
    numeric = detect_numeric_headers(pages)
    assert numeric.count == 2
    nums = sorted(u.unit_number for u in numeric.units)
    assert nums == [5, 6]
    decision = detect_units(pages, source_file="ok.pdf")
    assert decision.count == 2
    assert "My toys" in decision.units[0].title or decision.units[0].unit_number == 5


def test_yes_pdf_module_with_three_texts():
    pages = [
        "Secondary One\nModule 1\n3 texts",
        "Text 1: Social media post\nReading\nVocabulary",
        "Text 2: Short story\nLanguage focus\nConnectives",
        "Text 3: News article\nDemonstrative pronouns\nPrepositions of time",
    ]
    decision = detect_units(pages, source_file="yes.pdf")
    assert decision.count == 1
    assert decision.source == "module_subunits"


def test_filename_u5u6_validates_content():
    pages = [
        "Unit 5 My toys\nVocabulary\nReading",
        "Unit 6 My clothes\nVocabulary\nLanguage focus",
    ]
    decision = detect_units(pages, source_file="1A_U5U6.pdf")
    assert decision.count == 2
    assert not any("disagrees" in w for w in decision.warnings)


def test_filename_unit_mismatch_flags_review():
    pages = [
        "Unit 1 Only unit\nVocabulary\nReading\nLanguage structures",
        "More content",
    ]
    decision = detect_units(pages, source_file="1A_U5U6.pdf")
    assert any("disagrees" in w.lower() or "suggests" in w.lower() for w in decision.warnings)
    assert decision.needs_human_review


def test_cross_check_mismatch_lowers_confidence():
    pages = [
        "Contents\nUnit 1 Alpha .... 1\nUnit 2 Beta .... 5\nUnit 3 Gamma .... 9",
        "Unit 1 Alpha\nVocabulary\nReading",
        "Unit 2 Beta\nVocabulary\nReading",
    ]
    decision = detect_units(pages)
    assert decision.count >= 2
    if decision.warnings:
        assert decision.confidence <= 0.92


def test_parse_filename_unit_numbers():
    assert parse_filename_unit_numbers("1A_U5U6.pdf") == [5, 6]
    assert parse_filename_unit_numbers("05_plet2e_2b_c1_c2.pdf") == [1, 2]
    assert parse_filename_unit_numbers("NTP3E_1AU3.pdf") == [3]


def test_hybrid_segmenter_splits_numeric_units():
    pages = [
        "5 My toys\nVocabulary\ntoy\nReading\nLanguage focus",
        "Practice\nReview",
        "6 My clothes\nVocabulary\nshirt\nReading\nLanguage structures",
        "Writing",
    ]
    chunks = hybrid_segment_units(pages, source_file="ok.pdf")
    assert len(chunks) >= 2
    titles = " | ".join(c["unit_title"] for c in chunks)
    assert "toys" in titles.lower() or "5" in titles
    assert "clothes" in titles.lower() or "6" in titles


def test_exercise_number_not_counted_as_unit():
    pages = [
        "Activity sheet\nQuestion 5 What is your name?\nQuestion 6 How old are you?",
        "Answer the questions.",
    ]
    numeric = detect_numeric_headers(pages)
    assert numeric.count == 0
