"""Golden tests for hybrid unit segmentation (no LLM, no PDF required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "segmentation"
GOLDEN = Path(__file__).resolve().parent / "golden" / "segmentation"

sys.path.insert(0, str(ROOT))

from curriculum_os.engine.hybrid_segmenter import hybrid_segment_units  # noqa: E402
from curriculum_os.engine.extract import normalize_grade_label  # noqa: E402
from curriculum_os.engine.run import _text_quality  # noqa: E402


def _load_cases() -> list[tuple[str, dict, dict]]:
    cases = []
    for fixture_path in sorted(FIXTURES.glob("*.json")):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        golden_path = GOLDEN / fixture_path.name
        if not golden_path.exists():
            pytest.skip(f"no golden file for {fixture_path.name}")
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        cases.append((fixture["id"], fixture, golden))
    return cases


def _assert_golden(chunks: list[dict], golden: dict, case_id: str) -> None:
    n = len(chunks)
    assert golden["chunk_count_min"] <= n <= golden["chunk_count_max"], (
        f"{case_id}: chunk count {n} not in [{golden['chunk_count_min']}, {golden['chunk_count_max']}]"
    )

    titles = " | ".join(c.get("unit_title", "") for c in chunks)
    for forbidden in golden.get("titles_must_not", []):
        assert forbidden not in titles, f"{case_id}: forbidden title {forbidden!r} in {titles!r}"

    if golden.get("titles_contain_any"):
        assert any(
            needle in titles for needle in golden["titles_contain_any"]
        ), f"{case_id}: expected one of {golden['titles_contain_any']} in {titles!r}"

    allowed = set(golden.get("boundary_sources_allowed", []))
    if allowed:
        for c in chunks:
            src = c.get("boundary_source", "")
            assert src in allowed, f"{case_id}: boundary_source {src!r} not in {allowed}"


_CASES = _load_cases()


@pytest.mark.parametrize(
    "case_id,fixture,golden",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_hybrid_segmentation_golden(case_id: str, fixture: dict, golden: dict) -> None:
    pages = fixture["pages"]
    chunks = hybrid_segment_units(
        pages,
        source_file=fixture.get("source_file"),
    )
    _assert_golden(chunks, golden, case_id)


def test_fragment_prefers_module_over_document() -> None:
    fixture = json.loads((FIXTURES / "fragment_ntp_single.json").read_text(encoding="utf-8"))
    chunks = hybrid_segment_units(fixture["pages"], source_file=fixture["source_file"])
    assert len(chunks) == 1
    assert "Document" != chunks[0]["unit_title"]
    assert "meet you" in chunks[0]["unit_title"].lower() or "Module" in chunks[0]["unit_title"]


def test_open_english_toc_table_splits_units() -> None:
    pages = [
        "3B\nPRIMARY THREE\nPlaces and activities\nUnit 1\nA shopping day\nUnit 2 Be helpful\nUnit 3 Ming's busy week",
        "Copyright",
        """Contents
Module: Places and activities (Units 1-3)
Unit
Vocabulary
Language structures
A shopping
day
(pages 1-8)
shops and items
Be helpful
(pages 9-16)
actions
Ming's busy
week
(pages 17-24)
days and activities
Appendix
Self-assessment pages 25-26""",
        "Characters in the book",
        "Icons",
        "1\n1\nPre-reading\nA shopping day\nPlaces and activities\nUnit contents",
        "2\nReading\nA shopping day",
        "3\nPractice\nA shopping day",
        "4\nLanguage structures",
        "5\nVocabulary",
        "6\nPhonics",
        "7\nMain task",
        "8\nReview",
        "9\n1\nPre-reading\nBe helpful\nPlaces and activities\nUnit contents",
        "10\nReading\nBe helpful",
        "11\nPractice\nBe helpful",
        "12\nLanguage structures",
        "13\nVocabulary",
        "14\nPhonics",
        "15\nMain task",
        "16\nReview",
        "17\n1\nPre-reading\nMing's busy week\nPlaces and activities\nUnit contents",
        "18\nReading\nMing's busy week",
        "19\nPractice\nMing's busy week",
        "20\nLanguage structures",
        "21\nVocabulary",
        "22\nPhonics",
        "23\nMain task",
        "24\nReview",
    ]

    chunks = hybrid_segment_units(pages, source_file="3B-book1_20230120-student-2.pdf")
    titles = [c["unit_title"] for c in chunks]

    assert titles[:3] == [
        "Unit 1: A shopping day",
        "Unit 2: Be helpful",
        "Unit 3: Ming's busy week",
    ]
    assert [c["boundary_source"] for c in chunks[:3]] == ["toc", "toc", "toc"]


def test_primary_grade_label_with_semester_suffix_normalizes() -> None:
    assert normalize_grade_label("Primary 3B", "primary") == "P3"


def test_segmentation_does_not_depend_on_filename_unit_hint() -> None:
    pages = [
        "Language focus: past tense\nVocabulary: travel",
        "Reading passage about a school trip.",
        "Grammar practice exercises.",
        "Writing task: postcard.",
    ]
    hinted = hybrid_segment_units(pages, source_file="NTP3E_1BU5.pdf")
    generic = hybrid_segment_units(pages, source_file="file.pdf")

    assert [(c["start_page"], c["end_page"], c["boundary_source"]) for c in hinted] == [
        (c["start_page"], c["end_page"], c["boundary_source"]) for c in generic
    ]
    assert len(hinted) == len(generic) == 1


@pytest.mark.integration
def test_pdf_fixture_if_present() -> None:
    """When PDFs exist, segmentation must not collapse to a single 'Document' chunk."""
    try:
        import fitz
    except ImportError:
        pytest.skip("pymupdf not installed")

    from curriculum_os.config import RAW_PDF_DIR
    from curriculum_os.engine.ingest import ingest_document_pages_structured

    checked = 0
    for pdf_path in sorted(RAW_PDF_DIR.glob("*.pdf")):
        checked += 1
        page_records = ingest_document_pages_structured(pdf_path)
        pages = [p["text"] for p in page_records]
        chunks = hybrid_segment_units(pages, source_file=pdf_path.name)
        assert chunks, f"{pdf_path.name}: no chunks"
        if _text_quality(pages)["likely_insufficient"]:
            continue
        if len(pages) > 3:
            titles = [c.get("unit_title", "") for c in chunks]
            assert not all(t == "Document" for t in titles), (
                f"{pdf_path.name}: all chunks titled Document"
            )

    if checked == 0:
        pytest.skip("no PDFs in data/input_pdfs")
