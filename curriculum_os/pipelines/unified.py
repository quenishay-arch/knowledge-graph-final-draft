"""Unified PDF pipeline adapter."""

from __future__ import annotations

from pathlib import Path

from curriculum_os.schemas import ProcessedDocument


def process_pdf(
    pdf_path: Path,
    frozen_grade: str,
    *,
    max_chunks: int | None = None,
    routing_meta: dict | None = None,
    canonical_log: dict | None = None,
) -> ProcessedDocument:
    """Run the unified engine; filename grade is only a fallback inside the engine."""
    from curriculum_os.engine.run import process_document

    from curriculum_os.engine.extract import get_llm_client

    client = get_llm_client()
    raw = process_document(
        client, pdf_path, max_chunks=max_chunks, frozen_grade=None
    )

    output_grade = raw.get("grade") or raw.get("predicted_grade") or "UNKNOWN"
    grade_source = raw.get("canonical", {}).get("source_of_truth", "inference")

    return ProcessedDocument(
        grade=output_grade,
        source_file=raw["source_file"],
        extractions=raw.get("extractions", []),
        route="unified",
        frozen_grade=frozen_grade,
        grade_source=grade_source,
        canonical=raw.get("canonical", canonical_log or {}),
        routing=routing_meta or {},
        document_analysis=raw.get("document_analysis"),
        unit_hierarchy=raw.get("unit_hierarchy"),
        page_types=raw.get("page_types"),
        extra={
            "grade_provenance": raw.get("grade_provenance"),
            "total_units_detected": raw.get("total_units_detected"),
            "text_quality": raw.get("text_quality"),
            "unit_quality": raw.get("unit_quality"),
            "human_review_flags": raw.get("human_review_flags"),
            "visual_unit_boundary_plan": raw.get("visual_unit_boundary_plan"),
        },
    )
