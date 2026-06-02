"""Shared document contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RoutePlan:
    file: str
    pdf_path: str
    route: Literal["unified"]
    frozen_grade: str
    grade_source: str
    parsed: dict
    signals: list[str] = field(default_factory=list)
    page_count: int = 0
    toc_entries: int = 0


@dataclass
class ProcessedDocument:
    """Unified pipeline output envelope."""

    grade: str
    source_file: str
    extractions: list[dict]
    route: str
    frozen_grade: str
    grade_source: str
    canonical: dict
    routing: dict
    document_analysis: dict | None = None
    unit_hierarchy: list[dict] | None = None
    page_types: list[dict] | None = None
    extra: dict = field(default_factory=dict)

    def to_graph_document(self) -> dict:
        d: dict[str, Any] = {
            "grade": self.grade,
            "source_file": self.source_file,
            "extractions": self.extractions,
            "routing": self.routing,
            "canonical": self.canonical,
        }
        if self.document_analysis:
            d["document_analysis"] = self.document_analysis
        return d
