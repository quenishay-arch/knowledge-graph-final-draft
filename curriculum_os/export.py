"""Shared knowledge graph export."""

from __future__ import annotations

from pathlib import Path


def build_knowledge_graph(documents: list[dict]):
    from curriculum_os.engine.export import build_knowledge_graph as _build

    return _build(documents)


def export_to_json(graph, path: Path) -> None:
    from curriculum_os.engine.export import export_to_json as _export

    _export(graph, path)


def export_to_excel(graph, path: Path) -> None:
    from curriculum_os.engine.export import export_to_excel as _export

    _export(graph, path)
