"""Paths and environment for Curriculum OS."""

from __future__ import annotations

from pathlib import Path

# eng curr/ root
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "input_pdfs"
OUTPUT_DIR = DATA_DIR / "outputs"
BY_DOCUMENT_DIR = OUTPUT_DIR / "by_document"
ROUTING_JSON = DATA_DIR / "routing_decisions.json"
KNOWLEDGE_GRAPH_JSON = OUTPUT_DIR / "knowledge_graph.json"
KNOWLEDGE_GRAPH_XLSX = OUTPUT_DIR / "knowledge_graph.xlsx"

BENCHMARK_MANIFEST = ROOT / "benchmark" / "manifest.json"
