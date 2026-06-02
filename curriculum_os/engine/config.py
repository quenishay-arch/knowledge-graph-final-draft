import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PDF_DIR = PROJECT_ROOT / "data" / "input_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
BY_GRADE_DIR = OUTPUT_DIR / "by_grade"

GRADES = ["P1", "P2", "P3", "P4", "P5", "P6"]

KNOWLEDGE_GRAPH_JSON = OUTPUT_DIR / "knowledge_graph.json"
KNOWLEDGE_GRAPH_XLSX = OUTPUT_DIR / "knowledge_graph.xlsx"

DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    or "DeepSeek-V4-Flash"
)


def discover_pdfs() -> list[Path]:
    """Return all PDFs in the single input folder, sorted by name."""
    if not RAW_PDF_DIR.exists():
        return []

    return sorted(RAW_PDF_DIR.glob("*.pdf"))
