#!/usr/bin/env python3
"""
Curriculum OS — unified entry point.

  python -m orchestrator.cli run --pdf-dir data/input_pdfs
  python -m orchestrator.cli route --pdf-dir data/input_pdfs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from curriculum_os.batch_runner import run_batch
from curriculum_os.config import RAW_PDF_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="HK Curriculum OS")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Route, extract, merge knowledge graph")
    run_p.add_argument("--pdf-dir", type=Path, default=RAW_PDF_DIR)
    run_p.add_argument("--max-chunks", type=int, default=None)
    run_p.add_argument("--no-merge", action="store_true")

    route_p = sub.add_parser("route", help="Routing decisions only")
    route_p.add_argument("--pdf-dir", type=Path, default=RAW_PDF_DIR)

    args = parser.parse_args()

    if args.command == "route":
        run_batch(args.pdf_dir, route_only=True)
    else:
        run_batch(
            args.pdf_dir,
            max_chunks=args.max_chunks,
            route_only=False,
            no_merge=args.no_merge,
        )


if __name__ == "__main__":
    main()
