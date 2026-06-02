#!/usr/bin/env python3
"""Legacy-compatible entry — delegates to Curriculum OS orchestrator."""

from orchestrator.cli import main

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(["run", "--pdf-dir", "data/input_pdfs"])
    elif sys.argv[1] not in ("run", "route"):
        sys.argv.insert(1, "run")
    main()
