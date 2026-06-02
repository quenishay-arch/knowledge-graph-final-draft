#!/usr/bin/env python3
"""Smoke test: NTP hard locks and P/S filename hints (no LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from curriculum_os.canonical import (
    enforce_ntp_grade,
    filename_grade_hint,
    parse_canonical_document,
    resolve_locked_grade,
)
from curriculum_os.engine.grade_signals import filename_grade_signal, choose_grade

MANIFEST = Path(__file__).resolve().parent / "manifest.json"


def main() -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]
    ok = 0
    total = 0
    print("Grade signal smoke test (no LLM)\n")
    for e in entries:
        parsed = parse_canonical_document(e["file"])
        exp = e["expected_grade"]
        locked, source = resolve_locked_grade(parsed)
        if locked:
            total += 1
            if parsed.publisher == "NTP":
                enforce_ntp_grade(parsed, locked)
            match = locked == exp
            ok += int(match)
            mark = "OK" if match else "ERR"
            print(f"  [{mark}] {e['file']}: NTP lock={locked} exp={exp}")
            continue

        hint = filename_grade_hint(parsed)
        if not hint:
            continue
        signals = filename_grade_signal(e["file"])
        decision = choose_grade(signals)
        total += 1
        match = decision.grade == exp
        ok += int(match)
        mark = "OK" if match else "ERR"
        print(
            f"  [{mark}] {e['file']}: filename signal={decision.grade} "
            f"exp={exp} (hint={hint})"
        )
    print(f"\n{ok}/{total} passed")


if __name__ == "__main__":
    main()
