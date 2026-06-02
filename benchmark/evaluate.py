#!/usr/bin/env python3
"""
Benchmark evaluation — compares PIPELINE outputs to independent manifest labels.

Does NOT re-run curriculum rules inside evaluation (prevents label leakage).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
COS_OUT = ROOT / "data" / "outputs" / "by_document"


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]


def _best_output(stem: str, out_dir: Path) -> Path | None:
    """Prefer pipeline output tagged with canonical_parser (avoids stale P1/P3/Unknown files)."""
    matches = list(out_dir.glob(f"*_{stem}.json"))
    if not matches:
        return None

    canonical_tagged: list[Path] = []
    for path in matches:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        canon = doc.get("canonical") or {}
        prov = doc.get("grade_provenance") or {}
        final = prov.get("final") or {}
        if (
            canon.get("source_of_truth") == "canonical_parser"
            or canon.get("source") == "canonical_parser"
            or final.get("source_of_truth") == "canonical_parser"
        ):
            canonical_tagged.append(path)

    if canonical_tagged:
        return sorted(canonical_tagged, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _read_output(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_p_grade_mislabel(pred: str, expected: str) -> bool:
    return pred.startswith("P") and expected.startswith("S")


def _print_metrics(
    entries: list[dict],
    preds: dict[str, str],
    inference_preds: dict[str, str],
    label: str,
) -> None:
    scored_entries = entries
    ntp_entries = [e for e in scored_entries if "NTP" in e["file"].upper()]

    correct = 0
    graded = 0
    p_mislabel = 0
    unknown = 0
    inf_correct = 0
    inf_graded = 0

    print(f"\n=== {label} ===")
    for e in scored_entries:
        pred = preds.get(e["file"])
        if pred is None:
            print(f"  [SKIP] {e['file']}: no pipeline output")
            continue
        graded += 1
        exp = e["expected_grade"]
        ok = pred == exp
        correct += int(ok)
        if pred in ("UNKNOWN", "Unknown"):
            unknown += 1
        if _is_p_grade_mislabel(pred, exp):
            p_mislabel += 1

        inf = inference_preds.get(e["file"])
        inf_note = ""
        if inf is not None:
            inf_graded += 1
            if inf == exp:
                inf_correct += 1
            inf_note = f" | inference_only={inf}"

        mark = "OK" if ok else "ERR"
        print(f"  [{mark}] {e['file']}: final={pred} exp={exp}{inf_note}")

    if graded:
        print(f"\nFinal grade accuracy: {correct}/{graded} = {100 * correct / graded:.1f}%")
    if ntp_entries:
        ntp_graded = sum(1 for e in ntp_entries if preds.get(e["file"]) is not None)
        if ntp_graded:
            ntp_correct = sum(
                1 for e in ntp_entries if preds.get(e["file"]) == e["expected_grade"]
            )
            ntp_p = sum(
                1
                for e in ntp_entries
                if preds.get(e["file"])
                and _is_p_grade_mislabel(preds[e["file"]], e["expected_grade"])
            )
            ntp_unk = sum(
                1
                for e in ntp_entries
                if preds.get(e["file"]) in ("UNKNOWN", "Unknown")
            )
            print(f"NTP final accuracy: {ntp_correct}/{ntp_graded} = {100 * ntp_correct / ntp_graded:.1f}%")
            print(f"NTP P-grade misclassification: {ntp_p}/{ntp_graded} = {100 * ntp_p / ntp_graded:.1f}%")
            print(f"NTP UNKNOWN rate: {ntp_unk}/{ntp_graded} = {100 * ntp_unk / ntp_graded:.1f}%")
    if inf_graded:
        print(
            f"Inference-only accuracy (no rule layer): "
            f"{inf_correct}/{inf_graded} = {100 * inf_correct / inf_graded:.1f}%"
        )


def _pipeline_preds(
    entries: list[dict],
) -> tuple[dict[str, str], dict[str, str]]:
    """Read final grade from Curriculum OS outputs."""
    finals: dict[str, str] = {}
    inference: dict[str, str] = {}
    for e in entries:
        stem = Path(e["file"]).stem
        path = None
        if COS_OUT.exists():
            path = _best_output(stem, COS_OUT)
        if not path:
            continue
        doc = _read_output(path)
        finals[e["file"]] = doc.get("grade", "") or doc.get("predicted_grade", "")
        prov = doc.get("grade_provenance") or {}
        inf = prov.get("inference") or {}
        inference[e["file"]] = inf.get("predicted_grade", "")
    return finals, inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark — pipeline vs independent labels")
    parser.parse_args()

    entries = load_manifest()
    present = sum(1 for e in entries if (ROOT / e["path"]).exists())
    print(f"PDFs present: {present}/{len(entries)}")
    print("Ground truth: manifest.json (independent of pipeline rule code)")

    finals, inference = _pipeline_preds(entries)
    if not finals:
        print("\nNo pipeline outputs found. Run: python -m orchestrator.cli run")
        print("\n--- Rule spot-check (pipeline code, not used for scoring) ---")
        try:
            from curriculum_os.engine.extract import ntp_form_grade, primary_filename_grade

            for name in [
                "NTP3E_1AU1.pdf",
                "NTP3E_2AU1.pdf",
                "NTP3E_3AU1.pdf",
                "p3b.pdf",
            ]:
                ng, _ = ntp_form_grade(name)
                sg, _ = primary_filename_grade(name)
                print(f"  {name}: ntp_form={ng} primary_filename={sg}")
        except ImportError as exc:
            print(f"  (spot-check skipped: {exc})")
        return

    _print_metrics(entries, finals, inference, "PIPELINE OUTPUT vs MANIFEST")


if __name__ == "__main__":
    main()
