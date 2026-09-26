# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Run frozen IVF over corpus cases and store compact, label-free summaries.

Usage (from the repository root):
    env -u PYTHONPATH uv run --frozen python research/failure_corpus/eval/run_ivf.py \
        --split dev --arm cal [--ablations full validity_only ...] [--workers 8]

IVF never sees a label: the manifest contains only the two bundle paths and their locks.
Evidence bundles for the ``full`` ablation are kept under the data directory; ablation
evidence is deleted after its summary is extracted (it would be several GB).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import manifests  # noqa: E402
from localize import localize  # noqa: E402


def summarize(result, oracle_order: list[str]) -> dict[str, Any]:
    validity = result.validity.to_jsonable() if result.validity is not None else {"checks": []}
    oracles = [o.to_jsonable() for o in result.outcomes]
    slim_oracles = []
    for o in oracles:
        div = o.get("divergence") or {}
        slim_oracles.append({
            "name": o["name"], "type": o["type"], "status": o["status"],
            "reason_codes": o.get("reason_codes", []),
            "metrics": {k: v for k, v in (o.get("metrics") or {}).items()
                        if not isinstance(v, list) or len(v) <= 32},
            "divergence": {k: div.get(k) for k in (
                "signal", "classification", "confidence", "first_numerical_difference_step",
                "first_tolerance_violation_step", "first_event_disagreement_step",
                "affected_components", "affected_env_ids")} if div else None,
        })
    return {
        "verdict": result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict),
        "reason_codes": list(result.reason_codes),
        "error": result.error,
        "validity": {"valid": validity.get("valid"),
                     "checks": [{k: c.get(k) for k in ("check_id", "name", "status", "reason_code", "detail")}
                                for c in validity.get("checks", [])]},
        "oracles": slim_oracles,
        "localization": localize(validity, oracles, oracle_order),
        "evidence": str(result.bundle_path),
    }


def run_one(split: str, case: dict[str, Any], arm: str, ablation: str, tol: dict[str, Any],
            evidence_root: str, keep: bool) -> tuple[str, dict[str, Any]]:
    from ivf.manifest import parse_manifest
    from ivf.runner import validate

    base = common.bundle_dir(split, case["reference_id"])
    cand = common.bundle_dir(split, case["capture_id"])
    if not (cand / "COMPLETE").exists():
        return case["capture_id"], {"verdict": "NO_CAPTURE", "localization": {"category": "none"},
                                    "note": "candidate capture raised; see result.json"}
    text = manifests.build_manifest(case, base, cand, tol, arm=arm, ablation=ablation)
    mpath = Path(evidence_root) / "manifests" / f"{case['capture_id']}.yaml"
    manifests.write_manifest(text, mpath)
    manifest = parse_manifest(text, source_path=str(mpath))
    result = validate(manifest, results_root=evidence_root, seed=0)
    summary = summarize(result, manifests.ABLATIONS[ablation]["oracles"])
    if not keep:
        shutil.rmtree(result.bundle_path, ignore_errors=True)
        summary["evidence"] = None
    return case["capture_id"], summary


def tolerances_for(thresholds: dict[str, Any], arm: str, platform: str, condition: str) -> dict[str, Any]:
    if arm == "permissive":
        return manifests.permissive_tolerances()
    return thresholds[arm][platform][condition]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--arm", required=True, choices=["cal", "strict", "permissive"])
    ap.add_argument("--ablations", nargs="*", default=["full"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None, help="restrict to these capture ids")
    args = ap.parse_args()
    common.assert_ivf_frozen()

    thresholds = {}
    if args.arm != "permissive":
        thresholds = json.loads((common.RESULTS / "thresholds.json").read_text())
    cases = common.evaluation_cases(args.split)
    if args.cases:
        cases = [c for c in cases if c["capture_id"] in set(args.cases)]
    for ablation in args.ablations:
        out_dir = common.RESULTS / args.split / "ivf" / f"{args.arm}__{ablation}"
        evidence_root = str(common.DATA / "evidence" / args.split / f"{args.arm}__{ablation}")
        keep = ablation == "full"
        todo = [c for c in cases if not (out_dir / f"{c['capture_id']}.json").exists()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {}
            for c in todo:
                # a case's condition decides which calibration applies; "same" also covers A/A reruns
                cond = "same" if c["condition"] == "same" else "cross"
                tol = tolerances_for(thresholds, args.arm, c["platform"], cond)
                futs[pool.submit(run_one, args.split, c, args.arm, ablation, tol, evidence_root, keep)] = c
            for fut in as_completed(futs):
                cid, summary = fut.result()
                common.write_json(out_dir / f"{cid}.json", summary)
        print(f"{args.split} {args.arm} {ablation}: {len(todo)} run, {len(cases)} total", flush=True)


if __name__ == "__main__":
    main()
