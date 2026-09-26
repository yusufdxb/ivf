# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic stratum: IVF's own 17-class taxonomy against the same simple baselines.

IVF results come from ``ivf calibrate --keep-bundles`` (seeds 11 23 47). Baseline
thresholds come from a separate run with seeds 101 102 103, using only the ``none``
(clean) trials. Every baseline reads the exact arrays IVF compared (evidence ``signals/``).
This stratum is the one IVF's authors designed and tuned the calibration workload against,
so it is reported separately and never pooled with the external corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

MARGIN, FLOOR = 1.25, 1e-6
SYN = common.DATA / "synthetic"
# IVF classification that names each synthetic category (exploratory localization check)
EXPECTED_CLASS = {
    "reset": "reset_mismatch",
    "frame": "coordinate_frame_mismatch",
    "convention": "quaternion_convention_mismatch",
    "action": "action_mismatch",
    "observation": "sensor_semantic_difference",
    "sensor": "sensor_semantic_difference",
    "units": "setup_mismatch",
    "dynamics": "solver_parameter_difference",
    "numerical": "numerical_drift",
    "capability": "unsupported_feature",
}


def trials(run: str):
    for d in sorted((SYN / run).iterdir()):
        if not (d / "verdict.json").exists():
            continue
        subj = json.loads((d / "manifest.resolved.json").read_text())["subjects"]
        fault = subj["candidate"].get("fault") or "none"
        with np.load(d / "signals" / "baseline.npz") as a, np.load(d / "signals" / "candidate.npz") as b:
            base = {k: a[k].astype(np.float64) for k in a.files}
            cand = {k: b[k].astype(np.float64) for k in b.files}
        yield d, fault, base, cand


def rmse_all(base, cand):
    out = {}
    for k in base:
        if k.startswith("__") or k not in cand:
            continue
        a, b = base[k], cand[k]
        out[k] = float("inf") if a.shape != b.shape else float(np.sqrt(np.mean((a - b) ** 2)))
    missing = set(base) ^ set(cand)
    if missing:
        out["__signal_set_mismatch__"] = float("inf")
    return out


def main() -> None:
    from ivf.faults import TAXONOMY

    cat = (
        {f.name: f.category for f in TAXONOMY.values()}
        if isinstance(TAXONOMY, dict)
        else {f.name: f.category for f in TAXONOMY}
    )
    clean = [rmse_all(b, c) for _, f, b, c in trials("calib") if f == "none"]
    keys = sorted({k for r in clean for k in r})
    thr = {k: max(FLOOR, MARGIN * max(r.get(k, 0.0) for r in clean)) for k in keys}
    rows = []
    for d, fault, base, cand in trials("eval"):
        v = json.loads((d / "verdict.json").read_text())
        oracles = json.loads((d / "oracles.json").read_text())
        validity = json.loads((d / "validity.json").read_text())
        r = rmse_all(base, cand)
        finite = all(np.isfinite(x).all() for k, x in cand.items() if not k.startswith("__"))
        first = sorted(
            [o for o in oracles if o["status"] == "fail" and o.get("divergence")],
            key=lambda o: o["divergence"].get("first_tolerance_violation_step") or 10**9,
        )
        cls = first[0]["divergence"].get("classification") if first else None
        rows.append(
            {
                "fault": fault,
                "category": cat.get(fault, "control"),
                "defect": fault != "none",
                "ivf": v["verdict"] != "PASS",
                "ivf_verdict": v["verdict"],
                "ivf_validity_failed": not validity["valid"],
                "ivf_classification": cls,
                "B1_native": not finite,
                "B3_pole_angle": r.get("pole_angle", 0.0) > thr.get("pole_angle", FLOOR),
                "B3all": any(r.get(k, 0.0) > thr.get(k, FLOOR) for k in r),
            }
        )
    res = {"thresholds": thr, "n_trials": len(rows)}
    dfx = [x for x in rows if x["defect"]]
    cln = [x for x in rows if not x["defect"]]
    for arm in ("ivf", "B1_native", "B3_pole_angle", "B3all"):
        res[arm] = {
            "recall": f"{sum(x[arm] for x in dfx)}/{len(dfx)}",
            "fpr": f"{sum(x[arm] for x in cln)}/{len(cln)}",
        }
    res["composite_B1_B3all"] = {
        "recall": f"{sum(x['B1_native'] or x['B3all'] for x in dfx)}/{len(dfx)}",
        "fpr": f"{sum(x['B1_native'] or x['B3all'] for x in cln)}/{len(cln)}",
    }
    res["ivf_only"] = sorted(
        {x["fault"] for x in dfx if x["ivf"] and not (x["B1_native"] or x["B3all"])}
    )
    res["baseline_only"] = sorted(
        {x["fault"] for x in dfx if not x["ivf"] and (x["B1_native"] or x["B3all"])}
    )
    res["both_miss"] = sorted(
        {x["fault"] for x in dfx if not x["ivf"] and not (x["B1_native"] or x["B3all"])}
    )
    loc = [
        x for x in dfx if x["ivf"] and not x["ivf_validity_failed"] and x["category"] in EXPECTED_CLASS
    ]
    res["ivf_classification_matches_category"] = (
        f"{sum(x['ivf_classification'] == EXPECTED_CLASS[x['category']] for x in loc)}/{len(loc)}"
    )
    res["per_fault"] = {
        f: {
            "ivf": sum(x["ivf"] for x in rows if x["fault"] == f),
            "B3all": sum(x["B3all"] for x in rows if x["fault"] == f),
            "B1": sum(x["B1_native"] for x in rows if x["fault"] == f),
            "classification": sorted({str(x["ivf_classification"]) for x in rows if x["fault"] == f}),
        }
        for f in sorted({x["fault"] for x in rows})
    }
    common.write_json(common.RESULTS / "synthetic" / "scores.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_fault", "thresholds")}, indent=1))


if __name__ == "__main__":
    main()
