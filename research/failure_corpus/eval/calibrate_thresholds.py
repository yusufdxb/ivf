# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Derive every threshold (IVF-cal, IVF-strict, B2, B3, B3-all) from clean calibration pairs only.

Rule (pre-registered): threshold = 1.25 x the worst value on clean calibration pairs of the
same platform and condition, floor 1e-6. Cross = PhysX reference vs Newton clean; same =
PhysX reference vs PhysX rerun. IVF-cal trajectory/event/decision/statistical tolerances use
the statistics IVF itself reported under the measurement-only ("permissive") manifest; where
IVF refuses to run (bit-identical PhysX reruns, V-19), the same statistic is computed here
with IVF's formula and the refusal is recorded.

Usage: env -u PYTHONPATH uv run --frozen python research/failure_corpus/eval/calibrate_thresholds.py
Prerequisite: run_ivf.py --split calibration --arm permissive
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import manifests
import stats

MARGIN, FLOOR = 1.25, 1e-6
PLATFORMS = ["go2", "g1", "h1", "anymal_d"]


def main() -> None:
    labels = common.load_labels("calibration")
    by_id = {lab["capture_id"]: lab for lab in labels}
    cases = common.evaluation_cases("calibration")
    raw: dict = {}
    notes: list[str] = []
    for case in cases:
        p, cond = case["platform"], case["condition"]
        ref = common.load_result("calibration", case["reference_id"])
        res = common.load_result("calibration", case["capture_id"])
        base_arr = stats.load_arrays(common.bundle_dir("calibration", case["reference_id"]))
        cand_arr = stats.load_arrays(common.bundle_dir("calibration", case["capture_id"]))
        summ = json.loads(
            (
                common.RESULTS
                / "calibration"
                / "ivf"
                / "permissive__full"
                / f"{case['capture_id']}.json"
            ).read_text()
        )
        entry = raw.setdefault(p, {}).setdefault(
            cond, {"ivf": [], "perf": [], "traj": [], "ivf_refused": []}
        )
        entry["perf"].append(stats.perf_stats(ref, res))
        entry["traj"].append(stats.trajectory_stats(base_arr, cand_arr))
        if summ["verdict"] == "INVALID_EXPERIMENT":
            entry["ivf_refused"].append(
                {"case": case["capture_id"], "reason_codes": summ["reason_codes"]}
            )
            m = {
                f"traj_{s}": stats.ivf_style_worst(s, base_arr[s], cand_arr[s])
                for s in manifests.TRAJ_SIGNALS
            }
            m.update(
                {
                    "fall_timing": 0,
                    "contact_timing": 0,
                    "survival_disagreement": 0.0,
                    "mean_base_height": float(
                        np.max(np.abs(base_arr["base_height"].mean(0) - cand_arr["base_height"].mean(0)))
                    ),
                }
            )
            m["computed_outside_ivf"] = True
        else:
            o = {x["name"]: x["metrics"] for x in summ["oracles"]}
            m = {f"traj_{s}": o[f"traj_{s}"]["worst_aggregated_error"] for s in manifests.TRAJ_SIGNALS}
            m["fall_timing"] = o["fall_timing"].get("worst_timing_delta_steps", 0)
            m["contact_timing"] = o["contact_timing"].get("worst_timing_delta_steps", 0)
            m["survival_disagreement"] = 1.0 - float(o["survival"].get("agreement_rate", 1.0))
            m["mean_base_height"] = max(
                abs(o["mean_base_height"]["ci_low"]), abs(o["mean_base_height"]["ci_high"])
            )
            m["event_occurrence_mismatch"] = {
                k: o[k].get("baseline_occurrences") != o[k].get("candidate_occurrences")
                for k in ("fall_timing", "contact_timing")
            }
            m["computed_outside_ivf"] = False
        entry["ivf"].append(m)
    del by_id

    def worst(vals):
        return max(vals) if vals else 0.0

    th = {
        "rule": f"threshold = {MARGIN} x worst clean calibration value, floor {FLOOR}",
        "cal": {},
        "strict": {},
        "B2": {},
        "B3": {},
        "B3all": {},
        "raw": raw,
        "notes": notes,
    }
    for p in PLATFORMS:
        # strict: from the three calibration PhysX reference captures of this platform
        refs = [lab for lab in labels if lab["is_reference"] and lab["platform"] == p]
        p95 = {s: [] for s in manifests.TRAJ_SIGNALS}
        for lab in refs:
            arr = stats.load_arrays(common.bundle_dir("calibration", lab["capture_id"]))
            for s in manifests.TRAJ_SIGNALS:
                p95[s].append(stats.per_step_change_p95(arr[s], s))
        strict_traj = {s: max(FLOOR, float(np.median(v))) for s, v in p95.items()}
        strict = {
            "traj": strict_traj,
            "event": {"fall_timing": 1, "contact_timing": 1},
            "decision": {"survival": 1.0},
            "statistical": {"mean_base_height": max(FLOOR, 0.5 * strict_traj["base_height"])},
        }
        for cond in ("cross", "same"):
            e = raw[p][cond]
            ivf = e["ivf"]
            th["cal"].setdefault(p, {})[cond] = {
                "traj": {
                    s: max(FLOOR, MARGIN * worst([m[f"traj_{s}"] for m in ivf]))
                    for s in manifests.TRAJ_SIGNALS
                },
                "event": {
                    k: max(1, math.ceil(MARGIN * worst([m[k] for m in ivf])))
                    for k in ("fall_timing", "contact_timing")
                },
                "decision": {
                    "survival": max(0.0, 1.0 - MARGIN * worst([m["survival_disagreement"] for m in ivf]))
                },
                "statistical": {
                    "mean_base_height": max(FLOOR, MARGIN * worst([m["mean_base_height"] for m in ivf]))
                },
            }
            th["strict"].setdefault(p, {})[cond] = strict
            th["B2"].setdefault(p, {})[cond] = {
                k: max(FLOOR, MARGIN * worst([x[k] for x in e["perf"]])) for k in stats.PERF_KEYS
            }
            th["B3"].setdefault(p, {})[cond] = {
                s: max(FLOOR, MARGIN * worst([x[s] for x in e["traj"]])) for s in stats.B3_SIGNALS
            }
            th["B3all"].setdefault(p, {})[cond] = {
                s: max(FLOOR, MARGIN * worst([x[s] for x in e["traj"]])) for s in stats.B3ALL_SIGNALS
            }
            mism = [
                m.get("event_occurrence_mismatch") for m in ivf if m.get("event_occurrence_mismatch")
            ]
            if any(any(v.values()) for v in mism):
                notes.append(
                    f"{p}/{cond}: an event occurrence mismatch on a clean calibration pair; "
                    "IVF's event oracle fails on any occurrence mismatch regardless of tolerance"
                )
            if e["ivf_refused"]:
                notes.append(
                    f"{p}/{cond}: IVF refused {len(e['ivf_refused'])} clean calibration pair(s) "
                    f"({sorted({c for r in e['ivf_refused'] for c in r['reason_codes']})}); "
                    "statistics computed outside IVF with IVF's formula"
                )
    common.write_json(common.RESULTS / "thresholds.json", th)
    for n in notes:
        print("NOTE", n)
    print("wrote", common.RESULTS / "thresholds.json")


if __name__ == "__main__":
    main()
