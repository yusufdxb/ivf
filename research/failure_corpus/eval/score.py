# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Join labels, baselines and IVF summaries; compute the pre-registered metrics.

Usage: env -u PYTHONPATH uv run --frozen python research/failure_corpus/eval/score.py --split dev
Writes results/<split>/cases.csv and results/<split>/scores.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import manifests  # noqa: E402
import stats  # noqa: E402

IVF_ARMS = ["cal__full", "strict__full"] + [f"cal__{a}" for a in manifests.ABLATIONS if a != "full"]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def rate(flags: list[bool]) -> dict[str, Any]:
    k, n = int(sum(flags)), len(flags)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict[str, Any]:
    """Two-sided exact McNemar on paired binary outcomes."""
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    b01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b10 + b01
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(b10, b01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b10, "b_only": b01, "p": round(p, 6)}


def stratified_bootstrap_diff(a: list[bool], b: list[bool], strata: list[str], reps: int = 10000,
                              seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    a_, b_ = np.array(a, float), np.array(b, float)
    groups = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    idx_groups = [np.array(v) for v in groups.values()]
    diffs = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in idx_groups])
        diffs[r] = a_[idx].mean() - b_[idx].mean()
    return [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]


def ivf_summary(split: str, arm: str, cid: str) -> dict[str, Any] | None:
    p = common.RESULTS / split / "ivf" / arm / f"{cid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def first_violation_step(s: dict[str, Any]) -> int | None:
    steps = []
    for o in s.get("oracles", []):
        if o["status"] == "fail" and o.get("divergence"):
            d = o["divergence"]
            st = d.get("first_tolerance_violation_step")
            if st is None:
                st = d.get("first_event_disagreement_step")
            if st is not None:
                steps.append(st)
    return min(steps) if steps else None


def evaluate_case(split: str, case: dict[str, Any], th: dict[str, Any], labels_by_block) -> dict[str, Any]:
    p, cond = case["platform"], ("same" if case["condition"] == "same" else "cross")
    ref = common.load_result(split, case["reference_id"])
    res = common.load_result(split, case["capture_id"])
    row: dict[str, Any] = {k: case[k] for k in ("capture_id", "platform", "seed", "condition", "family",
                                                "variant", "defect", "category")}
    crashed = res is None or res["native"].get("exception") is not None
    row["loud_crash"] = crashed
    row["B1_native"] = crashed or not res["native"].get("native_pass", False)
    if crashed:
        for k in ("B2_perf", "B3_traj", "B3all_traj", "B4_config"):
            row[k] = True
        row["consequential"] = True
    else:
        perf = stats.perf_stats(ref, res)
        row.update({f"perf_{k}": round(v, 5) for k, v in perf.items()})
        row["B2_perf"] = any(perf[k] > th["B2"][p][cond][k] for k in stats.PERF_KEYS)
        base_arr = stats.load_arrays(common.bundle_dir(split, case["reference_id"]))
        cand_arr = stats.load_arrays(common.bundle_dir(split, case["capture_id"]))
        tr = stats.trajectory_stats(base_arr, cand_arr)
        row.update({f"rmse_{k}": round(v, 6) for k, v in tr.items()})
        row["B3_traj"] = any(tr[s] > th["B3"][p][cond][s] for s in stats.B3_SIGNALS)
        row["B3all_traj"] = any(tr[s] > th["B3all"][p][cond][s] for s in stats.B3ALL_SIGNALS)
        meta = lambda cid: json.loads((common.bundle_dir(split, cid) / "metadata.json").read_text())  # noqa: E731
        row["B4_config"] = (meta(case["reference_id"])["capture_contract"]["task"]["full_cfg_digest_sha256"]
                            != meta(case["capture_id"])["capture_contract"]["task"]["full_cfg_digest_sha256"])
        # consequential: closed-loop outcome differs from the clean counterpart beyond the B2 threshold
        clean_key = "clean_newton" if cond == "cross" else "clean_physx_rerun"
        clean = labels_by_block.get((case["platform"], case["seed"], clean_key))
        if case["defect"] and clean is not None:
            cres = common.load_result(split, clean)
            cperf = stats.perf_stats(cres, res)
            row["consequential"] = any(cperf[k] > th["B2"][p][cond][k] for k in stats.PERF_KEYS)
        else:
            row["consequential"] = None
    row["composite"] = bool(row["B1_native"] or row["B2_perf"] or row["B3_traj"])
    row["composite_all"] = bool(row["B1_native"] or row["B2_perf"] or row["B3all_traj"])
    for arm in IVF_ARMS:
        s = ivf_summary(split, arm, case["capture_id"])
        if s is None:
            row[f"ivf_{arm}"] = None
            continue
        row[f"ivf_{arm}"] = s["verdict"] != "PASS" if s["verdict"] != "NO_CAPTURE" else True
        if arm == "cal__full":
            row["ivf_verdict"] = s["verdict"]
            row["ivf_reason_codes"] = ";".join(s.get("reason_codes", [])[:6])
            row["ivf_loc_category"] = s["localization"]["category"]
            row["ivf_loc_basis"] = s["localization"].get("basis")
            row["ivf_first_violation_step"] = first_violation_step(s)
            row["ivf_selfcmp_only"] = s.get("reason_codes") == ["IVF-EXPERIMENT-SELF-COMPARISON"]
            row["ivf_cal_sens"] = row[f"ivf_{arm}"] and not row["ivf_selfcmp_only"]
        if arm == "strict__full":
            row["ivf_strict_verdict"] = s["verdict"]
    return row


ARMS_REPORTED = ["B1_native", "B2_perf", "B3_traj", "B3all_traj", "B4_config", "composite", "composite_all",
                 "ivf_cal__full", "ivf_cal_sens", "ivf_strict__full"] + [f"ivf_cal__{a}" for a in manifests.ABLATIONS
                                                                          if a != "full"]


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cond in ("cross", "same"):
        rs = [r for r in rows if r["condition"] == cond]
        dfx = [r for r in rs if r["defect"]]
        cln = [r for r in rs if not r["defect"]]
        block: dict[str, Any] = {"n_defect": len(dfx), "n_clean": len(cln)}
        for arm in ARMS_REPORTED:
            if any(r.get(arm) is None for r in rs):
                continue
            block[arm] = {"recall": rate([bool(r[arm]) for r in dfx]),
                          "fpr": rate([bool(r[arm]) for r in cln]),
                          "false_acceptance": rate([not r[arm] for r in dfx])}
        if dfx and all(r.get("ivf_cal__full") is not None for r in dfx):
            a = [bool(r["ivf_cal__full"]) for r in dfx]
            b = [bool(r["composite"]) for r in dfx]
            fam = [r["family"] for r in dfx]
            block["primary_recall_diff_ivfcal_minus_composite"] = {
                "diff": round(float(np.mean(a) - np.mean(b)), 4),
                "ci95_stratified_bootstrap": stratified_bootstrap_diff(a, b, fam),
                "mcnemar": mcnemar_exact(a, b)}
            acc = [r for r in dfx if not r["composite"]]
            block["ivf_flags_among_composite_accepted"] = rate([bool(r["ivf_cal__full"]) for r in acc])
            acc_all = [r for r in dfx if not r["composite_all"]]
            block["ivf_flags_among_composite_all_accepted"] = rate([bool(r["ivf_cal__full"]) for r in acc_all])
            comp_only = [r for r in dfx if r["composite"] and not r["ivf_cal__full"]]
            block["composite_flags_ivf_misses"] = sorted(f"{r['platform']}/{r['family']}" for r in comp_only)
        if cln and all(r.get("ivf_cal__full") is not None for r in cln):
            a = [bool(r["ivf_cal__full"]) for r in cln]
            b = [bool(r["composite"]) for r in cln]
            block["fpr_diff_ivfcal_minus_composite"] = {
                "diff": round(float(np.mean(a) - np.mean(b)), 4),
                "ci95_bootstrap": stratified_bootstrap_diff(a, b, [r["platform"] for r in cln])}
        det = [r for r in dfx if r.get("ivf_cal__full")]
        if det:
            correct = [r["ivf_loc_category"] == r["category"] for r in det]
            cats = [r["category"] for r in dfx]
            majority = max(set(cats), key=cats.count)
            block["localization"] = {
                "top1": rate(correct),
                "majority_category": majority,
                "majority_guess_accuracy_on_flagged": round(float(np.mean([r["category"] == majority for r in det])), 4),
                "predicted_categories": {c: sum(1 for r in det if r["ivf_loc_category"] == c)
                                         for c in sorted({r["ivf_loc_category"] for r in det})},
            }
            onset = [r["ivf_first_violation_step"] for r in det if r.get("ivf_first_violation_step") is not None]
            block["onset_frac_first_violation_le_1"] = round(float(np.mean([s <= 1 for s in onset])), 4) if onset else None
        fam_tab = {}
        for fam in sorted({r["family"] for r in rs}):
            fr = [r for r in rs if r["family"] == fam]
            fam_tab[fam] = {"n": len(fr), "defect": fr[0]["defect"],
                            **{arm: int(sum(bool(r.get(arm)) for r in fr)) for arm in
                               ("B1_native", "B2_perf", "B3_traj", "B3all_traj", "B4_config", "composite",
                                "ivf_cal__full", "ivf_strict__full")},
                            "consequential": int(sum(bool(r.get("consequential")) for r in fr)),
                            "ivf_loc_correct": int(sum(1 for r in fr if r.get("ivf_cal__full")
                                                       and r.get("ivf_loc_category") == r["category"]))}
        block["per_family"] = fam_tab
        out[cond] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    args = ap.parse_args()
    th = json.loads((common.RESULTS / "thresholds.json").read_text())
    labels = common.load_labels(args.split)
    by_block = {}
    for lab in labels:
        tail = lab["capture_id"].split("__")[-1] if args.split != "holdout" else None
        key = lab["family"] if lab["family"] != "none" else None
        if lab["family"] == "none" and not lab["is_reference"]:
            key = "clean_newton" if lab["condition"] == "cross" else "clean_physx_rerun"
        if key:
            by_block[(lab["platform"], lab["seed"], key)] = lab["capture_id"]
        del tail
    rows = [evaluate_case(args.split, c, th, by_block) for c in common.evaluation_cases(args.split)]
    out_dir = common.RESULTS / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with (out_dir / "cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    result = score(rows)
    common.write_json(out_dir / "scores.json", result)
    for cond, blk in result.items():
        print(f"== {args.split} {cond}: {blk['n_defect']} defect, {blk['n_clean']} clean")
        for arm in ARMS_REPORTED:
            if arm in blk:
                b = blk[arm]
                print(f"  {arm:28s} recall {b['recall']['k']}/{b['recall']['n']}  fpr {b['fpr']['k']}/{b['fpr']['n']}")
        for k in ("primary_recall_diff_ivfcal_minus_composite", "ivf_flags_among_composite_accepted",
                  "fpr_diff_ivfcal_minus_composite", "localization", "onset_frac_first_violation_le_1"):
            if k in blk:
                print(f"  {k}: {json.dumps(blk[k])}")


if __name__ == "__main__":
    main()
