# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""v2 evaluation: revised IVF vs conventional sim-to-sim validation vs their union.

Subcommands (run from the repository root with ``env -u PYTHONPATH uv run --frozen python``):

    v2.py ivf        --split S [--arm permissive|cal] [--workers N]
    v2.py calibrate                       # thresholds from clean calibration pairs only
    v2.py score      --split S

Differences from v1 (``research/failure_corpus/eval``): captures come from the v2 producer
(``captures_v2``); the manifest additionally requires the experiment-input controls, each
subject declares ``expect_backend`` (the backend the experimenter asked for, from the plan,
not from the label), the event oracles carry a calibrated occurrence allowance, and root
cause is read from IVF's evidence attribution instead of the divergence classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
V1 = HERE.parents[1] / "eval"
sys.path.insert(0, str(V1))
import common  # noqa: E402
import manifests as m1  # noqa: E402
import stats  # noqa: E402
from score import mcnemar_exact, rate, stratified_bootstrap_diff  # noqa: E402

CAPTURES = common.DATA / "captures_v2"
RESULTS = common.RESEARCH / "v2" / "results"
MARGIN, FLOOR = 1.25, 1e-6
INPUT_CONTROLS = ["joint_ordering", "policy_interface", "effective_model_parameters", "termination_semantics",
                  "initial_state_realization", "backend_identity", "resource_health"]
REALIZATION_FLOOR = 1e-4   # absolute, rad / (m or rad)/s; float32 readback noise is far below
MODEL_FLOOR = 1e-4         # relative
FROZEN_V2_TREE_FILE = common.RESEARCH / "v2" / "FROZEN_IVF_V2.txt"

#: IVF attribution cause -> corpus category (fixed before any v2 evaluation)
CAUSE_CATEGORY = {
    "backend_identity": "ineffective_stale_config",
    "solver_capacity": "actuator_solver_config",
    "initial_state": "reset_randomization",
    "joint_order": "joint_body_ordering",
    "policy_interface.joint_order": "joint_body_ordering",
    "policy_interface.observation_layout": "obs_action_ordering",
    "policy_interface.action_scale": "checkpoint_schema",
    "policy_interface.other": "obs_action_ordering",
    "termination_semantics": "termination_metric",
    "model_parameters.actuation": "actuator_solver_config",
    "model_parameters.mass_properties": "reset_randomization",
    "timing": "timestep_decimation",
}


def bundle(split: str, cid: str) -> Path:
    return CAPTURES / split / cid / "bundle"


def result(split: str, cid: str) -> dict[str, Any] | None:
    p = CAPTURES / split / cid / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def expected_backend(split: str, cid: str) -> str:
    plan_dir = common.RESEARCH / ("plans" if split in ("calibration", "dev", "holdout") else "v2/plans")
    for plan in plan_dir.glob(f"{split}__*.json"):
        for job in json.loads(plan.read_text())["jobs"]:
            if job["capture_id"] == cid:
                return "physx" if job["backend_mode"] == "physx" else "newton"
    raise KeyError(cid)


def labels(split: str) -> list[dict[str, Any]]:
    p = common.RESEARCH / "corpus" / f"labels_{split}.json"
    if not p.exists():
        p = common.RESEARCH / "v2" / "corpus" / f"labels_{split}.json"
    return json.loads(p.read_text())


def cases(split: str) -> list[dict[str, Any]]:
    labs = labels(split)
    out = []
    for lab in labs:
        if lab["is_reference"]:
            continue
        ref = next(x for x in labs if x["is_reference"] and x["platform"] == lab["platform"]
                   and x["seed"] == lab["seed"])
        out.append({**lab, "reference_id": ref["capture_id"]})
    return out


# ------------------------------------------------------------------------------ manifest

def build_manifest(split: str, case: dict[str, Any], tol: dict[str, Any], arm: str) -> str:
    base, cand = bundle(split, case["reference_id"]), bundle(split, case["capture_id"])
    text = m1.build_manifest(case, base, cand, tol, arm="permissive" if arm == "permissive" else "cal")
    doc = yaml.safe_load(text)
    doc["subjects"]["baseline"]["expect_backend"] = expected_backend(split, case["reference_id"])
    doc["subjects"]["candidate"]["expect_backend"] = expected_backend(split, case["capture_id"])
    doc["controls"]["require_same"] = list(doc["controls"]["require_same"]) + INPUT_CONTROLS
    why = ("Calibrated from clean calibration pairs (1.25 x worst, floored); realized values of clean "
           "PhysX and Newton runs agree to this level.")
    doc["controls"]["tolerances"] = {
        "effective_model_parameters": {"value": float(tol["inputs"]["effective_model_parameters"]),
                                       "unit": "dimensionless", "scope": "max relative difference per array",
                                       "rationale": why, "aggregation": "max", "min_samples": 1,
                                       "kind": "numerical"},
        "initial_state_realization": {"value": float(tol["inputs"]["initial_state_realization"]),
                                      "unit": "dimensionless",
                                      "scope": "max absolute requested-vs-readback difference (rad, m/s, rad/s)",
                                      "rationale": why, "aggregation": "max", "min_samples": 1,
                                      "kind": "numerical"},
    }
    for o in doc["oracles"]:
        if o["type"] == "event_equivalence":
            o["max_occurrence_mismatch_fraction"] = float(tol["event_allowance"][o["name"]])
    doc["name"] = doc["name"].replace("-cal-", "-v2-").replace("-permissive-", "-v2perm-")
    return yaml.safe_dump(doc, sort_keys=False)


def permissive() -> dict[str, Any]:
    t = m1.permissive_tolerances()
    t["inputs"] = {"effective_model_parameters": 1e9, "initial_state_realization": 1e9}
    t["event_allowance"] = {"fall_timing": 1.0, "contact_timing": 1.0}
    return t


# ------------------------------------------------------------------------------ run IVF

def frozen_guard(permissive_ok: bool) -> None:
    tree = subprocess.run(["git", "-C", str(common.REPO), "rev-parse", "HEAD:src/ivf"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(common.REPO), "status", "--porcelain", "--", "src/ivf"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit("src/ivf has uncommitted changes; refusing")
    if FROZEN_V2_TREE_FILE.exists():
        frozen = FROZEN_V2_TREE_FILE.read_text().split()[0]
        if tree != frozen:
            raise SystemExit(f"src/ivf tree {tree} is not the frozen v2 tree {frozen}; refusing")
    elif not permissive_ok:
        print(f"WARNING: v2 IVF not frozen yet (tree {tree}); development run", flush=True)


def summarize(res, oracle_order):
    from localize import localize as loc_v1
    validity = res.validity.to_jsonable() if res.validity is not None else {"checks": []}
    oracles = [o.to_jsonable() for o in res.outcomes]
    att = res.attribution or {"primary": None, "causes": [], "findings": []}
    return {
        "verdict": res.verdict.value, "reason_codes": list(res.reason_codes), "error": res.error,
        "validity": {"valid": validity.get("valid"),
                     "checks": [{k: c.get(k) for k in ("check_id", "name", "status", "reason_code", "detail",
                                                       "fields")} for c in validity.get("checks", [])]},
        "oracles": [{"name": o["name"], "status": o["status"], "reason_codes": o.get("reason_codes", []),
                     "metrics": {k: v for k, v in (o.get("metrics") or {}).items()
                                 if not isinstance(v, list) or len(v) <= 32}} for o in oracles],
        "attribution": {"primary": att["primary"], "causes": att["causes"],
                        "findings": [{k: f.get(k) for k in ("cause", "evidence", "reason_code", "fields")}
                                     for f in att["findings"]]},
        "v1_localization": loc_v1(validity, oracles, oracle_order),
    }


def run_one(split, case, tol, arm, evroot):
    from ivf.manifest import parse_manifest
    from ivf.runner import validate
    if not (bundle(split, case["capture_id"]) / "COMPLETE").exists():
        return case["capture_id"], {"verdict": "NO_CAPTURE", "attribution": {"primary": None, "causes": []}}
    text = build_manifest(split, case, tol, arm)
    mp = Path(evroot) / "manifests" / f"{case['capture_id']}.yaml"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(text)
    res = validate(parse_manifest(text, source_path=str(mp)), results_root=evroot, seed=0)
    s = summarize(res, m1.ALL_ORACLES)
    shutil.rmtree(res.bundle_path, ignore_errors=True)  # summaries are the committed record
    return case["capture_id"], s


def cmd_ivf(args):
    frozen_guard(args.arm == "permissive")
    th = {} if args.arm == "permissive" else json.loads((RESULTS / "thresholds_v2.json").read_text())
    out = RESULTS / args.split / "ivf" / args.arm
    evroot = str(common.DATA / "evidence_v2" / args.split / args.arm)
    todo = [c for c in cases(args.split) if not (out / f"{c['capture_id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = []
        for c in todo:
            cond = "same" if c["condition"] == "same" else "cross"
            tol = permissive() if args.arm == "permissive" else th["ivf"][c["platform"]][cond]
            futs.append(pool.submit(run_one, args.split, c, tol, args.arm, evroot))
        for f in as_completed(futs):
            cid, s = f.result()
            common.write_json(out / f"{cid}.json", s)
    print(f"{args.split} {args.arm}: {len(todo)} run", flush=True)


# ------------------------------------------------------------------------------ calibrate

def cmd_calibrate(args):
    raw: dict = {}
    for c in cases("calibration"):
        p, cond = c["platform"], c["condition"]
        s = json.loads((RESULTS / "calibration" / "ivf" / "permissive" / f"{c['capture_id']}.json").read_text())
        ref, res = result("calibration", c["reference_id"]), result("calibration", c["capture_id"])
        ba, ca = stats.load_arrays(bundle("calibration", c["reference_id"])), stats.load_arrays(
            bundle("calibration", c["capture_id"]))
        e = raw.setdefault(p, {}).setdefault(cond, {"traj": [], "perf": [], "b3": [], "ev": [], "stat": [],
                                                    "surv": [], "model": [], "real": [], "refused": []})
        e["perf"].append(stats.perf_stats(ref, res))
        e["b3"].append(stats.trajectory_stats(ba, ca))
        refused = s["verdict"] == "INVALID_EXPERIMENT"
        if refused:
            e["refused"].append(s["reason_codes"])
            e["traj"].append({sg: stats.ivf_style_worst(sg, ba[sg], ca[sg]) for sg in m1.TRAJ_SIGNALS})
            e["ev"].append({"fall_timing": (0, 0.0), "contact_timing": (0, 0.0)})
            e["stat"].append(float(np.max(np.abs(ba["base_height"].mean(0) - ca["base_height"].mean(0)))))
            e["surv"].append(0.0)
        else:
            o = {x["name"]: x["metrics"] for x in s["oracles"]}
            e["traj"].append({sg: o[f"traj_{sg}"]["worst_aggregated_error"] for sg in m1.TRAJ_SIGNALS})
            e["ev"].append({k: (o[k].get("worst_timing_delta_steps", 0), o[k].get("occurrence_mismatch_fraction", 0.0))
                            for k in ("fall_timing", "contact_timing")})
            e["stat"].append(max(abs(o["mean_base_height"]["ci_low"]), abs(o["mean_base_height"]["ci_high"])))
            e["surv"].append(1.0 - float(o["survival"].get("agreement_rate", 1.0)))
        # experiment-input realized values, read straight from the bundles
        a_in = json.loads((bundle("calibration", c["reference_id"]) / "metadata.json").read_text())[
            "capture_contract"]["experiment_inputs"]
        b_in = json.loads((bundle("calibration", c["capture_id"]) / "metadata.json").read_text())[
            "capture_contract"]["experiment_inputs"]
        from ivf.experiment_inputs import _max_rel_diff
        am, bm = a_in["effective_model"], b_in["effective_model"]
        e["model"].append(max([_max_rel_diff(am[k], bm[k]) for k in am if not k.endswith("_names")] + [0.0]))
        for inp in (a_in, b_in):
            rr = inp["reset_realization"]
            e["real"].append(max(float(np.max(np.abs(np.asarray(rr["requested"][k]) - np.asarray(rr["readback"][k]))))
                                 for k in rr["requested"]))

    def worst(v):
        return max(v) if v else 0.0
    th: dict = {"rule": f"{MARGIN} x worst clean calibration value, floored", "ivf": {}, "B2": {}, "B3": {},
                "raw_notes": {}}
    for p, conds in raw.items():
        for cond, e in conds.items():
            th["ivf"].setdefault(p, {})[cond] = {
                "traj": {sg: max(FLOOR, MARGIN * worst([x[sg] for x in e["traj"]])) for sg in m1.TRAJ_SIGNALS},
                "event": {k: max(1, math.ceil(MARGIN * worst([x[k][0] for x in e["ev"]])))
                          for k in ("fall_timing", "contact_timing")},
                "event_allowance": {k: min(1.0, MARGIN * worst([x[k][1] for x in e["ev"]]))
                                    for k in ("fall_timing", "contact_timing")},
                "decision": {"survival": max(0.0, 1.0 - MARGIN * worst(e["surv"]))},
                "statistical": {"mean_base_height": max(FLOOR, MARGIN * worst(e["stat"]))},
                "inputs": {"effective_model_parameters": max(MODEL_FLOOR, MARGIN * worst(e["model"])),
                           "initial_state_realization": max(REALIZATION_FLOOR, MARGIN * worst(e["real"]))},
            }
            th["B2"].setdefault(p, {})[cond] = {k: max(FLOOR, MARGIN * worst([x[k] for x in e["perf"]]))
                                                for k in stats.PERF_KEYS}
            th["B3"].setdefault(p, {})[cond] = {sg: max(FLOOR, MARGIN * worst([x[sg] for x in e["b3"]]))
                                                for sg in stats.B3_SIGNALS}
            th["raw_notes"].setdefault(p, {})[cond] = {"ivf_refused": e["refused"], "model_rel": e["model"],
                                                       "realization": e["real"]}
    common.write_json(RESULTS / "thresholds_v2.json", th)
    for p in th["ivf"]:
        for cond in th["ivf"][p]:
            t = th["ivf"][p][cond]
            print(p, cond, "event_allowance", t["event_allowance"], "inputs", t["inputs"], "surv", t["decision"])


# ------------------------------------------------------------------------------ score

def conventional(split, c, th):
    p, cond = c["platform"], ("same" if c["condition"] == "same" else "cross")
    ref, res = result(split, c["reference_id"]), result(split, c["capture_id"])
    crashed = res is None or res["native"].get("exception") is not None
    row = {"B1": crashed or not res["native"].get("native_pass", False)}
    if crashed:
        row.update(B2=True, B3=True)
    else:
        perf = stats.perf_stats(ref, res)
        row["B2"] = any(perf[k] > th["B2"][p][cond][k] for k in stats.PERF_KEYS)
        tr = stats.trajectory_stats(stats.load_arrays(bundle(split, c["reference_id"])),
                                    stats.load_arrays(bundle(split, c["capture_id"])))
        row["B3"] = any(tr[s] > th["B3"][p][cond][s] for s in stats.B3_SIGNALS)
    row["conventional"] = bool(row["B1"] or row["B2"] or row["B3"])
    return row


def cmd_score(args):
    th = json.loads((RESULTS / "thresholds_v2.json").read_text())
    rows = []
    for c in cases(args.split):
        r = {k: c[k] for k in ("capture_id", "platform", "seed", "condition", "family", "defect", "category")}
        r.update(conventional(args.split, c, th))
        s = json.loads((RESULTS / args.split / "ivf" / "cal" / f"{c['capture_id']}.json").read_text())
        r["ivf_verdict"], r["ivf_codes"] = s["verdict"], ";".join(s.get("reason_codes", [])[:5])
        r["ivf"] = s["verdict"] != "PASS"
        r["union"] = bool(r["ivf"] or r["conventional"])
        prim = s["attribution"]["primary"]
        r["ivf_cause"] = prim
        r["ivf_causes"] = ";".join(s["attribution"]["causes"])
        r["ivf_category"] = CAUSE_CATEGORY.get(prim, "unattributed" if prim else "none")
        r["ivf_category_set"] = ";".join(sorted({CAUSE_CATEGORY.get(x, "unattributed")
                                                 for x in s["attribution"]["causes"]}))
        r["ivf_v1_category"] = s["v1_localization"]["category"]
        rows.append(r)
    out = RESULTS / args.split
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    report = {}
    for cond in ("cross", "same"):
        rs = [r for r in rows if r["condition"] == cond]
        d, cl = [r for r in rs if r["defect"]], [r for r in rs if not r["defect"]]
        if not rs:
            continue
        blk = {"n_defect": len(d), "n_clean": len(cl)}
        for arm in ("B1", "B2", "B3", "conventional", "ivf", "union"):
            blk[arm] = {"recall": rate([r[arm] for r in d]), "fpr": rate([r[arm] for r in cl]),
                        "false_acceptance": rate([not r[arm] for r in d]),
                        "flag_rate_all": round(float(np.mean([r[arm] for r in rs])), 4)}
        if d:
            a, b, u = [r["ivf"] for r in d], [r["conventional"] for r in d], [r["union"] for r in d]
            fam = [r["family"] for r in d]
            blk["ivf_minus_conventional_recall"] = {"diff": round(float(np.mean(a) - np.mean(b)), 4),
                                                    "ci95": stratified_bootstrap_diff(a, b, fam),
                                                    "mcnemar": mcnemar_exact(a, b)}
            blk["union_minus_conventional_recall"] = {"diff": round(float(np.mean(u) - np.mean(b)), 4),
                                                      "ci95": stratified_bootstrap_diff(u, b, fam)}
            blk["union_minus_ivf_recall"] = {"diff": round(float(np.mean(u) - np.mean(a)), 4),
                                             "ci95": stratified_bootstrap_diff(u, a, fam)}
            cells = {"both": 0, "ivf_only": 0, "conventional_only": 0, "neither": 0}
            by_cat: dict = {}
            for r in d:
                k = ("both" if r["ivf"] and r["conventional"] else "ivf_only" if r["ivf"]
                     else "conventional_only" if r["conventional"] else "neither")
                cells[k] += 1
                by_cat.setdefault(r["category"], dict.fromkeys(cells, 0))[k] += 1
            blk["overlap"], blk["overlap_by_category"] = cells, by_cat
            det = [r for r in d if r["ivf"]]
            if det:
                cats = [r["category"] for r in d]
                maj = max(set(cats), key=cats.count)
                blk["localization"] = {
                    "top1": rate([r["ivf_category"] == r["category"] for r in det]),
                    "set_contains_truth": rate([r["category"] in r["ivf_category_set"].split(";") for r in det]),
                    "unattributed": rate([r["ivf_category"] == "unattributed" for r in det]),
                    "majority_category": maj,
                    "majority_guess": round(float(np.mean([r["category"] == maj for r in det])), 4),
                    "v1_classifier_top1": rate([r["ivf_v1_category"] == r["category"] for r in det]),
                }
        blk["per_family"] = {f: {"n": len(fr), "defect": fr[0]["defect"],
                                 **{a: sum(bool(r[a]) for r in fr) for a in ("conventional", "ivf", "union")},
                                 "loc_correct": sum(1 for r in fr if r["ivf"] and r["ivf_category"] == r["category"])}
                             for f in sorted({r["family"] for r in rs})
                             for fr in [[r for r in rs if r["family"] == f]]}
        blk["per_platform"] = {p: {a: f"{sum(r[a] for r in pr if r['defect'])}/{sum(r['defect'] for r in pr)} "
                                      f"fp {sum(r[a] for r in pr if not r['defect'])}/{sum(not r['defect'] for r in pr)}"
                                   for a in ("conventional", "ivf", "union")}
                               for p in sorted({r["platform"] for r in rs})
                               for pr in [[r for r in rs if r["platform"] == p]]}
        report[cond] = blk
    common.write_json(out / "scores.json", report)
    for cond, blk in report.items():
        print(f"== {args.split} {cond}: {blk['n_defect']} defect, {blk['n_clean']} clean")
        for arm in ("B1", "B2", "B3", "conventional", "ivf", "union"):
            b = blk[arm]
            print(f"  {arm:13s} recall {b['recall']['k']}/{b['recall']['n']}  fpr {b['fpr']['k']}/{b['fpr']['n']}"
                  f"  FA {b['false_acceptance']['rate']}")
        for k in ("ivf_minus_conventional_recall", "union_minus_conventional_recall", "overlap", "localization"):
            if k in blk:
                print(f"  {k}: {json.dumps(blk[k])}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ivf")
    a.add_argument("--split", required=True)
    a.add_argument("--arm", default="cal", choices=["cal", "permissive"])
    a.add_argument("--workers", type=int, default=6)
    sub.add_parser("calibrate")
    s = sub.add_parser("score")
    s.add_argument("--split", required=True)
    args = ap.parse_args()
    {"ivf": cmd_ivf, "calibrate": cmd_calibrate, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
