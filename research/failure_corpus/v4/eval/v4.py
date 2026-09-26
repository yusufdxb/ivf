# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""v4 confirmatory evaluation: conventional validation vs IVF configuration/state checks vs union.

Frozen before the v4 holdout was authored. Subcommands (repository root,
``env -u PYTHONPATH uv run --frozen python``):

    v4.py ivf --split S [--workers N]     run IVF with the config/state-only manifest
    v4.py thresholds                      conventional thresholds (clean calibration only)
    v4.py score --split S                 primary analysis, once

IVF arm (pre-registered): a case is flagged if and only if at least one of these IVF
validity checks has status ``fail``:

    V-24 initial_state_realization   reset state actually applied (either subject)
    V-25 backend_identity            runtime backend is the declared one (either subject)
    V-28 solver_conformance          running solver settings vs intended preset (either subject)
    V-22 effective_model_parameters  realized masses, gains, armature, contact materials
    V-27 randomization_semantics     randomization terms, modes, targets, ranges
    V-20 joint_ordering              joint arrays in the same order
    V-21 policy_interface            observation layout, joint order, action scale fed to the policy
    V-23 termination_semantics       live termination terms, bodies, thresholds

No trajectory, event, decision, statistical, or invariant oracle contributes. Every other
IVF check (including the self-comparison guard and timing) is recorded but ignored.

Conventional arm: unchanged from v3 (native smoke OR closed-loop performance OR open-loop
trajectory RMSE; 1.25 x worst clean calibration value). Union = either.
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
import stats  # noqa: E402

V4 = common.RESEARCH / "v4"
CAPTURES = common.DATA / "captures_v3"          # calibration shared with v3; v4holdout is new
RESULTS = V4 / "results"
FROZEN = V4 / "FROZEN_V4.txt"
INTENDED = json.loads((V4 / "intended_presets.json").read_text())
CONFIG_CHECKS = {"V-20", "V-21", "V-22", "V-23", "V-24", "V-25", "V-27", "V-28"}
CONFIG_CONTROLS = ["joint_ordering", "policy_interface", "effective_model_parameters", "termination_semantics",
                   "initial_state_realization", "backend_identity", "randomization_semantics", "solver_conformance"]
CONTROL_TOL = 1e-4   # realized-value tolerance; every clean calibration pair in v3 sat at 0 (floor)
MARGIN, FLOOR = 1.25, 1e-6
PLATFORMS = ["go2", "g1", "h1", "anymal_d", "cassie", "spot", "go2_rough"]
BOOT = 10000


def plan_dir(split):
    return V4 / "plans" if split == "v4holdout" or split == "calibration_rough" else common.RESEARCH / "v3" / "plans"


def labels(split):
    p = V4 / "corpus" / f"labels_{split}.json"
    if not p.exists():
        p = common.RESEARCH / "v3" / "corpus" / f"labels_{split}.json"
    return json.loads(p.read_text())


def cases(split):
    labs = labels(split)
    out = []
    for lab in labs:
        if lab["is_reference"]:
            continue
        ref = next(x for x in labs if x["is_reference"] and x["platform"] == lab["platform"]
                   and x["seed"] == lab["seed"])
        out.append({**lab, "reference_id": ref["capture_id"]})
    return out


def bundle(split, cid):
    return CAPTURES / split / cid / "bundle"


def result(split, cid):
    p = CAPTURES / split / cid / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def expected_backend(split, cid):
    for plan in plan_dir(split).glob(f"{split}__*.json"):
        for job in json.loads(plan.read_text())["jobs"]:
            if job["capture_id"] == cid:
                return "physx" if job["backend_mode"] == "physx" else "newton"
    raise KeyError(cid)


# ----------------------------------------------------------------------------- IVF (config/state only)

def build_manifest(split, case):
    base, cand = bundle(split, case["reference_id"]), bundle(split, case["capture_id"])

    def lock(b):
        return json.loads((b / "COMPLETE").read_text())["checksums_sha256"]
    subj = {}
    for role, b, cid in (("baseline", base, case["reference_id"]), ("candidate", cand, case["capture_id"])):
        be = expected_backend(split, cid)
        subj[role] = {"kind": "parity_bundle", "label": role, "path": str(b), "bundle_sha256": lock(b),
                      "expect_backend": be}
        if be == "newton":
            subj[role]["expect_solver"] = dict(INTENDED[case["platform"]])
    tol = {"value": CONTROL_TOL, "unit": "dimensionless", "scope": "max realized difference per array",
           "rationale": "Every clean PhysX/Newton calibration pair in v3 agreed exactly (floor); fixed before v4.",
           "aggregation": "max", "min_samples": 1, "kind": "numerical"}
    doc = {"schema_version": "ivf.validation/v1",
           "name": f"v4-{case['capture_id'][:40]}".replace("__", "-"),
           "description": "v4 configuration/state-only IVF run; labels are not visible to IVF.",
           "subjects": subj,
           "workload": {"task": f"{case['platform']}_velocity_open_loop", "num_envs": 16, "steps": 250,
                        "warmup_steps": 0, "seeds": [int(case["seed"])]},
           "controls": {"require_same": CONFIG_CONTROLS, "allow_different": ["solver_specific_parameters"],
                        "tolerances": {"effective_model_parameters": tol, "initial_state_realization": tol}},
           # the schema needs one oracle; it cannot contribute (flag = config checks only) and never runs
           # when a config check fails
           "oracles": [{"type": "metamorphic", "name": "observation_shape_stability",
                        "relation": "observation_definition_stability"}]}
    return yaml.safe_dump(doc, sort_keys=False)


def run_one(split, case, evroot):
    from ivf.manifest import parse_manifest
    from ivf.runner import validate
    text = build_manifest(split, case)
    mp = Path(evroot) / "manifests" / f"{case['capture_id']}.yaml"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(text)
    res = validate(parse_manifest(text, source_path=str(mp)), results_root=evroot, seed=0)
    v = res.validity.to_jsonable() if res.validity is not None else {"checks": []}
    checks = [{k: c.get(k) for k in ("check_id", "name", "status", "reason_code", "detail", "fields")}
              for c in v.get("checks", [])]
    att = res.attribution or {"primary": None, "causes": []}
    shutil.rmtree(res.bundle_path, ignore_errors=True)
    return case["capture_id"], {"verdict": res.verdict.value, "reason_codes": list(res.reason_codes),
                                "error": res.error, "checks": checks,
                                "config_flag": any(c["status"] == "fail" and c["check_id"] in CONFIG_CHECKS
                                                   for c in checks),
                                "config_failed": sorted({c["check_id"] for c in checks
                                                         if c["status"] == "fail" and c["check_id"] in CONFIG_CHECKS}),
                                "attribution": {"primary": att.get("primary"), "causes": att.get("causes", [])}}


def frozen_guard():
    tree = subprocess.run(["git", "-C", str(common.REPO), "rev-parse", "HEAD:src/ivf"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(common.REPO), "status", "--porcelain", "--", "src/ivf"],
                           capture_output=True, text=True, check=True).stdout.strip()
    frozen = FROZEN.read_text().split()[0]
    if dirty or tree != frozen:
        raise SystemExit(f"src/ivf is not the frozen tree {frozen} (have {tree}, dirty={bool(dirty)})")


def cmd_ivf(args):
    frozen_guard()
    out = RESULTS / args.split / "ivf"
    evroot = str(common.DATA / "evidence_v4" / args.split)
    todo = [c for c in cases(args.split) if not (out / f"{c['capture_id']}.json").exists()
            and (bundle(args.split, c["capture_id"]) / "COMPLETE").exists()
            and (bundle(args.split, c["reference_id"]) / "COMPLETE").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run_one, args.split, c, evroot) for c in todo]):
            cid, s = f.result()
            common.write_json(out / f"{cid}.json", s)
    print(f"{args.split}: {len(todo)} run", flush=True)


# ----------------------------------------------------------------------------- conventional thresholds

def cmd_thresholds(args):
    raw: dict = {}
    for split in ("calibration", "calibration_rough"):
        try:
            cs = cases(split)
        except FileNotFoundError:
            continue
        for c in cs:
            if c["condition"] != "cross":
                continue
            ref, res = result(split, c["reference_id"]), result(split, c["capture_id"])
            e = raw.setdefault(c["platform"], {"perf": [], "b3": []})
            e["perf"].append(stats.perf_stats(ref, res))
            e["b3"].append(stats.trajectory_stats(stats.load_arrays(bundle(split, c["reference_id"])),
                                                  stats.load_arrays(bundle(split, c["capture_id"]))))
    th = {"rule": f"{MARGIN} x worst clean cross calibration value, floor {FLOOR}", "B2": {}, "B3": {}}
    for p, e in raw.items():
        th["B2"][p] = {k: max(FLOOR, MARGIN * max(x[k] for x in e["perf"])) for k in stats.PERF_KEYS}
        th["B3"][p] = {s: max(FLOOR, MARGIN * max(x[s] for x in e["b3"])) for s in stats.B3_SIGNALS}
        th.setdefault("n_pairs", {})[p] = len(e["perf"])
    common.write_json(RESULTS / "thresholds_v4.json", th)
    print(json.dumps(th["n_pairs"]))


def conventional(split, c, th):
    p = c["platform"]
    ref, res = result(split, c["reference_id"]), result(split, c["capture_id"])
    crashed = res is None or res["native"].get("exception") is not None
    row = {"B1": crashed or not res["native"].get("native_pass", False)}
    if crashed:
        row.update(B2=True, B3=True)
    else:
        perf = stats.perf_stats(ref, res)
        row["B2"] = any(perf[k] > th["B2"][p][k] for k in stats.PERF_KEYS)
        tr = stats.trajectory_stats(stats.load_arrays(bundle(split, c["reference_id"])),
                                    stats.load_arrays(bundle(split, c["capture_id"])))
        row["B3"] = any(tr[s] > th["B3"][p][s] for s in stats.B3_SIGNALS)
    row["conventional"] = bool(row["B1"] or row["B2"] or row["B3"])
    return row


# ----------------------------------------------------------------------------- statistics

def cluster_bootstrap(rows, key, stat, reps=BOOT, seed=0):
    """Percentile CI of ``stat(rows)`` resampling whole clusters (``key``) with replacement."""
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    keys = list(groups)
    vals = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [r for k in pick for r in groups[keys[k]]]
        vals[i] = stat(sample)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)], len(keys)


def two_way_bootstrap(rows, stat, reps=BOOT, seed=1):
    """Resample families and robots independently; keep cells in both resamples (with multiplicity)."""
    rng = np.random.default_rng(seed)
    fams = sorted({r["family"] for r in rows})
    robs = sorted({r["platform"] for r in rows})
    cell: dict = {}
    for r in rows:
        cell.setdefault((r["family"], r["platform"]), []).append(r)
    vals = []
    for _ in range(reps):
        fs = rng.choice(fams, len(fams))
        rs = rng.choice(robs, len(robs))
        sample = [r for f in fs for p in rs for r in cell.get((f, p), [])]
        if sample:
            vals.append(stat(sample))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def fa_reduction(sample):
    return float(np.mean([not r["conventional"] for r in sample]) - np.mean([not r["union"] for r in sample]))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cmd_score(args):
    th = json.loads((RESULTS / "thresholds_v4.json").read_text())
    rows = []
    for c in cases(args.split):
        r = {k: c[k] for k in ("capture_id", "platform", "seed", "family", "defect", "category")}
        r.update(conventional(args.split, c, th))
        sp = RESULTS / args.split / "ivf" / f"{c['capture_id']}.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            r["ivf"], r["ivf_checks"] = bool(s["config_flag"]), ";".join(s["config_failed"])
        else:  # no bundle (capture crashed): IVF cannot evaluate; the crash is loud via B1
            r["ivf"], r["ivf_checks"] = False, "no_capture"
        r["union"] = bool(r["ivf"] or r["conventional"])
        rows.append(r)
    out = RESULTS / args.split
    with (out / "cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    d = [r for r in rows if r["defect"]]
    cl = [r for r in rows if not r["defect"]]
    rep: dict = {"n_defect": len(d), "n_clean": len(cl),
                 "n_defect_families": len({r["family"] for r in d})}
    for arm in ("B1", "B2", "B3", "conventional", "ivf", "union"):
        k, kc = sum(r[arm] for r in d), sum(r[arm] for r in cl)
        rep[arm] = {"recall": f"{k}/{len(d)}", "recall_rate": round(k / len(d), 4),
                    "false_acceptance": round(1 - k / len(d), 4), "false_alarms": f"{kc}/{len(cl)}",
                    "fpr_wilson95": wilson(kc, len(cl))}
    red = fa_reduction(d)
    ci_fam, nfam = cluster_bootstrap(d, lambda r: r["family"], fa_reduction)
    ci_rob, nrob = cluster_bootstrap(d, lambda r: r["platform"], fa_reduction)
    ci_two = two_way_bootstrap(d, fa_reduction)
    added = sum(r["union"] for r in cl) - sum(r["conventional"] for r in cl)
    rep["primary"] = {
        "fa_conventional": round(1 - rep["conventional"]["recall_rate"], 4),
        "fa_union": round(1 - rep["union"]["recall_rate"], 4),
        "fa_reduction": round(red, 4),
        "ci95_family_cluster_bootstrap (primary)": ci_fam, "n_family_clusters": nfam,
        "ci95_robot_cluster_bootstrap (sensitivity)": ci_rob, "n_robot_clusters": nrob,
        "ci95_two_way_family_robot (sensitivity)": ci_two,
        "union_added_false_alarms": int(added),
        "criteria": {"reduction_ge_0.10": red >= 0.10, "ci_excludes_0": ci_fam[0] > 0, "added_fa_le_1": added <= 1},
    }
    rep["primary"]["claim_passes"] = all(rep["primary"]["criteria"].values())
    cells = {"both": 0, "ivf_only": 0, "conventional_only": 0, "neither": 0}
    by_cat: dict = {}
    for r in d:
        k = ("both" if r["ivf"] and r["conventional"] else "ivf_only" if r["ivf"]
             else "conventional_only" if r["conventional"] else "neither")
        cells[k] += 1
        by_cat.setdefault(r["category"], dict.fromkeys(cells, 0))[k] += 1
    rep["overlap"], rep["overlap_by_category"] = cells, by_cat
    rep["unique_catch_categories"] = {
        "ivf_only": _counts([r["category"] for r in d if r["ivf"] and not r["conventional"]]),
        "conventional_only": _counts([r["category"] for r in d if r["conventional"] and not r["ivf"]])}
    rep["ivf_checks_on_ivf_only"] = _counts([x for r in d if r["ivf"] and not r["conventional"]
                                             for x in r["ivf_checks"].split(";") if x])
    rep["ivf_false_alarm_checks"] = _counts([x for r in cl if r["ivf"] for x in r["ivf_checks"].split(";") if x])
    rep["per_family"] = {f: {"n": len(fr), "defect": fr[0]["defect"], "category": fr[0]["category"],
                             **{a: sum(r[a] for r in fr) for a in ("conventional", "ivf", "union")}}
                         for f in sorted({r["family"] for r in rows}) for fr in [[r for r in rows if r["family"] == f]]}
    rep["per_platform"] = {p: {a: f"{sum(r[a] for r in pr if r['defect'])}/{sum(r['defect'] for r in pr)} "
                                  f"fa {sum(r[a] for r in pr if not r['defect'])}/{sum(not r['defect'] for r in pr)}"
                               for a in ("conventional", "ivf", "union")}
                           for p in sorted({r["platform"] for r in rows}) for pr in [[r for r in rows if r["platform"] == p]]}
    common.write_json(out / "scores.json", rep)
    print(json.dumps({k: rep[k] for k in ("n_defect", "n_clean", "n_defect_families", "conventional", "ivf", "union",
                                          "primary", "overlap", "unique_catch_categories")}, indent=1))


def _counts(xs):
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ivf")
    a.add_argument("--split", required=True)
    a.add_argument("--workers", type=int, default=6)
    sub.add_parser("thresholds")
    s = sub.add_parser("score")
    s.add_argument("--split", required=True)
    args = ap.parse_args()
    {"ivf": cmd_ivf, "thresholds": cmd_thresholds, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
