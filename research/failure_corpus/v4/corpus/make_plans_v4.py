# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""v4 plans: Rough GO2 clean calibration (seeds 0-2) and the v4 holdout (seeds 15-16).

Usage: python make_plans_v4.py calibration_rough | holdout
"""
import hashlib
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
PLATFORMS = ["go2", "g1", "h1", "anymal_d", "cassie", "spot", "go2_rough"]
POLICY = "research/failure_corpus/policies/{p}/policy.pt"


def job(plans, labels, split, p, seed, cid, mode, fam, params, cond, defect, category, is_ref, sources=()):
    plans.setdefault(p, []).append({"capture_id": cid, "platform": p, "seed": seed, "backend_mode": mode,
                                    "fault": {"family": fam, "params": params}, "replay_ref": f"{p}:{seed}",
                                    "is_reference": is_ref})
    labels.append({"capture_id": cid, "split": split, "platform": p, "seed": seed, "condition": cond,
                   "family": fam, "defect": defect, "category": category, "is_reference": is_ref,
                   "sources": list(sources)})


def write(split, plans, labels, fault_module=None):
    for p, jobs in plans.items():
        plan = {"split": split, "platform": p, "policy": POLICY.format(p=p), "jobs": jobs}
        if fault_module:
            plan["fault_module"] = fault_module
        (V4 / "plans" / f"{split}__{p}.json").write_text(json.dumps(plan, indent=1) + "\n")
    (V4 / "corpus" / f"labels_{split}.json").write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")


def calibration_rough():
    plans, labels = {}, []
    for seed in (0, 1, 2):
        b = f"calibration_rough__go2_rough__s{seed}"
        for key, mode, cond, ref in (("ref_physx", "physx", "reference", True),
                                     ("clean_physx_rerun", "physx", "same", False),
                                     ("clean_newton", "newton", "cross", False)):
            job(plans, labels, "calibration_rough", "go2_rough", seed, f"{b}__{key}", mode, "none", {}, cond,
                False, "none", ref)
    write("calibration_rough", plans, labels)


def holdout():
    cases = json.loads((V4 / "holdout" / "fault_cases.json").read_text())
    plans, labels = {}, []

    def cid(key):
        return "v4h_" + hashlib.sha256(f"ivf-v4-holdout::{key}".encode()).hexdigest()[:12]
    for p in PLATFORMS:
        for seed in (15, 16):
            job(plans, labels, "v4holdout", p, seed, cid(f"{p}:{seed}:ref"), "physx", "none", {}, "reference",
                False, "none", True)
            job(plans, labels, "v4holdout", p, seed, cid(f"{p}:{seed}:clean"), "newton", "none", {}, "cross",
                False, "none", False)
            for fam in cases:
                if p not in fam["params"]:
                    continue
                job(plans, labels, "v4holdout", p, seed, cid(f"{p}:{seed}:{fam['family']}"), "newton",
                    fam["family"], fam["params"][p], "cross", bool(fam["defect"]),
                    fam["category"] if fam["defect"] else "none", False,
                    [s.get("url") for s in fam.get("sources", [])])
    write("v4holdout", plans, labels, "research/failure_corpus/v4/holdout/faults.py")
    print("holdout", len(labels), "defect", sum(x["defect"] for x in labels),
          "clean", sum(1 for x in labels if not x["defect"] and not x["is_reference"]))


if __name__ == "__main__":
    {"calibration_rough": calibration_rough, "holdout": holdout}[sys.argv[1]]()
