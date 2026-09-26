# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Plans and labels for v3: clean calibration (seeds 0-2) and the fresh holdout (seeds 13-14).

Usage: python make_plans_v3.py calibration | holdout
"""

import hashlib
import json
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
PLATFORMS = ["go2", "g1", "h1", "anymal_d", "cassie", "spot"]
POLICY = "research/failure_corpus/policies/{p}/policy.pt"


def write(split, plans, labels, fault_module=None):
    for p, jobs in plans.items():
        plan = {"split": split, "platform": p, "policy": POLICY.format(p=p), "jobs": jobs}
        if fault_module:
            plan["fault_module"] = fault_module
        (V3 / "plans" / f"{split}__{p}.json").write_text(json.dumps(plan, indent=1) + "\n")
    (V3 / "corpus" / f"labels_{split}.json").write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")


def job(plans, labels, split, p, seed, cid, mode, fam, params, cond, defect, category, is_ref, sources=()):
    plans.setdefault(p, []).append({"capture_id": cid, "platform": p, "seed": seed, "backend_mode": mode,
                                    "fault": {"family": fam, "params": params}, "replay_ref": f"{p}:{seed}",
                                    "is_reference": is_ref})
    labels.append({"capture_id": cid, "split": split, "platform": p, "seed": seed, "condition": cond,
                   "family": fam, "defect": defect, "category": category, "is_reference": is_ref,
                   "sources": list(sources)})


def calibration():
    plans, labels = {}, []
    for p in PLATFORMS:
        for seed in (0, 1, 2):
            b = f"calibration__{p}__s{seed}"
            job(plans, labels, "calibration", p, seed, f"{b}__ref_physx", "physx", "none", {}, "reference", False,
                "none", True)
            job(plans, labels, "calibration", p, seed, f"{b}__clean_physx_rerun", "physx", "none", {}, "same",
                False, "none", False)
            job(plans, labels, "calibration", p, seed, f"{b}__clean_newton", "newton", "none", {}, "cross", False,
                "none", False)
    write("calibration", plans, labels)


def holdout():
    cases = json.loads((V3 / "holdout" / "fault_cases.json").read_text())
    plans, labels = {}, []

    def cid(key):
        return "v3h_" + hashlib.sha256(f"ivf-v3-holdout::{key}".encode()).hexdigest()[:12]
    for p in PLATFORMS:
        for seed in (13, 14):
            job(plans, labels, "v3holdout", p, seed, cid(f"{p}:{seed}:ref"), "physx", "none", {}, "reference",
                False, "none", True)
            job(plans, labels, "v3holdout", p, seed, cid(f"{p}:{seed}:clean"), "newton", "none", {}, "cross",
                False, "none", False)
            for fam in cases:
                if p not in fam["params"]:
                    continue
                job(plans, labels, "v3holdout", p, seed, cid(f"{p}:{seed}:{fam['family']}"), "newton",
                    fam["family"], fam["params"][p], "cross", bool(fam["defect"]),
                    fam["category"] if fam["defect"] else "none", False,
                    [s.get("url") for s in fam.get("sources", [])])
    write("v3holdout", plans, labels, "research/failure_corpus/v3/holdout/faults.py")
    print("holdout", len(labels), "defect", sum(x["defect"] for x in labels),
          "clean", sum(1 for x in labels if not x["defect"] and not x["is_reference"]))


if __name__ == "__main__":
    {"calibration": calibration, "holdout": holdout}[sys.argv[1]]()
