# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Plans and labels for the v2 holdout (independently authored faults) and Cassie calibration.

Holdout: platforms go2, g1, h1, anymal_d, cassie; seeds 9 and 10 (never used before); per
(platform, seed): PhysX reference, Newton clean, and every family in fault_cases.json applied
to a Newton candidate. Capture ids are opaque. Cassie calibration: clean seeds 0 to 2.
"""

import hashlib
import json
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
CASES = json.loads((V2 / "holdout" / "fault_cases.json").read_text())
PLATFORMS = ["go2", "g1", "h1", "anymal_d", "cassie"]
POLICY = "research/failure_corpus/policies/{p}/policy.pt"


def opaque(key):
    return "v2h_" + hashlib.sha256(f"ivf-v2-holdout::{key}".encode()).hexdigest()[:12]


def main():
    plans, labels = {}, []
    for p in PLATFORMS:
        for seed in (9, 10):
            ref = f"{p}:{seed}"
            def add(key, mode, fam, params, defect, category, is_ref=False, sources=()):
                cid = opaque(key)
                plans.setdefault(p, []).append({"capture_id": cid, "platform": p, "seed": seed, "backend_mode": mode,
                                                "fault": {"family": fam, "params": params}, "replay_ref": ref,
                                                "is_reference": is_ref})
                labels.append({"capture_id": cid, "split": "v2holdout", "platform": p, "seed": seed,
                               "condition": "reference" if is_ref else "cross", "family": fam, "defect": defect,
                               "category": category, "is_reference": is_ref, "sources": list(sources)})
            add(f"{p}:{seed}:ref", "physx", "none", {}, False, "none", is_ref=True)
            add(f"{p}:{seed}:clean", "newton", "none", {}, False, "none")
            for fam in CASES:
                if p not in fam["params"]:
                    continue
                add(f"{p}:{seed}:{fam['family']}", "newton", fam["family"], fam["params"][p], bool(fam["defect"]),
                    fam["category"] if fam["defect"] else "none",
                    sources=[s.get("url") for s in fam.get("sources", [])])
    for p, jobs in plans.items():
        (V2 / "plans" / f"v2holdout__{p}.json").write_text(json.dumps(
            {"split": "v2holdout", "platform": p, "policy": POLICY.format(p=p),
             "fault_module": "research/failure_corpus/v2/holdout/faults.py", "jobs": jobs}, indent=1) + "\n")
    (V2 / "corpus" / "labels_v2holdout.json").write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")
    # Cassie clean calibration (seeds 0 to 2): reference, PhysX rerun, Newton clean
    cal_jobs, cal_labels = [], []
    for seed in (0, 1, 2):
        for key, mode, cond, is_ref in (("ref_physx", "physx", "reference", True),
                                        ("clean_physx_rerun", "physx", "same", False),
                                        ("clean_newton", "newton", "cross", False)):
            cid = f"calibration__cassie__s{seed}__{key}"
            cal_jobs.append({"capture_id": cid, "platform": "cassie", "seed": seed, "backend_mode": mode,
                             "fault": {"family": "none"}, "replay_ref": f"cassie:{seed}", "is_reference": is_ref})
            cal_labels.append({"capture_id": cid, "split": "calibration_cassie", "platform": "cassie", "seed": seed,
                               "condition": cond, "family": "none", "defect": False, "category": "none",
                               "is_reference": is_ref, "sources": []})
    (V2 / "plans" / "calibration_cassie__cassie.json").write_text(json.dumps(
        {"split": "calibration_cassie", "platform": "cassie", "policy": POLICY.format(p="cassie"),
         "jobs": cal_jobs}, indent=1) + "\n")
    (V2 / "corpus" / "labels_calibration_cassie.json").write_text(json.dumps(cal_labels, indent=1) + "\n")
    n_def = sum(1 for x in labels if x["defect"])
    n_clean = sum(1 for x in labels if not x["defect"] and not x["is_reference"])
    print("holdout captures", len(labels), "defect", n_def, "clean", n_clean)


if __name__ == "__main__":
    main()
