# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Generate the capture plans and ground-truth labels for every corpus split.

Deterministic. Writes:
  plans/<split>__<platform>.json   capture jobs (what the producer runs; no category labels)
  corpus/labels_<split>.json       ground truth per case (read only by the scorer)
  corpus/LABEL_HASHES.txt          sha256 of each label file, committed before capture

Holdout capture ids are opaque hashes so directory names do not reveal the family.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ["go2", "g1", "h1", "anymal_d"]
TRAINED_ACTION_SCALE = {"go2": 0.25, "g1": 0.5, "h1": 0.5, "anymal_d": 0.5}
BASE_MASS_BODY = {"go2": "base", "g1": "torso_link", "h1": "torso_link", "anymal_d": "base"}

CATEGORY = {
    "joint_order_interface": "joint_body_ordering",
    "capture_not_canonicalized": "joint_body_ordering",
    "obs_term_swap": "obs_action_ordering",
    "timestep_dt_decimation": "timestep_decimation",
    "randomization_asymmetry": "reset_randomization",
    "reset_velocity_dropped": "reset_randomization",
    "reset_joint_offsets_ignored": "reset_randomization",
    "armature_mismatch": "actuator_solver_config",
    "contact_capacity": "actuator_solver_config",
    "actuator_gain_scale": "actuator_solver_config",
    "preset_not_applied": "ineffective_stale_config",
    "action_scale": "checkpoint_schema",
    "termination_body_mismatch": "termination_metric",
}
SOURCES = {
    "capture_not_canonicalized": ["isaaclab-doc-sim2sim-joint-order", "isaaclab-issue6485-checkpoint-has-no-ordering-record",
                                  "native joint-order difference verified in this checkout (deviation D3)"],
    "joint_order_interface": ["isaaclab-doc-sim2sim-joint-order", "isaaclab-pr6913-g1-anymald-missing-ordering-overrides",
                               "isaaclab-issue6485-checkpoint-has-no-ordering-record", "unitree-rl-lab-145-g1-permutation-and-history-layout"],
    "obs_term_swap": ["unitree-rl-gym-32-go2-obs-order"],
    "timestep_dt_decimation": ["deploy-tienkung-8-decimation-mismatch", "hover-38-mujoco-obs-update-frequency"],
    "randomization_asymmetry": ["isaaclab-issue7786-base-com-dr-disabled-on-newton", "isaaclab-issue7097-com-randomization-unusable-newton",
                                "isaaclab-pr7992-featherstone-inertia-dr-no-effect"],
    "reset_velocity_dropped": ["isaaclab-issue7236-instep-reset-stale-fk-newton", "ivf-internal-upstream-reset-defect"],
    "reset_joint_offsets_ignored": ["isaaclab-issue7236-instep-reset-stale-fk-newton", "ivf-internal-upstream-reset-defect"],
    "armature_mismatch": ["isaaclab-pr7612-armature-split", "isaaclab-pr7607-backend-conditioned-task-config",
                         "unitree-rl-lab-31-g1-arm-armature-mujoco", "unitree-rl-gym-47-g1-mjcf-missing-joint-damping-armature"],
    "contact_capacity": ["isaaclab-pr6850-newton-contact-buffer-overflow", "isaacsim-doc-mjwarp-nconmax-drops-contacts",
                         "mjwarp-doc-overflow-undefined-behavior"],
    "actuator_gain_scale": ["newton-issue3698-mjcf-dampratio-ignored", "unitree-rl-gym-47-g1-mjcf-missing-joint-damping-armature"],
    "preset_not_applied": ["isaaclab-pr7103-smoke-test-preset-path (verified locally, this study)"],
    "action_scale": ["unilab-579-cross-backend-config-drift"],
    "termination_body_mismatch": ["isaaclab-doc-body-ordering-silent", "isaaclab-pr6913-g1-anymald-missing-ordering-overrides"],
}
CROSS_ONLY = {"armature_mismatch", "contact_capacity", "preset_not_applied", "capture_not_canonicalized"}
TRAINING_ARMATURE_IS_ZERO = {"go2": True, "g1": False, "h1": False, "anymal_d": True}


def fault_params(family: str, platform: str, variant: str, condition: str = "cross") -> dict:
    dev = variant == "dev"
    if family == "joint_order_interface":
        if not dev:
            return {"interface": "lr_swap"}
        return {"interface": "native" if condition == "cross" else "newton_order"}
    if family == "armature_mismatch":
        return {"armature": 0.02 if TRAINING_ARMATURE_IS_ZERO[platform] else 0.0}
    if family == "obs_term_swap":
        return {"a": "base_ang_vel", "b": "projected_gravity"} if dev else {"a": "base_lin_vel", "b": "velocity_commands"}
    if family == "timestep_dt_decimation":
        return {"dt": 0.005, "decimation": 2} if dev else {"dt": 0.01, "decimation": 2}
    if family == "randomization_asymmetry":
        return {"body": BASE_MASS_BODY[platform], "mass_range": [0.8, 1.25] if dev else [0.9, 1.1]}
    if family == "contact_capacity":
        return {"nconmax": 2, "njmax": 8} if dev else {"nconmax": 3, "njmax": 12}
    if family == "actuator_gain_scale":
        return {"stiffness_scale": 1.0, "damping_scale": 0.0} if dev else {"stiffness_scale": 0.7, "damping_scale": 0.7}
    if family == "action_scale":
        return {"scale": TRAINED_ACTION_SCALE[platform] * (2.0 if dev else 0.5)}
    if family == "termination_body_mismatch":
        return {"body_index": 1 if dev else 2}
    return {}


def families_for(variant: str) -> list[str]:
    fams = list(CATEGORY)
    if variant == "dev":
        fams.remove("reset_joint_offsets_ignored")
    else:
        fams.remove("reset_velocity_dropped")
    return fams


def case_id(split: str, key: str) -> str:
    if split == "holdout":
        return "h_" + hashlib.sha256(f"ivf-corpus-holdout::{key}".encode()).hexdigest()[:12]
    return key


def jobs_for(split: str, platform: str, seed: int, variant: str, *, clean_only: bool):
    ref = f"{platform}:{seed}"
    jobs, labels = [], []

    def add(key, backend_mode, fault, condition, defect, *, is_reference=False):
        cid = case_id(split, key)
        jobs.append({"capture_id": cid, "platform": platform, "seed": seed, "backend_mode": backend_mode,
                     "fault": fault, "replay_ref": ref, "is_reference": is_reference})
        fam = fault["family"]
        labels.append({"capture_id": cid, "split": split, "platform": platform, "seed": seed,
                       "condition": condition, "family": fam, "variant": variant,
                       "defect": defect, "category": CATEGORY.get(fam, "none"),
                       "sources": SOURCES.get(fam, []), "is_reference": is_reference})

    base = f"{split}__{platform}__s{seed}"
    add(f"{base}__ref_physx", "physx", {"family": "none"}, "reference", False, is_reference=True)
    add(f"{base}__clean_physx_rerun", "physx", {"family": "none"}, "same", False)
    add(f"{base}__clean_newton", "newton", {"family": "none"}, "cross", False)
    if clean_only:
        return jobs, labels
    add(f"{base}__benign_newton_capacity", "newton", {"family": "benign_capacity", "params": {"factor": 2}}, "cross", False)
    for fam in families_for(variant):
        mode = "newton_via_pre7103_test_path" if fam == "preset_not_applied" else "newton"
        add(f"{base}__cross__{fam}", mode, {"family": fam, "params": fault_params(fam, platform, variant, "cross")},
            "cross", True)
        if fam not in CROSS_ONLY:
            add(f"{base}__same__{fam}", "physx", {"family": fam, "params": fault_params(fam, platform, variant, "same")},
                "same", True)
    return jobs, labels


def main() -> None:
    splits = {
        "calibration": [(p, s, "dev", True) for p in PLATFORMS for s in (0, 1, 2)],
        "dev": [(p, 3, "dev", False) for p in ("go2", "g1", "h1")],
        "holdout": [("anymal_d", 3, "dev", False), ("anymal_d", 4, "dev", False)]
                   + [(p, 4, "holdout", False) for p in ("go2", "g1", "h1")]
                   + [(p, s, "holdout", True) for p in PLATFORMS for s in (5, 6)],
    }
    plans_dir = ROOT / "plans"
    plans_dir.mkdir(exist_ok=True)
    hashes = []
    for split, entries in splits.items():
        all_labels = []
        per_platform: dict[str, list] = {}
        for platform, seed, variant, clean_only in entries:
            jobs, labels = jobs_for(split, platform, seed, variant, clean_only=clean_only)
            per_platform.setdefault(platform, []).extend(jobs)
            all_labels.extend(labels)
        for platform, jobs in per_platform.items():
            plan = {"split": split, "platform": platform,
                    "policy": f"research/failure_corpus/policies/{platform}/policy.pt", "jobs": jobs}
            (plans_dir / f"{split}__{platform}.json").write_text(json.dumps(plan, indent=1) + "\n")
        path = ROOT / "corpus" / f"labels_{split}.json"
        path.write_text(json.dumps(all_labels, indent=1, sort_keys=True) + "\n")
        hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  corpus/labels_{split}.json")
        print(split, len(all_labels), "cases")
    (ROOT / "corpus" / "LABEL_HASHES.txt").write_text("\n".join(hashes) + "\n")


if __name__ == "__main__":
    main()
