# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""The fixed IVF manifest template for corpus cases, and its ablations.

Oracle set, controls and verdict policy are the pre-registered template (amended by D1:
the fall event and survival decision watch ``upright`` instead of ``base_height``).
Only tolerance values vary between arms; they come from ``thresholds.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

TRAJ_SIGNALS = [
    "joint_pos",
    "joint_vel",
    "joint_pos_target",
    "root_link_pos_w",
    "root_link_quat_w",
    "base_height",
    "policy_obs",
]
TRAJ_UNITS = {
    "joint_pos": "rad",
    "joint_vel": "rad/s",
    "joint_pos_target": "rad",
    "root_link_pos_w": "m",
    "root_link_quat_w": "dimensionless",
    "base_height": "m",
    "policy_obs": "dimensionless",
}
REQUIRE_SAME = [
    "asset_identity",
    "action_sequence",
    "initial_state_distribution",
    "observation_definition",
    "control_frequency",
    "seeds",
    "num_envs",
    "horizon",
    "task_variant",
    "frame_convention",
    "quaternion_convention",
    "reset_semantics",
    "environment_ordering",
    "action_timing",
]
STEPS, ENVS = 250, 16

ORACLE_GROUPS = {
    "invariant": ["finite_state", "unit_quaternion"],
    "trajectory": [f"traj_{s}" for s in TRAJ_SIGNALS],
    "event": ["fall_timing", "contact_timing", "survival"],
    "statistical": ["mean_base_height"],
    "metamorphic": ["observation_shape_stability", "action_replay"],
}
ALL_ORACLES = [o for g in ORACLE_GROUPS.values() for o in g]

ABLATIONS = {
    "full": {"oracles": ALL_ORACLES, "controls": True},
    "validity_only": {"oracles": ["finite_state"], "controls": True},
    "no_control_checks": {"oracles": ALL_ORACLES, "controls": False},
    "trajectory_only": {"oracles": ORACLE_GROUPS["trajectory"], "controls": True},
    "event_decision_only": {"oracles": ORACLE_GROUPS["event"], "controls": True},
    "statistical_only": {"oracles": ORACLE_GROUPS["statistical"], "controls": True},
    "invariants_only": {"oracles": ORACLE_GROUPS["invariant"], "controls": True},
    "minus_policy_obs": {
        "oracles": [o for o in ALL_ORACLES if o != "traj_policy_obs"],
        "controls": True,
    },
    "minus_joint_pos_target": {
        "oracles": [o for o in ALL_ORACLES if o != "traj_joint_pos_target"],
        "controls": True,
    },
    # D8 (post hoc, exploratory): drop the oracles that produced every clean false positive
    "minus_event_decision": {
        "oracles": [o for o in ALL_ORACLES if o not in ORACLE_GROUPS["event"]],
        "controls": True,
    },
    "joint_signals_only": {"oracles": ["traj_joint_pos", "traj_joint_vel"], "controls": True},
}


def _tol(value, unit, scope, rationale, aggregation, min_samples, kind, mme=None) -> dict[str, Any]:
    t = {
        "value": float(value),
        "unit": unit,
        "scope": scope,
        "rationale": rationale,
        "aggregation": aggregation,
        "min_samples": int(min_samples),
        "kind": kind,
    }
    if mme is not None:
        t["minimum_meaningful_effect"] = float(mme)
    return t


def oracle_specs(tol: dict[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    """All oracle specs keyed by name, with tolerances from ``tol`` (one platform, one condition)."""
    why = {
        "cal": "Calibrated: 1.25 x the worst value IVF itself reported on clean calibration pairs "
        "(seeds 0 to 2) of this platform and condition; no fault case was used.",
        "strict": "IVF flagship method: one-control-period allowance, the 95th percentile of the "
        "per-step change of this signal in the PhysX reference capture.",
        "permissive": "Measurement only: deliberately loose so the oracle reports its statistic "
        "on clean calibration pairs; never used to decide a corpus case.",
    }[arm]
    specs: dict[str, dict[str, Any]] = {
        "finite_state": {"type": "invariant", "name": "finite_state", "check": "finite_state"},
        "unit_quaternion": {
            "type": "invariant",
            "name": "unit_quaternion",
            "check": "unit_quaternion",
            "signals": ["root_link_quat_w"],
            "tolerance": _tol(
                1e-3,
                "dimensionless",
                "deviation of |q| from 1 per element",
                "float32 root quaternions; 1e-3 is a numerical guard, not accuracy",
                "max",
                1,
                "numerical",
            ),
        },
    }
    for s in TRAJ_SIGNALS:
        specs[f"traj_{s}"] = {
            "type": "trajectory_equivalence",
            "name": f"traj_{s}",
            "signal": s,
            "metric": "geodesic" if s == "root_link_quat_w" else "absolute",
            "tolerance": _tol(
                tol["traj"][s],
                TRAJ_UNITS[s],
                f"per-step {s} error reduced over envs",
                why,
                "second_largest",
                STEPS * ENVS,
                "engineering",
            ),
        }
    specs["fall_timing"] = {
        "type": "event_equivalence",
        "name": "fall_timing",
        "event": "fall",
        "signal": "upright",
        "condition": "below",
        "threshold": 0.5,
        "tolerance": _tol(
            tol["event"]["fall_timing"],
            "steps",
            "first step with tilt beyond 60 degrees",
            why,
            "max",
            8,
            "event",
        ),
    }
    specs["contact_timing"] = {
        "type": "event_equivalence",
        "name": "contact_timing",
        "event": "base_contact",
        "signal": "illegal_contact",
        "condition": "above",
        "threshold": 0.5,
        "tolerance": _tol(
            tol["event"]["contact_timing"],
            "steps",
            "first step the configured base-contact termination condition holds",
            why,
            "max",
            8,
            "event",
        ),
    }
    specs["survival"] = {
        "type": "decision_equivalence",
        "name": "survival",
        "decision": "survives_horizon",
        "signal": "upright",
        "condition": "below",
        "threshold": 0.5,
        "polarity": "survived",
        "tolerance": _tol(
            tol["decision"]["survival"],
            "fraction",
            "fraction of paired envs with the same survive-or-fall decision",
            why,
            "all",
            8,
            "engineering",
        ),
    }
    margin = tol["statistical"]["mean_base_height"]
    specs["mean_base_height"] = {
        "type": "statistical_equivalence",
        "name": "mean_base_height",
        "signal": "base_height",
        "summary": "mean",
        "tolerance": _tol(
            margin,
            "m",
            "horizon-mean base height per env, paired",
            why,
            "mean",
            16,
            "statistical",
            mme=0.1 * margin,
        ),
    }
    specs["observation_shape_stability"] = {
        "type": "metamorphic",
        "name": "observation_shape_stability",
        "relation": "observation_definition_stability",
    }
    specs["action_replay"] = {
        "type": "metamorphic",
        "name": "action_replay",
        "relation": "action_replay_consumption",
    }
    return specs


def build_manifest(
    case: dict[str, Any],
    baseline_bundle: Path,
    candidate_bundle: Path,
    tol: dict[str, Any],
    *,
    arm: str,
    ablation: str = "full",
) -> str:
    ab = ABLATIONS[ablation]
    specs = oracle_specs(tol, arm)

    def lock(b: Path) -> str:
        return json.loads((b / "COMPLETE").read_text())["checksums_sha256"]

    doc = {
        "schema_version": "ivf.validation/v1",
        "name": f"corpus-{case['capture_id'][:40]}-{arm}-{ablation}".replace("__", "-"),
        "description": "Failure-corpus case. Labels are not visible to IVF.",
        "subjects": {
            "baseline": {
                "kind": "parity_bundle",
                "label": "PhysX reference",
                "path": str(baseline_bundle),
                "bundle_sha256": lock(baseline_bundle),
            },
            "candidate": {
                "kind": "parity_bundle",
                "label": "candidate",
                "path": str(candidate_bundle),
                "bundle_sha256": lock(candidate_bundle),
            },
        },
        "workload": {
            "task": f"{case['platform']}_velocity_flat_open_loop",
            "num_envs": ENVS,
            "steps": STEPS,
            "warmup_steps": 0,
            "seeds": [int(case["seed"])],
        },
        "controls": {
            "require_same": REQUIRE_SAME if ab["controls"] else [],
            "allow_different": ["solver_specific_parameters"],
            "unsupported_or_unverifiable": ["asset_binary_identity", "backend_internal_state"],
        },
        "oracles": [specs[name] for name in ab["oracles"]],
        "verdict_policy": {
            "invalid_experiment_on_control_violation": True,
            "inconclusive_on_insufficient_samples": True,
            "unsupported_is_failure": False,
        },
    }
    return yaml.safe_dump(doc, sort_keys=False)


def permissive_tolerances() -> dict[str, Any]:
    """Tolerances loose enough that every oracle passes on shape: used only to *measure*."""
    return {
        "traj": {s: 1e9 for s in TRAJ_SIGNALS},
        "event": {"fall_timing": 1e6, "contact_timing": 1e6},
        "decision": {"survival": 0.0},
        "statistical": {"mean_base_height": 1e9},
    }


def write_manifest(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
