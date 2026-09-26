# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Per-case statistics for the conventional baselines and for threshold calibration.

Nothing here imports IVF. The simple trajectory baselines are deliberately plain:
root-mean-square error over every step, env and component, per signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

B3_SIGNALS = ["joint_pos", "root_link_pos_w"]                      # registered B3
B3ALL_SIGNALS = ["joint_pos", "joint_vel", "joint_pos_target", "root_link_pos_w", "root_link_quat_w",
                 "base_height", "policy_obs"]                        # D7: IVF's signal set
PERF_KEYS = ["return_rel", "falls", "tilted", "tracking"]


def load_arrays(bundle: Path) -> dict[str, np.ndarray]:
    with np.load(bundle / "trajectories.npz") as d:
        return {k: d[k].astype(np.float64) for k in d.files}


def geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    chord = np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
    return 4.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def rmse(sig: str, a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if sig == "root_link_quat_w":
        return float(np.sqrt(np.mean(geodesic(a, b) ** 2)))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def trajectory_stats(base: dict[str, np.ndarray], cand: dict[str, np.ndarray]) -> dict[str, float]:
    return {s: rmse(s, base[s], cand[s]) for s in B3ALL_SIGNALS}


def perf_stats(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, float]:
    r, c = ref["closed_loop"], cand["closed_loop"]
    return {
        "return_rel": abs(c["mean_return_per_env"] - r["mean_return_per_env"]) / max(abs(r["mean_return_per_env"]), 1e-6),
        "falls": abs(c["fall_terminations_per_env"] - r["fall_terminations_per_env"]),
        "tilted": abs(c["tilted_fraction"] - r["tilted_fraction"]),
        "tracking": abs(c["mean_tracking_error"] - r["mean_tracking_error"]),
    }


def per_step_change_p95(x: np.ndarray, sig: str) -> float:
    """IVF flagship-style one-control-period allowance: p95 of per-step change."""
    if sig == "root_link_quat_w":
        d = geodesic(x[1:], x[:-1])
    else:
        d = np.linalg.norm(x[1:] - x[:-1], axis=-1)
    return float(np.percentile(d, 95))


def ivf_style_worst(sig: str, a: np.ndarray, b: np.ndarray) -> float:
    """Replicates IVF's trajectory statistic (max over steps of second-largest env error)."""
    err = geodesic(a, b) if sig == "root_link_quat_w" else np.linalg.norm(b - a, axis=-1)
    srt = np.sort(err, axis=1)
    return float(np.max(srt[:, -2]))
