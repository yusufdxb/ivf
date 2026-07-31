# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Divergence localization: where it first went wrong, and what it looks like.

Printing a final error tells an engineer that something is wrong. It does not tell
them what to open. This module produces a structured record naming the first step,
the environments, the components, the surrounding values, and a *rule-based*
classification with an explicit confidence and an explicit statement of what would
falsify it.

The classifier is deliberately deterministic and rule-based. There is no model and
no LLM: a classification that cannot be reproduced from the record by hand is not
evidence, and a debugging aid that hallucinates is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CLASSIFICATIONS = (
    "setup_mismatch",
    "asset_mismatch",
    "reset_mismatch",
    "cloning_mismatch",
    "coordinate_frame_mismatch",
    "quaternion_convention_mismatch",
    "action_mismatch",
    "unsupported_feature",
    "numerical_drift",
    "contact_model_difference",
    "solver_parameter_difference",
    "sensor_semantic_difference",
    "task_level_regression",
    "unknown",
)
"""The classification vocabulary. ``unknown`` is a legitimate answer and is used
whenever no rule fires with adequate support."""


@dataclass
class DivergenceRecord:
    """A localized, human-actionable description of one divergence."""

    signal: str
    oracle: str

    first_numerical_difference_step: int | None = None
    """First step at which the two subjects differ at all, above float noise."""

    first_tolerance_violation_step: int | None = None
    """First step at which the declared tolerance was exceeded."""

    first_event_disagreement_step: int | None = None
    """First step at which a semantic event disagreed."""

    affected_env_ids: list[int] = field(default_factory=list)
    affected_components: list[int] = field(default_factory=list)
    """Indices within the signal's last dimension (body/joint/axis, depending on signal)."""

    baseline_values: list[float] = field(default_factory=list)
    candidate_values: list[float] = field(default_factory=list)
    """Values at the first violating step, for the worst affected environment."""

    window: dict[str, Any] = field(default_factory=dict)
    """A short surrounding time window of the error curve, for the report plot."""

    action_at_violation: list[float] | None = None
    contact_state: dict[str, Any] | None = None

    config_differences: dict[str, Any] = field(default_factory=dict)
    """Controlled or uncontrolled configuration differences between the subjects."""

    classification: str = "unknown"
    confidence: str = "low"
    """``low`` | ``medium`` | ``high``. Never numeric: a made-up probability would be
    a stronger claim than the evidence supports."""

    classification_basis: str = ""
    limitations: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "signal": self.signal,
            "oracle": self.oracle,
            "first_numerical_difference_step": self.first_numerical_difference_step,
            "first_tolerance_violation_step": self.first_tolerance_violation_step,
            "first_event_disagreement_step": self.first_event_disagreement_step,
            "affected_env_ids": self.affected_env_ids,
            "affected_components": self.affected_components,
            "baseline_values": self.baseline_values,
            "candidate_values": self.candidate_values,
            "window": self.window,
            "action_at_violation": self.action_at_violation,
            "contact_state": self.contact_state,
            "config_differences": self.config_differences,
            "classification": self.classification,
            "confidence": self.confidence,
            "classification_basis": self.classification_basis,
            "limitations": self.limitations,
        }


def localize(
    *,
    signal: str,
    oracle: str,
    baseline: np.ndarray,
    candidate: np.ndarray,
    per_step_env_error: np.ndarray,
    threshold: float,
    actions: np.ndarray | None = None,
    config_differences: dict[str, Any] | None = None,
    numerical_noise: float = 1e-12,
    window_radius: int = 8,
) -> DivergenceRecord:
    """Build a :class:`DivergenceRecord` from two aligned signals and their error curve.

    ``per_step_env_error`` is ``(steps, envs)``: already reduced over the signal's last
    dimension by the caller, because the right reduction (Euclidean, geodesic angle,
    per-component) depends on what the signal means.
    """
    record = DivergenceRecord(signal=signal, oracle=oracle)
    record.config_differences = dict(config_differences or {})

    finite_mask = np.isfinite(per_step_env_error)
    above_noise = np.where(finite_mask & (per_step_env_error > numerical_noise))
    if above_noise[0].size:
        record.first_numerical_difference_step = int(above_noise[0].min())

    violating = np.where(~finite_mask | (per_step_env_error > threshold))
    if violating[0].size:
        step = int(violating[0].min())
        record.first_tolerance_violation_step = step
        envs_at_step = np.where(~finite_mask[step] | (per_step_env_error[step] > threshold))[0]
        record.affected_env_ids = [int(e) for e in envs_at_step]
        row = np.where(finite_mask[step], per_step_env_error[step], np.inf)
        worst_env = int(np.argmax(row))
        base_vec = np.atleast_1d(baseline[step, worst_env])
        cand_vec = np.atleast_1d(candidate[step, worst_env])
        record.baseline_values = [float(v) for v in base_vec]
        record.candidate_values = [float(v) for v in cand_vec]
        with np.errstate(invalid="ignore"):
            comp_err = np.abs(cand_vec - base_vec)
        record.affected_components = [
            int(i) for i in np.where(~np.isfinite(comp_err) | (comp_err > threshold))[0]
        ]
        lo = max(0, step - window_radius)
        hi = min(per_step_env_error.shape[0], step + window_radius + 1)
        window_slice = per_step_env_error[lo:hi, worst_env]
        record.window = {
            "start_step": lo,
            "end_step": hi,
            "env_id": worst_env,
            "error": [None if not np.isfinite(v) else float(v) for v in window_slice],
            "threshold": threshold,
        }
        if actions is not None and step < actions.shape[0]:
            action_slice = np.atleast_1d(np.asarray(actions[step]).reshape(-1))
            record.action_at_violation = [float(v) for v in action_slice[:8]]

    classify(record, baseline=baseline, candidate=candidate, per_step_env_error=per_step_env_error,
             threshold=threshold)
    return record


def classify(
    record: DivergenceRecord,
    *,
    baseline: np.ndarray,
    candidate: np.ndarray,
    per_step_env_error: np.ndarray,
    threshold: float,
) -> DivergenceRecord:
    """Assign a classification to ``record`` in place, with basis and limitations.

    Rules are ordered from most specific to least. Each rule states what it keys on so
    a reader can disagree with it from the record alone.
    """
    diffs = record.config_differences

    # 1. A declared configuration difference explains the divergence more simply than
    #    any dynamical story, so it wins outright.
    if diffs:
        keys = sorted(diffs)
        if any(k in ("asset_identity",) for k in keys):
            return _set(record, "asset_mismatch", "high",
                        f"subjects differ in asset identity: {keys}",
                        "Assumes the recorded asset identity is trustworthy; a weak "
                        "identity (scenario name + joint count) can collide.")
        if any(k in ("num_envs", "horizon") for k in keys):
            return _set(record, "cloning_mismatch", "high",
                        f"subjects differ in replication or horizon: {keys}",
                        "Says nothing about whether the underlying dynamics also differ.")
        if any("solver" in k for k in keys):
            return _set(record, "solver_parameter_difference", "medium",
                        f"subjects differ in solver settings: {keys}",
                        "A solver-parameter difference is not automatically a defect; "
                        "backends legitimately expose different solvers.")
        if any(k in ("physics_dt", "control_frequency_hz") for k in keys):
            return _set(record, "setup_mismatch", "high",
                        f"subjects differ in timing configuration: {keys}",
                        "Timing differences make the comparison uninterpretable rather "
                        "than merely different.")

    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(baseline)):
        return _set(record, "numerical_drift", "high",
                    "non-finite values present in at least one subject",
                    "Non-finite output localizes the failure but not its cause; the "
                    "underlying defect may be any of setup, solver or dynamics.")

    if record.first_tolerance_violation_step is None:
        return _set(record, "unknown", "low", "no tolerance violation to explain", "")

    # 2. Quaternion convention: a component permutation makes the signals agree.
    if baseline.shape[-1] == 4 and _permutation_recovers(baseline, candidate):
        return _set(record, "quaternion_convention_mismatch", "high",
                    "candidate matches baseline under a wxyz<->xyzw component permutation",
                    "Detects only the two common orderings, not an arbitrary basis change.")

    step0 = record.first_tolerance_violation_step
    errors = per_step_env_error
    env_ids = record.affected_env_ids or list(range(errors.shape[1]))
    sub = errors[:, env_ids]

    # 3. A per-env constant offset that is already present at the first step reads as a
    #    frame or origin problem; the same offset appearing later does not.
    delta = candidate - baseline
    if delta.ndim == 3 and delta.shape[0] > 2:
        per_env_offset = delta.mean(axis=0)
        residual = float(np.abs(delta - per_env_offset[None]).max())
        offset_scale = float(np.abs(per_env_offset).max())
        if offset_scale > threshold and residual <= max(threshold, 1e-9):
            return _set(record, "coordinate_frame_mismatch", "high",
                        "the difference is a per-environment constant offset over the whole horizon",
                        "A constant offset is also what a legitimate frame convention change "
                        "looks like; IVF cannot tell a bug from an intended reframing without "
                        "an explicit frame declaration in the manifest.")

    # 4. A divergence that is present from the very first captured step and is already a
    #    large fraction of its eventual magnitude did not accumulate: the two runs started
    #    from different states. Keyed on the shape of the error curve rather than on the
    #    step index, so it fires the same way whether the tolerance happens to be crossed
    #    at step 1 or step 5.
    if record.first_numerical_difference_step == 0 and errors.size:
        with np.errstate(invalid="ignore"):
            overall = float(np.nanmax(errors))
            early = float(np.nanmax(errors[: min(3, errors.shape[0])]))
        if np.isfinite(overall) and overall > 0 and early >= 0.2 * overall:
            return _set(record, "reset_mismatch", "medium",
                        f"the difference is present from step 0 and already {early / overall:.0%} "
                        "of its eventual magnitude, so it did not accumulate",
                        "Cannot separate an initial-state difference from a first-step "
                        "actuation difference; both manifest at step 0. Check the action "
                        "stream digest and the initial-state digest to disambiguate.")
    if step0 <= 1:
        return _set(record, "reset_mismatch", "medium",
                    f"tolerance exceeded at step {step0}, before dynamics could accumulate",
                    "Cannot separate an initial-state difference from a first-step "
                    "actuation difference; both manifest at step 0. Check the action "
                    "stream digest and the initial-state digest to disambiguate.")

    # 5. A step change in error with a flat run before and after reads as a held /
    #    stale value or a dropped update, not as accumulation.
    if _is_step_change(sub, step0, threshold):
        return _set(record, "sensor_semantic_difference", "medium",
                    f"error is flat before step {step0} and jumps discontinuously there",
                    "A discontinuity is consistent with a stale sensor, a dropped update, "
                    "or a genuine contact event; without contact data IVF cannot separate them.")

    # 6. Monotonic growth from near zero is the signature of accumulation.
    if _is_monotonic_growth(sub, step0):
        if record.action_at_violation and any(abs(a) > 0 for a in record.action_at_violation):
            return _set(record, "numerical_drift", "low",
                        "error grows monotonically from near zero while an action is active",
                        "An actuated divergence that grows smoothly is equally consistent with "
                        "an action-scaling defect; the low confidence is the honest reading.")
        return _set(record, "numerical_drift", "medium",
                    "error grows monotonically from near zero",
                    "Growth alone does not distinguish accumulated float error from a small "
                    "genuine dynamics difference; over a long horizon they look identical.")

    return _set(record, "unknown", "low",
                "no rule matched the error-curve shape",
                "The record still localizes the first step and the affected environments; "
                "classification is the part that failed, not detection.")


def _set(record: DivergenceRecord, classification: str, confidence: str, basis: str,
         limitations: str) -> DivergenceRecord:
    """Assign classification fields and return the record."""
    assert classification in CLASSIFICATIONS, classification
    record.classification = classification
    record.confidence = confidence
    record.classification_basis = basis
    record.limitations = limitations
    return record


def _permutation_recovers(baseline: np.ndarray, candidate: np.ndarray, atol: float = 1e-9) -> bool:
    """Whether reordering the candidate's last dimension recovers the baseline."""
    for perm in ((3, 0, 1, 2), (1, 2, 3, 0)):
        if np.allclose(candidate[..., list(perm)], baseline, atol=atol, rtol=0.0):
            return True
    return False


def _is_step_change(errors: np.ndarray, step: int, threshold: float) -> bool:
    """Whether the error curve is flat, jumps at ``step``, and stays put."""
    if step < 3 or step + 3 >= errors.shape[0]:
        return False
    before = errors[:step]
    after = errors[step:]
    before_span = float(np.ptp(before)) if before.size else 0.0
    jump = float(np.max(after[:1]) - np.max(before[-1:]))
    return before_span <= threshold * 0.1 and jump > threshold


def _is_monotonic_growth(errors: np.ndarray, step: int, fraction: float = 0.8) -> bool:
    """Whether the error is non-decreasing over most of the run up to ``step``."""
    if step < 3:
        return False
    curve = np.nanmax(errors[: step + 1], axis=1)
    if curve.size < 3:
        return False
    increases = int(np.sum(np.diff(curve) >= -1e-15))
    return increases >= fraction * (curve.size - 1)
