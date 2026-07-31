# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical equivalence oracles: exact, absolute/relative, ULP, and envelope."""

from __future__ import annotations

import numpy as np

from ..divergence import localize
from .base import OracleContext, OracleOutcome, aggregate, register

_NOISE = 1e-12


def _element_error(baseline: np.ndarray, candidate: np.ndarray, metric: str) -> np.ndarray:
    """Return a ``(steps, envs)`` error curve under the requested metric."""
    if metric == "absolute":
        return np.linalg.norm(candidate - baseline, axis=-1)
    if metric == "relative":
        scale = np.maximum(np.linalg.norm(baseline, axis=-1), _NOISE)
        return np.linalg.norm(candidate - baseline, axis=-1) / scale
    if metric == "geodesic":
        # Quaternion geodesic angle, robust to the double cover (q and -q are the same
        # rotation), which a naive componentwise difference gets wrong by up to 2 rad.
        #
        # Computed from the chord as 4*arcsin(d/2) rather than as 2*arccos(|q_a . q_b|).
        # The two are identical in exact arithmetic: with d = min(||q_a - q_b||,
        # ||q_a + q_b||) and q_a . q_b = 1 - d^2/2, the half-angle identity gives
        # sin(phi/4) = d/2. The arccos form is catastrophically ill-conditioned exactly
        # where this oracle spends its time: near-identical orientations push the dot
        # product to 1, where arccos amplifies float64 rounding into ~3e-8 rad of
        # spurious angle, enough to fail a tight tolerance on two identical rotations.
        # The chord form is well-conditioned at small angles, which is the end that
        # matters here.
        if baseline.shape[-1] != 4:
            raise ValueError(f"metric 'geodesic' needs a 4-vector signal, got shape {baseline.shape}")
        near = np.linalg.norm(candidate - baseline, axis=-1)
        far = np.linalg.norm(candidate + baseline, axis=-1)
        chord = np.minimum(near, far)
        return 4.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))
    raise ValueError(f"unknown metric {metric!r} (expected absolute | relative | geodesic)")


def _config_differences(ctx: OracleContext) -> dict:
    """Return the metadata differences that could explain a divergence."""
    out = {}
    for key in ("asset_identity", "physics_dt", "control_frequency_hz", "solver_settings",
                "task_variant", "seed"):
        a, b = ctx.baseline.metadata.get(key), ctx.candidate.metadata.get(key)
        if a != b:
            out[key] = {"baseline": a, "candidate": b}
    if ctx.baseline.num_envs != ctx.candidate.num_envs:
        out["num_envs"] = {"baseline": ctx.baseline.num_envs, "candidate": ctx.candidate.num_envs}
    if ctx.baseline.steps != ctx.candidate.steps:
        out["horizon"] = {"baseline": ctx.baseline.steps, "candidate": ctx.candidate.steps}
    return out


@register("trajectory_equivalence")
def trajectory_equivalence(ctx: OracleContext) -> OracleOutcome:
    """Compare two trajectories against a declared numerical tolerance.

    Reduces the signal's last dimension with the requested metric, aggregates over
    environments with the tolerance's declared aggregation, and takes the worst step.
    A violation produces a full divergence record, not just a number.
    """
    signal = str(ctx.spec.params.get("signal", ""))
    if not signal:
        raise ValueError(f"oracle {ctx.spec.name!r}: 'signal' is required")
    metric = str(ctx.spec.params.get("metric", "absolute"))
    tol = ctx.tolerance
    baseline, candidate = ctx.signal_pair(signal)

    if baseline.shape != candidate.shape:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=(f"{signal}: shapes differ ({baseline.shape} vs {candidate.shape}); "
                     "no element-wise comparison is defined"),
            reason_codes=["IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH"],
            tolerance=tol.to_jsonable(),
            limitations="Shape disagreement is reported by the validity layer as the primary finding.",
        )

    n_samples = int(baseline.shape[0] * baseline.shape[1])
    errors = _element_error(baseline, candidate, metric)
    per_step = np.array([aggregate(errors[t], tol.aggregation) for t in range(errors.shape[0])])
    worst = float(np.nanmax(per_step)) if per_step.size else float("nan")
    worst_step = int(np.nanargmax(per_step)) if per_step.size else -1

    metrics = {
        "metric": metric,
        "aggregation": tol.aggregation,
        "worst_aggregated_error": worst,
        "worst_step": worst_step,
        "median_aggregated_error": float(np.nanmedian(per_step)) if per_step.size else None,
        "n_samples": n_samples,
        "steps": int(baseline.shape[0]),
        "num_envs": int(baseline.shape[1]),
    }

    if n_samples < tol.min_samples:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=f"{signal}: {n_samples} samples is below the declared minimum of {tol.min_samples}",
            reason_codes=["IVF-SAMPLE-INSUFFICIENT"], metrics=metrics, tolerance=tol.to_jsonable(),
        )

    if np.isfinite(worst) and worst <= tol.value:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=(f"{signal}: worst {tol.aggregation} {metric} error {worst:.3e} {tol.unit} "
                     f"<= {tol.value:.3e} {tol.unit}"),
            metrics=metrics, tolerance=tol.to_jsonable(),
            limitations="Agreement within a tolerance is not evidence that either subject is correct.",
        )

    record = localize(
        signal=signal, oracle=ctx.spec.name, baseline=baseline, candidate=candidate,
        per_step_env_error=errors, threshold=tol.value,
        actions=ctx.candidate.actions, config_differences=_config_differences(ctx),
    )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=(f"{signal}: worst {tol.aggregation} {metric} error {worst:.3e} {tol.unit} "
                 f"exceeds {tol.value:.3e} {tol.unit}; first violation at step "
                 f"{record.first_tolerance_violation_step}"),
        reason_codes=["IVF-ORACLE-NON_EQUIVALENT"], metrics=metrics, tolerance=tol.to_jsonable(),
        divergence=record,
        limitations="Exceeding a tolerance says the subjects differ, not which one is wrong.",
    )


@register("exact_equivalence")
def exact_equivalence(ctx: OracleContext) -> OracleOutcome:
    """Require bitwise-identical arrays.

    Appropriate only where identity is a real contract: replayed actions, recorded
    metadata, or a same-backend same-seed rerun. Using it across backends is a
    category error and the manifest guide says so.
    """
    signal = str(ctx.spec.params.get("signal", ""))
    baseline, candidate = ctx.signal_pair(signal)
    if baseline.shape != candidate.shape:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="fail",
            summary=f"{signal}: shapes differ ({baseline.shape} vs {candidate.shape})",
            reason_codes=["IVF-ORACLE-NON_EQUIVALENT"],
        )
    identical = bool(
        np.array_equal(baseline, candidate)
        and np.array_equal(np.isnan(baseline), np.isnan(candidate))
    )
    if identical:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=f"{signal}: bitwise identical over {baseline.shape[0]} steps",
            metrics={"n_samples": int(baseline.size)},
        )
    mismatches = int(np.sum(baseline != candidate))
    errors = np.linalg.norm(candidate - baseline, axis=-1)
    record = localize(
        signal=signal, oracle=ctx.spec.name, baseline=baseline, candidate=candidate,
        per_step_env_error=errors, threshold=0.0, actions=ctx.candidate.actions,
        config_differences=_config_differences(ctx),
    )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=f"{signal}: {mismatches} of {baseline.size} elements are not bitwise identical",
        reason_codes=["IVF-ORACLE-NON_EQUIVALENT"],
        metrics={"mismatching_elements": mismatches, "n_samples": int(baseline.size)},
        divergence=record,
    )


@register("ulp_equivalence")
def ulp_equivalence(ctx: OracleContext) -> OracleOutcome:
    """Compare in units in the last place against a declared ULP budget.

    ULP distance is the right yardstick when the claim is "the same computation on the
    same hardware", which is what the pre-existing free-rigid-body result (~1 FP32 ULP
    in its validated scope) actually established. It is the wrong yardstick across
    backends, where the computations differ by design.
    """
    signal = str(ctx.spec.params.get("signal", ""))
    precision = str(ctx.spec.params.get("precision", "float32"))
    tol = ctx.tolerance
    baseline, candidate = ctx.signal_pair(signal)
    if baseline.shape != candidate.shape:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=f"{signal}: shapes differ ({baseline.shape} vs {candidate.shape})",
            reason_codes=["IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH"], tolerance=tol.to_jsonable(),
        )
    dtype = {"float32": np.float32, "float64": np.float64}.get(precision)
    if dtype is None:
        raise ValueError(f"oracle {ctx.spec.name!r}: precision must be float32 or float64")

    a = baseline.astype(dtype)
    b = candidate.astype(dtype)
    ulps = _ulp_distance(a, b)
    per_step = np.array([aggregate(ulps[t], tol.aggregation) for t in range(ulps.shape[0])])
    worst = float(np.nanmax(per_step)) if per_step.size else float("nan")
    metrics = {
        "precision": precision,
        "worst_ulp": worst,
        "worst_step": int(np.nanargmax(per_step)) if per_step.size else -1,
        "n_samples": int(a.size),
        "aggregation": tol.aggregation,
    }
    if int(a.size) < tol.min_samples:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=f"{signal}: {a.size} samples is below the declared minimum of {tol.min_samples}",
            reason_codes=["IVF-SAMPLE-INSUFFICIENT"], metrics=metrics, tolerance=tol.to_jsonable(),
        )
    if np.isfinite(worst) and worst <= tol.value:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=f"{signal}: worst distance {worst:.1f} {precision} ULP <= {tol.value:g} ULP",
            metrics=metrics, tolerance=tol.to_jsonable(),
            limitations="A ULP claim is scoped to this precision and this hardware; it does not "
            "transfer to a different device or a different backend.",
        )
    record = localize(
        signal=signal, oracle=ctx.spec.name, baseline=baseline, candidate=candidate,
        per_step_env_error=np.max(ulps, axis=-1), threshold=tol.value,
        config_differences=_config_differences(ctx),
    )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=f"{signal}: worst distance {worst:.1f} {precision} ULP exceeds {tol.value:g} ULP",
        reason_codes=["IVF-ORACLE-ULP-EXCEEDED"], metrics=metrics, tolerance=tol.to_jsonable(),
        divergence=record,
    )


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the elementwise distance between ``a`` and ``b`` in ULPs.

    Uses the monotone integer ordering of IEEE-754: reinterpreting the bits of a float
    as a sign-magnitude integer and mapping to two's complement makes adjacent
    representable floats differ by exactly one. Non-finite pairs yield ``inf`` unless
    both are the identical non-finite value.
    """
    dtype = a.dtype
    int_dtype = np.int32 if dtype == np.float32 else np.int64
    offset = np.array(np.iinfo(int_dtype).min, dtype=int_dtype)

    def to_ordered(x: np.ndarray) -> np.ndarray:
        bits = x.view(int_dtype)
        return np.where(bits < 0, offset - bits, bits)

    finite = np.isfinite(a) & np.isfinite(b)
    ordered = np.abs(to_ordered(np.ascontiguousarray(a)).astype(np.float64)
                     - to_ordered(np.ascontiguousarray(b)).astype(np.float64))
    same_nonfinite = (~finite) & (a == b)
    return np.where(finite, ordered, np.where(same_nonfinite, 0.0, np.inf))


@register("trajectory_envelope")
def trajectory_envelope(ctx: OracleContext) -> OracleOutcome:
    """Require the candidate to stay inside a baseline-derived envelope.

    The envelope is the per-step min/max across the baseline's environments, widened by
    the declared tolerance. This is the right oracle when the baseline is an *ensemble*
    and the question is membership, not pointwise agreement. It says nothing when the
    baseline has one environment, and reports ``skipped`` in that case rather than
    manufacturing a degenerate band.
    """
    signal = str(ctx.spec.params.get("signal", ""))
    tol = ctx.tolerance
    baseline, candidate = ctx.signal_pair(signal)
    if baseline.shape[1] < 2:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="skipped",
            summary=f"{signal}: envelope needs a baseline ensemble (>=2 environments), got "
                    f"{baseline.shape[1]}",
            tolerance=tol.to_jsonable(),
            limitations="A one-member envelope is a point, not a band; reporting a pass here "
            "would be meaningless.",
        )
    lo = baseline.min(axis=1, keepdims=True) - tol.value
    hi = baseline.max(axis=1, keepdims=True) + tol.value
    breach = np.maximum(lo - candidate, candidate - hi)
    per_step_env = np.max(breach, axis=-1)
    worst = float(np.nanmax(per_step_env))
    metrics = {"worst_breach": worst, "n_samples": int(candidate.shape[0] * candidate.shape[1])}
    if worst <= 0.0:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=f"{signal}: candidate stays inside the baseline envelope (margin {-worst:.3e})",
            metrics=metrics, tolerance=tol.to_jsonable(),
        )
    record = localize(
        signal=signal, oracle=ctx.spec.name, baseline=baseline, candidate=candidate,
        per_step_env_error=per_step_env, threshold=0.0,
        config_differences=_config_differences(ctx),
    )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=f"{signal}: candidate leaves the baseline envelope by {worst:.3e} {tol.unit}",
        reason_codes=["IVF-ORACLE-NON_EQUIVALENT"], metrics=metrics, tolerance=tol.to_jsonable(),
        divergence=record,
    )
