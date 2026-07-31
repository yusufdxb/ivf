# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Metamorphic and capability oracles.

Metamorphic relations are the right tool where a golden trajectory is inappropriate:
they assert that a *transformation of the input* produces a known transformation of the
output, without anyone having to know the correct output. Only relations that are
defensible for the workload under test are implemented; a relation that is merely
plausible is worse than none, because a false alarm from a validator is expensive.
"""

from __future__ import annotations

import numpy as np

from .base import OracleContext, OracleOutcome, register

RELATIONS = (
    "env_permutation_invariance",
    "action_replay_consumption",
    "observation_definition_stability",
    "quaternion_double_cover",
)
"""Relations this build implements. Each is documented in ``docs/oracles.md`` with the
workload conditions under which it is valid."""


@register("metamorphic")
def metamorphic(ctx: OracleContext) -> OracleOutcome:
    """Check one declared metamorphic relation between the two subjects."""
    relation = str(ctx.spec.params.get("relation", ""))
    if relation not in RELATIONS:
        raise ValueError(
            f"oracle {ctx.spec.name!r}: unknown relation {relation!r}; known: {list(RELATIONS)}"
        )
    return _RELATION_FNS[relation](ctx)


def _env_permutation_invariance(ctx: OracleContext) -> OracleOutcome:
    """Per-environment outcomes must not depend on environment ordering.

    Valid whenever environments are independent, which is the normal case for cloned
    Isaac Lab environments and is false only when environments interact (shared ground
    contact across origins, global randomization coupling). The manifest guide records
    that precondition; this oracle cannot verify it.
    """
    signal = str(ctx.spec.params.get("signal", ""))
    tol = ctx.tolerance
    baseline, candidate = ctx.signal_pair(signal)
    if baseline.shape != candidate.shape:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=f"{signal}: shapes differ ({baseline.shape} vs {candidate.shape})",
            reason_codes=["IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH"], tolerance=tol.to_jsonable(),
        )
    n_envs = baseline.shape[1]
    # Greedy matching: for each candidate environment, find the baseline environment it
    # is closest to. The relation holds when that assignment is a permutation and every
    # matched pair agrees within tolerance.
    cost = np.array([
        [float(np.max(np.linalg.norm(candidate[:, j] - baseline[:, i], axis=-1)))
         for i in range(n_envs)]
        for j in range(n_envs)
    ])
    assignment = np.argmin(cost, axis=1)
    residuals = cost[np.arange(n_envs), assignment]
    is_permutation = len(set(assignment.tolist())) == n_envs
    worst = float(residuals.max()) if residuals.size else 0.0
    metrics = {
        "relation": "env_permutation_invariance", "n_envs": int(n_envs),
        "assignment": [int(a) for a in assignment], "is_permutation": bool(is_permutation),
        "worst_matched_residual": worst,
    }
    if is_permutation and worst <= tol.value:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=(f"{signal}: candidate environments are a permutation of the baseline set "
                     f"(worst matched residual {worst:.3e} {tol.unit})"),
            metrics=metrics, tolerance=tol.to_jsonable(),
            limitations="Assumes environments are independent; that precondition is declared "
                        "in the manifest, not verified here.",
        )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=(f"{signal}: per-environment outcomes are not preserved under reordering "
                 f"(permutation={is_permutation}, worst matched residual {worst:.3e} {tol.unit})"),
        reason_codes=["IVF-ORACLE-METAMORPHIC-VIOLATION"], metrics=metrics,
        tolerance=tol.to_jsonable(),
    )


def _action_replay_consumption(ctx: OracleContext) -> OracleOutcome:
    """A replayed run must consume exactly the recorded action sequence."""
    a, b = ctx.baseline.action_digest(), ctx.candidate.action_digest()
    metrics = {"relation": "action_replay_consumption", "baseline_digest": a, "candidate_digest": b}
    if not a or not b:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="skipped",
            summary="no action stream or action digest recorded for at least one subject",
            metrics=metrics,
            limitations="A source that records neither the actions nor their hash cannot "
                        "participate in this relation; that is a gap in the source, not a pass.",
        )
    if a == b:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=f"both subjects consumed the same action stream ({a[:16]}…)", metrics=metrics,
        )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary="subjects consumed different action streams; no dynamical comparison is meaningful",
        reason_codes=["IVF-CONTROL-ACTION-SEQUENCE-MISMATCH"], metrics=metrics,
    )


def _observation_definition_stability(ctx: OracleContext) -> OracleOutcome:
    """Changing output formatting must not change the observation definition."""
    a = ctx.baseline.observation_definition()
    b = ctx.candidate.observation_definition()
    metrics = {"relation": "observation_definition_stability", "baseline": a, "candidate": b}
    if a == b:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=f"both subjects define the same {len(a)} signal(s) with identical shapes",
            metrics=metrics,
        )
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    reshaped = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=(f"observation definitions differ: baseline-only {only_a}, candidate-only "
                 f"{only_b}, reshaped {reshaped}"),
        reason_codes=["IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH"], metrics=metrics,
    )


def _quaternion_double_cover(ctx: OracleContext) -> OracleOutcome:
    """``q`` and ``-q`` are the same rotation, so any pose metric must agree on both."""
    signal = str(ctx.spec.params.get("signal", ""))
    tol = ctx.tolerance
    baseline, _ = ctx.signal_pair(signal)
    if baseline.shape[-1] != 4:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="skipped",
            summary=f"{signal}: not a 4-vector signal, the double-cover relation does not apply",
            tolerance=tol.to_jsonable(),
        )
    dot = np.abs(np.sum(baseline * (-baseline), axis=-1))
    angle = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
    worst = float(np.max(angle))
    metrics = {"relation": "quaternion_double_cover", "worst_angle": worst}
    if worst <= tol.value:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=f"{signal}: geodesic metric treats q and -q as identical (worst {worst:.3e} rad)",
            metrics=metrics, tolerance=tol.to_jsonable(),
        )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=f"{signal}: the pose metric distinguishes q from -q by up to {worst:.3e} rad",
        reason_codes=["IVF-ORACLE-METAMORPHIC-VIOLATION"], metrics=metrics,
        tolerance=tol.to_jsonable(),
    )


_RELATION_FNS = {
    "env_permutation_invariance": _env_permutation_invariance,
    "action_replay_consumption": _action_replay_consumption,
    "observation_definition_stability": _observation_definition_stability,
    "quaternion_double_cover": _quaternion_double_cover,
}


@register("required_features")
def required_features(ctx: OracleContext) -> OracleOutcome:
    """Verify the environment actually provides the features the manifest depends on.

    Returns ``unsupported`` (never ``fail``) when a feature is missing: an absent
    backend capability is a property of the machine, not a defect in the change under
    test. Features are matched against what each subject *declares*; IVF does not
    independently probe a backend's capability surface, and the fault taxonomy records
    that gap explicitly.
    """
    required = [str(f) for f in ctx.spec.params.get("features", [])]
    if not required:
        raise ValueError(f"oracle {ctx.spec.name!r}: 'features' is required and must be non-empty")
    declared = set()
    for sset in (ctx.baseline, ctx.candidate):
        declared |= set(sset.metadata.get("declared_features", []) or [])
    missing = [f for f in required if f not in declared]
    metrics = {"required": required, "declared": sorted(declared), "missing": missing}
    if missing:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="unsupported",
            summary=f"required feature(s) not available: {missing}",
            reason_codes=["IVF-BACKEND-FEATURE-UNSUPPORTED"], metrics=metrics,
            limitations="IVF trusts the subject's own feature declaration; a run that "
                        "misreports a capability is not caught by this oracle.",
        )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="pass",
        summary=f"all {len(required)} required feature(s) are declared available", metrics=metrics,
        limitations="Declared, not probed.",
    )
