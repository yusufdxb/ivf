# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Per-run invariant oracles: properties that must hold without any reference run.

Invariants are the cheapest and most trustworthy checks IVF has, because they need no
baseline and make no claim about which subject is correct. They are also the defense
against *identical wrongness*: a defect that affects both subjects equally passes every
pairwise comparison and is caught only here.
"""

from __future__ import annotations

import numpy as np

from ..divergence import DivergenceRecord
from .base import OracleContext, OracleOutcome, register

_CHECKS = ("finite_state", "unit_quaternion", "bounded")


@register("invariant")
def invariant(ctx: OracleContext) -> OracleOutcome:
    """Apply a per-run invariant to both subjects independently.

    Checks:

    ``finite_state``
        every element is finite in both subjects.
    ``unit_quaternion``
        every 4-vector has unit norm within the declared tolerance.
    ``bounded``
        every element lies within ``[lower, upper]``.
    """
    check = str(ctx.spec.params.get("check", ctx.spec.name))
    if check not in _CHECKS:
        raise ValueError(f"oracle {ctx.spec.name!r}: unknown invariant check {check!r}; known {_CHECKS}")
    requested = ctx.spec.params.get("signals")
    names = sorted(set(ctx.baseline.signals) & set(ctx.candidate.signals))
    if requested:
        names = [str(n) for n in requested]

    failures: list[str] = []
    first_bad: DivergenceRecord | None = None
    inspected = 0

    for role, sset in (("baseline", ctx.baseline), ("candidate", ctx.candidate)):
        for name in names:
            if name not in sset.signals:
                raise KeyError(f"invariant {ctx.spec.name!r}: {role} has no signal {name!r}")
            arr = sset.signals[name][ctx.warmup_steps:]
            inspected += int(arr.size)
            bad = _violations(check, arr, ctx)
            if bad is None:
                continue
            step, env = bad
            failures.append(f"{role}.{name} at step {step}, env {env}")
            if first_bad is None:
                first_bad = DivergenceRecord(
                    signal=name, oracle=ctx.spec.name,
                    first_tolerance_violation_step=int(step),
                    affected_env_ids=[int(env)],
                    baseline_values=[float(v) for v in np.atleast_1d(
                        ctx.baseline.signals[name][ctx.warmup_steps:][step, env])],
                    candidate_values=[float(v) for v in np.atleast_1d(
                        ctx.candidate.signals[name][ctx.warmup_steps:][step, env])],
                    classification="numerical_drift" if check == "finite_state" else "unknown",
                    confidence="high" if check == "finite_state" else "low",
                    classification_basis=f"invariant {check!r} violated in {role}",
                    limitations="An invariant violation localizes the symptom, not the cause.",
                )

    metrics = {"check": check, "signals": names, "elements_inspected": inspected}
    if failures:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="fail",
            summary=f"invariant {check!r} violated: " + "; ".join(failures[:4])
                    + ("" if len(failures) <= 4 else f" (+{len(failures) - 4} more)"),
            reason_codes=["IVF-ORACLE-INVARIANT-VIOLATION"], metrics=metrics,
            tolerance=None if ctx.spec.tolerance is None else ctx.spec.tolerance.to_jsonable(),
            divergence=first_bad,
        )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="pass",
        summary=f"invariant {check!r} held over {inspected} elements across {len(names)} signal(s)",
        metrics=metrics,
        tolerance=None if ctx.spec.tolerance is None else ctx.spec.tolerance.to_jsonable(),
        limitations="An invariant holding says the run is well-formed, not that it is right.",
    )


def _violations(check: str, arr: np.ndarray, ctx: OracleContext) -> tuple[int, int] | None:
    """Return the first ``(step, env)`` violating ``check``, or ``None``."""
    if check == "finite_state":
        mask = ~np.isfinite(arr)
    elif check == "unit_quaternion":
        if arr.shape[-1] != 4:
            return None
        if ctx.spec.tolerance is None:  # guarded by manifest parsing; protects direct callers
            raise ValueError("unit_quaternion requires an explicit tolerance")
        tol = ctx.spec.tolerance.value
        mask = np.abs(np.linalg.norm(arr, axis=-1) - 1.0) > tol
        mask = mask[..., None]
    elif check == "bounded":
        lower = float(ctx.spec.params.get("lower", -np.inf))
        upper = float(ctx.spec.params.get("upper", np.inf))
        with np.errstate(invalid="ignore"):
            mask = (arr < lower) | (arr > upper) | ~np.isfinite(arr)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(check)
    hits = np.argwhere(mask.any(axis=-1) if mask.ndim == 3 else mask)
    if hits.size == 0:
        return None
    return int(hits[0][0]), int(hits[0][1])
