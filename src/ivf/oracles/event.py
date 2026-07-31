# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Event and downstream-decision oracles.

These are the oracles that connect a numerical difference to something a user cares
about. A 3e-3 rad difference in a joint angle is not interesting on its own; a
termination that fires 40 steps earlier, or an episode that now fails, is.
"""

from __future__ import annotations

import numpy as np

from ..divergence import DivergenceRecord
from .base import OracleContext, OracleOutcome, register


def _first_crossing(arr: np.ndarray, threshold: float, condition: str) -> np.ndarray:
    """Return, per environment, the first step satisfying the condition, or ``-1``.

    ``arr`` is ``(steps, envs, dim)``; the condition is evaluated on the maximum over
    the last dimension, which makes a multi-component signal behave like "any component
    crossed".
    """
    reduced = np.nanmax(arr, axis=-1)
    if condition == "above":
        hit = reduced > threshold
    elif condition == "below":
        hit = reduced < threshold
    else:
        raise ValueError(f"unknown condition {condition!r} (expected above | below)")
    any_hit = hit.any(axis=0)
    first = np.argmax(hit, axis=0)
    return np.where(any_hit, first, -1)


@register("event_equivalence")
def event_equivalence(ctx: OracleContext) -> OracleOutcome:
    """Compare when and whether a semantic event fires in each subject.

    Two distinct failures are reported separately, because they mean different things:
    an event that fires in one subject and not the other is a *count* mismatch (a
    behavioral change), while an event that fires in both at different times is a
    *timing* delta measured in steps.
    """
    signal = str(ctx.spec.params.get("signal", ""))
    if not signal:
        raise ValueError(f"oracle {ctx.spec.name!r}: 'signal' is required")
    event_name = str(ctx.spec.params.get("event", ctx.spec.name))
    threshold = float(ctx.spec.params.get("threshold", 0.0))
    condition = str(ctx.spec.params.get("condition", "above"))
    tol = ctx.tolerance
    if tol.unit != "steps":
        raise ValueError(
            f"oracle {ctx.spec.name!r}: an event tolerance must be declared in 'steps', got {tol.unit!r}"
        )

    baseline, candidate = ctx.signal_pair(signal)
    base_first = _first_crossing(baseline, threshold, condition)
    cand_first = _first_crossing(candidate, threshold, condition)

    n_envs = int(min(base_first.size, cand_first.size))
    if n_envs < tol.min_samples:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=f"event {event_name!r}: {n_envs} environments is below the declared minimum "
                    f"of {tol.min_samples}",
            reason_codes=["IVF-SAMPLE-INSUFFICIENT"], tolerance=tol.to_jsonable(),
            metrics={"n_envs": n_envs},
        )

    base_first, cand_first = base_first[:n_envs], cand_first[:n_envs]
    occurred_base = base_first >= 0
    occurred_cand = cand_first >= 0
    count_mismatch = np.where(occurred_base != occurred_cand)[0]
    both = occurred_base & occurred_cand
    deltas = np.abs(cand_first - base_first)[both]
    worst_delta = int(deltas.max()) if deltas.size else 0

    metrics = {
        "event": event_name,
        "n_envs": n_envs,
        "baseline_occurrences": int(occurred_base.sum()),
        "candidate_occurrences": int(occurred_cand.sum()),
        "envs_with_count_mismatch": [int(i) for i in count_mismatch],
        "worst_timing_delta_steps": worst_delta,
        "median_timing_delta_steps": float(np.median(deltas)) if deltas.size else 0.0,
        "max_allowed_delta_steps": tol.value,
    }

    if count_mismatch.size:
        env = int(count_mismatch[0])
        record = DivergenceRecord(
            signal=signal, oracle=ctx.spec.name,
            first_event_disagreement_step=int(max(base_first[env], cand_first[env])),
            affected_env_ids=[int(i) for i in count_mismatch],
            classification="task_level_regression", confidence="medium",
            classification_basis=f"event {event_name!r} occurs in one subject and not the other "
                                 f"in {count_mismatch.size} environment(s)",
            limitations="A changed event count is a behavioral change; whether it is a "
                        "regression depends on which behavior the workload wants.",
        )
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="fail",
            summary=(f"event {event_name!r}: occurred in {int(occurred_base.sum())} baseline vs "
                     f"{int(occurred_cand.sum())} candidate environments"),
            reason_codes=["IVF-ORACLE-EVENT-COUNT-MISMATCH"], metrics=metrics,
            tolerance=tol.to_jsonable(), divergence=record,
        )

    if worst_delta > tol.value:
        env = int(np.argmax(np.where(both, np.abs(cand_first - base_first), -1)))
        record = DivergenceRecord(
            signal=signal, oracle=ctx.spec.name,
            first_event_disagreement_step=int(min(base_first[env], cand_first[env])),
            affected_env_ids=[env],
            classification="task_level_regression", confidence="medium",
            classification_basis=f"event {event_name!r} timing differs by {worst_delta} steps",
            limitations="Timing deltas grow naturally with horizon in chaotic regimes; this "
                        "oracle is informative only over horizons where the baseline timing is stable.",
        )
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="fail",
            summary=(f"event {event_name!r}: worst timing delta {worst_delta} steps exceeds the "
                     f"declared {tol.value:g} steps"),
            reason_codes=["IVF-ORACLE-EVENT-TIMING-DELTA"], metrics=metrics,
            tolerance=tol.to_jsonable(), divergence=record,
        )

    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="pass",
        summary=(f"event {event_name!r}: same occurrence pattern across {n_envs} environments, "
                 f"worst timing delta {worst_delta} steps <= {tol.value:g}"),
        metrics=metrics, tolerance=tol.to_jsonable(),
    )


@register("decision_equivalence")
def decision_equivalence(ctx: OracleContext) -> OracleOutcome:
    """Compare a downstream binary decision derived from each subject's trajectory.

    This is the oracle that answers "does the change alter what I would *do*". A
    decision is defined from an event: ``survived`` (the event never fired) or
    ``triggered`` (it did). Agreement is measured per environment.
    """
    signal = str(ctx.spec.params.get("signal", ""))
    decision = str(ctx.spec.params.get("decision", ctx.spec.name))
    threshold = float(ctx.spec.params.get("threshold", 0.0))
    condition = str(ctx.spec.params.get("condition", "above"))
    polarity = str(ctx.spec.params.get("polarity", "survived"))
    tol = ctx.tolerance
    if tol.unit != "fraction":
        raise ValueError(
            f"oracle {ctx.spec.name!r}: a decision tolerance must be declared in 'fraction', "
            f"got {tol.unit!r}"
        )

    baseline, candidate = ctx.signal_pair(signal)
    base_first = _first_crossing(baseline, threshold, condition)
    cand_first = _first_crossing(candidate, threshold, condition)
    n_envs = int(min(base_first.size, cand_first.size))
    base_first, cand_first = base_first[:n_envs], cand_first[:n_envs]

    if polarity == "survived":
        base_dec, cand_dec = base_first < 0, cand_first < 0
    elif polarity == "triggered":
        base_dec, cand_dec = base_first >= 0, cand_first >= 0
    else:
        raise ValueError(f"oracle {ctx.spec.name!r}: polarity must be 'survived' or 'triggered'")

    agree = base_dec == cand_dec
    rate = float(agree.mean()) if n_envs else float("nan")
    disagreeing = [int(i) for i in np.where(~agree)[0]]
    metrics = {
        "decision": decision,
        "n_envs": n_envs,
        "agreement_rate": rate,
        "minimum_agreement": tol.value,
        "baseline_positive_rate": float(base_dec.mean()) if n_envs else None,
        "candidate_positive_rate": float(cand_dec.mean()) if n_envs else None,
        "disagreeing_envs": disagreeing,
    }

    if n_envs < tol.min_samples:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=(f"decision {decision!r}: {n_envs} environments cannot resolve an agreement "
                     f"rate of {tol.value:g} (declared minimum {tol.min_samples})"),
            reason_codes=["IVF-SAMPLE-INSUFFICIENT"], metrics=metrics, tolerance=tol.to_jsonable(),
            limitations="With n environments the finest resolvable agreement rate is 1-1/n; "
                        "asking for a finer one is not answerable by this sample.",
        )

    if rate >= tol.value:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=f"decision {decision!r}: agreement {rate:.3f} >= {tol.value:g} over {n_envs} envs",
            metrics=metrics, tolerance=tol.to_jsonable(),
        )
    record = DivergenceRecord(
        signal=signal, oracle=ctx.spec.name, affected_env_ids=disagreeing,
        classification="task_level_regression", confidence="high",
        classification_basis=f"downstream decision {decision!r} differs in "
                             f"{len(disagreeing)}/{n_envs} environments",
        limitations="Identifies that the decision changed, not which subject decides correctly.",
    )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=f"decision {decision!r}: agreement {rate:.3f} is below the declared {tol.value:g}",
        reason_codes=["IVF-ORACLE-DECISION-DISAGREEMENT"], metrics=metrics,
        tolerance=tol.to_jsonable(), divergence=record,
    )
