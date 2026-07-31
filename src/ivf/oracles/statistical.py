# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Statistical equivalence, done so that a large sample cannot manufacture a failure.

The oracle here is **equivalence testing**, not significance testing. The question is
not "can I detect a difference" (with enough environments, always yes) but "is the
difference small enough not to matter". The decision rule is the confidence interval
against a declared equivalence margin:

* interval entirely inside ``±margin``  -> equivalent, ``pass``
* interval entirely outside, and the observed effect is at least the declared minimum
  meaningful effect -> ``fail``
* interval entirely outside but the effect is below the minimum meaningful effect ->
  ``pass`` with ``IVF-EFFECT-BELOW-MEANINGFUL``, because a real but operationally
  irrelevant difference is not a regression
* interval straddling the margin -> ``inconclusive`` with ``IVF-EQUIVALENCE-UNDECIDED``

That third case is the one this design exists for, and the fourth is the one most
harnesses report as a pass. Failing to prove equivalence is not proof of equivalence.

Intervals are percentile bootstrap on the *paired* per-environment differences: paired
because both subjects ran the same seeds and environments, and bootstrap because it
needs no distributional assumption and no SciPy. The resample seed is recorded so the
interval is reproducible.
"""

from __future__ import annotations

import numpy as np

from .base import OracleContext, OracleOutcome, register

_SUMMARIES = ("mean", "final", "max", "abs_max", "rms")

_BOOTSTRAP_MIN_PAIRS = 8
"""Hard floor on paired samples, independent of what the manifest declares.

A percentile bootstrap resamples the observed values; with a handful of pairs the
resulting interval describes those few points, not the population, and it collapses to a
point whenever the observed spread is small. Below this floor the oracle returns
``inconclusive`` no matter how confident the arithmetic looks. This floor is a property
of the method, so a manifest cannot lower it.
"""


def _summarize(arr: np.ndarray, how: str) -> np.ndarray:
    """Reduce ``(steps, envs, dim)`` to one scalar per environment."""
    magnitude = np.linalg.norm(arr, axis=-1)
    if how == "mean":
        return magnitude.mean(axis=0)
    if how == "final":
        return magnitude[-1]
    if how == "max":
        return magnitude.max(axis=0)
    if how == "abs_max":
        return np.abs(magnitude).max(axis=0)
    if how == "rms":
        return np.sqrt((magnitude ** 2).mean(axis=0))
    raise ValueError(f"unknown summary {how!r}; known: {_SUMMARIES}")


def _bootstrap_ci(diffs: np.ndarray, alpha: float, seed: int, draws: int = 5000) -> tuple[float, float]:
    """Return the percentile bootstrap interval for the mean paired difference."""
    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(draws, n))
    means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


@register("statistical_equivalence")
def statistical_equivalence(ctx: OracleContext) -> OracleOutcome:
    """Test paired equivalence of a per-environment summary against a declared margin."""
    signal = str(ctx.spec.params.get("signal", ""))
    if not signal:
        raise ValueError(f"oracle {ctx.spec.name!r}: 'signal' is required")
    summary = str(ctx.spec.params.get("summary", "mean"))
    tol = ctx.tolerance
    if tol.kind != "statistical":
        raise ValueError(
            f"oracle {ctx.spec.name!r}: tolerance.kind must be 'statistical', got {tol.kind!r}"
        )

    baseline, candidate = ctx.signal_pair(signal)
    if baseline.shape[1] != candidate.shape[1]:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=(f"{signal}: {baseline.shape[1]} vs {candidate.shape[1]} environments; "
                     "a paired analysis needs matched environments"),
            reason_codes=["IVF-CONTROL-ENV-COUNT-MISMATCH"], tolerance=tol.to_jsonable(),
        )

    base_summary = _summarize(baseline, summary)
    cand_summary = _summarize(candidate, summary)
    diffs = cand_summary - base_summary
    n = int(diffs.size)
    margin = tol.value
    mme = tol.minimum_meaningful_effect

    metrics: dict = {
        "summary": summary,
        "n_pairs": n,
        "mean_difference": float(np.mean(diffs)) if n else None,
        "sd_difference": float(np.std(diffs, ddof=1)) if n > 1 else None,
        "equivalence_margin": margin,
        "minimum_meaningful_effect": mme,
        "alpha": ctx.alpha,
        "bootstrap_seed": ctx.seed,
    }

    floor = max(tol.min_samples, _BOOTSTRAP_MIN_PAIRS)
    if n < floor:
        limitation = (
            "" if n >= _BOOTSTRAP_MIN_PAIRS else
            f"Below {_BOOTSTRAP_MIN_PAIRS} paired samples a percentile bootstrap interval "
            "describes the observed points rather than the population, and collapses to a "
            "point when their spread is small. IVF refuses the claim rather than reporting "
            "a confident-looking degenerate interval."
        )
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=(f"{signal}: {n} paired samples is below the required minimum of {floor} "
                     f"(manifest declares {tol.min_samples}, method floor is {_BOOTSTRAP_MIN_PAIRS})"),
            reason_codes=["IVF-SAMPLE-INSUFFICIENT"], metrics=metrics, tolerance=tol.to_jsonable(),
            limitations=limitation,
        )
    if not np.all(np.isfinite(diffs)):
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=f"{signal}: non-finite values make the paired difference undefined",
            reason_codes=["IVF-ORACLE-INVARIANT-VIOLATION"], metrics=metrics,
            tolerance=tol.to_jsonable(),
        )

    lo, hi = _bootstrap_ci(diffs, ctx.alpha, ctx.seed)
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    dz = float(np.mean(diffs) / sd) if sd > 0 else (0.0 if np.mean(diffs) == 0 else float("inf"))
    observed = float(abs(np.mean(diffs)))
    metrics.update({
        "ci_low": lo, "ci_high": hi, "ci_level": 1.0 - ctx.alpha,
        "cohens_dz": dz, "observed_absolute_effect": observed,
    })

    inside = (lo >= -margin) and (hi <= margin)
    outside = (lo > margin) or (hi < -margin)

    if inside:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=(f"{signal}: mean paired difference {np.mean(diffs):+.3e} {tol.unit}, "
                     f"{100 * (1 - ctx.alpha):.0f}% CI [{lo:+.3e}, {hi:+.3e}] lies inside "
                     f"±{margin:g} {tol.unit}"),
            metrics=metrics, tolerance=tol.to_jsonable(),
            limitations="Equivalence is established only for this summary of this signal at "
                        "this sample size; it does not generalize to other quantities.",
        )
    if outside:
        if mme is not None and observed < mme:
            return OracleOutcome(
                name=ctx.spec.name, type=ctx.spec.type, status="pass",
                summary=(f"{signal}: difference {np.mean(diffs):+.3e} {tol.unit} is statistically "
                         f"resolvable but below the declared minimum meaningful effect "
                         f"of {mme:g} {tol.unit}"),
                reason_codes=["IVF-EFFECT-BELOW-MEANINGFUL"], metrics=metrics,
                tolerance=tol.to_jsonable(),
                limitations="This is an operational judgement encoded in the manifest, not a "
                            "statistical one: the difference is real.",
            )
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="fail",
            summary=(f"{signal}: mean paired difference {np.mean(diffs):+.3e} {tol.unit}, "
                     f"{100 * (1 - ctx.alpha):.0f}% CI [{lo:+.3e}, {hi:+.3e}] lies entirely outside "
                     f"±{margin:g} {tol.unit} (Cohen's dz {dz:+.2f})"),
            reason_codes=["IVF-ORACLE-NON_EQUIVALENT"], metrics=metrics, tolerance=tol.to_jsonable(),
            limitations="Establishes a difference of this size, not which subject is correct.",
        )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
        summary=(f"{signal}: {100 * (1 - ctx.alpha):.0f}% CI [{lo:+.3e}, {hi:+.3e}] straddles the "
                 f"±{margin:g} {tol.unit} margin; neither equivalence nor difference is established "
                 f"at n={n}"),
        reason_codes=["IVF-EQUIVALENCE-UNDECIDED"], metrics=metrics, tolerance=tol.to_jsonable(),
        limitations="Failing to prove equivalence is not proof of equivalence. More paired "
                    "samples, or a wider declared margin with a stated rationale, would resolve it.",
    )
