# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Oracle behaviour, one oracle at a time.

Every oracle is exercised for all three of the outcomes it can produce, because the
interesting failure mode of a validator is not "it missed a bug" but "it returned a
confident answer where it should have said it could not tell".
"""

from __future__ import annotations

import numpy as np
import pytest

from ivf.manifest import OracleSpec, Tolerance
from ivf.oracles import OracleContext, get_oracle, registered_oracles

from .conftest import make_signals

pytestmark = pytest.mark.unit


def tol(value, unit="rad", kind="numerical", aggregation="max", min_samples=1, mme=None):
    """Build a fully qualified tolerance for a test."""
    return Tolerance(
        value=value, unit=unit, scope="per-step error over the horizon",
        rationale="chosen for this fixture so the assertion is unambiguous",
        aggregation=aggregation, min_samples=min_samples, kind=kind,
        minimum_meaningful_effect=mme,
    )


def run(oracle_type, name, params, baseline, candidate, *, tolerance=None, alpha=0.05, warmup=0):
    """Invoke one oracle against a hand-built pair of signal sets."""
    spec = OracleSpec(type=oracle_type, name=name, params=params, tolerance=tolerance)
    ctx = OracleContext(
        manifest=None, spec=spec, baseline=baseline, candidate=candidate,
        warmup_steps=warmup, alpha=alpha, seed=7,
    )
    return get_oracle(oracle_type)(ctx)


def test_every_registered_oracle_is_documented():
    """The doctor advertises this list, so it must not contain surprises."""
    assert set(registered_oracles()) == {
        "decision_equivalence", "event_equivalence", "exact_equivalence", "invariant",
        "metamorphic", "required_features", "statistical_equivalence",
        "trajectory_envelope", "trajectory_equivalence", "ulp_equivalence",
    }


# -- trajectory equivalence --------------------------------------------------------------

def test_trajectory_equivalence_passes_on_identical_signals():
    a, b = make_signals("baseline"), make_signals("candidate")
    out = run("trajectory_equivalence", "x", {"signal": "x"}, a, b, tolerance=tol(1e-9))
    assert out.status == "pass"
    assert out.metrics["worst_aggregated_error"] == 0.0


def test_trajectory_equivalence_fails_and_localizes():
    a = make_signals("baseline")
    b = make_signals("candidate")
    b.signals["x"][10, 2, 0] += 0.5
    out = run("trajectory_equivalence", "x", {"signal": "x"}, a, b, tolerance=tol(1e-3))
    assert out.status == "fail"
    assert out.reason_codes == ["IVF-ORACLE-NON_EQUIVALENT"]
    assert out.divergence.first_tolerance_violation_step == 10
    assert out.divergence.affected_env_ids == [2]


def test_trajectory_equivalence_is_inconclusive_below_min_samples():
    """Too little data must not become a pass."""
    a, b = make_signals("baseline", steps=2, envs=2), make_signals("candidate", steps=2, envs=2)
    out = run("trajectory_equivalence", "x", {"signal": "x"}, a, b,
              tolerance=tol(1e-9, min_samples=10_000))
    assert out.status == "inconclusive"
    assert "IVF-SAMPLE-INSUFFICIENT" in out.reason_codes


def test_warmup_steps_are_discarded():
    a = make_signals("baseline")
    b = make_signals("candidate")
    b.signals["x"][0:5] += 10.0
    strict = run("trajectory_equivalence", "x", {"signal": "x"}, a, b, tolerance=tol(1e-3))
    trimmed = run("trajectory_equivalence", "x", {"signal": "x"}, a, b, tolerance=tol(1e-3), warmup=5)
    assert strict.status == "fail"
    assert trimmed.status == "pass"


def test_geodesic_metric_handles_the_quaternion_double_cover():
    """q and -q are the same rotation; a componentwise metric would report 2 rad."""
    steps, envs = 20, 3
    angles = np.linspace(0.0, 1.0, steps)[:, None]
    half = 0.5 * np.repeat(angles, envs, axis=1)
    quat = np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=-1)
    a = make_signals("baseline", steps=steps, envs=envs)
    b = make_signals("candidate", steps=steps, envs=envs)
    a.signals["q"], b.signals["q"] = quat, -quat
    out = run("trajectory_equivalence", "q", {"signal": "q", "metric": "geodesic"}, a, b,
              tolerance=tol(1e-9))
    assert out.status == "pass"


def test_geodesic_metric_equals_the_known_rotation_angle():
    """Anchors the metric to ground truth rather than only to self-consistency.

    Two quaternions built from z-rotations of theta_a and theta_b are separated by
    exactly |theta_a - theta_b|. A same-shape metric that is off by a constant factor
    passes every symmetry and double-cover check while silently halving or doubling
    every orientation error it reports, so the scale has to be pinned directly.
    """
    from ivf.oracles.numeric import _element_error

    for delta in (1e-6, 1e-3, 0.1, 1.0, 2.5):
        a = np.array([[[np.cos(0.0), 0.0, 0.0, np.sin(0.0)]]])
        b = np.array([[[np.cos(delta / 2), 0.0, 0.0, np.sin(delta / 2)]]])
        assert _element_error(a, b, "geodesic")[0, 0] == pytest.approx(delta, rel=1e-9, abs=1e-12)


def test_geodesic_metric_is_well_conditioned_at_tiny_angles():
    """The reason for the chord form: arccos would report ~3e-8 rad for identical inputs."""
    from ivf.oracles.numeric import _element_error

    q = np.array([[[np.cos(0.3), 0.0, 0.0, np.sin(0.3)]]])
    assert _element_error(q, q.copy(), "geodesic")[0, 0] == 0.0


def test_unknown_signal_names_available_ones():
    a, b = make_signals("baseline"), make_signals("candidate")
    with pytest.raises(KeyError, match="available"):
        run("trajectory_equivalence", "z", {"signal": "z"}, a, b, tolerance=tol(1e-3))


# -- aggregation -------------------------------------------------------------------------

def test_second_largest_keeps_single_env_sensitivity_but_trims_one_tail():
    """The reduction that motivates the whole aggregation field."""
    from ivf.oracles.base import aggregate

    values = np.array([0.0, 0.0, 0.0, 5.0])
    assert aggregate(values, "max") == 5.0
    assert aggregate(values, "second_largest") == 0.0
    assert aggregate(np.array([0.0, 3.0, 5.0]), "second_largest") == 3.0


def test_aggregation_of_non_finite_values_is_infinite_not_silently_dropped():
    from ivf.oracles.base import aggregate

    assert aggregate(np.array([0.0, np.nan]), "max") == float("inf")


# -- exact and ULP ------------------------------------------------------------------------

def test_exact_equivalence_distinguishes_identity_from_near_identity():
    a, b = make_signals("baseline"), make_signals("candidate")
    assert run("exact_equivalence", "x", {"signal": "x"}, a, b).status == "pass"
    b.signals["x"][0, 0, 0] = np.nextafter(b.signals["x"][0, 0, 0], 1.0)
    assert run("exact_equivalence", "x", {"signal": "x"}, a, b).status == "fail"


def test_ulp_distance_counts_representable_steps():
    from ivf.oracles.numeric import _ulp_distance

    a = np.array([1.0], dtype=np.float32)
    b = np.array([np.nextafter(np.float32(1.0), np.float32(2.0))], dtype=np.float32)
    assert _ulp_distance(a, b)[0] == 1.0
    assert _ulp_distance(a, a)[0] == 0.0
    assert not np.isfinite(_ulp_distance(a, np.array([np.inf], dtype=np.float32))[0])


def test_ulp_oracle_passes_within_budget_and_fails_beyond_it():
    a, b = make_signals("baseline"), make_signals("candidate")
    b.signals["x"] = np.nextafter(b.signals["x"].astype(np.float32), np.float32(1e9)).astype(np.float64)
    within = run("ulp_equivalence", "x", {"signal": "x"}, a, b, tolerance=tol(2.0, unit="ulp"))
    beyond = run("ulp_equivalence", "x", {"signal": "x"}, a, b, tolerance=tol(0.0, unit="ulp"))
    assert within.status == "pass"
    assert beyond.status == "fail"
    assert "IVF-ORACLE-ULP-EXCEEDED" in beyond.reason_codes


# -- invariants ---------------------------------------------------------------------------

def test_finite_state_invariant_catches_a_single_nan():
    a, b = make_signals("baseline"), make_signals("candidate")
    b.signals["x"][7, 1, 0] = np.nan
    out = run("invariant", "finite_state", {"check": "finite_state"}, a, b)
    assert out.status == "fail"
    assert out.divergence.first_tolerance_violation_step == 7


def test_bounded_invariant_uses_declared_limits():
    a, b = make_signals("baseline"), make_signals("candidate")
    out = run("invariant", "bounded", {"check": "bounded", "lower": -0.5, "upper": 0.5,
                                       "signals": ["x"]}, a, b)
    assert out.status == "fail"


def test_invariant_needs_no_reference_and_checks_both_subjects():
    """The only defense against a defect that affects both subjects identically.

    A shared-layer bug that corrupts both runs the same way agrees perfectly with itself,
    so every pairwise comparison passes. Only a reference-free invariant catches it.
    """
    a, b = make_signals("baseline"), make_signals("candidate")
    a.signals["x"][3, 0, 0] = 500.0
    b.signals["x"][3, 0, 0] = 500.0
    trajectory = run("trajectory_equivalence", "x", {"signal": "x"}, a, b, tolerance=tol(1e-9))
    invariant = run("invariant", "bounded", {"check": "bounded", "lower": -10.0, "upper": 10.0,
                                             "signals": ["x"]}, a, b)
    assert trajectory.status == "pass"  # identical wrongness is invisible pairwise
    assert invariant.status == "fail"


# -- events and decisions -------------------------------------------------------------------

def _event_pair(shift: int = 0, envs: int = 8):
    """Build signals whose threshold crossing is shifted by ``shift`` steps."""
    steps = 60
    ramp = np.linspace(0.0, 2.0, steps)[:, None, None]
    a = make_signals("baseline", steps=steps, envs=envs)
    b = make_signals("candidate", steps=steps, envs=envs)
    a.signals["e"] = np.repeat(ramp, envs, axis=1)
    shifted = np.roll(np.repeat(ramp, envs, axis=1), shift, axis=0)
    if shift > 0:
        shifted[:shift] = 0.0
    b.signals["e"] = shifted
    return a, b


def test_event_equivalence_passes_within_the_timing_budget():
    a, b = _event_pair(shift=1)
    out = run("event_equivalence", "term", {"signal": "e", "threshold": 1.0, "condition": "above"},
              a, b, tolerance=tol(1, unit="steps", kind="event", min_samples=4))
    assert out.status == "pass"


def test_event_equivalence_fails_on_a_timing_delta():
    a, b = _event_pair(shift=5)
    out = run("event_equivalence", "term", {"signal": "e", "threshold": 1.0, "condition": "above"},
              a, b, tolerance=tol(1, unit="steps", kind="event", min_samples=4))
    assert out.status == "fail"
    assert out.reason_codes == ["IVF-ORACLE-EVENT-TIMING-DELTA"]


def test_event_equivalence_reports_a_count_mismatch_separately():
    a, b = _event_pair()
    b.signals["e"][:, 0, :] = 0.0
    out = run("event_equivalence", "term", {"signal": "e", "threshold": 1.0, "condition": "above"},
              a, b, tolerance=tol(1, unit="steps", kind="event", min_samples=4))
    assert out.status == "fail"
    assert out.reason_codes == ["IVF-ORACLE-EVENT-COUNT-MISMATCH"]


def test_event_tolerance_must_be_declared_in_steps():
    a, b = _event_pair()
    with pytest.raises(ValueError, match="declared in 'steps'"):
        run("event_equivalence", "term", {"signal": "e", "threshold": 1.0}, a, b,
            tolerance=tol(1, unit="rad", kind="event"))


def test_decision_equivalence_measures_agreement_per_environment():
    a, b = _event_pair()
    b.signals["e"][:, 0, :] = 0.0
    out = run("decision_equivalence", "survived",
              {"signal": "e", "threshold": 1.0, "polarity": "survived"}, a, b,
              tolerance=tol(1.0, unit="fraction", kind="engineering", aggregation="all", min_samples=4))
    assert out.status == "fail"
    assert out.metrics["agreement_rate"] == pytest.approx(7 / 8)
    assert out.metrics["disagreeing_envs"] == [0]


def test_decision_equivalence_refuses_a_rate_the_sample_cannot_resolve():
    a, b = _event_pair(envs=3)
    out = run("decision_equivalence", "survived", {"signal": "e", "threshold": 1.0}, a, b,
              tolerance=tol(0.99, unit="fraction", kind="engineering", aggregation="all",
                            min_samples=100))
    assert out.status == "inconclusive"
    assert "IVF-SAMPLE-INSUFFICIENT" in out.reason_codes


# -- statistical equivalence -----------------------------------------------------------------

def _paired(delta: float, envs: int = 40, spread: float = 0.01, jitter: float = 0.0):
    """Build a paired fixture whose per-environment difference is ``delta`` plus jitter.

    ``jitter`` is what gives the paired differences any spread at all; without it every
    environment differs by exactly ``delta`` and the interval is a point, which is a
    degenerate case rather than a realistic one.
    """
    rng = np.random.default_rng(0)
    steps = 30
    base = rng.normal(0.0, spread, size=(steps, envs, 1)) + 1.0
    per_env = rng.normal(0.0, jitter, size=(1, envs, 1)) if jitter else 0.0
    a = make_signals("baseline", steps=steps, envs=envs)
    b = make_signals("candidate", steps=steps, envs=envs)
    a.signals["s"] = base
    b.signals["s"] = base + delta + per_env
    return a, b


def test_statistical_equivalence_establishes_equivalence_inside_the_margin():
    a, b = _paired(delta=0.0)
    out = run("statistical_equivalence", "s", {"signal": "s", "summary": "mean"}, a, b,
              tolerance=tol(0.05, kind="statistical", aggregation="mean", min_samples=10, mme=0.01))
    assert out.status == "pass"
    assert out.metrics["ci_low"] <= 0.0 <= out.metrics["ci_high"]


def test_statistical_equivalence_fails_outside_the_margin():
    a, b = _paired(delta=0.5)
    out = run("statistical_equivalence", "s", {"signal": "s", "summary": "mean"}, a, b,
              tolerance=tol(0.05, kind="statistical", aggregation="mean", min_samples=10, mme=0.01))
    assert out.status == "fail"


def test_a_real_but_operationally_irrelevant_difference_is_not_a_regression():
    """The guard against a huge sample turning a trivial difference into a failure."""
    a, b = _paired(delta=0.002, envs=400, spread=0.001, jitter=0.0005)
    out = run("statistical_equivalence", "s", {"signal": "s", "summary": "mean"}, a, b,
              tolerance=tol(0.001, kind="statistical", aggregation="mean", min_samples=10, mme=0.05))
    assert out.status == "pass"
    assert out.reason_codes == ["IVF-EFFECT-BELOW-MEANINGFUL"]


def test_an_interval_straddling_the_margin_is_undecided_not_a_pass():
    """Failing to prove equivalence is not proof of equivalence."""
    a, b = _paired(delta=0.02, envs=12, spread=0.01, jitter=0.02)
    out = run("statistical_equivalence", "s", {"signal": "s", "summary": "mean"}, a, b,
              tolerance=tol(0.02, kind="statistical", aggregation="mean", min_samples=10, mme=0.001))
    assert out.status == "inconclusive"
    assert "IVF-EQUIVALENCE-UNDECIDED" in out.reason_codes


def test_bootstrap_floor_cannot_be_lowered_by_a_manifest():
    a, b = _paired(delta=0.0, envs=4)
    out = run("statistical_equivalence", "s", {"signal": "s"}, a, b,
              tolerance=tol(0.05, kind="statistical", aggregation="mean", min_samples=1, mme=0.01))
    assert out.status == "inconclusive"
    assert "method floor" in out.summary


def test_bootstrap_interval_is_reproducible_for_a_given_seed():
    a, b = _paired(delta=0.1, jitter=0.05)
    kwargs = dict(tolerance=tol(0.05, kind="statistical", aggregation="mean", min_samples=10, mme=0.01))
    first = run("statistical_equivalence", "s", {"signal": "s"}, a, b, **kwargs)
    second = run("statistical_equivalence", "s", {"signal": "s"}, a, b, **kwargs)
    assert first.metrics["ci_low"] == second.metrics["ci_low"]


# -- metamorphic and capability ----------------------------------------------------------------

def test_env_permutation_invariance_holds_under_reordering():
    a = make_signals("baseline", envs=5)
    b = make_signals("candidate", envs=5)
    order = [3, 1, 4, 0, 2]
    b.signals["x"] = a.signals["x"][:, order, :]
    out = run("metamorphic", "perm", {"relation": "env_permutation_invariance", "signal": "x"},
              a, b, tolerance=tol(1e-9))
    assert out.status == "pass"
    assert out.metrics["is_permutation"] is True


def test_env_permutation_invariance_fails_when_an_outcome_changed():
    a = make_signals("baseline", envs=5)
    b = make_signals("candidate", envs=5)
    b.signals["x"][:, 2, :] += 1.0
    out = run("metamorphic", "perm", {"relation": "env_permutation_invariance", "signal": "x"},
              a, b, tolerance=tol(1e-6))
    assert out.status == "fail"
    assert "IVF-ORACLE-METAMORPHIC-VIOLATION" in out.reason_codes


def test_action_replay_relation_skips_rather_than_passes_without_evidence():
    a, b = make_signals("baseline"), make_signals("candidate")
    a.actions = None
    a.metadata.pop("action_stream_sha256", None)
    out = run("metamorphic", "replay", {"relation": "action_replay_consumption"}, a, b)
    assert out.status == "skipped"


def test_observation_definition_stability_reports_what_moved():
    a, b = make_signals("baseline"), make_signals("candidate")
    b.signals["extra"] = b.signals["x"].copy()
    out = run("metamorphic", "obs", {"relation": "observation_definition_stability"}, a, b)
    assert out.status == "fail"
    assert "extra" in out.summary


def test_unknown_metamorphic_relation_is_rejected():
    a, b = make_signals("baseline"), make_signals("candidate")
    with pytest.raises(ValueError, match="unknown relation"):
        run("metamorphic", "bogus", {"relation": "gravity_is_down"}, a, b)


def test_missing_feature_is_unsupported_not_a_failure():
    """An absent backend capability is a property of the machine, not a defect."""
    a, b = make_signals("baseline"), make_signals("candidate")
    out = run("required_features", "features", {"features": ["articulation_tendons"]}, a, b)
    assert out.status == "unsupported"
    assert out.reason_codes == ["IVF-BACKEND-FEATURE-UNSUPPORTED"]


def test_trajectory_envelope_skips_on_a_degenerate_baseline():
    a = make_signals("baseline", envs=1)
    b = make_signals("candidate", envs=1)
    out = run("trajectory_envelope", "env", {"signal": "x"}, a, b, tolerance=tol(0.1))
    assert out.status == "skipped"


def test_trajectory_envelope_detects_a_breach():
    a = make_signals("baseline", envs=6)
    b = make_signals("candidate", envs=6)
    b.signals["x"] += 5.0
    out = run("trajectory_envelope", "env", {"signal": "x"}, a, b, tolerance=tol(0.1))
    assert out.status == "fail"
