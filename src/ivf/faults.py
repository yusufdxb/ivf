# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned fault taxonomy and the detectability matrix it supports.

A validator that has never been shown a defect it failed to catch is a validator
whose detection claims are decoration. This module packages the fault classes so
they are versioned, importable and testable, and so every detection claim IVF makes
is backed by a matrix row rather than by an assertion in a README.

Two injection surfaces, matching the two places a defect can enter:

``generation``
    Mutates the *parameters* of a synthetic run before it is integrated, so the
    resulting trajectory is self-consistent. This emulates a real simulator defect:
    nothing about the recorded data looks corrupt, it is simply wrong.

``trace``
    Mutates the recorded arrays or metadata after the fact. This emulates a recorder,
    serialization or plumbing defect.

The taxonomy records, per fault, whether IVF is *designed* to detect it and which
oracle is responsible. ``expected_detectable=False`` entries are load-bearing: they
are the honest statement of what this validator does not see, and the calibration
run asserts they behave as declared rather than quietly passing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .signals import SignalSet

TAXONOMY_VERSION = "ivf.faults/v1"
"""Version of the fault taxonomy. Bumped when a fault's semantics change."""


@dataclass(frozen=True)
class FaultSpec:
    """One fault class and IVF's declared expectation about it."""

    name: str
    category: str
    """One of ``reset``, ``frame``, ``cloning``, ``convention``, ``action``,
    ``observation``, ``sensor``, ``timing``, ``units``, ``dynamics``, ``numerical``,
    ``protocol``, ``metadata``, ``capability``, ``control``."""

    surface: str
    """``"generation"``, ``"trace"``, or ``"none"`` for the negative control."""

    description: str
    expected_detectable: bool
    responsible_oracle: str
    """Which oracle type is expected to catch it, or ``"none"``."""

    minimum_severity: str
    """The smallest parameterization at which detection is claimed. Below this,
    detection is not claimed and the matrix records the miss without calling it a bug."""

    limitations: str = ""
    default_params: dict[str, Any] = field(default_factory=dict)


TAXONOMY: dict[str, FaultSpec] = {
    "none": FaultSpec(
        name="none",
        category="control",
        surface="none",
        description="Negative control: no fault injected.",
        expected_detectable=False,
        responsible_oracle="none",
        minimum_severity="n/a",
        limitations="Any detection here is a false positive and fails calibration.",
    ),
    "ignored_reset_velocity": FaultSpec(
        name="ignored_reset_velocity",
        category="reset",
        surface="generation",
        description="A reset writes position but silently drops the velocity component.",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="initial velocity >= 0.05 rad/s against a 2e-3 rad tolerance",
        limitations="Undetectable when the declared initial velocity is already zero.",
    ),
    "wrong_env_origin": FaultSpec(
        name="wrong_env_origin",
        category="frame",
        surface="trace",
        description="Environment origins are added to (or omitted from) a world-frame signal.",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="offset >= tolerance value",
        limitations="A per-env constant offset is indistinguishable from a genuine frame change "
        "without an explicit frame declaration; classification, not detection, is the weak part.",
        default_params={"spacing": 2.0},
    ),
    "incorrect_clone_count": FaultSpec(
        name="incorrect_clone_count",
        category="cloning",
        surface="trace",
        description="The candidate reports fewer environments than the baseline.",
        expected_detectable=True,
        responsible_oracle="experiment_validity",
        minimum_severity="any difference of one environment",
    ),
    "quaternion_ordering": FaultSpec(
        name="quaternion_ordering",
        category="convention",
        surface="trace",
        description="A 4-vector signal is emitted xyzw where wxyz was declared.",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="any non-identity rotation",
        limitations="Only applies to signals of last-dimension 4; a pure-identity trajectory "
        "is invariant under the swap and is honestly undetectable.",
    ),
    "dropped_action": FaultSpec(
        name="dropped_action",
        category="action",
        surface="generation",
        description="One commanded action is silently not applied (the Isaac Lab first-step wrench class).",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="drive amplitude >= 0.5 with a 2e-3 rad tolerance",
        limitations="Undetectable in a passive scenario, where there is no action to drop.",
        default_params={"step": 0},
    ),
    "shifted_action_timing": FaultSpec(
        name="shifted_action_timing",
        category="action",
        surface="generation",
        description="The action stream is applied one control step late.",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="drive amplitude >= 1.7 on the calibration workload, measured: at "
        "amplitude 1.5 a one-step shift produces 1.78e-3 rad of second-largest error against "
        "a 2e-3 rad budget and is NOT detected; the crossing is near amplitude 1.68",
        limitations="Undetectable for a constant action stream, and undetectable for a slowly "
        "varying one: the signal is the action's slope times one step, so a low-frequency or "
        "low-amplitude command hides a one-step delay below any operationally sane budget. "
        "This limit was measured, not assumed.",
        default_params={"offset": 1},
    ),
    "observation_field_swap": FaultSpec(
        name="observation_field_swap",
        category="observation",
        surface="trace",
        description="Two observation fields are transposed in the recorder.",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="the two fields differ by more than the tolerance somewhere",
    ),
    "stale_sensor_state": FaultSpec(
        name="stale_sensor_state",
        category="sensor",
        surface="trace",
        description="A signal is held at its previous value for a run of steps.",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="hold length >= 5 steps on a moving signal",
        limitations="Undetectable on a signal that is genuinely constant over the window.",
        default_params={"start": 20, "length": 10},
    ),
    "incorrect_timestep": FaultSpec(
        name="incorrect_timestep",
        category="timing",
        surface="generation",
        description="The integrator runs at a different dt than declared.",
        expected_detectable=True,
        responsible_oracle="experiment_validity",
        minimum_severity="any difference detectable in float64",
        limitations="Caught as an invalid experiment, not as a physics difference: comparing "
        "runs at different control frequencies is uninterpretable, so IVF refuses rather than scores.",
        default_params={"factor": 1.1},
    ),
    "unit_scaling": FaultSpec(
        name="unit_scaling",
        category="units",
        surface="generation",
        description="A physical parameter is supplied in the wrong unit (cm vs m class).",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="scale factor differing from 1 by >= 1%",
        default_params={"gravity_scale": 100.0},
    ),
    "altered_friction": FaultSpec(
        name="altered_friction",
        category="dynamics",
        surface="generation",
        description="A dissipation parameter differs between subjects.",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="damping delta >= 0.01 over a 400-step horizon",
        limitations="At small deltas over short horizons this is genuinely below the noise "
        "of any defensible tolerance; the matrix records the miss rather than hiding it.",
        default_params={"damping": 0.08},
    ),
    "altered_mass": FaultSpec(
        name="altered_mass",
        category="dynamics",
        surface="generation",
        description="An inertial parameter differs between subjects (length proxy on the pendulum).",
        expected_detectable=True,
        responsible_oracle="trajectory_equivalence",
        minimum_severity="length delta >= 1%",
        default_params={"length": 0.55},
    ),
    "silent_nan": FaultSpec(
        name="silent_nan",
        category="numerical",
        surface="trace",
        description="Non-finite values appear in a signal without an error being raised.",
        expected_detectable=True,
        responsible_oracle="invariant",
        minimum_severity="a single non-finite element",
        default_params={"step": 50, "env": 0},
    ),
    "truncated_rollout": FaultSpec(
        name="truncated_rollout",
        category="protocol",
        surface="trace",
        description="The run stops early and the evidence is shorter than declared.",
        expected_detectable=True,
        responsible_oracle="experiment_validity",
        minimum_severity="any missing step",
        default_params={"keep_fraction": 0.5},
    ),
    "corrupted_metadata": FaultSpec(
        name="corrupted_metadata",
        category="metadata",
        surface="trace",
        description="Provenance metadata no longer describes the run it is attached to.",
        expected_detectable=True,
        responsible_oracle="experiment_validity",
        minimum_severity="any change to a controlled field",
        default_params={"field": "asset_identity", "value": "corrupted-asset"},
    ),
    "unsupported_feature_misreported": FaultSpec(
        name="unsupported_feature_misreported",
        category="capability",
        surface="trace",
        description="A capability the run did not actually have is advertised as available.",
        expected_detectable=False,
        responsible_oracle="none",
        minimum_severity="n/a",
        limitations="IVF cannot detect a lie about a capability it never independently probes. "
        "This row exists to keep that gap visible; closing it needs a capability probe in "
        "`ivf doctor`, not a comparison oracle.",
        default_params={"feature": "articulation_tendons"},
    ),
}
"""The shipped taxonomy, keyed by fault name."""


class UnknownFault(KeyError):
    """Raised when a manifest or calibration run names a fault that is not in the taxonomy."""


def get_fault(name: str) -> FaultSpec:
    """Return the spec for ``name`` or raise :class:`UnknownFault`."""
    try:
        return TAXONOMY[name]
    except KeyError:
        raise UnknownFault(f"unknown fault {name!r}; known: {sorted(TAXONOMY)}") from None


# --------------------------------------------------------------------------------------
# Generation-surface injection
# --------------------------------------------------------------------------------------

def apply_generation_fault(name: str, kwargs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Return synthetic-system kwargs mutated by the named fault.

    Faults whose surface is not ``generation`` return the kwargs untouched; they are
    applied later by :func:`apply_trace_fault`.
    """
    spec = get_fault(name)
    if spec.surface != "generation":
        return kwargs
    merged = {**spec.default_params, **params}
    out = dict(kwargs)
    if name == "ignored_reset_velocity":
        out["apply_reset_velocity"] = False
    elif name == "dropped_action":
        # The pendulum's drive is the action; removing it entirely is the strongest
        # form of this class and keeps the fixture interpretable.
        out["drive_amplitude"] = 0.0
    elif name == "shifted_action_timing":
        out["drive_step_offset"] = int(merged["offset"])
    elif name == "incorrect_timestep":
        out["dt"] = float(out.get("dt", 0.005)) * float(merged["factor"])
    elif name == "unit_scaling":
        out["gravity"] = float(out.get("gravity", 9.81)) * float(merged["gravity_scale"])
    elif name == "altered_friction":
        out["damping"] = float(merged["damping"])
    elif name == "altered_mass":
        out["length"] = float(merged["length"])
    else:  # pragma: no cover - guarded by the surface check above
        raise UnknownFault(f"generation fault {name!r} has no implementation")
    return out


# --------------------------------------------------------------------------------------
# Trace-surface injection
# --------------------------------------------------------------------------------------

def apply_trace_fault(name: str, signal_set: SignalSet, params: dict[str, Any]) -> SignalSet:
    """Return a copy of ``signal_set`` mutated by the named fault.

    Faults whose surface is not ``trace`` are returned unchanged.
    """
    spec = get_fault(name)
    if spec.surface != "trace":
        return signal_set
    merged = {**spec.default_params, **params}
    out = SignalSet(
        role=signal_set.role,
        signals={k: v.copy() for k, v in signal_set.signals.items()},
        metadata=copy.deepcopy(signal_set.metadata),
        actions=None if signal_set.actions is None else signal_set.actions.copy(),
        complete=signal_set.complete,
    )
    target = str(merged.get("signal", "")) or _first_moving_signal(out)

    if name == "wrong_env_origin":
        spacing = float(merged["spacing"])
        offsets = np.arange(out.num_envs, dtype=np.float64)[None, :, None] * spacing
        out.signals[target] = out.signals[target] + offsets
    elif name == "incorrect_clone_count":
        keep = max(1, out.num_envs - 1)
        out.signals = {k: v[:, :keep, :] for k, v in out.signals.items()}
        if out.actions is not None and out.actions.ndim >= 2:
            out.actions = out.actions[:, :keep]
    elif name == "quaternion_ordering":
        quat = next((k for k, v in out.signals.items() if v.shape[-1] == 4), None)
        if quat is None:
            raise UnknownFault(
                "quaternion_ordering needs a signal with last dimension 4; none present"
            )
        out.signals[quat] = out.signals[quat][:, :, [1, 2, 3, 0]]
    elif name == "observation_field_swap":
        names = sorted(out.signals)
        if len(names) < 2:
            raise UnknownFault("observation_field_swap needs at least two signals")
        a, b = names[0], names[1]
        if out.signals[a].shape != out.signals[b].shape:
            raise UnknownFault(f"observation_field_swap needs same-shaped signals; {a} vs {b} differ")
        out.signals[a], out.signals[b] = out.signals[b], out.signals[a]
    elif name == "stale_sensor_state":
        start = int(merged["start"])
        length = int(merged["length"])
        end = min(out.steps, start + length)
        if start < out.steps:
            held = out.signals[target][start - 1 if start > 0 else 0]
            out.signals[target][start:end] = held
    elif name == "silent_nan":
        step = min(int(merged["step"]), out.steps - 1)
        env = min(int(merged["env"]), out.num_envs - 1)
        out.signals[target][step, env, 0] = np.nan
    elif name == "truncated_rollout":
        keep = max(1, int(out.steps * float(merged["keep_fraction"])))
        out.signals = {k: v[:keep] for k, v in out.signals.items()}
        if out.actions is not None:
            out.actions = out.actions[:keep]
        out.complete = False
    elif name == "corrupted_metadata":
        out.metadata[str(merged["field"])] = merged["value"]
    elif name == "unsupported_feature_misreported":
        out.metadata.setdefault("declared_features", []).append(str(merged["feature"]))
    else:  # pragma: no cover - guarded by the surface check above
        raise UnknownFault(f"trace fault {name!r} has no implementation")
    return out


def _first_moving_signal(signal_set: SignalSet) -> str:
    """Return the name of a signal that actually varies, so a perturbation is visible."""
    best_name, best_range = "", -1.0
    for name in sorted(signal_set.signals):
        arr = signal_set.signals[name]
        finite = arr[np.isfinite(arr)]
        span = float(np.ptp(finite)) if finite.size else 0.0
        if span > best_range:
            best_name, best_range = name, span
    if not best_name:
        raise UnknownFault("no signals available to inject into")
    return best_name
