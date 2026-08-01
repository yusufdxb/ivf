# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The capture specification: what to run, on what, and with which declared semantics.

Kept separate from the capture itself so it can be parsed, validated and unit-tested
without Isaac Lab present. That matters more than it sounds: the most common capture
failure is a typo in a spec, and finding it after a 40-second Kit boot is the wrong
place to find it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SPEC_SCHEMA = "parity.capture/v1"

SUPPORTED_TASKS = frozenset({"cartpole_passive"})
"""One workload, supported completely. Adding a second is a deliberate act, not a
plugin registration: every task needs its own declared reset and termination semantics,
and a generic loader would let someone add a task without declaring them."""

SUPPORTED_BACKENDS = frozenset({"physx", "newton"})


class SpecError(ValueError):
    """Raised when a capture spec is malformed."""


@dataclass(frozen=True)
class Defect:
    """A deliberate, test-only defect injected into the simulator run.

    This exists so the validation framework can be shown catching a defect that was
    genuinely produced by a simulator, not by mutating a recorded array afterwards.
    Every field is off by default and the capture prints a loud banner when any is on,
    because a defective bundle that nobody can tell is defective is a trap.
    """

    drop_reset_velocity: bool = False
    """Reset writes joint position together with an explicit zero joint velocity.

    This explicit simulator-level perturbation is modeled after a reset-semantics
    failure observed while developing the parity harness. The bundle records the switch
    for auditability; the verdict logic does not consume that bookkeeping field.
    """

    damping_scale: float = 1.0
    """Multiplies cart-actuator damping. A value near 1 is the benign-difference knob:
    a real physical configuration change producing a real, small numerical difference."""

    def any_enabled(self) -> bool:
        """Whether any defect is active."""
        return self.drop_reset_velocity or self.damping_scale != 1.0

    def describe(self) -> list[str]:
        """Human-readable list of the active defects."""
        out = []
        if self.drop_reset_velocity:
            out.append("drop_reset_velocity: reset writes zero joint velocity")
        if self.damping_scale != 1.0:
            out.append(f"damping_scale: cart damping multiplied by {self.damping_scale}")
        return out

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {"drop_reset_velocity": self.drop_reset_velocity, "damping_scale": self.damping_scale}


@dataclass(frozen=True)
class CaptureSpec:
    """A parsed capture specification."""

    name: str
    task: str
    backend: str
    device: str
    num_envs: int
    steps: int
    seed: int
    initial_pole_angle: float
    initial_pole_angle_spread: float
    initial_pole_velocity: float
    termination_angle: float
    defect: Defect = field(default_factory=Defect)
    source_path: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view, used as the task configuration identity."""
        return {
            "schema": SPEC_SCHEMA,
            "name": self.name,
            "task": self.task,
            "backend": self.backend,
            "device": self.device,
            "num_envs": self.num_envs,
            "steps": self.steps,
            "seed": self.seed,
            "initial_pole_angle": self.initial_pole_angle,
            "initial_pole_angle_spread": self.initial_pole_angle_spread,
            "initial_pole_velocity": self.initial_pole_velocity,
            "termination_angle": self.termination_angle,
            "defect": self.defect.to_jsonable(),
        }

    def config_identity(self) -> dict[str, Any]:
        """Return the configuration fields that define *what experiment this is*.

        The device and the injected defect are excluded on purpose. Two captures that
        differ only by a defect are the same declared experiment, which is exactly what
        makes them comparable; if the defect changed the identity, the validity layer
        would refuse the comparison and the defect could never be caught by an oracle.
        """
        payload = self.to_jsonable()
        for key in ("backend", "device", "defect", "name"):
            payload.pop(key, None)
        return payload


def parse_spec(text: str, *, source_path: str | None = None) -> CaptureSpec:
    """Parse and validate a capture spec from YAML text."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecError("top level: expected a mapping")

    schema = str(raw.get("schema_version", ""))
    if schema != SPEC_SCHEMA:
        raise SpecError(f"schema_version: expected {SPEC_SCHEMA!r}, got {schema!r}")

    known = {"schema_version", "name", "task", "backend", "device", "num_envs", "steps",
             "seed", "initial_pole_angle", "initial_pole_angle_spread", "initial_pole_velocity",
             "termination_angle", "defect"}
    unknown = set(raw) - known
    if unknown:
        raise SpecError(f"top level: unknown key(s) {sorted(unknown)}")

    task = str(raw.get("task", ""))
    if task not in SUPPORTED_TASKS:
        raise SpecError(f"task: {task!r} is not supported; this adapter supports {sorted(SUPPORTED_TASKS)}")
    backend = str(raw.get("backend", ""))
    if backend not in SUPPORTED_BACKENDS:
        raise SpecError(f"backend: {backend!r} is not one of {sorted(SUPPORTED_BACKENDS)}")

    for key in ("name", "num_envs", "steps", "seed"):
        if key not in raw:
            raise SpecError(f"{key}: required")

    defect_raw = raw.get("defect") or {}
    if not isinstance(defect_raw, dict):
        raise SpecError("defect: expected a mapping")
    unknown_defect = set(defect_raw) - {"drop_reset_velocity", "damping_scale"}
    if unknown_defect:
        raise SpecError(f"defect: unknown key(s) {sorted(unknown_defect)}")

    steps = int(raw["steps"])
    num_envs = int(raw["num_envs"])
    if steps < 1 or num_envs < 1:
        raise SpecError("steps and num_envs must both be at least 1")

    drop_reset_velocity = defect_raw.get("drop_reset_velocity", False)
    if not isinstance(drop_reset_velocity, bool):
        raise SpecError("defect.drop_reset_velocity: expected a YAML boolean")

    return CaptureSpec(
        name=str(raw["name"]),
        task=task,
        backend=backend,
        device=str(raw.get("device", "cuda:0")),
        num_envs=num_envs,
        steps=steps,
        seed=int(raw["seed"]),
        initial_pole_angle=float(raw.get("initial_pole_angle", 0.4)),
        initial_pole_angle_spread=float(raw.get("initial_pole_angle_spread", 0.0)),
        initial_pole_velocity=float(raw.get("initial_pole_velocity", 0.5)),
        termination_angle=float(raw.get("termination_angle", 1.0)),
        defect=Defect(
            drop_reset_velocity=drop_reset_velocity,
            damping_scale=float(defect_raw.get("damping_scale", 1.0)),
        ),
        source_path=source_path,
    )


def load_spec(path: str | Path) -> CaptureSpec:
    """Load and validate a capture spec from disk."""
    p = Path(path)
    if not p.is_file():
        raise SpecError(f"capture spec not found: {p}")
    return parse_spec(p.read_text(encoding="utf-8"), source_path=str(p))
