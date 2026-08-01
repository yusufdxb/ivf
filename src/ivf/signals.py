# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Signal sources: how a subject in a manifest becomes arrays and provenance.

Every source produces the same thing, a :class:`SignalSet`, so oracles never learn
where the data came from. Three sources ship:

``synthetic``
    A deterministic reference system integrated in numpy. This is what makes IVF
    usable, testable and demonstrable without a GPU, and it is the substrate for
    fault-injection calibration.

``parity_bundle``
    A trajectory bundle produced by ``isaaclab_contrib.parity``. Read with numpy and
    ``json`` only: **Isaac Lab is not imported**, because the bundle format is a
    directory of ``metadata.json`` + ``trajectories.npz`` and nothing about reading it
    requires a simulator.

``isaaclab``
    A live run. Requires the Isaac Lab runtime; raises :class:`RuntimeUnavailable`
    when it is absent, which the runner turns into ``UNSUPPORTED``, never ``FAIL``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import Subject


class RuntimeUnavailable(RuntimeError):
    """Raised when a source needs a runtime that is not installed.

    Distinct from a plain error so the runner can report ``UNSUPPORTED`` rather than
    pretending the change under test was at fault.
    """


class SignalSourceError(RuntimeError):
    """Raised when a source is configured wrongly or its inputs are unreadable."""


@dataclass
class SignalSet:
    """Signals plus the provenance the validity layer needs to judge comparability.

    Arrays are always ``(steps, num_envs, dim)`` float64. A scalar-per-env signal has
    ``dim == 1``; keeping the rank fixed removes an entire class of broadcasting bugs
    from the oracles.
    """

    role: str
    """``"baseline"`` or ``"candidate"``."""

    signals: dict[str, np.ndarray] = field(default_factory=dict)
    """Signal name to ``(steps, envs, dim)`` array."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provenance. Keys consumed by :mod:`ivf.validity` are documented in
    :data:`CONTROL_METADATA_KEYS`."""

    actions: np.ndarray | None = None
    """The action stream actually consumed, if the source records one."""

    complete: bool = True
    """``False`` when the run was truncated or failed part-way."""

    @property
    def steps(self) -> int:
        """Number of captured steps, or 0 when there are no signals."""
        return 0 if not self.signals else int(next(iter(self.signals.values())).shape[0])

    @property
    def num_envs(self) -> int:
        """Number of environments, or 0 when there are no signals."""
        return 0 if not self.signals else int(next(iter(self.signals.values())).shape[1])

    def observation_definition(self) -> dict[str, list[int]]:
        """Return the signal-name to shape map used as the observation-definition control."""
        return {name: list(arr.shape[1:]) for name, arr in sorted(self.signals.items())}

    def action_digest(self) -> str:
        """Return a stable digest of the action stream, or the recorded one.

        Falls back to the metadata hash so a bundle that stores only a hash (the
        parity bundle format does exactly this for large streams) still participates
        in the action-sequence control.
        """
        if self.actions is not None:
            return hashlib.sha256(np.ascontiguousarray(self.actions, dtype=np.float64).tobytes()).hexdigest()
        return str(self.metadata.get("action_stream_sha256", ""))


CONTROL_METADATA_KEYS = (
    "asset_identity",
    "initial_state_digest",
    "control_frequency_hz",
    "physics_dt",
    "seed",
    "task_variant",
    "solver_settings",
)
"""Metadata keys the validity layer reads. A source that omits one makes the
corresponding control *unverifiable*, which the validity layer reports explicitly
rather than treating as a match."""


# --------------------------------------------------------------------------------------
# Synthetic reference systems
# --------------------------------------------------------------------------------------

def _damped_pendulum(
    *,
    steps: int,
    num_envs: int,
    seed: int,
    dt: float = 0.005,
    gravity: float = 9.81,
    length: float = 0.5,
    damping: float = 0.02,
    initial_angle: float = 0.35,
    initial_angle_spread: float = 0.05,
    initial_velocity: float = 0.4,
    apply_reset_velocity: bool = True,
    drive_amplitude: float = 0.0,
    declared_drive_amplitude: float | None = None,
    drive_step_offset: int = 0,
    termination_angle: float = 1.2,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Integrate a damped driven pendulum with semi-implicit Euler.

    Chosen as the synthetic reference because it is the smallest system that exhibits
    every property the oracles need to be exercised against: a continuous trajectory,
    a conserved-ish quantity, a discrete event (angle limit crossing), and a downstream
    decision (did the episode survive). It is *not* a physics claim; it is a fixture
    whose exact values are reproducible on any machine.

    ``apply_reset_velocity=False`` reproduces the real Isaac Lab defect class where a
    reset writes position but drops velocity.

    Injected defects are deliberately **silent in the recorded metadata**: the asset
    identity, the initial-state digest and the recorded action stream all describe what
    was *declared*, not what the integrator actually did. That is what makes the
    corresponding real bugs hard, and a fixture whose defects announce themselves in
    provenance would calibrate the validity layer rather than the oracles.
    """
    rng = np.random.default_rng(seed)
    theta = initial_angle + rng.uniform(-initial_angle_spread, initial_angle_spread, size=num_envs)
    omega = np.full(num_envs, initial_velocity if apply_reset_velocity else 0.0)

    declared_amplitude = drive_amplitude if declared_drive_amplitude is None else declared_drive_amplitude

    angle = np.empty((steps, num_envs), dtype=np.float64)
    ang_vel = np.empty((steps, num_envs), dtype=np.float64)
    commanded = np.zeros((steps, num_envs), dtype=np.float64)

    for t in range(steps):
        # The commanded action is what the controller asked for; the applied torque is
        # what the (possibly defective) integrator used. Recording the former is what a
        # real action recorder does, and it is why a dropped or delayed action cannot be
        # caught by hashing the action stream.
        commanded[t, :] = declared_amplitude * np.sin(0.05 * t)
        drive_index = t - drive_step_offset
        torque = drive_amplitude * np.sin(0.05 * drive_index) if 0 <= drive_index < steps else 0.0
        accel = -(gravity / length) * np.sin(theta) - damping * omega + torque
        omega = omega + dt * accel
        theta = theta + dt * omega
        angle[t, :] = theta
        ang_vel[t, :] = omega

    half = 0.5 * angle
    quat = np.stack(
        [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=-1
    )
    signals = {
        "pole_angle": angle[:, :, None],
        "pole_ang_vel": ang_vel[:, :, None],
        "abs_pole_angle": np.abs(angle)[:, :, None],
        "pole_quat": quat,
    }
    metadata = {
        "system": "damped_pendulum",
        # Declared identity: the same asset declaration on both sides. A parameter defect
        # changes behaviour, not the declaration, which is exactly the case IVF must catch
        # with an oracle rather than with a metadata comparison.
        "asset_identity": "damped_pendulum.v1",
        "physics_dt": dt,
        "control_frequency_hz": 1.0 / dt,
        "seed": seed,
        "task_variant": "damped_pendulum.v1",
        "solver_settings": {"integrator": "semi_implicit_euler", "substeps": 1},
        "initial_state_digest": hashlib.sha256(
            json.dumps(
                {
                    "angle": initial_angle,
                    "spread": initial_angle_spread,
                    "velocity": initial_velocity,
                    "seed": seed,
                    "num_envs": num_envs,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "termination_angle": termination_angle,
        "effective_parameters": {
            "gravity": gravity, "length": length, "damping": damping,
            "applied_drive_amplitude": drive_amplitude, "drive_step_offset": drive_step_offset,
            "reset_velocity_applied": bool(apply_reset_velocity),
        },
    }
    return signals, commanded, metadata


SYNTHETIC_SYSTEMS: dict[str, Callable[..., tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]]] = {
    "damped_pendulum": _damped_pendulum,
}
"""Registry of synthetic reference systems, keyed by the manifest's ``system`` field."""


def _load_synthetic(subject: Subject, *, steps: int, num_envs: int, seed: int) -> SignalSet:
    """Build a :class:`SignalSet` from a synthetic reference system."""
    from .faults import apply_generation_fault, apply_trace_fault  # local: avoid import cycle

    params = dict(subject.params)
    system = str(params.pop("system", "damped_pendulum"))
    if system not in SYNTHETIC_SYSTEMS:
        raise SignalSourceError(
            f"subjects.{subject.role}.system: unknown synthetic system {system!r}; "
            f"known: {sorted(SYNTHETIC_SYSTEMS)}"
        )
    fault = params.pop("fault", None)
    fault_params = params.pop("fault_params", {}) or {}

    kwargs: dict[str, Any] = {"steps": steps, "num_envs": num_envs, "seed": seed, **params}
    # Pin the declared action amplitude before injection so an action-surface fault
    # changes what is applied without changing what was recorded as commanded.
    kwargs.setdefault("declared_drive_amplitude", kwargs.get("drive_amplitude", 0.0))
    if fault is not None:
        kwargs = apply_generation_fault(str(fault), kwargs, dict(fault_params))

    signals, actions, metadata = SYNTHETIC_SYSTEMS[system](**kwargs)
    signal_set = SignalSet(role=subject.role, signals=signals, metadata=metadata, actions=actions)
    if fault is not None:
        signal_set = apply_trace_fault(str(fault), signal_set, dict(fault_params))
        signal_set.metadata["injected_fault"] = fault
    return signal_set


# --------------------------------------------------------------------------------------
# Parity trajectory bundles
# --------------------------------------------------------------------------------------

def load_parity_bundle(path: str | Path, *, role: str = "baseline") -> SignalSet:
    """Read an ``isaaclab_contrib.parity`` trajectory bundle with numpy only.

    The bundle format is a directory containing ``metadata.json`` and
    ``trajectories.npz`` whose arrays are ``(horizon, num_envs, dim)``, with the action
    stream under the reserved key ``"__actions__"``. Nothing here imports Isaac Lab, so
    a laptop with no GPU can inspect evidence produced on a workstation.
    """
    root = Path(path)
    meta_path = root / "metadata.json"
    npz_path = root / "trajectories.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        raise SignalSourceError(
            f"{root}: not a parity trajectory bundle (expected metadata.json and trajectories.npz)"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    signals: dict[str, np.ndarray] = {}
    actions: np.ndarray | None = None
    with np.load(npz_path) as payload:
        for key in payload.files:
            array = np.asarray(payload[key], dtype=np.float64)
            if key == "__actions__":
                actions = array
                continue
            if array.ndim == 2:
                array = array[:, :, None]
            signals[key] = array

    declared = metadata.get("quantities", {})
    for name, shape in declared.items():
        if name not in signals:
            raise SignalSourceError(f"{root}: metadata declares quantity {name!r} but the payload lacks it")
        if list(signals[name].shape) != list(shape):
            raise SignalSourceError(
                f"{root}: quantity {name!r} shape {list(signals[name].shape)} does not match "
                f"the declared {list(shape)}"
            )

    resolved = metadata.get("resolved_config", {})
    scenario_cfg = resolved.get("scenario")
    if isinstance(scenario_cfg, str):
        try:
            scenario_cfg = json.loads(scenario_cfg)
        except json.JSONDecodeError:
            scenario_cfg = {"raw": scenario_cfg}

    mapped = {
        "source": "parity_bundle",
        "bundle_path": str(root),
        "backend": metadata.get("backend", ""),
        "device": metadata.get("device", ""),
        "asset_identity": _asset_identity(metadata, scenario_cfg),
        "physics_dt": metadata.get("physics_dt"),
        "control_frequency_hz": (1.0 / metadata["physics_dt"]) if metadata.get("physics_dt") else None,
        "seed": metadata.get("seed"),
        "task_variant": metadata.get("scenario_name", ""),
        "solver_settings": resolved.get("solver_settings", {}) or {},
        "action_stream_sha256": metadata.get("action_stream_sha256", ""),
        "initial_state_digest": _initial_state_digest(metadata, scenario_cfg),
        "capture_hook": metadata.get("capture_hook", ""),
        "joint_names": metadata.get("joint_names", []),
        "library_versions": metadata.get("library_versions", {}),
        "engine_versions": metadata.get("engine_versions", {}),
        "hardware": metadata.get("hardware", {}),
        "git_sha": metadata.get("git_sha", ""),
        "bundle_schema_version": metadata.get("schema_version", ""),
        "quantity_hashes": metadata.get("quantity_hashes", {}),
        "measured_physics_dt": metadata.get("measured_physics_dt"),
    }
    return SignalSet(role=role, signals=signals, metadata=mapped, actions=actions)


def _asset_identity(metadata: dict[str, Any], scenario_cfg: Any) -> str:
    """Derive an asset-identity string from bundle metadata.

    Prefers explicit asset hashes; falls back to scenario name plus joint ordering,
    which is the strongest identity a bundle without asset hashes can support. The
    fallback is recorded as such so the validity layer can say "unverifiable" rather
    than "matched".
    """
    asset_hashes = metadata.get("asset_hashes") or {}
    if asset_hashes:
        return "sha256:" + hashlib.sha256(
            json.dumps(asset_hashes, sort_keys=True).encode()
        ).hexdigest()[:32]
    name = metadata.get("scenario_name", "")
    joints = metadata.get("joint_names", [])
    return f"weak:{name}:joints={len(joints)}"


def _initial_state_digest(metadata: dict[str, Any], scenario_cfg: Any) -> str:
    """Derive an initial-state digest from the resolved scenario config, when present."""
    if not isinstance(scenario_cfg, dict):
        return ""
    keys = ("initial_state", "init_state", "reset", "actuation")
    payload = {k: scenario_cfg[k] for k in keys if k in scenario_cfg}
    payload["seed"] = metadata.get("seed")
    if len(payload) == 1:
        return ""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def load_trajectory_bundle_v1(path: str | Path, *, role: str = "baseline") -> SignalSet:
    """Read a ``trajectory_bundle/v1`` capture, refusing anything that violates the contract.

    Everything the v1 contract declares is mapped onto the metadata the validity layer
    consumes, so a strict capture makes controls *verifiable* rather than merely
    plausible: the quaternion layout, the frame convention, the reset semantics and the
    environment ordering all arrive as declarations instead of assumptions.
    """
    from .bundle import load_v1  # local import: keeps the legacy path free of the strict reader

    bundle = load_v1(path)
    contract = bundle.contract
    signals = {name: arr if arr.ndim == 3 else arr[:, :, None] for name, arr in bundle.arrays.items()}
    metadata = {
        "source": "trajectory_bundle/v1",
        "bundle_path": str(bundle.root),
        "capture_schema": "trajectory_bundle/v1",
        "run_status": contract.run_status,
        "declared_steps": contract.declared_steps,
        "captured_steps": contract.captured_steps,
        "task_variant": str(contract.task.get("id", "")),
        "asset_identity": str(contract.task.get("config_digest_sha256", "")),
        "backend": str(contract.backend.get("id", "")),
        "solver_settings": contract.backend.get("solver_settings", {}) or {},
        "declared_features": list(contract.backend.get("features", []) or []),
        "physics_dt": float(contract.timing["physics_dt"]),
        "control_frequency_hz": 1.0 / float(contract.timing["control_dt"]),
        "control_dt": float(contract.timing["control_dt"]),
        "action_applied": contract.timing.get("action_applied"),
        "capture_hook": contract.timing.get("capture_hook"),
        "seed": contract.seed.get("value"),
        "env_ids": list(contract.seed.get("env_ids", [])),
        "initial_state_digest": str(contract.reset.get("initial_state_digest", "")),
        "reset_semantics": str(contract.reset.get("semantics", "")),
        "termination": contract.termination,
        "frame_convention": str(contract.frames.get("convention", "")),
        "quaternion_layout": str(contract.quaternion.get("layout", "")),
        "library_versions": dict(contract.software),
        "units": {n: a.unit for n, a in contract.arrays.items()},
        "checksums": bundle.checksums,
    }
    return SignalSet(
        role=role, signals=signals, metadata=metadata, actions=bundle.actions,
        complete=(contract.run_status == "completed" and not contract.is_partial),
    )


def _load_parity_bundle_subject(subject: Subject) -> SignalSet:
    """Resolve a ``parity_bundle`` subject, strict when the capture declares v1.

    Routing on the capture contract rather than on a manifest flag is deliberate: a
    producer that has upgraded to the strict boundary should not need every consumer's
    manifest edited before the strictness takes effect.
    """
    from .bundle import is_v1

    path = subject.params.get("path")
    if not path:
        raise SignalSourceError(f"subjects.{subject.role}.path: required for kind 'parity_bundle'")
    if Path(path).is_dir() and is_v1(path):
        return load_trajectory_bundle_v1(path, role=subject.role)
    return load_parity_bundle(path, role=subject.role)


def _load_isaaclab(subject: Subject, *, steps: int, num_envs: int, seed: int) -> SignalSet:
    """Resolve an ``isaaclab`` subject by executing a live run.

    Not implemented in this build. The import probe is deliberate: it fails with
    :class:`RuntimeUnavailable` when Isaac Lab is absent (which is the common case and
    must not read as a regression), and with a clear "not implemented" when it is
    present, so nobody mistakes a missing feature for a passing one.
    """
    try:
        import isaaclab  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the host environment
        raise RuntimeUnavailable(
            f"subjects.{subject.role}: kind 'isaaclab' needs an importable Isaac Lab ({exc})"
        ) from exc
    raise RuntimeUnavailable(
        f"subjects.{subject.role}: live Isaac Lab execution is not implemented in IVF "
        f"{_version()}. Generate a trajectory bundle with isaaclab_contrib.parity and "
        "point this subject at it with kind 'parity_bundle'."
    )


def _version() -> str:
    """Return the installed IVF version string."""
    from . import __version__

    return __version__


def load_subject(subject: Subject, *, steps: int, num_envs: int, seed: int) -> SignalSet:
    """Resolve one manifest subject into a :class:`SignalSet`."""
    if subject.kind == "synthetic":
        return _load_synthetic(subject, steps=steps, num_envs=num_envs, seed=seed)
    if subject.kind == "parity_bundle":
        return _load_parity_bundle_subject(subject)
    if subject.kind == "isaaclab":
        return _load_isaaclab(subject, steps=steps, num_envs=num_envs, seed=seed)
    raise SignalSourceError(
        f"subjects.{subject.role}.kind: unknown source {subject.kind!r} "
        "(expected synthetic | parity_bundle | isaaclab)"
    )
