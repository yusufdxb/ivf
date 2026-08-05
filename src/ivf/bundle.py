# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""``trajectory_bundle/v1``: the versioned capture boundary.

This is the contract between whatever produced a rollout and whatever judges it. It is
deliberately a **profile layered on the existing parity bundle**, not a replacement: the
physical layout is still ``metadata.json`` plus ``trajectories.npz``, so every bundle
already on disk stays readable and the capture side did not have to be rewritten. What
v1 adds is the set of declarations that make a bundle *self-describing enough to judge*,
plus two files that make it verifiable:

```
<bundle>/
  metadata.json        parity RunMetadata, plus a "capture_contract" block (this spec)
  trajectories.npz     (steps, envs, dim) arrays, unchanged
  integrity.json       parity capture-integrity report, optional
  CHECKSUMS.sha256     sha256 of every other file
  COMPLETE             completion marker, written LAST
```

The ordering rule is the whole point of the marker. ``COMPLETE`` is written after the
payload and after the checksums, so a bundle from a crashed, killed, or out-of-disk
capture is missing it. A reader therefore never has to guess whether a short rollout was
intentional: an undeclared partial run is structurally detectable rather than inferred
from the data looking odd.

Why the declarations below and not others: each one is a question whose wrong answer
silently invalidates a comparison rather than making it fail loudly. A quaternion read
``xyzw`` when it was written ``wxyz`` produces a plausible, wrong rotation. A run at a
different control frequency produces a plausible, meaningless step-by-step diff. Those
are the failure modes worth structural defence.

Validation refuses a bundle rather than degrading it. A refused bundle never reaches an
oracle, because an oracle's job is to compare two interpretable things, and deciding
whether they are interpretable is not the oracle's job.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "trajectory_bundle/v1"
"""Capture-boundary schema identifier."""

COMPLETION_MARKER = "COMPLETE"
"""Written last. Its absence means the capture did not finish."""

CHECKSUM_FILE = "CHECKSUMS.sha256"
METADATA_FILE = "metadata.json"
PAYLOAD_FILE = "trajectories.npz"

CONTRACT_KEY = "capture_contract"
"""Key inside ``metadata.json`` holding the v1 declarations."""

RUN_STATUSES = frozenset({"completed", "partial", "failed"})

QUATERNION_LAYOUTS = {"wxyz": True, "xyzw": False}
"""Layout to ``scalar_first``. Any other layout, or a ``scalar_first`` that contradicts
the layout, is an ambiguity and is refused: there is no safe default here, and guessing
produces a rotation that is wrong by up to 180 degrees while looking entirely plausible."""

FRAME_CONVENTIONS = frozenset({
    "world_z_up_right_handed",
    "world_y_up_right_handed",
    "env_local_z_up_right_handed",
})

RESET_SEMANTICS = frozenset({
    "writes_pose_and_velocity",
    "writes_pose_only",
    "writes_nothing",
})
"""``writes_pose_only`` is not a bug marker. It is a legitimate declaration for a
workload whose reset genuinely only sets pose, and declaring it is what lets a reader
tell that case apart from the defect where velocity was dropped by accident."""

REQUIRED_CONTRACT_SECTIONS = (
    "schema", "run_status", "declared_steps", "captured_steps",
    "task", "backend", "software", "seed", "timing", "frames",
    "quaternion", "reset", "termination", "arrays",
)


class BundleContractError(ValueError):
    """Raised when a bundle does not satisfy ``trajectory_bundle/v1``.

    Carries a stable :attr:`reason_code` so a refusal is greppable in CI and diffable
    across runs, exactly like an oracle outcome.
    """

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(f"[{reason_code}] {message}")
        self.reason_code = reason_code
        self.message = message


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ArraySpec:
    """The declaration for one captured array."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    unit: str
    frame: str
    semantics: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "shape": list(self.shape), "dtype": self.dtype, "unit": self.unit,
            "frame": self.frame, "semantics": self.semantics,
        }


@dataclass
class CaptureContract:
    """The parsed and validated ``capture_contract`` block."""

    run_status: str
    declared_steps: int
    captured_steps: int
    task: dict[str, Any]
    backend: dict[str, Any]
    software: dict[str, Any]
    hardware: dict[str, Any]
    seed: dict[str, Any]
    timing: dict[str, Any]
    frames: dict[str, Any]
    quaternion: dict[str, Any]
    reset: dict[str, Any]
    termination: dict[str, Any]
    arrays: dict[str, ArraySpec] = field(default_factory=dict)
    capture: dict[str, Any] = field(default_factory=dict)
    """Optional capture-identity block: ``capture_id``, ``created_utc``, ``producer``,
    ``execution``. Added additively, so every bundle written before it stays loadable and
    simply reports no capture identity rather than a fabricated one."""

    @property
    def is_partial(self) -> bool:
        """Whether fewer steps were captured than declared."""
        return self.captured_steps < self.declared_steps

    @property
    def scalar_first(self) -> bool:
        """Whether quaternions are stored scalar-first (``wxyz``)."""
        return bool(self.quaternion["scalar_first"])

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "schema": SCHEMA,
            "run_status": self.run_status,
            "declared_steps": self.declared_steps,
            "captured_steps": self.captured_steps,
            "task": self.task, "backend": self.backend, "software": self.software,
            "hardware": self.hardware,
            "seed": self.seed, "timing": self.timing, "frames": self.frames,
            "quaternion": self.quaternion, "reset": self.reset,
            "termination": self.termination,
            "capture": self.capture,
            "arrays": {n: a.to_jsonable() for n, a in sorted(self.arrays.items())},
        }


@dataclass
class TrajectoryBundleV1:
    """A validated ``trajectory_bundle/v1`` on disk."""

    root: Path
    contract: CaptureContract
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    actions: np.ndarray | None = None
    checksums: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def _require(condition: bool, code: str, message: str) -> None:
    """Raise :class:`BundleContractError` when ``condition`` is false."""
    if not condition:
        raise BundleContractError(code, message)


def _parse_contract(raw: Any, where: str) -> CaptureContract:
    """Parse and validate the ``capture_contract`` block."""
    _require(isinstance(raw, dict), "IVF-BUNDLE-CONTRACT-MISSING",
             f"{where}: metadata.json has no '{CONTRACT_KEY}' block, so this is not a "
             f"{SCHEMA} bundle")

    schema = str(raw.get("schema", ""))
    _require(schema == SCHEMA, "IVF-BUNDLE-SCHEMA-UNSUPPORTED",
             f"{where}: capture schema {schema!r} is not {SCHEMA!r}")

    missing = [k for k in REQUIRED_CONTRACT_SECTIONS if k not in raw]
    _require(not missing, "IVF-BUNDLE-CONTRACT-INCOMPLETE",
             f"{where}: capture contract is missing required declaration(s): {missing}")

    # -- completion and partial-run honesty --------------------------------------------
    run_status = str(raw["run_status"])
    _require(run_status in RUN_STATUSES, "IVF-BUNDLE-CONTRACT-INCOMPLETE",
             f"{where}: run_status {run_status!r} is not one of {sorted(RUN_STATUSES)}")
    declared_steps = int(raw["declared_steps"])
    captured_steps = int(raw["captured_steps"])
    _require(declared_steps > 0 and captured_steps > 0, "IVF-BUNDLE-CONTRACT-INCOMPLETE",
             f"{where}: declared_steps and captured_steps must both be positive")
    _require(captured_steps <= declared_steps, "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT",
             f"{where}: captured_steps ({captured_steps}) exceeds declared_steps "
             f"({declared_steps}), which no honest capture can do")
    _require(not (captured_steps < declared_steps and run_status == "completed"),
             "IVF-BUNDLE-PARTIAL-UNDECLARED",
             f"{where}: only {captured_steps} of {declared_steps} declared steps were "
             "captured, but run_status says 'completed'. A short rollout must declare "
             "itself partial; otherwise a consumer cannot tell truncation from intent")

    # -- quaternion convention ----------------------------------------------------------
    quat = raw["quaternion"]
    _require(isinstance(quat, dict) and "layout" in quat, "IVF-BUNDLE-QUATERNION-AMBIGUOUS",
             f"{where}: quaternion.layout is not declared. Reading wxyz data as xyzw "
             "produces a plausible, wrong rotation, so there is no safe default")
    layout = str(quat["layout"])
    _require(layout in QUATERNION_LAYOUTS, "IVF-BUNDLE-QUATERNION-AMBIGUOUS",
             f"{where}: quaternion.layout {layout!r} is not one of {sorted(QUATERNION_LAYOUTS)}")
    expected_scalar_first = QUATERNION_LAYOUTS[layout]
    _require("scalar_first" in quat, "IVF-BUNDLE-QUATERNION-AMBIGUOUS",
             f"{where}: quaternion.scalar_first is not declared")
    _require(bool(quat["scalar_first"]) == expected_scalar_first,
             "IVF-BUNDLE-QUATERNION-AMBIGUOUS",
             f"{where}: quaternion.layout={layout!r} implies scalar_first="
             f"{expected_scalar_first}, but the bundle declares "
             f"{bool(quat['scalar_first'])}. The two declarations contradict each other")

    # -- frames, reset, termination ------------------------------------------------------
    frames = raw["frames"]
    _require(isinstance(frames, dict) and str(frames.get("convention", "")) in FRAME_CONVENTIONS,
             "IVF-BUNDLE-FRAME-UNDECLARED",
             f"{where}: frames.convention must be one of {sorted(FRAME_CONVENTIONS)}, got "
             f"{frames.get('convention') if isinstance(frames, dict) else frames!r}")
    reset = raw["reset"]
    _require(isinstance(reset, dict) and str(reset.get("semantics", "")) in RESET_SEMANTICS,
             "IVF-BUNDLE-RESET-UNDECLARED",
             f"{where}: reset.semantics must be one of {sorted(RESET_SEMANTICS)}")
    termination = raw["termination"]
    _require(isinstance(termination, dict) and "declared" in termination,
             "IVF-BUNDLE-TERMINATION-UNDECLARED",
             f"{where}: termination.declared is required (use false for a workload that "
             "cannot terminate, so silence is never mistaken for 'never happened')")

    # -- timing ---------------------------------------------------------------------------
    timing = raw["timing"]
    for key in ("physics_dt", "control_dt", "action_applied", "capture_hook"):
        _require(isinstance(timing, dict) and key in timing, "IVF-BUNDLE-CONTRACT-INCOMPLETE",
                 f"{where}: timing.{key} is required")
    _require(float(timing["physics_dt"]) > 0 and float(timing["control_dt"]) > 0,
             "IVF-BUNDLE-CONTRACT-INCOMPLETE", f"{where}: timing steps must be positive")

    # -- seed and environment ordering -----------------------------------------------------
    seed = raw["seed"]
    _require(isinstance(seed, dict) and "value" in seed and "env_ids" in seed,
             "IVF-BUNDLE-CONTRACT-INCOMPLETE",
             f"{where}: seed.value and seed.env_ids are required. Environment ordering is "
             "what makes a paired comparison meaningful, so it is declared, not assumed")
    env_ids = list(seed["env_ids"])
    _require(len(set(env_ids)) == len(env_ids), "IVF-BUNDLE-CONTRACT-INCOMPLETE",
             f"{where}: seed.env_ids contains duplicates")

    # -- arrays -----------------------------------------------------------------------------
    arrays_raw = raw["arrays"]
    _require(isinstance(arrays_raw, dict) and arrays_raw, "IVF-BUNDLE-CONTRACT-INCOMPLETE",
             f"{where}: at least one array must be declared")
    arrays: dict[str, ArraySpec] = {}
    for name, spec in arrays_raw.items():
        _require(isinstance(spec, dict), "IVF-BUNDLE-CONTRACT-INCOMPLETE",
                 f"{where}: arrays.{name} must be a mapping")
        for key in ("shape", "dtype", "unit", "frame"):
            _require(key in spec, "IVF-BUNDLE-CONTRACT-INCOMPLETE",
                     f"{where}: arrays.{name}.{key} is required")
        shape = tuple(int(v) for v in spec["shape"])
        _require(len(shape) == 3, "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT",
                 f"{where}: arrays.{name}.shape must be (steps, envs, dim), got {list(shape)}")
        _require(shape[0] == captured_steps, "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT",
                 f"{where}: arrays.{name} declares {shape[0]} steps but the contract "
                 f"declares captured_steps={captured_steps}")
        _require(shape[1] == len(env_ids), "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT",
                 f"{where}: arrays.{name} declares {shape[1]} environments but "
                 f"seed.env_ids lists {len(env_ids)}")
        arrays[str(name)] = ArraySpec(
            name=str(name), shape=shape, dtype=str(spec["dtype"]), unit=str(spec["unit"]),
            frame=str(spec["frame"]), semantics=str(spec.get("semantics", "")),
        )

    for section in ("task", "backend", "software"):
        _require(isinstance(raw[section], dict) and raw[section],
                 "IVF-BUNDLE-CONTRACT-INCOMPLETE", f"{where}: {section} must be a non-empty mapping")
    for key in ("id", "config_digest_sha256"):
        _require(key in raw["task"], "IVF-BUNDLE-CONTRACT-INCOMPLETE",
                 f"{where}: task.{key} is required")
    _require("id" in raw["backend"], "IVF-BUNDLE-CONTRACT-INCOMPLETE",
             f"{where}: backend.id is required")

    return CaptureContract(
        run_status=run_status, declared_steps=declared_steps, captured_steps=captured_steps,
        task=dict(raw["task"]), backend=dict(raw["backend"]), software=dict(raw["software"]),
        hardware=dict(raw.get("hardware", {})),
        seed=dict(seed), timing=dict(timing), frames=dict(frames), quaternion=dict(quat),
        reset=dict(reset), termination=dict(termination), arrays=arrays,
        capture=dict(raw.get("capture", {}) or {}),
    )


def _verify_checksums(root: Path) -> dict[str, str]:
    """Verify every recorded digest, returning the recorded mapping."""
    checksum_path = root / CHECKSUM_FILE
    _require(checksum_path.is_file(), "IVF-BUNDLE-CHECKSUM-MISSING",
             f"{root}: {CHECKSUM_FILE} is absent, so the payload cannot be verified")
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        recorded[name] = digest
    _require(bool(recorded), "IVF-BUNDLE-CHECKSUM-MISSING",
             f"{root}: {CHECKSUM_FILE} records no files")

    for name, digest in sorted(recorded.items()):
        path = root / name
        _require(path.is_file(), "IVF-BUNDLE-CHECKSUM-MISMATCH",
                 f"{root}: {name} is recorded in {CHECKSUM_FILE} but missing from the bundle")
        actual = sha256_file(path)
        _require(actual == digest, "IVF-BUNDLE-CHECKSUM-MISMATCH",
                 f"{root}: {name} does not match its recorded digest "
                 f"(recorded {digest[:16]}, found {actual[:16]})")

    present = {
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and p.name not in (CHECKSUM_FILE, COMPLETION_MARKER)
    }
    extra = sorted(present - set(recorded))
    _require(not extra, "IVF-BUNDLE-CHECKSUM-MISMATCH",
             f"{root}: file(s) present but not recorded in {CHECKSUM_FILE}: {extra}")
    return recorded


def load_v1(path: str | Path, *, verify_checksums: bool = True) -> TrajectoryBundleV1:
    """Load and fully validate a ``trajectory_bundle/v1``.

    Raises :class:`BundleContractError` on any contract violation. Nothing partially
    valid is returned: a caller either gets an interpretable bundle or an explanation.
    """
    root = Path(path)
    _require(root.is_dir(), "IVF-BUNDLE-CONTRACT-MISSING", f"{root}: not a directory")

    # Completion first. Everything else is meaningless if the capture did not finish.
    marker = root / COMPLETION_MARKER
    _require(marker.is_file(), "IVF-BUNDLE-INCOMPLETE",
             f"{root}: no {COMPLETION_MARKER} marker. It is written last, so its absence "
             "means the capture was interrupted and the payload may be truncated")
    try:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleContractError("IVF-BUNDLE-INCOMPLETE",
                                  f"{root}: {COMPLETION_MARKER} is not valid JSON ({exc})") from exc
    _require(str(marker_payload.get("schema", "")) == SCHEMA, "IVF-BUNDLE-SCHEMA-UNSUPPORTED",
             f"{root}: {COMPLETION_MARKER} declares schema "
             f"{marker_payload.get('schema')!r}, expected {SCHEMA!r}")

    checksums = _verify_checksums(root) if verify_checksums else {}
    if verify_checksums and "checksums_sha256" in marker_payload:
        actual = sha256_file(root / CHECKSUM_FILE)
        _require(actual == marker_payload["checksums_sha256"], "IVF-BUNDLE-CHECKSUM-MISMATCH",
                 f"{root}: {CHECKSUM_FILE} does not match the digest sealed in "
                 f"{COMPLETION_MARKER}")

    metadata_path = root / METADATA_FILE
    payload_path = root / PAYLOAD_FILE
    _require(metadata_path.is_file() and payload_path.is_file(), "IVF-BUNDLE-CONTRACT-MISSING",
             f"{root}: expected {METADATA_FILE} and {PAYLOAD_FILE}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = _parse_contract(metadata.get(CONTRACT_KEY), str(root))

    arrays: dict[str, np.ndarray] = {}
    actions: np.ndarray | None = None
    with np.load(payload_path) as payload:
        for key in payload.files:
            raw = np.asarray(payload[key])
            if key == "__actions__":
                actions = raw.astype(np.float64)
                continue
            arrays[key] = raw

    declared = set(contract.arrays)
    actual_names = set(arrays)
    _require(declared == actual_names, "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT",
             f"{root}: declared arrays {sorted(declared)} do not match the payload "
             f"{sorted(actual_names)}")
    for name, spec in contract.arrays.items():
        got = arrays[name]
        _require(tuple(got.shape) == spec.shape, "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT",
                 f"{root}: array {name!r} has shape {list(got.shape)} but declares "
                 f"{list(spec.shape)}")
        _require(got.dtype.name == spec.dtype, "IVF-BUNDLE-ARRAY-DTYPE-INCONSISTENT",
                 f"{root}: array {name!r} has dtype {got.dtype.name!r} but declares "
                 f"{spec.dtype!r}")
        if got.shape[-1] == 4 and "quat" in name:
            norms = np.linalg.norm(got.astype(np.float64), axis=-1)
            _require(bool(np.all(np.abs(norms - 1.0) < 1e-3)), "IVF-BUNDLE-QUATERNION-AMBIGUOUS",
                     f"{root}: array {name!r} declares a quaternion layout but contains "
                     "non-unit 4-vectors, so the declaration cannot be trusted")

    return TrajectoryBundleV1(
        root=root, contract=contract, metadata=metadata,
        arrays={k: v.astype(np.float64) for k, v in arrays.items()},
        actions=actions, checksums=checksums,
    )


def is_v1(path: str | Path) -> bool:
    """Whether ``path`` looks like a ``trajectory_bundle/v1`` (has a completion marker).

    Used to route between the strict v1 path and the legacy parity-bundle reader. It
    deliberately keys on the marker rather than on the metadata block: a bundle that
    declares the contract but never finished writing is exactly the case v1 exists to
    catch, and it must take the strict path so it is refused rather than skipped.
    """
    root = Path(path)
    return (root / COMPLETION_MARKER).is_file() or (
        (root / METADATA_FILE).is_file()
        and CONTRACT_KEY in json.loads((root / METADATA_FILE).read_text(encoding="utf-8"))
    )


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------

def finalize_v1(root: str | Path, *, run_status: str) -> dict[str, str]:
    """Checksum a written bundle and stamp the completion marker last.

    The order is the contract: payload, then ``CHECKSUMS.sha256``, then ``COMPLETE``.
    Any interruption leaves the bundle detectably unfinished.
    """
    path = Path(root)
    _require(run_status in RUN_STATUSES, "IVF-BUNDLE-CONTRACT-INCOMPLETE",
             f"run_status {run_status!r} is not one of {sorted(RUN_STATUSES)}")
    checksums = {
        p.relative_to(path).as_posix(): sha256_file(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name not in (CHECKSUM_FILE, COMPLETION_MARKER)
    }
    _require(bool(checksums), "IVF-BUNDLE-CHECKSUM-MISSING", f"{path}: nothing to checksum")
    lines = [f"{digest}  {name}" for name, digest in sorted(checksums.items())]
    (path / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")

    marker = {
        "schema": SCHEMA,
        "run_status": run_status,
        "n_files": len(checksums),
        "checksums_sha256": sha256_file(path / CHECKSUM_FILE),
    }
    (path / COMPLETION_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checksums
