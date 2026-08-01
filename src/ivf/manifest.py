# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The versioned experiment manifest: IVF's unit of intent.

A manifest declares *what comparison is being made and what would count as a
failure*, before any data exists. It is parsed into typed objects here rather than
being passed around as a dictionary, so a malformed experiment fails at load time
with a precise message instead of producing a confident, meaningless report.

Two design rules are enforced by the parser and are not negotiable at runtime:

* **No naked epsilon.** A tolerance is a :class:`Tolerance` with a unit, a scope, a
  rationale, an aggregation rule, a minimum sample count, and a declared kind. A bare
  float is rejected. The reason is empirical, not stylistic: the pre-existing parity
  work found that unqualified thresholds are how comparison harnesses end up
  publishing "0.002 rad" without anyone able to say over what, across how many
  environments, or why 0.002.
* **Controls are declared, not inferred.** ``controls.require_same`` lists the
  properties the comparison depends on. :mod:`ivf.validity` checks exactly those and
  refuses the experiment when one is violated.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "ivf.validation/v1"
"""Manifest schema identifier. A major change here is a breaking change for users."""

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")

TOLERANCE_KINDS = frozenset({"exact", "numerical", "statistical", "event", "engineering"})
"""How a tolerance is to be read.

``exact`` bitwise, ``numerical`` a floating-point budget, ``statistical`` an
equivalence margin subject to sampling uncertainty, ``event`` a discrete-timing
budget, ``engineering`` a value chosen from domain judgement rather than derived.
An ``engineering`` tolerance is never presented as a physical claim.
"""

AGGREGATIONS = frozenset({"max", "mean", "p95", "second_largest", "final", "any", "all"})
"""How per-element errors reduce to one number.

``second_largest`` is the reduction the pre-existing parity harness settled on for
reductions over environments: it deliberately discards the single largest environment,
so at least two environments must exceed a threshold before the reduction fails.
"""

KNOWN_CONTROLS = frozenset({
    "asset_identity",
    "action_sequence",
    "initial_state_distribution",
    "observation_definition",
    "control_frequency",
    "seeds",
    "num_envs",
    "horizon",
    "task_variant",
    "solver_specific_parameters",
    "frame_convention",
    "quaternion_convention",
    "reset_semantics",
    "environment_ordering",
    "action_timing",
    "asset_binary_identity",
    "initial_state_realization",
    "backend_internal_state",
})
"""Properties the validity layer knows how to check. Unknown names are rejected at
load: silently ignoring a control the user asked for is the worst possible failure."""

UNVERIFIABLE_ONLY_CONTROLS = frozenset({
    "asset_binary_identity",
    "initial_state_realization",
    "backend_internal_state",
})
"""Controls the current bundle contract can name but cannot compare."""


class ManifestError(ValueError):
    """Raised when a manifest cannot be parsed or is internally inconsistent.

    The message always names the offending path within the document, because the
    caller is a human editing YAML.
    """


@dataclass(frozen=True)
class Tolerance:
    """A fully qualified acceptance threshold.

    Every field is required. ``value`` alone means nothing without the rest.
    """

    value: float
    """The threshold magnitude, in :attr:`unit`."""

    unit: str
    """Physical or dimensionless unit, e.g. ``"rad"``, ``"m"``, ``"ulp"``, ``"steps"``,
    ``"fraction"``. Recorded verbatim in the evidence bundle."""

    scope: str
    """What the threshold applies to, e.g. ``"per-step per-env absolute error"``.
    Prose, but mandatory: it is what a reviewer reads first."""

    rationale: str
    """Why this number and not another. Manifests without it are rejected."""

    aggregation: str
    """One of :data:`AGGREGATIONS`."""

    min_samples: int
    """Fewest samples for which a decision may be made. Below this the oracle returns
    ``inconclusive`` with ``IVF-SAMPLE-INSUFFICIENT`` rather than guessing."""

    kind: str
    """One of :data:`TOLERANCE_KINDS`."""

    minimum_meaningful_effect: float | None = None
    """For ``statistical`` tolerances: the smallest difference that would actually
    matter operationally. Guards against a large sample turning an irrelevant
    difference into a failure."""

    @classmethod
    def parse(cls, raw: Any, where: str) -> Tolerance:
        """Parse ``raw`` into a tolerance, rejecting bare numbers."""
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            raise ManifestError(
                f"{where}: a bare number is not a tolerance. Declare value, unit, scope, "
                "rationale, aggregation, min_samples and kind."
            )
        if not isinstance(raw, dict):
            raise ManifestError(f"{where}: expected a tolerance mapping, got {type(raw).__name__}")
        required = ("value", "unit", "scope", "rationale", "aggregation", "min_samples", "kind")
        missing = [k for k in required if k not in raw or raw[k] in (None, "")]
        if missing:
            raise ManifestError(f"{where}: tolerance is missing required field(s): {', '.join(missing)}")
        unknown = set(raw) - set(required) - {"minimum_meaningful_effect"}
        if unknown:
            raise ManifestError(f"{where}: unknown tolerance field(s): {', '.join(sorted(unknown))}")
        if raw["aggregation"] not in AGGREGATIONS:
            raise ManifestError(
                f"{where}.aggregation: {raw['aggregation']!r} is not one of {sorted(AGGREGATIONS)}"
            )
        if raw["kind"] not in TOLERANCE_KINDS:
            raise ManifestError(f"{where}.kind: {raw['kind']!r} is not one of {sorted(TOLERANCE_KINDS)}")
        value = float(raw["value"])
        if value < 0:
            raise ManifestError(f"{where}.value: must be non-negative, got {value}")
        min_samples = int(raw["min_samples"])
        if min_samples < 1:
            raise ManifestError(f"{where}.min_samples: must be at least 1, got {min_samples}")
        mme = raw.get("minimum_meaningful_effect")
        if raw["kind"] == "statistical" and mme is None:
            raise ManifestError(
                f"{where}: a statistical tolerance must declare minimum_meaningful_effect, "
                "otherwise a large sample turns a trivial difference into a failure."
            )
        rationale = str(raw["rationale"]).strip()
        if len(rationale) < 10:
            raise ManifestError(f"{where}.rationale: too short to be a rationale ({rationale!r})")
        return cls(
            value=value,
            unit=str(raw["unit"]),
            scope=str(raw["scope"]),
            rationale=rationale,
            aggregation=str(raw["aggregation"]),
            min_samples=min_samples,
            kind=str(raw["kind"]),
            minimum_meaningful_effect=None if mme is None else float(mme),
        )

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        payload = {
            "value": self.value,
            "unit": self.unit,
            "scope": self.scope,
            "rationale": self.rationale,
            "aggregation": self.aggregation,
            "min_samples": self.min_samples,
            "kind": self.kind,
        }
        if self.minimum_meaningful_effect is not None:
            payload["minimum_meaningful_effect"] = self.minimum_meaningful_effect
        return payload


@dataclass(frozen=True)
class Subject:
    """One side of the comparison: baseline or candidate."""

    role: str
    """``"baseline"`` or ``"candidate"``."""

    kind: str
    """Signal source kind: ``"synthetic"``, ``"parity_bundle"``, or ``"isaaclab"``."""

    params: dict[str, Any] = field(default_factory=dict)
    """Source-specific parameters, validated by the source implementation."""

    label: str = ""
    """Short human label used in reports; defaults to a rendering of the source."""

    @classmethod
    def parse(cls, role: str, raw: Any) -> Subject:
        """Parse one subject entry."""
        where = f"subjects.{role}"
        if not isinstance(raw, dict):
            raise ManifestError(f"{where}: expected a mapping")
        if "kind" not in raw:
            raise ManifestError(f"{where}.kind: required (synthetic | parity_bundle | isaaclab)")
        params = {k: v for k, v in raw.items() if k not in ("kind", "label")}
        return cls(role=role, kind=str(raw["kind"]), params=params, label=str(raw.get("label", "")))

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {"role": self.role, "kind": self.kind, "label": self.label, **self.params}


@dataclass(frozen=True)
class Workload:
    """What was run, independent of who ran it."""

    task: str
    steps: int
    num_envs: int = 1
    warmup_steps: int | None = None
    seeds: tuple[int, ...] = (0,)

    @classmethod
    def parse(cls, raw: Any) -> Workload:
        """Parse the workload block."""
        if not isinstance(raw, dict):
            raise ManifestError("workload: expected a mapping")
        for key in ("task", "steps"):
            if key not in raw:
                raise ManifestError(f"workload.{key}: required")
        seeds = raw.get("seeds", [0])
        if isinstance(seeds, int):
            seeds = [seeds]
        if not isinstance(seeds, list) or not seeds:
            raise ManifestError("workload.seeds: expected a non-empty list of integers")
        steps = int(raw["steps"])
        if steps < 1:
            raise ManifestError(f"workload.steps: must be at least 1, got {steps}")
        warmup = raw.get("warmup_steps")
        if warmup is not None and int(warmup) >= steps:
            raise ManifestError(
                f"workload.warmup_steps ({warmup}) must be smaller than workload.steps ({steps})"
            )
        return cls(
            task=str(raw["task"]),
            steps=steps,
            num_envs=int(raw.get("num_envs", 1)),
            warmup_steps=None if warmup is None else int(warmup),
            seeds=tuple(int(s) for s in seeds),
        )

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "task": self.task,
            "steps": self.steps,
            "num_envs": self.num_envs,
            "warmup_steps": self.warmup_steps,
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True)
class Controls:
    """The properties that must match for the comparison to mean anything."""

    require_same: tuple[str, ...] = ()
    allow_different: tuple[str, ...] = ()
    unsupported_or_unverifiable: tuple[str, ...] = ()

    @classmethod
    def parse(cls, raw: Any) -> Controls:
        """Parse the controls block, rejecting unknown or contradictory names."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ManifestError("controls: expected a mapping")
        known_keys = {"require_same", "allow_different", "unsupported_or_unverifiable"}
        unknown_keys = set(raw) - known_keys
        if unknown_keys:
            raise ManifestError(f"controls: unknown key(s) {sorted(unknown_keys)}")
        same = tuple(str(x) for x in raw.get("require_same", ()) or ())
        diff = tuple(str(x) for x in raw.get("allow_different", ()) or ())
        unsupported = tuple(str(x) for x in raw.get("unsupported_or_unverifiable", ()) or ())
        for name in (*same, *diff, *unsupported):
            if name not in KNOWN_CONTROLS:
                raise ManifestError(
                    f"controls: unknown control {name!r}. Known controls: {sorted(KNOWN_CONTROLS)}"
                )
        overlap = (set(same) & set(diff)) | (set(same) & set(unsupported)) | (set(diff) & set(unsupported))
        if overlap:
            raise ManifestError(
                f"controls: {sorted(overlap)} appear in more than one control partition"
            )
        misplaced = (set(same) | set(diff)) & UNVERIFIABLE_ONLY_CONTROLS
        if misplaced:
            raise ManifestError(
                "controls: "
                f"{sorted(misplaced)} can only appear in unsupported_or_unverifiable; "
                "trajectory_bundle/v1 does not expose enough information to compare them"
            )
        return cls(
            require_same=same,
            allow_different=diff,
            unsupported_or_unverifiable=unsupported,
        )

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "require_same": list(self.require_same),
            "allow_different": list(self.allow_different),
            "unsupported_or_unverifiable": list(self.unsupported_or_unverifiable),
        }


@dataclass(frozen=True)
class OracleSpec:
    """One declared acceptance criterion, as written in the manifest."""

    type: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    tolerance: Tolerance | None = None

    @classmethod
    def parse(cls, raw: Any, index: int) -> OracleSpec:
        """Parse one oracle entry."""
        where = f"oracles[{index}]"
        if not isinstance(raw, dict):
            raise ManifestError(f"{where}: expected a mapping")
        if "type" not in raw:
            raise ManifestError(f"{where}.type: required")
        otype = str(raw["type"])
        name = str(raw.get("name", otype))
        if not _NAME_RE.match(name):
            raise ManifestError(f"{where}.name: {name!r} is not a valid identifier-like name")
        tolerance = Tolerance.parse(raw["tolerance"], f"{where}.tolerance") if "tolerance" in raw else None
        params = {k: v for k, v in raw.items() if k not in ("type", "name", "tolerance")}
        if otype == "invariant" and params.get("check") == "unit_quaternion" and tolerance is None:
            raise ManifestError(f"{where}.tolerance: required for unit_quaternion; no implicit epsilon")
        return cls(type=otype, name=name, params=params, tolerance=tolerance)

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        payload: dict[str, Any] = {"type": self.type, "name": self.name, **self.params}
        if self.tolerance is not None:
            payload["tolerance"] = self.tolerance.to_jsonable()
        return payload


@dataclass(frozen=True)
class VerdictPolicy:
    """How oracle outcomes roll up into one verdict."""

    invalid_experiment_on_control_violation: bool = True
    """When a control is violated, refuse the experiment instead of reporting results."""

    inconclusive_on_insufficient_samples: bool = True
    """When an oracle has too few samples, return INCONCLUSIVE rather than passing it."""

    unsupported_is_failure: bool = False
    """Whether an unsupported feature should be escalated to FAIL. Default: no. An
    unsupported feature is a property of the environment, not a defect in the change."""

    @classmethod
    def parse(cls, raw: Any) -> VerdictPolicy:
        """Parse the verdict-policy block."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ManifestError("verdict_policy: expected a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ManifestError(f"verdict_policy: unknown field(s) {sorted(unknown)}")
        return cls(**{k: bool(v) for k, v in raw.items()})

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "invalid_experiment_on_control_violation": self.invalid_experiment_on_control_violation,
            "inconclusive_on_insufficient_samples": self.inconclusive_on_insufficient_samples,
            "unsupported_is_failure": self.unsupported_is_failure,
        }


@dataclass(frozen=True)
class Manifest:
    """A parsed, validated experiment declaration."""

    schema_version: str
    name: str
    subjects: dict[str, Subject]
    workload: Workload
    controls: Controls
    oracles: tuple[OracleSpec, ...]
    verdict_policy: VerdictPolicy
    description: str = ""
    source_path: str | None = None
    source_text: str = ""

    @property
    def baseline(self) -> Subject:
        """Return the baseline subject."""
        return self.subjects["baseline"]

    @property
    def candidate(self) -> Subject:
        """Return the candidate subject."""
        return self.subjects["candidate"]

    def to_jsonable(self) -> dict[str, Any]:
        """Return the resolved manifest as a JSON-serializable mapping.

        This is what gets hashed and stored in the evidence bundle. It is the
        *resolved* form: defaults are materialized, so a bundle records what actually
        ran rather than what the user chose to leave implicit.
        """
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "subjects": {role: s.to_jsonable() for role, s in sorted(self.subjects.items())},
            "workload": self.workload.to_jsonable(),
            "controls": self.controls.to_jsonable(),
            "oracles": [o.to_jsonable() for o in self.oracles],
            "verdict_policy": self.verdict_policy.to_jsonable(),
        }

    def canonical_json(self) -> str:
        """Return the canonical serialization used for hashing.

        Sorted keys, no insignificant whitespace, ``repr``-stable floats. Two manifests
        that mean the same thing produce the same bytes regardless of YAML formatting,
        key order, or which defaults the author spelled out.
        """
        return json.dumps(self.to_jsonable(), sort_keys=True, separators=(",", ":"), default=_json_default)

    def digest(self) -> str:
        """Return the SHA-256 of :meth:`canonical_json`."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    """Serialize the few non-JSON types a manifest may carry."""
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__} in a manifest")


def parse_manifest(text: str, *, source_path: str | None = None) -> Manifest:
    """Parse and validate a manifest from YAML text.

    Raises :class:`ManifestError` with a path-qualified message on any problem.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ManifestError(f"not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("top level: expected a mapping")

    version = raw.get("schema_version")
    if version is None:
        raise ManifestError(f"schema_version: required (expected {SCHEMA_VERSION!r})")
    if str(version) != SCHEMA_VERSION:
        raise ManifestError(
            f"schema_version: {version!r} is not supported by this IVF build (expected {SCHEMA_VERSION!r})"
        )

    name = str(raw.get("name", "")).strip()
    if not _NAME_RE.match(name):
        raise ManifestError(f"name: {name!r} must start with a letter and contain only [A-Za-z0-9_.-]")

    subjects_raw = raw.get("subjects")
    if not isinstance(subjects_raw, dict) or set(subjects_raw) != {"baseline", "candidate"}:
        raise ManifestError("subjects: exactly two entries are required, 'baseline' and 'candidate'")
    subjects = {role: Subject.parse(role, subjects_raw[role]) for role in ("baseline", "candidate")}

    oracles_raw = raw.get("oracles")
    if not isinstance(oracles_raw, list) or not oracles_raw:
        raise ManifestError(
            "oracles: at least one oracle is required; a run with no criterion decides nothing"
        )
    oracles = tuple(OracleSpec.parse(o, i) for i, o in enumerate(oracles_raw))
    seen: set[str] = set()
    for oracle in oracles:
        if oracle.name in seen:
            raise ManifestError(f"oracles: duplicate oracle name {oracle.name!r}")
        seen.add(oracle.name)

    known_top = {
        "schema_version", "name", "description", "subjects", "workload",
        "controls", "oracles", "verdict_policy",
    }
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ManifestError(f"top level: unknown key(s) {sorted(unknown_top)}")

    return Manifest(
        schema_version=SCHEMA_VERSION,
        name=name,
        description=str(raw.get("description", "")),
        subjects=subjects,
        workload=Workload.parse(raw.get("workload")),
        controls=Controls.parse(raw.get("controls")),
        oracles=oracles,
        verdict_policy=VerdictPolicy.parse(raw.get("verdict_policy")),
        source_path=source_path,
        source_text=text,
    )


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a manifest from disk."""
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"manifest not found: {p}")
    return parse_manifest(p.read_text(encoding="utf-8"), source_path=str(p))
