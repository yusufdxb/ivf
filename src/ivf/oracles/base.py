# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The oracle interface: small, explicit, and extensible without editing a comparator.

An oracle answers one question about one pair of subjects and returns an
:class:`OracleOutcome`. It never decides the run verdict; :mod:`ivf.runner` does that
by rolling outcomes up under the manifest's verdict policy. This split is what keeps
"one oracle failed" from being confused with "the change is bad", and what lets an
oracle honestly return ``inconclusive`` instead of guessing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..divergence import DivergenceRecord
from ..manifest import Manifest, OracleSpec, Tolerance
from ..signals import SignalSet

OracleStatus = Literal["pass", "fail", "inconclusive", "unsupported", "skipped"]
"""Per-oracle outcome vocabulary.

``skipped`` means the oracle did not apply (for example, a metamorphic relation whose
precondition is absent). It is distinct from ``pass``: a skipped oracle provides no
evidence and must never read as reassurance.
"""


@dataclass
class OracleOutcome:
    """The result of applying one oracle."""

    name: str
    type: str
    status: OracleStatus
    summary: str
    reason_codes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    """Effect sizes, intervals, observed maxima. Everything a reader needs to disagree."""

    tolerance: dict[str, Any] | None = None
    divergence: DivergenceRecord | None = None
    limitations: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "name": self.name,
            "type": self.type,
            "status": self.status,
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
            "metrics": _jsonable(self.metrics),
            "tolerance": self.tolerance,
            "divergence": None if self.divergence is None else self.divergence.to_jsonable(),
            "limitations": self.limitations,
        }


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars and arrays into JSON-safe values."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class OracleContext:
    """Everything an oracle is allowed to see."""

    manifest: Manifest
    spec: OracleSpec
    baseline: SignalSet
    candidate: SignalSet
    warmup_steps: int = 0
    """Steps to discard before evaluating; declared in the manifest workload."""

    alpha: float = 0.05
    """Per-oracle false-alarm budget, already Šidák-split across statistical oracles."""

    seed: int = 0
    """Seed for any resampling, so intervals are reproducible."""

    def signal_pair(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the warm-up-trimmed baseline and candidate arrays for ``name``.

        Raises :class:`KeyError` with a message naming what is available, because a
        typo in a signal name is the single most common manifest mistake.
        """
        for role, sset in (("baseline", self.baseline), ("candidate", self.candidate)):
            if name not in sset.signals:
                raise KeyError(
                    f"signal {name!r} not present in {role}; available: {sorted(sset.signals)}"
                )
        w = self.warmup_steps
        return self.baseline.signals[name][w:], self.candidate.signals[name][w:]

    @property
    def tolerance(self) -> Tolerance:
        """Return the oracle's tolerance, raising if the manifest omitted it."""
        if self.spec.tolerance is None:
            raise ValueError(f"oracle {self.spec.name!r} of type {self.spec.type!r} requires a tolerance")
        return self.spec.tolerance


OracleFn = Callable[[OracleContext], OracleOutcome]

_REGISTRY: dict[str, OracleFn] = {}


def register(oracle_type: str) -> Callable[[OracleFn], OracleFn]:
    """Register an oracle implementation under a manifest ``type`` string."""

    def decorator(fn: OracleFn) -> OracleFn:
        if oracle_type in _REGISTRY:
            raise KeyError(f"oracle type {oracle_type!r} is already registered")
        _REGISTRY[oracle_type] = fn
        return fn

    return decorator


def get_oracle(oracle_type: str) -> OracleFn:
    """Return the implementation for ``oracle_type`` or raise :class:`KeyError`."""
    try:
        return _REGISTRY[oracle_type]
    except KeyError:
        raise KeyError(f"unknown oracle type {oracle_type!r}; registered: {registered_oracles()}") from None


def registered_oracles() -> list[str]:
    """Return the sorted list of registered oracle types."""
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------------------
# Shared reductions
# --------------------------------------------------------------------------------------

def aggregate(values: np.ndarray, how: str) -> float:
    """Reduce an error array to one number using a manifest aggregation rule.

    ``second_largest`` is the reduction over environments the pre-existing parity work
    settled on: ``mean`` dilutes a single diverging environment by the environment
    count, while ``max`` gates on the single worst bifurcation tail. Second-largest
    trims one tail event, so at least two elements must exceed a threshold to fail.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return float("nan")
    if not np.all(np.isfinite(flat)):
        return float("inf")
    if how == "max":
        return float(np.max(flat))
    if how == "mean":
        return float(np.mean(flat))
    if how == "p95":
        return float(np.percentile(flat, 95))
    if how == "second_largest":
        if flat.size == 1:
            return float(flat[0])
        return float(np.partition(flat, -2)[-2])
    if how == "final":
        return float(flat[-1])
    if how == "any":
        return float(np.max(flat))
    if how == "all":
        return float(np.min(flat))
    raise ValueError(f"unknown aggregation {how!r}")
