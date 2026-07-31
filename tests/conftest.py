# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures.

Nothing here touches a GPU or a simulator. Tests that would need one are marked and
skipped with a message naming what is missing, never silently passed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ivf.manifest import parse_manifest
from ivf.signals import SignalSet

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "validation" / "examples"
BUNDLES = REPO_ROOT / "validation" / "bundles"


@pytest.fixture
def repo_root() -> Path:
    """The repository root, so tests can find example manifests and fixtures."""
    return REPO_ROOT


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    """An isolated evidence-bundle root, so no test writes into the working tree."""
    return tmp_path / "ivf-results"


MINIMAL_MANIFEST = """
schema_version: ivf.validation/v1
name: minimal
subjects:
  baseline:
    kind: synthetic
    system: damped_pendulum
  candidate:
    kind: synthetic
    system: damped_pendulum
workload:
  task: damped_pendulum.v1
  num_envs: 4
  steps: 50
  warmup_steps: 0
  seeds: [3]
controls:
  require_same: [asset_identity, observation_definition, control_frequency, num_envs, horizon]
oracles:
  - type: invariant
    name: finite_state
    check: finite_state
"""


@pytest.fixture
def minimal_manifest():
    """A parsed manifest that passes on any machine in well under a second."""
    return parse_manifest(MINIMAL_MANIFEST)


def make_signals(role: str, *, steps: int = 40, envs: int = 6, offset: float = 0.0,
                 dim: int = 1, **metadata) -> SignalSet:
    """Build a deterministic :class:`SignalSet` for oracle-level tests."""
    t = np.linspace(0.0, 4.0, steps)[:, None, None]
    e = np.arange(envs, dtype=np.float64)[None, :, None]
    base = np.sin(t + 0.1 * e) + offset
    signal = np.repeat(base, dim, axis=2)
    meta = {
        "asset_identity": "fixture.v1",
        "physics_dt": 0.005,
        "control_frequency_hz": 200.0,
        "seed": 3,
        "task_variant": "fixture.v1",
        "solver_settings": {"integrator": "test"},
        "initial_state_digest": "digest",
    }
    meta.update(metadata)
    return SignalSet(
        role=role,
        signals={"x": signal, "y": np.abs(signal)},
        metadata=meta,
        actions=np.zeros((steps, envs)),
    )
