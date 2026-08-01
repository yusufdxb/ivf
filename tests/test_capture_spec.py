# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Offline guards for the narrow simulator capture specification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ADAPTER_SRC = Path(__file__).resolve().parents[1] / "adapters" / "parity_capture" / "src"
sys.path.insert(0, str(ADAPTER_SRC))

from parity_capture.spec import SpecError, parse_spec  # noqa: E402

pytestmark = pytest.mark.unit

BASE = """schema_version: parity.capture/v1
name: guard
task: cartpole_passive
backend: physx
num_envs: 2
steps: 4
seed: 0
"""


def test_reset_perturbation_is_inactive_when_omitted_or_explicitly_false():
    assert parse_spec(BASE).defect.drop_reset_velocity is False
    assert parse_spec(BASE + "defect:\n  drop_reset_velocity: false\n").defect.drop_reset_velocity is False


def test_reset_perturbation_requires_explicit_boolean_true():
    assert parse_spec(BASE + "defect:\n  drop_reset_velocity: true\n").defect.drop_reset_velocity is True
    with pytest.raises(SpecError, match="expected a YAML boolean"):
        parse_spec(BASE + 'defect:\n  drop_reset_velocity: "false"\n')


def test_backend_and_test_switch_do_not_change_task_identity():
    physx = parse_spec(BASE)
    newton = parse_spec(BASE.replace("backend: physx", "backend: newton"))
    perturbed = parse_spec(BASE + "defect:\n  drop_reset_velocity: true\n")
    assert physx.config_identity() == newton.config_identity() == perturbed.config_identity()
