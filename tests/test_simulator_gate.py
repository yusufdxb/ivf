# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The gate that stops a simulator job from going green without a simulator.

These run on CPU. They do not exercise the GPU path and are not evidence that it works;
they are evidence that a job which *claims* to exercise it cannot pass by skipping.
"""

from __future__ import annotations

import pytest
from _pytest.outcomes import Failed, Skipped

from tests.simulator_gate import REQUIRE_ENV, simulator_required, simulator_unavailable

pytestmark = [pytest.mark.unit]


def test_unset_environment_skips_so_a_laptop_run_stays_honest(monkeypatch):
    monkeypatch.delenv(REQUIRE_ENV, raising=False)
    assert simulator_required() is False
    with pytest.raises(Skipped) as excinfo:
        simulator_unavailable("Isaac Lab is not installed")
    assert "Isaac Lab is not installed" in str(excinfo.value)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_required_environment_turns_a_missing_simulator_into_a_failure(monkeypatch, value):
    monkeypatch.setenv(REQUIRE_ENV, value)
    assert simulator_required() is True
    with pytest.raises(Failed) as excinfo:
        simulator_unavailable("Isaac Lab is not installed")
    message = str(excinfo.value)
    assert REQUIRE_ENV in message
    assert "Isaac Lab is not installed" in message


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_only_explicit_affirmatives_arm_the_gate(monkeypatch, value):
    monkeypatch.setenv(REQUIRE_ENV, value)
    assert simulator_required() is False
    with pytest.raises(Skipped):
        simulator_unavailable("Isaac Lab is not installed")


def test_both_simulator_modules_route_their_prerequisite_checks_through_the_gate():
    """A new bare ``pytest.skip`` in a simulator module would reopen the hole."""
    from pathlib import Path

    tests_dir = Path(__file__).resolve().parent
    for name in ("test_simulator_end_to_end.py", "test_simulator_newton_end_to_end.py"):
        source = (tests_dir / name).read_text(encoding="utf-8")
        assert "simulator_unavailable" in source, f"{name} does not use the gate"
        assert "pytest.skip(" not in source, (
            f"{name} skips directly; route it through simulator_unavailable so "
            f"{REQUIRE_ENV} can turn it into a failure"
        )
