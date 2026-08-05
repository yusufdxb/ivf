# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The one switch that decides whether a missing simulator is a skip or a failure.

Every simulator-backed test skips when Isaac Lab is absent, which is correct on a laptop
and correct on the CPU CI runners: a skip is an honest statement that the GPU path was
not exercised. It is *not* correct on a job whose entire purpose is to exercise that
path. There, a skip and a pass are indistinguishable in the check list, so a runner that
silently lost its Isaac Lab install would keep reporting green forever while covering
nothing.

``IVF_REQUIRE_SIMULATOR=1`` closes that hole. When it is set, every missing prerequisite
becomes a test failure naming the prerequisite, so "this runner never had a simulator"
and "this runner has one and the chain works" stop looking the same from outside.
"""

from __future__ import annotations

import os

import pytest

REQUIRE_ENV = "IVF_REQUIRE_SIMULATOR"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def simulator_required() -> bool:
    """Whether this environment declares that the simulator path must actually run."""
    return os.environ.get(REQUIRE_ENV, "").strip().lower() in _TRUTHY


def simulator_unavailable(reason: str) -> None:
    """Skip because the simulator is missing, or fail if this job promised to run it.

    Never returns.
    """
    if simulator_required():
        pytest.fail(
            f"{REQUIRE_ENV} is set, so this job asserts the simulator path runs here, "
            f"but it cannot: {reason}"
        )
    pytest.skip(reason)
