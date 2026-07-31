# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Validation oracles.

Importing this package registers every shipped oracle. Third-party oracles register
themselves the same way, by importing :func:`ivf.oracles.register` and decorating a
function; nothing in the runner needs to change.
"""

from __future__ import annotations

# Importing for the registration side effect is the point; the names are re-exported
# so a user can call an oracle directly in a test.
from . import event, invariant, metamorphic, numeric, statistical  # noqa: F401
from .base import (
    OracleContext,
    OracleFn,
    OracleOutcome,
    OracleStatus,
    aggregate,
    get_oracle,
    register,
    registered_oracles,
)

__all__ = [
    "OracleContext",
    "OracleFn",
    "OracleOutcome",
    "OracleStatus",
    "aggregate",
    "get_oracle",
    "register",
    "registered_oracles",
]
