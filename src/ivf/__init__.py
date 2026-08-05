# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""IVF: an auditable validation and acceptance layer for Isaac Lab.

IVF answers one question, "can I trust this change for my workload, and what evidence
supports that decision", by executing a declared experiment, applying declared
acceptance criteria, and sealing the result into an evidence bundle another engineer can
verify offline.

It does not claim that any simulator or physics backend is universally correct.
"""

from __future__ import annotations

__version__ = "0.1.0rc2"

__all__ = ["__version__"]
