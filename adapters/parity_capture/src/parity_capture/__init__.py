# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab capture adapter for IVF.

Lives outside IVF core so that IVF core never depends on Isaac Lab. This package is
installed into an environment that already has the simulator; IVF is installed
everywhere else.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
