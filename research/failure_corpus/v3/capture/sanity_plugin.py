# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Positive controls for the three v3 capabilities, used only on development seeds.

Written by the IVF author, so never part of any holdout.
"""


class SolverIntegratorChanged:
    def cfg(self, cfg, platform, params):
        cfg.sim.physics.solver_cfg.integrator = params.get("integrator", "euler")


class MaterialEventRemoved:
    def cfg(self, cfg, platform, params):
        cfg.events.physics_material = None


class TerminationThresholdChanged:
    def cfg(self, cfg, platform, params):
        cfg.terminations.base_contact.params["threshold"] = float(params.get("threshold", 50.0))


FAMILIES = {
    "sanity_solver": SolverIntegratorChanged(),
    "sanity_material": MaterialEventRemoved(),
    "sanity_termination": TerminationThresholdChanged(),
}
