# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Resolve each task's registered newton_mjwarp preset into the solver settings an experimenter asks for.

Written once to research/failure_corpus/v3/intended_presets.json and committed; the v3 harness
declares these as each Newton subject's ``expect_solver``. Resource capacities (njmax, nconmax)
are deliberately excluded: MuJoCo-Warp rewrites them at build time (measured), and saturation is
covered separately by ``resource_health``.
"""
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

TASKS = {"go2": "Isaac-Velocity-Flat-UnitreeGo2", "g1": "Isaac-Velocity-Flat-G1", "h1": "Isaac-Velocity-Flat-H1",
         "anymal_d": "Isaac-Velocity-Flat-AnymalD", "cassie": "Isaac-Velocity-Flat-Cassie",
         "spot": "Isaac-Velocity-Flat-Spot"}
KEYS = ("integrator", "cone", "solver", "iterations", "ls_iterations", "impratio", "tolerance")
out = {}
for p, task in TASKS.items():
    cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    phys = cfg.sim.physics
    sc = phys.solver_cfg
    out[p] = {k: getattr(sc, k) for k in KEYS}
    out[p]["num_substeps"] = int(phys.num_substeps)
    print(p, out[p], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
app.close()
