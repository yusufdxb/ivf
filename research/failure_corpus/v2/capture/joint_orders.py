# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Record PhysX and Newton joint enumeration for platforms (Isaac Lab interpreter)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app
import gymnasium as gym  # noqa: E402
import isaaclab.sim as su  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401

import corpus_capture_v2 as cc  # noqa: E402

for platform in sys.argv[2:]:
    out = {}
    for bm in ("physx", "newton"):
        su.create_new_stage()
        cfg = cc.build_cfg(platform, bm, 2, 0)
        cc.controlled_cfg(cfg, platform, mode="open_loop")
        env = gym.make(cc.TASKS[platform], cfg=cfg)
        out[bm] = list(env.unwrapped.scene["robot"].joint_names)
        out[bm + "_manager"] = str(env.unwrapped.sim.physics_manager)
        cc.close_env(env)
    out["identical"] = out["physx"] == out["newton"]
    dest = Path(sys.argv[1]) / platform
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "joint_orders.json").write_text(json.dumps(out, indent=1) + "\n")
    print("ORDERS", platform, out["identical"], flush=True)
app.close()
