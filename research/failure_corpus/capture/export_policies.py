# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Export the latest PhysX-trained rsl_rl checkpoint of each platform to TorchScript.

Mirrors the checkpoint-loading path of Isaac Lab's rsl_rl play script.
Usage: $ISAACLAB_PYTHON export_policies.py <isaaclab_root> <out_dir> <platform> [...]
"""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import hashlib  # noqa: E402
import importlib.metadata as md  # noqa: E402
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

TASKS = {
    "go2": "Isaac-Velocity-Flat-UnitreeGo2",
    "g1": "Isaac-Velocity-Flat-G1",
    "h1": "Isaac-Velocity-Flat-H1",
    "anymal_d": "Isaac-Velocity-Flat-AnymalD",
    "cassie": "Isaac-Velocity-Flat-Cassie",
    "spot": "Isaac-Velocity-Flat-Spot",
    "go2_rough": "Isaac-Velocity-Rough-UnitreeGo2",
}
root, out = Path(sys.argv[1]), Path(sys.argv[2])
for platform in sys.argv[3:]:
    task = TASKS[platform]
    sim_utils.create_new_stage()
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
    runs = sorted((root / "logs" / "rsl_rl" / agent_cfg.experiment_name).glob("*/"))
    ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    ckpt = ckpts[-1]
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(str(ckpt))
    dest = out / platform
    runner.export_policy_to_jit(path=str(dest), filename="policy.pt")
    info = {
        "task": task,
        "checkpoint": f"{agent_cfg.experiment_name}/{runs[-1].name}/{ckpt.name}",
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "policy_pt_sha256": hashlib.sha256((dest / "policy.pt").read_bytes()).hexdigest(),
        "trained_backend": "physx (default preset)",
        "seed": agent_cfg.seed,
    }
    (dest / "policy_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print("EXPORTED", platform, info, flush=True)
    env.close()
    SimulationContext.clear_instance()
app.close()
