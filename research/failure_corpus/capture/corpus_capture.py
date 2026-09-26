# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Failure-corpus capture producer for Isaac Lab velocity-locomotion tasks.

Research tooling, not part of IVF. Like ``adapters/parity_capture`` it writes the
``trajectory_bundle/v1`` file format directly and never imports IVF, so IVF is judged
through the same file boundary it was designed around.

Producer rule (pre-registered): every contract field is read from the live environment
at runtime, never from the case label. The case label (fault family, category, whether a
defect is present) is written ONLY to ``case_label.json`` beside the bundle, outside the
bundle directory, so neither IVF nor any baseline can read it.

One Kit process runs a *plan*: an ordered list of capture jobs for one platform. Each job
builds the env, runs an open-loop replay capture (the IVF protocol) and a closed-loop
policy evaluation (the conventional sim-to-sim protocol), then tears the env down.

Usage (inside the Isaac Lab interpreter):
    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES $ISAACLAB_PYTHON corpus_capture.py \
        --plan plan.json --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform as _platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

TASKS = {
    "go2": "Isaac-Velocity-Flat-UnitreeGo2",
    "g1": "Isaac-Velocity-Flat-G1",
    "h1": "Isaac-Velocity-Flat-H1",
    "anymal_d": "Isaac-Velocity-Flat-AnymalD",
}
# Base body used for the fall/termination signal, matching each task's base_contact term.
BASE_BODY = {"go2": "base", "g1": "torso_link", "h1": "torso_link", "anymal_d": "base"}
# Height below which the base is considered fallen (about 45-55% of nominal standing height).
FALL_HEIGHT = {"go2": 0.15, "g1": 0.40, "h1": 0.50, "anymal_d": 0.30}

OPEN_LOOP_ENVS = 16
OPEN_LOOP_STEPS = 250          # control steps (5 s at 50 Hz)
CLOSED_LOOP_ENVS = 64
CLOSED_LOOP_STEPS = 1000       # control steps (20 s at 50 Hz, one default episode)


# ----------------------------------------------------------------------------------------
# helpers that do not need Kit
# ----------------------------------------------------------------------------------------

def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize_v1(root: Path, run_status: str) -> str:
    """Write CHECKSUMS.sha256 then COMPLETE, in normative v1 order. Returns the root hash."""
    checksums = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in ("CHECKSUMS.sha256", "COMPLETE")
    }
    (root / "CHECKSUMS.sha256").write_text(
        "".join(f"{d}  {n}\n" for n, d in sorted(checksums.items())), encoding="utf-8")
    root_hash = sha256_file(root / "CHECKSUMS.sha256")
    (root / "COMPLETE").write_text(json.dumps({
        "schema": "trajectory_bundle/v1", "run_status": run_status,
        "n_files": len(checksums), "checksums_sha256": root_hash,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root_hash


def command_schedule(num_envs: int) -> np.ndarray:
    """Deterministic per-env velocity command (vx, vy, wz); identical for every job."""
    vx = np.linspace(0.2, 0.8, num_envs)
    wz = np.array([(-0.3, 0.0, 0.3)[i % 3] for i in range(num_envs)])
    return np.stack([vx, np.zeros(num_envs), wz], axis=1).astype(np.float32)


def requested_initial_state(num_joints: int, num_envs: int, seed: int) -> dict[str, np.ndarray]:
    """Requested reset state: small seeded joint offsets and a forward base velocity."""
    rng = np.random.default_rng(10_000 + seed)
    return {
        "joint_offset": rng.uniform(-0.05, 0.05, size=(num_envs, num_joints)).astype(np.float32),
        "root_lin_vel": np.tile(np.array([[0.3, 0.0, 0.0]], dtype=np.float32), (num_envs, 1)),
        "root_ang_vel": rng.uniform(-0.2, 0.2, size=(num_envs, 3)).astype(np.float32),
    }


# ----------------------------------------------------------------------------------------
# config construction and fault injection (cfg-level)
# ----------------------------------------------------------------------------------------

def build_cfg(platform: str, backend_mode: str, num_envs: int, seed: int):
    """Return a resolved env cfg for the requested backend path.

    ``backend_mode``:
      * ``physx``: stock default preset.
      * ``newton``: raw registry cfg resolved with ``newton_mjwarp`` selected (the path the
        upstream smoke test uses after PR #7103).
      * ``newton_via_pre7103_test_path``: the pre-#7103 smoke-test path, verbatim: parse the
        cfg (which resolves presets to default) and then apply ``newton_mjwarp`` as a global
        override afterwards.
    """
    from isaaclab_tasks.utils.hydra import apply_overrides, collect_presets, resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    task = TASKS[platform]
    if backend_mode == "physx":
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
    elif backend_mode == "newton":
        cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=("newton_mjwarp",))
        cfg.sim.device = "cuda:0"
        cfg.scene.num_envs = num_envs
    elif backend_mode == "newton_via_pre7103_test_path":
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
        raw = load_cfg_from_registry(task, "env_cfg_entry_point")
        apply_overrides(cfg, None, {"env": cfg.to_dict(), "agent": None}, ["newton_mjwarp"], [], [],
                        {"env": collect_presets(raw), "agent": {}})
        cfg.scene.num_envs = num_envs
    else:
        raise ValueError(backend_mode)
    cfg.seed = seed
    return cfg


def controlled_cfg(cfg, *, mode: str) -> None:
    """Apply the controlled evaluation protocol shared by every job (clean and faulty).

    PLAY-style: observation noise off, pushes off, random startup mass/COM off, fixed
    command schedule. Open-loop additionally removes auto-reset terminations so trajectories
    stay aligned (the termination condition is still evaluated and recorded).
    """
    cfg.observations.policy.enable_corruption = False
    ev = cfg.events
    for name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com"):
        if hasattr(ev, name):
            setattr(ev, name, None)
    cmd = cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = False
    if hasattr(cfg, "curriculum"):
        cfg.curriculum.terrain_levels = None
    if mode == "open_loop":
        cfg.episode_length_s = 1.0e6
        cfg.terminations.base_contact = None


def _actuators(cfg):
    return cfg.scene.robot.actuators


def apply_cfg_fault(cfg, fault: dict[str, Any], *, mode: str) -> None:
    """Mutate the env cfg for cfg-level fault families. Runtime families are no-ops here."""
    fam, p = fault.get("family", "none"), fault.get("params", {})
    if fam == "timestep_substeps":
        # Newton-only knob: physics substeps per sim.dt. sim.dt and decimation unchanged.
        cfg.sim.physics.num_substeps = int(p["num_substeps"])
    elif fam == "timestep_dt_decimation":
        cfg.sim.dt = float(p["dt"])
        cfg.decimation = int(p["decimation"])
        cfg.sim.render_interval = cfg.decimation
    elif fam == "actuator_gain_scale":
        for a in _actuators(cfg).values():
            if isinstance(a.stiffness, dict):
                a.stiffness = {k: v * p["stiffness_scale"] for k, v in a.stiffness.items()}
            elif a.stiffness is not None:
                a.stiffness = a.stiffness * p["stiffness_scale"]
            if isinstance(a.damping, dict):
                a.damping = {k: v * p["damping_scale"] for k, v in a.damping.items()}
            elif a.damping is not None:
                a.damping = a.damping * p["damping_scale"]
    elif fam == "armature_dropped":
        for a in _actuators(cfg).values():
            a.armature = 0.0
    elif fam == "contact_capacity":
        cfg.sim.physics.solver_cfg.nconmax = int(p["nconmax"])
        cfg.sim.physics.solver_cfg.njmax = int(p["njmax"])
    elif fam == "action_scale":
        cfg.actions.joint_pos.scale = float(p["scale"])
    elif fam == "obs_term_scale":
        term = getattr(cfg.observations.policy, p["term"])
        term.scale = float(p["scale"])
    elif fam == "benign_capacity":
        sc = cfg.sim.physics.solver_cfg
        sc.nconmax = int(sc.nconmax * p["factor"])
        sc.njmax = int(sc.njmax * p["factor"])
    elif fam == "randomization_asymmetry":
        # re-enable a startup randomization that the controlled protocol removes
        import isaaclab_tasks.core.velocity.mdp as mdp
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        cfg.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=p["body"]),
                    "mass_distribution_params": tuple(p["mass_range"]), "operation": "scale",
                    "distribution": "log_uniform"})
    elif fam in ("none", "termination_body_mismatch", "joint_order_action", "joint_order_obs_action", "obs_term_swap",
                 "reset_velocity_dropped", "reset_joint_offsets_ignored", "preset_not_applied",
                 "physx_only_setting"):
        pass
    else:
        raise ValueError(f"unknown fault family {fam!r}")


def apply_physx_only_setting(cfg, fault) -> None:
    """Set a PhysX articulation-solver field that the Newton backend does not consume."""
    p = fault.get("params", {})
    props = cfg.scene.robot.spawn.articulation_props
    props.solver_position_iteration_count = int(p["pos_iters"])
    props.solver_velocity_iteration_count = int(p["vel_iters"])


# ----------------------------------------------------------------------------------------
# runtime helpers (need Kit)
# ----------------------------------------------------------------------------------------

def t(x):
    """Return a torch tensor from an Isaac Lab data field (ProxyArray or tensor)."""
    return x.torch if hasattr(x, "torch") else x


def joint_permutation(joint_names: list[str], kind: str) -> list[int]:
    """Index permutation modelling a documented ordering mismatch.

    ``per_limb``: the policy's action vector (Isaac breadth-first order) is interpreted
    in a depth-first, per-limb order (the MJCF/URDF convention used by MuJoCo deployments).
    ``lr_swap``: left/right (or FL/FR, RL/RR) swapped.
    Returns ``perm`` such that the candidate applies ``a[perm]`` where it should apply ``a``.
    """
    n = len(joint_names)
    if kind == "per_limb":
        def limb_key(name: str):
            for tag_i, tag in enumerate(("FL", "FR", "RL", "RR", "LF", "RF", "LH", "RH")):
                if name.startswith(tag + "_") or name.startswith(tag):
                    return (tag_i, name)
            side = 0 if name.startswith("left") else 1 if name.startswith("right") else 2
            return (10 + side, name)
        depth_first = sorted(range(n), key=lambda i: limb_key(joint_names[i]))
        return depth_first
    if kind == "lr_swap":
        swap = {"left": "right", "right": "left", "FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL",
                "LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}
        perm = []
        for name in joint_names:
            other = name
            for a, b in swap.items():
                if name.startswith(a):
                    other = b + name[len(a):]
                    break
            perm.append(joint_names.index(other) if other in joint_names else joint_names.index(name))
        return perm
    raise ValueError(kind)


def pin_commands(env, schedule_np: np.ndarray) -> None:
    """Replace the command term's resampling with the fixed schedule (all modes)."""
    import torch
    term = env.command_manager.get_term("base_velocity")
    sched = torch.as_tensor(schedule_np, device=env.device)

    def _resample(env_ids):
        term.vel_command_b[env_ids] = sched[env_ids]
    term._resample_command = _resample
    term.vel_command_b[:] = sched


def obs_term_slices(env) -> dict[str, slice]:
    mgr = env.observation_manager
    names = mgr.active_terms["policy"]
    dims = mgr.group_obs_term_dim["policy"]
    out, start = {}, 0
    for n, d in zip(names, dims):
        width = int(np.prod(d))
        out[n] = slice(start, start + width)
        start += width
    return out


def transform_obs(obs, fault, slices, perm_idx):
    """Runtime observation faults: what the *policy* is fed in the candidate."""
    fam, p = fault.get("family", "none"), fault.get("params", {})
    if fam == "joint_order_obs_action":
        o = obs.clone()
        for term in ("joint_pos", "joint_vel", "actions"):
            if term in slices:
                s = slices[term]
                o[:, s] = obs[:, s][:, perm_idx]
        return o
    if fam == "obs_term_swap":
        a, b = slices[p["a"]], slices[p["b"]]
        o = obs.clone()
        o[:, a], o[:, b] = obs[:, b], obs[:, a]
        return o
    return obs


def transform_action(action, fault, perm_idx):
    fam = fault.get("family", "none")
    if fam in ("joint_order_action", "joint_order_obs_action"):
        return action[:, perm_idx]
    return action


def physics_identity(env) -> dict[str, Any]:
    sim = env.sim
    physics_cfg = sim.cfg.physics
    try:
        settings = json.loads(json.dumps(physics_cfg.to_dict(), default=str, sort_keys=True))
    except Exception as exc:  # pragma: no cover
        settings = {"unavailable": str(exc)}
    manager = str(sim.physics_manager)
    backend = "newton" if "newton" in manager.lower() else "physx" if "physx" in manager.lower() else manager
    return {"backend": backend, "manager": manager, "cfg_class": type(physics_cfg).__name__,
            "solver_settings": {"manager": manager, "cfg_class": type(physics_cfg).__name__, **settings}}


def full_cfg_digest(env) -> str:
    """Digest of the resolved env cfg with the physics subtree and sensor classes removed.

    Used only by the pre-registered 'rich identity' producer ablation.
    """
    d = env.cfg.to_dict()
    d.get("sim", {}).pop("physics", None)
    d.pop("seed", None)
    scene = d.get("scene", {})
    cs = scene.get("contact_forces")
    if isinstance(cs, dict):
        cs.pop("class_type", None)
    return sha256_json(d)


def software_versions() -> dict[str, Any]:
    import importlib.metadata as md
    out: dict[str, Any] = {"python": _platform.python_version()}
    for pkg in ("isaacsim", "isaaclab", "isaaclab_physx", "isaaclab_newton", "isaaclab_tasks",
                "newton", "mujoco-warp", "warp-lang", "torch", "numpy", "rsl-rl-lib"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            continue
    out["source_commits"] = {"corpus_capture": {"status": "see research/failure_corpus provenance"}}
    return out


def hardware() -> dict[str, Any]:
    import os
    import subprocess
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip().splitlines()
    return {"gpu": os.environ.get("IVF_HARDWARE_LABEL", "NVIDIA GPU"),
            "driver": drv[0] if drv else "unavailable", "platform": "Linux"}


# ----------------------------------------------------------------------------------------
# one job
# ----------------------------------------------------------------------------------------

def make_env(job: dict[str, Any], mode: str):
    import gymnasium as gym
    import isaaclab.sim as sim_utils

    platform, fault = job["platform"], job.get("fault", {"family": "none"})
    backend_mode = job["backend_mode"]
    n = OPEN_LOOP_ENVS if mode == "open_loop" else CLOSED_LOOP_ENVS
    sim_utils.create_new_stage()
    cfg = build_cfg(platform, backend_mode, n, job["seed"])
    controlled_cfg(cfg, mode=mode)
    apply_cfg_fault(cfg, fault, mode=mode)
    if fault.get("family") == "physx_only_setting":
        apply_physx_only_setting(cfg, fault)
    env = gym.make(TASKS[platform], cfg=cfg)
    env.unwrapped.sim._app_control_on_stop_handle = None
    return env


def wrong_termination_body(u, fault) -> str | None:
    """Contact-sensor body name at the configured wrong index (index-based body mismatch)."""
    if fault.get("family") != "termination_body_mismatch":
        return None
    names = list(u.scene["contact_forces"].body_names)
    return names[int(fault["params"]["body_index"])]


def retarget_termination(u, fault) -> None:
    """Point the live base-contact termination term at the wrong body (closed loop)."""
    import re as _re
    from isaaclab.managers import SceneEntityCfg
    name = wrong_termination_body(u, fault)
    if name is None or "base_contact" not in u.termination_manager.active_terms:
        return
    tc = u.termination_manager.get_term_cfg("base_contact")
    sc = SceneEntityCfg("contact_forces", body_names=[_re.escape(name)])
    sc.resolve(u.scene)
    tc.params["sensor_cfg"] = sc
    u.termination_manager.set_term_cfg("base_contact", tc)


def close_env(env):
    from isaaclab.sim import SimulationContext
    try:
        if env is not None:
            env.close()
    finally:
        SimulationContext.clear_instance()


def write_reset(u, req: dict[str, np.ndarray], fault: dict[str, Any]) -> dict[str, Any]:
    """Write the requested initial state; reset-family faults change what is *applied*."""
    import torch
    robot = u.scene["robot"]
    dev = u.device
    fam = fault.get("family", "none")
    jpos = t(robot.data.default_joint_pos).clone()
    jvel = torch.zeros_like(jpos)
    req_off = torch.as_tensor(req["joint_offset"], device=dev)
    req_jpos = jpos + req_off
    applied_jpos = jpos.clone() if fam == "reset_joint_offsets_ignored" else req_jpos
    root = t(robot.data.default_root_state).clone()
    root[:, :3] += u.scene.env_origins
    req_vel = torch.cat([torch.as_tensor(req["root_lin_vel"], device=dev),
                         torch.as_tensor(req["root_ang_vel"], device=dev)], dim=1)
    applied_vel = torch.zeros_like(req_vel) if fam == "reset_velocity_dropped" else req_vel
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(applied_vel)
    robot.write_joint_state_to_sim(applied_jpos, jvel)
    u.scene.write_data_to_sim()
    return {
        "requested_joint_pos_offset_digest": sha256_json(req["joint_offset"].round(7).tolist()),
        "applied_joint_pos_digest": sha256_json(applied_jpos.cpu().numpy().round(7).tolist()),
        "requested_root_vel": req_vel[0].cpu().tolist(),
        "applied_root_vel": applied_vel[0].cpu().tolist(),
    }


def illegal_contact(u, platform: str, fault: dict[str, Any]) -> tuple[Any, Any]:
    """Evaluate the task's base-contact termination as configured for this candidate."""
    import torch
    sensor = u.scene["contact_forces"]
    body_regex = BASE_BODY[platform]
    threshold = 1.0
    wrong = wrong_termination_body(u, fault)
    if wrong is not None:
        import re as _re
        body_regex = _re.escape(wrong)
    ids, _ = sensor.find_bodies(body_regex)
    hist = t(sensor.data.net_forces_w_history)
    if len(ids) == 0:
        force = torch.zeros(hist.shape[0], device=hist.device)
    else:
        force = torch.max(torch.norm(hist[:, :, ids], dim=-1), dim=1)[0].max(dim=1)[0]
    return force, (force > threshold).float()


def run_open_loop(job, out_dir: Path, replay_actions: np.ndarray | None, policy) -> dict[str, Any]:
    import torch
    platform, fault = job["platform"], job.get("fault", {"family": "none"})
    env = make_env(job, "open_loop")
    try:
        u = env.unwrapped
        robot = u.scene["robot"]
        ident = physics_identity(u)
        joint_names = list(robot.joint_names)
        nj = len(joint_names)
        n = OPEN_LOOP_ENVS
        pin_commands(u, command_schedule(n))
        obs, _ = env.reset()
        req = requested_initial_state(nj, n, job["seed"])
        reset_info = write_reset(u, req, fault)
        obs = u.observation_manager.compute()
        slices = obs_term_slices(u)
        perm_kind = fault.get("params", {}).get("perm", "per_limb")
        perm_idx = joint_permutation(joint_names, perm_kind)
        base_ids, _ = robot.find_bodies(BASE_BODY[platform])

        S = OPEN_LOOP_STEPS
        rec = {k: [] for k in ("joint_pos", "joint_vel", "joint_pos_target", "root_link_pos_w",
                               "root_link_quat_w", "root_lin_vel_b", "root_ang_vel_b", "base_height",
                               "policy_obs", "base_contact_force", "illegal_contact")}
        actions_out = np.zeros((S, n, nj), dtype=np.float32)
        finite = True
        for step in range(S):
            if replay_actions is not None:
                a = torch.as_tensor(replay_actions[step], device=u.device)
            else:  # reference job: generate the stream from the policy on clean observations
                with torch.inference_mode():
                    a = policy(obs["policy"])
            actions_out[step] = a.detach().cpu().numpy()
            obs, rew, term, trunc, _ = env.step(transform_action(a, fault, perm_idx))
            finite = finite and bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all())
            rec["joint_pos"].append(t(robot.data.joint_pos).cpu().numpy())
            rec["joint_vel"].append(t(robot.data.joint_vel).cpu().numpy())
            rec["joint_pos_target"].append(t(robot.data.joint_pos_target).cpu().numpy())
            rec["root_link_pos_w"].append(t(robot.data.root_link_pos_w).cpu().numpy())
            rec["root_link_quat_w"].append(t(robot.data.root_link_quat_w).cpu().numpy())
            rec["root_lin_vel_b"].append(t(robot.data.root_lin_vel_b).cpu().numpy())
            rec["root_ang_vel_b"].append(t(robot.data.root_ang_vel_b).cpu().numpy())
            z = t(robot.data.body_link_pos_w)[:, base_ids[0], 2] - u.scene.env_origins[:, 2]
            rec["base_height"].append(z.cpu().numpy()[:, None])
            rec["policy_obs"].append(transform_obs(obs["policy"], fault, slices, perm_idx).cpu().numpy())
            force, flag = illegal_contact(u, platform, fault)
            rec["base_contact_force"].append(force.cpu().numpy()[:, None])
            rec["illegal_contact"].append(flag.cpu().numpy()[:, None])
        arrays = {k: np.stack(v).astype(np.float32) for k, v in rec.items()}
        timing = {"physics_dt": float(u.physics_dt), "control_dt": float(u.step_dt),
                  "decimation": int(u.cfg.decimation)}
        full_digest = full_cfg_digest(u)
        preset_path_note = job["backend_mode"]
    finally:
        close_env(env)

    identity = {"platform": platform, "task": TASKS[platform], "num_envs": n, "steps": S,
                "seed": job["seed"], "command_schedule": command_schedule(n).tolist(),
                "requested_initial_state": {k: v.round(7).tolist() for k, v in req.items()},
                "action_stream": job.get("replay_ref", "self")}
    units = {"joint_pos": "rad", "joint_vel": "rad/s", "joint_pos_target": "rad",
             "root_link_pos_w": "m", "root_link_quat_w": "dimensionless", "root_lin_vel_b": "m/s",
             "root_ang_vel_b": "rad/s", "base_height": "m", "policy_obs": "dimensionless",
             "base_contact_force": "N", "illegal_contact": "dimensionless"}
    frames = {"joint_pos": "joint", "joint_vel": "joint", "joint_pos_target": "joint",
              "root_link_pos_w": "world", "root_link_quat_w": "world", "root_lin_vel_b": "base",
              "root_ang_vel_b": "base", "base_height": "world", "policy_obs": "policy",
              "base_contact_force": "world", "illegal_contact": "event"}
    contract = {
        "schema": "trajectory_bundle/v1", "run_status": "completed",
        "declared_steps": S, "captured_steps": S,
        "task": {"id": f"{platform}_velocity_flat_open_loop", "variant": TASKS[platform],
                 "config_digest_sha256": sha256_json(identity), "joint_names": joint_names,
                 "full_cfg_digest_sha256": full_digest,
                 "asset": {"id": f"isaaclab_assets:{platform}", "source_uri": "isaaclab nucleus",
                           "binary_identity": "unverifiable"}},
        "backend": {"id": ident["backend"], "solver_settings": ident["solver_settings"],
                    "features": ["articulation", "contact"]},
        "software": software_versions(), "hardware": hardware(),
        "seed": {"value": job["seed"], "env_ids": list(range(n)),
                 "env_order": "InteractiveScene clone index, ascending"},
        "timing": {**timing, "action_applied": "before_physics_step, held for decimation substeps",
                   "capture_hook": "post_env_step", "warmup_steps": 0,
                   "warmup_semantics": "none; sample 0 is the state after the first control step",
                   "timestamp_convention": "sample i is post-step state at (i + 1) * control_dt"},
        "frames": {"convention": "world_z_up_right_handed", "length_unit": "m", "angle_unit": "rad",
                   "note": "root arrays are world-frame and include the per-environment origin"},
        "quaternion": {"layout": "xyzw", "scalar_first": False, "normalized": True,
                       "hemisphere": "unconstrained"},
        "reset": {"semantics": "writes_pose_and_velocity",
                  "initial_state_digest": sha256_json(identity["requested_initial_state"]),
                  "initial_state": reset_info, "applied_before_step": 0, "randomized_fields": []},
        "termination": {"declared": True, "signal": "illegal_contact",
                        "condition": "illegal_contact > 0.5 (base contact force above threshold)",
                        "on_termination": "no_auto_reset (the rollout continues so trajectories stay aligned)"},
        "arrays": {k: {"shape": list(a.shape), "dtype": a.dtype.name, "unit": units[k], "frame": frames[k],
                       "semantics": f"captured post_env_step; joint order {joint_names}"}
                   for k, a in arrays.items()},
        "producer_notes": {"backend_mode": preset_path_note},
    }
    bundle = out_dir / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bundle / "trajectories.npz", **arrays, __actions__=actions_out)
    meta = {"scenario_name": contract["task"]["id"], "backend": ident["backend"], "device": "cuda:0",
            "seed": job["seed"], "physics_dt": timing["physics_dt"], "horizon": S, "num_envs": n,
            "capture_contract": contract}
    (bundle / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    root_hash = finalize_v1(bundle, "completed")
    return {"bundle": str(bundle), "bundle_sha256": root_hash, "finite": finite,
            "actions": actions_out, "physics": ident, "timing": timing}


def run_closed_loop(job, policy) -> dict[str, Any]:
    """Conventional sim-to-sim evaluation: run the policy with the env's own resets."""
    import torch
    platform, fault = job["platform"], job.get("fault", {"family": "none"})
    env = make_env(job, "closed_loop")
    try:
        u = env.unwrapped
        robot = u.scene["robot"]
        n = CLOSED_LOOP_ENVS
        sched = command_schedule(n)
        pin_commands(u, sched)
        retarget_termination(u, fault)
        obs, _ = env.reset()
        slices = obs_term_slices(u)
        perm_idx = joint_permutation(list(robot.joint_names), fault.get("params", {}).get("perm", "per_limb"))
        ret = torch.zeros(n, device=u.device)
        falls = torch.zeros(n, device=u.device)
        track_err, finite = [], True
        cmd = torch.as_tensor(sched, device=u.device)
        for _ in range(CLOSED_LOOP_STEPS):
            with torch.inference_mode():
                a = policy(transform_obs(obs["policy"], fault, slices, perm_idx))
            obs, rew, term, trunc, _ = env.step(transform_action(a, fault, perm_idx))
            finite = finite and bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all())
            ret += rew
            falls += (term & ~trunc).float() if term.dtype == torch.bool else term.float()
            v = t(robot.data.root_lin_vel_b)[:, :2]
            track_err.append(torch.linalg.norm(v - cmd[:, :2], dim=-1).mean().item())
        result = {
            "mean_return_per_env": float(ret.mean().item()),
            "fall_terminations_per_env": float(falls.mean().item()),
            "mean_tracking_error": float(np.mean(track_err)),
            "finite": finite,
            "physics": physics_identity(u),
        }
    finally:
        close_env(env)
    return result


def load_policy(path: str):
    import torch
    pol = torch.jit.load(path, map_location="cuda:0")
    pol.eval()
    return pol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    import isaaclab_tasks  # noqa: F401

    plan = json.loads(Path(args.plan).read_text())
    out_root = Path(args.out)
    policy = load_policy(plan["policy"])
    replay_cache: dict[str, np.ndarray] = {}
    for job in plan["jobs"]:
        jdir = out_root / job["capture_id"]
        if (jdir / "bundle" / "COMPLETE").exists() and (jdir / "result.json").exists():
            prev = json.loads((jdir / "result.json").read_text())
            if prev.get("replay_ref_key"):
                replay_cache[prev["replay_ref_key"]] = np.load(jdir / "reference_actions.npy")
            print(f"[corpus] skip existing {job['capture_id']}", flush=True)
            continue
        jdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        result: dict[str, Any] = {"capture_id": job["capture_id"], "native": {}}
        try:
            ref_key = job.get("replay_ref")
            replay = replay_cache.get(ref_key) if ref_key else None
            if ref_key and replay is None and not job.get("is_reference"):
                raise RuntimeError(f"reference stream {ref_key} not available")
            ol = run_open_loop(job, jdir, None if job.get("is_reference") else replay, policy)
            if job.get("is_reference"):
                np.save(jdir / "reference_actions.npy", ol["actions"])
                replay_cache[job["replay_ref"]] = ol["actions"]
                result["replay_ref_key"] = job["replay_ref"]
            cl = run_closed_loop(job, policy)
            result.update({
                "bundle": ol["bundle"], "bundle_sha256": ol["bundle_sha256"],
                "open_loop_physics": ol["physics"], "timing": ol["timing"],
                "closed_loop": {k: v for k, v in cl.items() if k != "physics"},
                "closed_loop_physics": cl["physics"],
                "native": {"exception": None, "finite_open_loop": ol["finite"],
                           "finite_closed_loop": cl["finite"],
                           "native_pass": bool(ol["finite"] and cl["finite"])},
            })
        except Exception as exc:
            result["native"] = {"exception": repr(exc)[:2000], "native_pass": False,
                                "traceback": traceback.format_exc()[-4000:]}
            print(f"[corpus] job {job['capture_id']} raised: {exc!r}", file=sys.stderr, flush=True)
        result["wall_s"] = round(time.time() - t0, 1)
        (jdir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        print(f"[corpus] done {job['capture_id']} in {result['wall_s']}s native_pass="
              f"{result['native'].get('native_pass')}", flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
