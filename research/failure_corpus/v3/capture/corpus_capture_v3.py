# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Failure-corpus capture producer for Isaac Lab velocity-locomotion tasks.

Research tooling, not part of IVF. Like ``adapters/parity_capture`` it writes the
``trajectory_bundle/v1`` file format directly and never imports IVF, so IVF is judged
through the same file boundary it was designed around.

Producer rules (pre-registered, amended by deviations D1-D4 before any fault case ran):

* Every contract field is read from the live environment at runtime, never from the case
  label. Labels live outside the bundle, so neither IVF nor any baseline can read them.
* Joint order. PhysX and Newton/MJWarp enumerate joints differently in this Isaac Lab
  checkout (breadth-first vs per-limb depth-first) and the checkout has no ordering remap.
  The policy was trained on PhysX, so its *canonical* order is the PhysX order. A correct
  experiment remaps canonical <-> native at the policy interface and records every joint
  array in canonical order, declaring that order in ``task.joint_names``. Joint-order
  fault families break exactly one of those two steps.

One Kit process runs a *plan*: an ordered list of capture jobs for one platform. Each job
builds the env, runs an open-loop replay capture (the IVF protocol) and a closed-loop
policy evaluation (the conventional sim-to-sim protocol), then tears the env down.

Usage (inside the Isaac Lab interpreter):
    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES $ISAACLAB_PYTHON corpus_capture.py \
        --plan plan.json --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform as _platform
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

TASKS = {
    "go2": "Isaac-Velocity-Flat-UnitreeGo2",
    "g1": "Isaac-Velocity-Flat-G1",
    "h1": "Isaac-Velocity-Flat-H1",
    "anymal_d": "Isaac-Velocity-Flat-AnymalD",
    "cassie": "Isaac-Velocity-Flat-Cassie",
    "spot": "Isaac-Velocity-Flat-Spot",
}
# Body watched by each task's base_contact termination term.
BASE_BODY = {"go2": "base", "g1": "torso_link", "h1": "torso_link", "anymal_d": "base", "cassie": "pelvis", "spot": "body"}
# Actuator armature the policies were trained with (PhysX resolved value). Only GO2 has a
# backend-conditioned armature preset in this checkout (0.0 PhysX / 0.02 Newton); the
# controlled protocol pins it to the training value on both backends.
PINNED_ARMATURE = {"go2": 0.0}

OPEN_LOOP_ENVS = 16
OPEN_LOOP_STEPS = 250  # control steps (5 s at 50 Hz)
CLOSED_LOOP_ENVS = 64
CLOSED_LOOP_STEPS = 1000  # control steps (20 s at 50 Hz, one default episode)
JOINT_OBS_TERMS = ("joint_pos", "joint_vel", "actions")
PRODUCER = {"name": "ivf-failure-corpus-capture", "version": "0.4.0"}
#: optional externally authored fault families (see load_fault_plugin); None when unused
PLUGIN = None


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
        "".join(f"{d}  {n}\n" for n, d in sorted(checksums.items())), encoding="utf-8"
    )
    root_hash = sha256_file(root / "CHECKSUMS.sha256")
    (root / "COMPLETE").write_text(
        json.dumps(
            {
                "schema": "trajectory_bundle/v1",
                "run_status": run_status,
                "n_files": len(checksums),
                "checksums_sha256": root_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root_hash


def command_schedule(num_envs: int) -> np.ndarray:
    """Deterministic per-env velocity command (vx, vy, wz); identical for every job."""
    vx = np.linspace(0.2, 0.8, num_envs)
    wz = np.array([(-0.3, 0.0, 0.3)[i % 3] for i in range(num_envs)])
    return np.stack([vx, np.zeros(num_envs), wz], axis=1).astype(np.float32)


def requested_initial_state(num_joints: int, num_envs: int, seed: int) -> dict[str, np.ndarray]:
    """Requested reset state in canonical joint order: seeded joint offsets, forward velocity."""
    rng = np.random.default_rng(10_000 + seed)
    return {
        "joint_offset": rng.uniform(-0.05, 0.05, size=(num_envs, num_joints)).astype(np.float32),
        "root_lin_vel": np.tile(np.array([[0.3, 0.0, 0.0]], dtype=np.float32), (num_envs, 1)),
        "root_ang_vel": rng.uniform(-0.2, 0.2, size=(num_envs, 3)).astype(np.float32),
    }


def lr_swapped(names: list[str]) -> list[str]:
    """``names`` with every left/right (FL/FR, RL/RR, LF/RF, LH/RH) pair exchanged."""
    swap = {
        "left": "right",
        "right": "left",
        "FL": "FR",
        "FR": "FL",
        "RL": "RR",
        "RR": "RL",
        "LF": "RF",
        "RF": "LF",
        "LH": "RH",
        "RH": "LH",
    }
    out = []
    for name in names:
        other = name
        for a, b in swap.items():
            if name.startswith(a):
                other = b + name[len(a) :]
                break
        out.append(other if other in names else name)
    return out


# ----------------------------------------------------------------------------------------
# config construction and cfg-level fault injection
# ----------------------------------------------------------------------------------------


def build_cfg(platform: str, backend_mode: str, num_envs: int, seed: int):
    """Return a resolved env cfg for the requested backend path.

    ``physx``: stock default preset. ``newton``: raw registry cfg resolved with
    ``newton_mjwarp`` selected (the upstream smoke-test path after PR #7103).
    ``newton_via_pre7103_test_path``: the pre-#7103 smoke-test path verbatim (parse, which
    resolves presets to default, then apply ``newton_mjwarp`` as a global override).
    """
    from isaaclab_tasks.utils.hydra import apply_overrides, collect_presets, resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    task = TASKS[platform]
    if backend_mode == "physx":
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
    elif backend_mode == "newton":
        cfg = resolve_presets(
            load_cfg_from_registry(task, "env_cfg_entry_point"), selected=("newton_mjwarp",)
        )
        cfg.sim.device = "cuda:0"
        cfg.scene.num_envs = num_envs
    elif backend_mode == "newton_via_pre7103_test_path":
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
        raw = load_cfg_from_registry(task, "env_cfg_entry_point")
        apply_overrides(
            cfg,
            None,
            {"env": cfg.to_dict(), "agent": None},
            ["newton_mjwarp"],
            [],
            [],
            {"env": collect_presets(raw), "agent": {}},
        )
        cfg.scene.num_envs = num_envs
    else:
        raise ValueError(backend_mode)
    cfg.seed = seed
    return cfg


def controlled_cfg(cfg, platform: str, *, mode: str) -> None:
    """Controlled evaluation protocol shared by every job, clean and faulty.

    PLAY-style: observation noise off, pushes off, random startup mass/COM off, armature
    pinned to the training value, fixed command schedule. Open-loop additionally removes
    auto-reset terminations so trajectories stay aligned (the termination condition is still
    evaluated and recorded).
    """
    cfg.observations.policy.enable_corruption = False
    ev = cfg.events
    for name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com"):
        if hasattr(ev, name):
            setattr(ev, name, None)
    # Startup material randomization that actually samples is disabled on both backends, like
    # mass/COM above; a degenerate range (a fixed material) is kept. Spot's stock config
    # applies a sampled range on PhysX only (measured on clean calibration pairs).
    pm = getattr(ev, "physics_material", None)
    if pm is not None:
        prm = pm.params or {}
        ranges = [prm.get(k) for k in ("static_friction_range", "dynamic_friction_range", "restitution_range")]
        if any(r is not None and tuple(r)[0] != tuple(r)[1] for r in ranges):
            ev.physics_material = None
    # Pin actuator armature to the training (PhysX-resolved) value on every backend, so a
    # backend-conditioned preset cannot silently change the comparison inputs.
    for name, a in cfg.scene.robot.actuators.items():
        if name in _physx_armature(platform):
            a.armature = _physx_armature(platform)[name]
    cmd = cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = False
    if hasattr(cfg, "curriculum"):
        cfg.curriculum.terrain_levels = None


_ARMATURE_CACHE: dict[str, dict[str, Any]] = {}


def _physx_armature(platform: str) -> dict[str, Any]:
    if platform not in _ARMATURE_CACHE:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        ref = parse_env_cfg(TASKS[platform], device="cuda:0", num_envs=1)
        _ARMATURE_CACHE[platform] = {k: a.armature for k, a in ref.scene.robot.actuators.items()}
    return _ARMATURE_CACHE[platform]


def _scale(value, factor):
    if isinstance(value, dict):
        return {k: v * factor for k, v in value.items()}
    return None if value is None else value * factor


RUNTIME_FAMILIES = {
    "none",
    "termination_body_mismatch",
    "joint_order_interface",
    "capture_not_canonicalized",
    "obs_term_swap",
    "reset_velocity_dropped",
    "reset_joint_offsets_ignored",
    "preset_not_applied",
}


def apply_cfg_fault(cfg, platform: str, fault: dict[str, Any]) -> None:
    """Mutate the env cfg for cfg-level fault families. Runtime families are no-ops here."""
    fam, p = fault.get("family", "none"), fault.get("params", {})
    if fam == "timestep_dt_decimation":
        cfg.sim.dt = float(p["dt"])
        cfg.decimation = int(p["decimation"])
        cfg.sim.render_interval = cfg.decimation
    elif fam == "actuator_gain_scale":
        for a in cfg.scene.robot.actuators.values():
            a.stiffness = _scale(a.stiffness, p["stiffness_scale"])
            a.damping = _scale(a.damping, p["damping_scale"])
    elif fam == "armature_mismatch":
        for a in cfg.scene.robot.actuators.values():
            a.armature = float(p["armature"])
    elif fam == "contact_capacity":
        cfg.sim.physics.solver_cfg.nconmax = int(p["nconmax"])
        cfg.sim.physics.solver_cfg.njmax = int(p["njmax"])
    elif fam == "benign_capacity":
        sc = cfg.sim.physics.solver_cfg
        sc.nconmax = int(sc.nconmax * p["factor"])
        sc.njmax = int(sc.njmax * p["factor"])
    elif fam == "action_scale":
        cfg.actions.joint_pos.scale = float(p["scale"])
    elif fam == "randomization_asymmetry":
        import isaaclab_tasks.core.velocity.mdp as mdp
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        cfg.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=p["body"]),
                "mass_distribution_params": tuple(p["mass_range"]),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
    elif fam not in RUNTIME_FAMILIES and plugin_family(fault) is None:
        raise ValueError(f"unknown fault family {fam!r}")


# ----------------------------------------------------------------------------------------
# runtime helpers (need Kit)
# ----------------------------------------------------------------------------------------


def t(x):
    """Return a torch tensor from an Isaac Lab data field (ProxyArray or tensor)."""
    return x.torch if hasattr(x, "torch") else x


class JointMap:
    """Canonical <-> native joint indexing for one live env and one fault.

    ``iface``: native index the policy interface uses for canonical slot i (correct = the
    joint with the same name; faults substitute another order). ``rec``: native index
    recorded in canonical slot i (correct = same name; ``capture_not_canonicalized``
    records native order and declares native names instead).
    """

    def __init__(self, native: list[str], orders: dict[str, list[str]], fault: dict[str, Any]):
        canon = orders["physx"]
        fam, p = fault.get("family", "none"), fault.get("params", {})
        iface_names = canon
        if fam == "joint_order_interface":
            kind = p["interface"]
            if kind == "native":  # no remap: canonical slot i drives native joint i
                iface_names = list(native)
            elif kind == "newton_order":  # deployment assumes the Newton enumeration
                iface_names = orders["newton"]
            elif kind == "lr_swap":  # remap table with left/right exchanged
                iface_names = lr_swapped(canon)
            else:
                raise ValueError(kind)
        self.iface = [native.index(n) for n in iface_names]
        self.correct = [native.index(n) for n in canon]
        if fam == "capture_not_canonicalized":
            self.rec = list(range(len(native)))
            self.recorded_names = list(native)
        else:
            self.rec = self.correct
            self.recorded_names = list(canon)

    def action_to_native(self, a_canon):
        out = a_canon.clone()
        out[:, self.iface] = a_canon
        return out

    def obs_to_policy(self, obs, slices):
        o = obs.clone()
        for term in JOINT_OBS_TERMS:
            if term in slices:
                s = slices[term]
                o[:, s] = obs[:, s][:, self.iface]
        return o

    def record(self, x):
        return x[..., self.rec]


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
    out, start = {}, 0
    for n, d in zip(mgr.active_terms["policy"], mgr.group_obs_term_dim["policy"]):
        width = int(np.prod(d))
        out[n] = slice(start, start + width)
        start += width
    return out


def policy_obs(obs, fault, slices, jmap: JointMap):
    """What the policy is fed: joint remap (possibly faulty), then observation-term faults."""
    o = jmap.obs_to_policy(obs, slices)
    if fault.get("family") == "obs_term_swap":
        p = fault["params"]
        a, b = slices[p["a"]], slices[p["b"]]
        oa, ob = o[:, a].clone(), o[:, b].clone()
        o[:, a], o[:, b] = ob, oa
    return o


def physics_identity(env) -> dict[str, Any]:
    physics_cfg = env.sim.cfg.physics
    try:
        settings = json.loads(json.dumps(physics_cfg.to_dict(), default=str, sort_keys=True))
    except Exception as exc:  # pragma: no cover
        settings = {"unavailable": str(exc)}
    manager = str(env.sim.physics_manager)
    backend = (
        "newton" if "newton" in manager.lower() else "physx" if "physx" in manager.lower() else manager
    )
    return {
        "backend": backend,
        "manager": manager,
        "cfg_class": type(physics_cfg).__name__,
        "solver_settings": {"manager": manager, "cfg_class": type(physics_cfg).__name__, **settings},
    }


def full_cfg_digest(env) -> str:
    """Digest of the resolved env cfg with the physics subtree and sensor class removed.

    Used only by the pre-registered 'rich identity' producer ablation.
    """
    d = env.cfg.to_dict()
    d.get("sim", {}).pop("physics", None)
    d.pop("seed", None)
    cs = d.get("scene", {}).get("contact_forces")
    if isinstance(cs, dict):
        cs.pop("class_type", None)
    return sha256_json(d)


def software_versions() -> dict[str, Any]:
    import importlib.metadata as md

    out: dict[str, Any] = {"python": _platform.python_version()}
    for pkg in (
        "isaacsim",
        "isaaclab",
        "isaaclab_physx",
        "isaaclab_newton",
        "isaaclab_tasks",
        "newton",
        "mujoco-warp",
        "warp-lang",
        "torch",
        "numpy",
        "rsl-rl-lib",
    ):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            continue
    out["source_commits"] = {"isaaclab": {"status": "recorded in research/failure_corpus/provenance"}}
    return out


def hardware() -> dict[str, Any]:
    import os
    import subprocess

    drv = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        .stdout.strip()
        .splitlines()
    )
    return {
        "gpu": os.environ.get("IVF_HARDWARE_LABEL", "NVIDIA GPU"),
        "driver": drv[0] if drv else "unavailable",
        "platform": "Linux",
    }


def load_fault_plugin(path: str | None):
    """Load an externally authored fault module exposing ``FAMILIES``.

    Each family may define ``cfg(cfg, platform, params)``, ``post_build(u, params)``,
    ``transform_action(a_native, u, params)``, ``transform_obs(obs_policy, u, params)`` and
    ``transform_reset(state, u, params)``. The producer applies them without inspecting
    what they do; its recording logic is unchanged, so a plugin fault is visible to IVF
    only through the generic evidence the producer always records.
    """
    global PLUGIN
    if not path:
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("ivf_corpus_fault_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    PLUGIN = mod


def plugin_family(fault):
    if PLUGIN is None:
        return None
    return getattr(PLUGIN, "FAMILIES", {}).get(fault.get("family"))


def plugin_call(fault, hook, *args, default=None):
    fam = plugin_family(fault)
    fn = getattr(fam, hook, None) if fam is not None else None
    if fn is None:
        return default
    return fn(*args, fault.get("params", {}))


def resolved_body_names(u) -> list[str]:
    return list(u.scene["robot"].body_names)


def effective_model(u, jmap: "JointMap") -> dict[str, Any]:
    """Model parameters read back from the simulator, bodies sorted by name, joints canonical."""
    import torch
    robot = u.scene["robot"]
    names = resolved_body_names(u)
    order = sorted(range(len(names)), key=lambda i: names[i])
    out: dict[str, Any] = {"body_names": [names[i] for i in order], "joint_names": jmap.recorded_names}
    mass = t(robot.data.body_mass)
    out["body_mass"] = np.round(mass[:, order].detach().cpu().numpy().astype(float), 6).tolist()
    for key in ("joint_stiffness", "joint_damping", "joint_armature"):
        val = getattr(robot.data, key, None)
        if val is not None:
            out[key] = np.round(jmap.record(t(val)).detach().cpu().numpy().astype(float), 6).tolist()
    nj = len(robot.joint_names)
    for gain in ("stiffness", "damping"):
        full = torch.full((u.num_envs, nj), float("nan"), device=u.device)
        for act in robot.actuators.values():
            g = getattr(act, gain, None)
            if isinstance(g, torch.Tensor):
                full[:, act.joint_indices] = g.to(full.dtype)
        out[f"actuator_{gain}"] = np.nan_to_num(
            np.round(jmap.record(full).detach().cpu().numpy().astype(float), 6), nan=-1.0).tolist()
    return out


def policy_interface(u, fault, slices, jmap: "JointMap") -> dict[str, Any]:
    """The interface the policy was actually fed, from the same code path that builds it."""
    layout = [[k, s.stop - s.start] for k, s in sorted(slices.items(), key=lambda kv: kv[1].start)]
    if fault.get("family") == "obs_term_swap":
        a, b = fault["params"]["a"], fault["params"]["b"]
        names = [x[0] for x in layout]
        ia, ib = names.index(a), names.index(b)
        layout[ia][0], layout[ib][0] = b, a
    iface = [u.scene["robot"].joint_names[i] for i in jmap.iface]
    term = u.action_manager.get_term("joint_pos")
    return {"obs_layout": layout, "obs_joint_order": iface, "action_joint_order": iface,
            "action_scale": float(term.cfg.scale) if not isinstance(term.cfg.scale, dict) else term.cfg.scale}


def termination_semantics(u, platform, fault) -> dict[str, Any]:
    """The live termination config (after fault hooks), with contact bodies resolved by name."""
    import re as _re
    stash = u._corpus_termination_stash
    sensor = u.scene["contact_forces"]
    out = {"episode_length_s": stash["episode_length_s"], "terms": {}}
    for name, term in sorted(stash["terms"].items()):
        entry = dict(term)
        sc = (term["params"] or {}).get("sensor_cfg")
        if isinstance(sc, dict) and sc.get("body_names") is not None:
            wrong = wrong_termination_body(u, fault) if name == "base_contact" else None
            regex = _re.escape(wrong) if wrong is not None else sc["body_names"]
            entry["resolved_bodies"] = sorted(sensor.find_bodies(regex)[1])
        out["terms"][name] = entry
    return out


def _termination_semantics_v2(u, platform, fault) -> list[dict[str, Any]]:
    import re as _re
    wrong = wrong_termination_body(u, fault)
    regex = _re.escape(wrong) if wrong is not None else BASE_BODY[platform]
    sensor = u.scene["contact_forces"]
    ids, names = sensor.find_bodies(regex)
    return [{"term": "base_contact", "func": "illegal_contact", "bodies": sorted(names), "threshold": 1.0},
            {"term": "time_out", "func": "time_out"}]


class ResourceMonitor:
    """Max buffer utilization over the run; available only where the backend exposes counters."""

    def __init__(self, u):
        self.solver, self.max_contact, self.max_constraint, self.reason = None, 0.0, 0.0, None
        try:
            solver = getattr(u.sim.physics_manager, "_solver", None)
            d = getattr(solver, "mjw_data", None) if solver is not None else None
            if d is not None and hasattr(d, "nacon") and hasattr(d, "naconmax"):
                self.solver = solver
            else:
                self.reason = f"{u.sim.physics_manager} exposes no buffer occupancy counters"
        except Exception as exc:  # pragma: no cover
            self.reason = repr(exc)

    def sample(self):
        if self.solver is None:
            return
        d = self.solver.mjw_data
        self.max_contact = max(self.max_contact, float(d.nacon.numpy()[0]) / max(float(d.naconmax), 1.0))
        njmax = float(getattr(d, "njmax", 0) or 0)
        if njmax > 0 and hasattr(d, "nefc"):
            self.max_constraint = max(self.max_constraint, float(d.nefc.numpy().max()) / njmax)

    def report(self):
        if self.solver is None:
            return {"status": "unavailable", "reason": self.reason}
        return {"status": "available", "max_contact_utilization": round(self.max_contact, 6),
                "max_constraint_utilization": round(self.max_constraint, 6)}


def _jsonable_param(v):
    from isaaclab.managers import SceneEntityCfg
    if isinstance(v, SceneEntityCfg):
        return {"name": v.name, "body_names": v.body_names, "joint_names": v.joint_names}
    if isinstance(v, dict):
        return {str(k): _jsonable_param(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable_param(x) for x in v]
    if isinstance(v, float):
        return round(v, 9)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return getattr(v, "__name__", type(v).__name__)


def stash_terminations(cfg) -> dict[str, Any]:
    """Snapshot the task's live termination config (after fault hooks, before open-loop removal)."""
    terms = {}
    for name, term in vars(cfg.terminations).items():
        if term is None or not hasattr(term, "func"):
            continue
        terms[name] = {"func": getattr(term.func, "__name__", type(term.func).__name__),
                       "time_out": bool(getattr(term, "time_out", False)),
                       "params": _jsonable_param(dict(term.params or {}))}
    return {"terms": terms, "episode_length_s": float(cfg.episode_length_s)}


def randomization_state(u) -> list[dict[str, Any]]:
    """Active event (randomization/reset) terms as configured in the live event manager."""
    out = []
    mgr = u.event_manager
    for mode, names in mgr.active_terms.items():
        for name in names:
            tc = mgr.get_term_cfg(name)
            out.append({"term": name, "mode": mode,
                        "func": getattr(tc.func, "__name__", type(tc.func).__name__),
                        "params": _jsonable_param(dict(tc.params or {})),
                        "interval_range_s": _jsonable_param(getattr(tc, "interval_range_s", None))})
    return sorted(out, key=lambda x: x["term"])


def material_readback(u) -> dict[str, Any]:
    """Per-body static friction and restitution read back from the simulator (bodies by name)."""
    import torch
    import warp as wp
    robot = u.scene["robot"]
    names = list(robot.body_names)
    backend = physics_identity(u)["backend"]
    if backend == "physx":
        view = robot.root_view
        counts = [robot._physics_sim_view.create_rigid_body_view(lp).max_shapes for lp in view.link_paths[0]]
        mats = wp.to_torch(view.get_material_properties()).to(torch.float64)  # (envs, shapes, 3)
        mu, rest = mats[..., 0], mats[..., 2]
    else:
        import isaaclab_newton.physics.newton_manager as nm
        model = nm.NewtonManager.get_model()
        counts = list(robot.num_shapes_per_body)
        mu = wp.to_torch(robot._root_view.get_attribute("shape_material_mu", model)[:, 0]).to(torch.float64)
        rest = wp.to_torch(robot._root_view.get_attribute("shape_material_restitution", model)[:, 0]).to(torch.float64)
        # PhysX material views cover collision shapes only; restrict Newton to the same set
        flags = wp.to_torch(robot._root_view.get_attribute("shape_flags", model)[:, 0])
        collide = (flags & 2) != 0
        mu = torch.where(collide, mu, torch.full_like(mu, float("nan")))
        rest = torch.where(collide, rest, torch.full_like(rest, float("nan")))
    # Body assignment of collision shapes differs between backends (measured on clean G1, H1,
    # ANYmal-D, Cassie), so compare the robot's collision-material distribution per env instead.
    del counts, names

    def stats(x):
        valid = ~torch.isnan(x)
        big, small = torch.full_like(x, float("inf")), torch.full_like(x, float("-inf"))
        n = valid.sum(dim=1).clamp(min=1)
        mean = torch.where(valid, x, torch.zeros_like(x)).sum(dim=1) / n
        lo = torch.where(valid, x, big).min(dim=1).values
        hi = torch.where(valid, x, small).max(dim=1).values
        return np.round(torch.stack([mean, lo, hi], dim=1).cpu().numpy(), 6).tolist()
    return {"status": "available", "material_static_friction_stats": stats(mu),
            "material_restitution_stats": stats(rest)}


def solver_effective(u) -> dict[str, Any]:
    """Solver settings read back from the running solver, in the config vocabulary."""
    solver = getattr(u.sim.physics_manager, "_solver", None)
    model = getattr(solver, "mjw_model", None) if solver is not None else None
    if model is None:
        return {"status": "unavailable", "reason": f"{u.sim.physics_manager} exposes no solver read-back"}
    from mujoco_warp._src.types import ConeType, IntegratorType, SolverType

    def scalar(x):
        try:
            return float(np.asarray(x.numpy()).reshape(-1)[0])
        except AttributeError:
            return float(x)
    opt = model.opt
    return {"status": "available",
            "integrator": IntegratorType(int(scalar(opt.integrator))).name.lower(),
            "cone": ConeType(int(scalar(opt.cone))).name.lower(),
            "solver": SolverType(int(scalar(opt.solver))).name.lower(),
            "iterations": int(scalar(opt.iterations)), "ls_iterations": int(scalar(opt.ls_iterations)),
            "impratio": round(1.0 / scalar(opt.impratio_invsqrt) ** 2, 6), "tolerance": round(scalar(opt.tolerance), 12),
            # substeps and solver dt come from the manager that drives solver.step; opt.timestep is
            # only filled in at step time and reads as MuJoCo's default before the first step
            "num_substeps": int(u.sim.physics_manager._num_substeps),
            "solver_dt": round(float(u.sim.physics_manager._solver_dt), 9)}


def make_env(job: dict[str, Any], mode: str):
    import gymnasium as gym
    import isaaclab.sim as sim_utils

    platform, fault = job["platform"], job.get("fault", {"family": "none"})
    n = OPEN_LOOP_ENVS if mode == "open_loop" else CLOSED_LOOP_ENVS
    sim_utils.create_new_stage()
    cfg = build_cfg(platform, job["backend_mode"], n, job["seed"])
    controlled_cfg(cfg, platform, mode=mode)
    if plugin_family(fault) is None:
        apply_cfg_fault(cfg, platform, fault)
    else:
        plugin_call(fault, "cfg", cfg, platform)
    stash = stash_terminations(cfg)
    if mode == "open_loop":
        # after every fault hook: the task's own termination config is what gets recorded and evaluated
        cfg.episode_length_s = 1.0e6
        for name in list(stash["terms"]):
            if not stash["terms"][name]["time_out"]:
                setattr(cfg.terminations, name, None)
    env = gym.make(TASKS[platform], cfg=cfg)
    env.unwrapped.sim._app_control_on_stop_handle = None
    env.unwrapped._corpus_termination_stash = stash
    plugin_call(fault, "post_build", env.unwrapped)
    return env


def close_env(env):
    from isaaclab.sim import SimulationContext

    try:
        if env is not None:
            env.close()
    finally:
        SimulationContext.clear_instance()


def wrong_termination_body(u, fault) -> str | None:
    """Contact-sensor body at the configured wrong index (index-based body mismatch)."""
    if fault.get("family") != "termination_body_mismatch":
        return None
    return list(u.scene["contact_forces"].body_names)[int(fault["params"]["body_index"])]


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


def illegal_contact(u, platform: str, fault: dict[str, Any]):
    """Evaluate the task's base-contact termination condition as configured for this run."""
    import re as _re

    import torch

    sensor = u.scene["contact_forces"]
    wrong = wrong_termination_body(u, fault)
    term = u._corpus_termination_stash["terms"].get("base_contact")
    threshold = float(term["params"]["threshold"]) if term else 1.0
    if wrong is not None:
        body_regex = _re.escape(wrong)
    elif term:
        body_regex = term["params"]["sensor_cfg"]["body_names"]
    else:
        body_regex = BASE_BODY[platform]
    ids, _ = sensor.find_bodies(body_regex)
    hist = t(sensor.data.net_forces_w_history)
    if len(ids) == 0:
        force = torch.zeros(hist.shape[0], device=hist.device)
    else:
        force = torch.max(torch.norm(hist[:, :, ids], dim=-1), dim=1)[0].max(dim=1)[0]
    return force, (force > threshold).float()


def write_reset(u, req: dict[str, np.ndarray], fault: dict[str, Any], jmap: JointMap) -> dict[str, Any]:
    """Write the requested initial state; reset-family faults change what is *applied*."""
    import torch

    robot = u.scene["robot"]
    dev = u.device
    fam = fault.get("family", "none")
    jpos = t(robot.data.default_joint_pos).clone()
    off_native = torch.zeros_like(jpos)
    off_native[:, jmap.correct] = torch.as_tensor(req["joint_offset"], device=dev)
    req_jpos = jpos + off_native
    applied_jpos = jpos.clone() if fam == "reset_joint_offsets_ignored" else req_jpos
    root = t(robot.data.default_root_state).clone()
    root[:, :3] += u.scene.env_origins
    req_vel = torch.cat(
        [
            torch.as_tensor(req["root_lin_vel"], device=dev),
            torch.as_tensor(req["root_ang_vel"], device=dev),
        ],
        dim=1,
    )
    applied_vel = torch.zeros_like(req_vel) if fam == "reset_velocity_dropped" else req_vel
    applied_jvel = torch.zeros_like(jpos)
    state = plugin_call(fault, "transform_reset",
                        {"joint_pos": applied_jpos, "joint_vel": applied_jvel, "root_vel": applied_vel}, u,
                        default=None)
    if state is not None:
        applied_jpos, applied_jvel, applied_vel = state["joint_pos"], state["joint_vel"], state["root_vel"]
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(applied_vel)
    robot.write_joint_state_to_sim(applied_jpos, applied_jvel)
    u.scene.write_data_to_sim()
    # read the state back from the simulator, not from what this function intended
    u.sim.forward()
    u.scene.update(dt=0.0)
    got_vel = t(robot.data.root_com_vel_w)

    def canon(x):
        return np.round(jmap.record(x).detach().cpu().numpy().astype(float), 6).tolist()

    realization = {
        "requested": {"joint_pos": canon(req_jpos), "joint_vel": canon(torch.zeros_like(jpos)),
                      "root_vel": np.round(req_vel.detach().cpu().numpy().astype(float), 6).tolist()},
        "readback": {"joint_pos": canon(t(robot.data.joint_pos)), "joint_vel": canon(t(robot.data.joint_vel)),
                     "root_vel": np.round(got_vel.detach().cpu().numpy().astype(float), 6).tolist()},
    }
    return {
        "requested_joint_offset_digest": sha256_json(req["joint_offset"].round(7).tolist()),
        "requested_root_vel_env0": req_vel[0].cpu().tolist(),
        "_realization": realization,
    }


# ----------------------------------------------------------------------------------------
# the two protocols
# ----------------------------------------------------------------------------------------

UNITS = {
    "joint_pos": "rad",
    "joint_vel": "rad/s",
    "joint_pos_target": "rad",
    "root_link_pos_w": "m",
    "root_link_quat_w": "dimensionless",
    "root_lin_vel_b": "m/s",
    "root_ang_vel_b": "rad/s",
    "base_height": "m",
    "upright": "dimensionless",
    "policy_obs": "dimensionless",
    "base_contact_force": "N",
    "illegal_contact": "dimensionless",
}
FRAMES = {
    "joint_pos": "joint",
    "joint_vel": "joint",
    "joint_pos_target": "joint",
    "root_link_pos_w": "world",
    "root_link_quat_w": "world",
    "root_lin_vel_b": "base",
    "root_ang_vel_b": "base",
    "base_height": "world",
    "upright": "base",
    "policy_obs": "policy",
    "base_contact_force": "world",
    "illegal_contact": "event",
}


def run_open_loop(
    job, out_dir: Path, replay_actions: np.ndarray | None, policy, orders
) -> dict[str, Any]:
    import torch

    platform, fault = job["platform"], job.get("fault", {"family": "none"})
    env = make_env(job, "open_loop")
    try:
        u = env.unwrapped
        robot = u.scene["robot"]
        ident = physics_identity(u)
        jmap = JointMap(list(robot.joint_names), orders, fault)
        nj, n, S = len(robot.joint_names), OPEN_LOOP_ENVS, OPEN_LOOP_STEPS
        pin_commands(u, command_schedule(n))
        retarget_termination(u, fault)
        env.reset()
        req = requested_initial_state(nj, n, job["seed"])
        reset_info = write_reset(u, req, fault, jmap)
        realization = reset_info.pop("_realization")
        obs = u.observation_manager.compute()
        slices = obs_term_slices(u)
        experiment_inputs = {
            "runtime": {"backend": ident["backend"], "manager": ident["manager"]},
            "policy_interface": policy_interface(u, fault, slices, jmap),
            "effective_model": {**effective_model(u, jmap),
                                **{k: v for k, v in material_readback(u).items() if k != "status"}},
            "randomization": randomization_state(u),
            "solver_effective": solver_effective(u),
            "termination_semantics": termination_semantics(u, platform, fault),
            "reset_realization": realization,
        }
        monitor = ResourceMonitor(u)
        base_ids, _ = robot.find_bodies(
            BASE_BODY[platform] if platform in ("go2", "anymal_d", "spot") else "pelvis"
        )

        rec: dict[str, list] = {k: [] for k in UNITS}
        actions_out = np.zeros((S, n, nj), dtype=np.float32)
        finite = True
        for step in range(S):
            if replay_actions is not None:
                a = torch.as_tensor(replay_actions[step], device=u.device)
            else:  # reference job: generate the stream from the policy
                with torch.inference_mode():
                    a = policy(policy_obs(obs["policy"], fault, slices, jmap))
            actions_out[step] = a.detach().cpu().numpy()
            a_native = jmap.action_to_native(a)
            a_native = plugin_call(fault, "transform_action", a_native, u, default=a_native)
            obs, rew, term, trunc, _ = env.step(a_native)
            monitor.sample()
            finite = finite and bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all())
            rec["joint_pos"].append(jmap.record(t(robot.data.joint_pos)).cpu().numpy())
            rec["joint_vel"].append(jmap.record(t(robot.data.joint_vel)).cpu().numpy())
            rec["joint_pos_target"].append(jmap.record(t(robot.data.joint_pos_target)).cpu().numpy())
            rec["root_link_pos_w"].append(t(robot.data.root_link_pos_w).cpu().numpy())
            rec["root_link_quat_w"].append(t(robot.data.root_link_quat_w).cpu().numpy())
            rec["root_lin_vel_b"].append(t(robot.data.root_lin_vel_b).cpu().numpy())
            rec["root_ang_vel_b"].append(t(robot.data.root_ang_vel_b).cpu().numpy())
            z = t(robot.data.body_link_pos_w)[:, base_ids[0], 2] - u.scene.env_origins[:, 2]
            rec["base_height"].append(z.cpu().numpy()[:, None])
            rec["upright"].append((-t(robot.data.projected_gravity_b)[:, 2]).cpu().numpy()[:, None])
            po = policy_obs(obs["policy"], fault, slices, jmap)
            po = plugin_call(fault, "transform_obs", po, u, default=po)
            rec["policy_obs"].append(po.cpu().numpy())
            force, flag = illegal_contact(u, platform, fault)
            rec["base_contact_force"].append(force.cpu().numpy()[:, None])
            rec["illegal_contact"].append(flag.cpu().numpy()[:, None])
        arrays = {k: np.stack(v).astype(np.float32) for k, v in rec.items()}
        timing = {
            "physics_dt": float(u.physics_dt),
            "control_dt": float(u.step_dt),
            "decimation": int(u.cfg.decimation),
        }
        full_digest = full_cfg_digest(u)
        recorded_names = jmap.recorded_names
    finally:
        close_env(env)

    identity = {
        "platform": platform,
        "task": TASKS[platform],
        "num_envs": n,
        "steps": S,
        "seed": job["seed"],
        "command_schedule": command_schedule(n).tolist(),
        "requested_initial_state": {k: v.round(7).tolist() for k, v in req.items()},
        "action_stream": job.get("replay_ref", "self"),
    }
    experiment_inputs["resource_health"] = monitor.report()
    contract = {
        "experiment_inputs": experiment_inputs,
        "schema": "trajectory_bundle/v1",
        "run_status": "completed",
        "declared_steps": S,
        "captured_steps": S,
        "capture": {
            "capture_id": str(uuid.uuid4()),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "producer": PRODUCER,
        },
        "task": {
            "id": f"{platform}_velocity_flat_open_loop",
            "variant": TASKS[platform],
            "config_digest_sha256": sha256_json(identity),
            "joint_names": recorded_names,
            "full_cfg_digest_sha256": full_digest,
            "asset": {
                "id": f"isaaclab_assets:{platform}",
                "source_uri": "isaaclab nucleus",
                "binary_identity": "unverifiable",
            },
        },
        "backend": {
            "id": ident["backend"],
            "solver_settings": ident["solver_settings"],
            "features": ["articulation", "contact"],
        },
        "software": software_versions(),
        "hardware": hardware(),
        "seed": {
            "value": job["seed"],
            "env_ids": list(range(n)),
            "env_order": "InteractiveScene clone index, ascending",
        },
        "timing": {
            **timing,
            "action_applied": "before_physics_step, held for decimation substeps",
            "capture_hook": "post_env_step",
            "warmup_steps": 0,
            "warmup_semantics": "none; sample 0 is the state after the first control step",
            "timestamp_convention": "sample i is post-step state at (i + 1) * control_dt",
        },
        "frames": {
            "convention": "world_z_up_right_handed",
            "length_unit": "m",
            "angle_unit": "rad",
            "note": "root arrays are world-frame and include the per-environment origin",
        },
        "quaternion": {
            "layout": "xyzw",
            "scalar_first": False,
            "normalized": True,
            "hemisphere": "unconstrained",
        },
        "reset": {
            "semantics": "writes_pose_and_velocity",
            "initial_state_digest": sha256_json(identity["requested_initial_state"]),
            "initial_state": reset_info,
            "applied_before_step": 0,
            "randomized_fields": [],
        },
        "termination": {
            "declared": True,
            "signal": "illegal_contact",
            "condition": "illegal_contact > 0.5 (base contact force above 1 N)",
            "on_termination": "no_auto_reset (the rollout continues so trajectories stay aligned)",
        },
        "arrays": {
            k: {
                "shape": list(a.shape),
                "dtype": a.dtype.name,
                "unit": UNITS[k],
                "frame": FRAMES[k],
                "semantics": f"captured post_env_step; joint order {recorded_names}",
            }
            for k, a in arrays.items()
        },
    }
    bundle = out_dir / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bundle / "trajectories.npz", **arrays, __actions__=actions_out)
    meta = {
        "scenario_name": contract["task"]["id"],
        "backend": ident["backend"],
        "device": "cuda:0",
        "seed": job["seed"],
        "physics_dt": timing["physics_dt"],
        "horizon": S,
        "num_envs": n,
        "capture_contract": contract,
    }
    (bundle / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    root_hash = finalize_v1(bundle, "completed")
    return {
        "bundle": str(bundle),
        "bundle_sha256": root_hash,
        "finite": finite,
        "actions": actions_out,
        "physics": ident,
        "timing": timing,
    }


def run_closed_loop(job, policy, orders) -> dict[str, Any]:
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
        jmap = JointMap(list(robot.joint_names), orders, fault)
        ret = torch.zeros(n, device=u.device)
        falls = torch.zeros(n, device=u.device)
        tilted = torch.zeros(n, device=u.device)
        track_err, finite = [], True
        cmd = torch.as_tensor(sched, device=u.device)
        for _ in range(CLOSED_LOOP_STEPS):
            with torch.inference_mode():
                po = policy_obs(obs["policy"], fault, slices, jmap)
                a = policy(plugin_call(fault, "transform_obs", po, u, default=po))
            a_native = jmap.action_to_native(a)
            obs, rew, term, trunc, _ = env.step(plugin_call(fault, "transform_action", a_native, u,
                                                            default=a_native))
            finite = finite and bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all())
            ret += rew
            falls += (term & ~trunc).float()
            tilted += (-t(robot.data.projected_gravity_b)[:, 2] < 0.5).float()
            v = t(robot.data.root_lin_vel_b)[:, :2]
            track_err.append(torch.linalg.norm(v - cmd[:, :2], dim=-1).mean().item())
        result = {
            "mean_return_per_env": float(ret.mean().item()),
            "fall_terminations_per_env": float(falls.mean().item()),
            "tilted_fraction": float((tilted / CLOSED_LOOP_STEPS).mean().item()),
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
    load_fault_plugin(plan.get("fault_module"))
    out_root = Path(args.out)
    policy = load_policy(plan["policy"])
    orders = json.loads((Path(plan["policy"]).parent / "joint_orders.json").read_text())
    replay_cache: dict[str, np.ndarray] = {}
    for job in plan["jobs"]:
        jdir = out_root / job["capture_id"]
        if (jdir / "bundle" / "COMPLETE").exists() and (jdir / "result.json").exists():
            if job.get("is_reference"):
                replay_cache[job["replay_ref"]] = np.load(jdir / "reference_actions.npy")
            print(f"[corpus] skip existing {job['capture_id']}", flush=True)
            continue
        jdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        result: dict[str, Any] = {"capture_id": job["capture_id"], "native": {}}
        try:
            ref_key = job.get("replay_ref")
            replay = None if job.get("is_reference") else replay_cache.get(ref_key)
            if not job.get("is_reference") and replay is None:
                raise RuntimeError(f"reference stream {ref_key} not available")
            ol = run_open_loop(job, jdir, replay, policy, orders)
            if job.get("is_reference"):
                np.save(jdir / "reference_actions.npy", ol["actions"])
                replay_cache[ref_key] = ol["actions"]
            cl = run_closed_loop(job, policy, orders)
            result.update(
                {
                    "bundle": ol["bundle"],
                    "bundle_sha256": ol["bundle_sha256"],
                    "open_loop_physics": ol["physics"],
                    "timing": ol["timing"],
                    "closed_loop": {k: v for k, v in cl.items() if k != "physics"},
                    "closed_loop_physics": cl["physics"],
                    "native": {
                        "exception": None,
                        "finite_open_loop": ol["finite"],
                        "finite_closed_loop": cl["finite"],
                        "native_pass": bool(ol["finite"] and cl["finite"]),
                    },
                }
            )
        except Exception as exc:
            result["native"] = {
                "exception": repr(exc)[:2000],
                "native_pass": False,
                "traceback": traceback.format_exc()[-4000:],
            }
            print(f"[corpus] job {job['capture_id']} raised: {exc!r}", file=sys.stderr, flush=True)
        result["wall_s"] = round(time.time() - t0, 1)
        (jdir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(
            f"[corpus] done {job['capture_id']} in {result['wall_s']}s native_pass="
            f"{result['native'].get('native_pass')}",
            flush=True,
        )
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
