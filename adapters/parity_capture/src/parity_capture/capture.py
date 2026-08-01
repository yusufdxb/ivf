# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The real Isaac Lab capture: boot Kit, build cart-pole, roll out, write ``trajectory_bundle/v1``.

Everything in this module runs *after* the simulator app has been launched, because
Isaac Lab's imports require a live Kit application. That ordering constraint is why the
Isaac Lab imports are inside functions rather than at module scope, and it is also why
this package exists at all: IVF core must stay importable on a laptop.

The cart-pole construction mirrors the proven ``isaaclab_contrib.parity`` scenario
(stock ``CARTPOLE_CFG``, pinned actuator gains, pole released off-centre) so this adapter
inherits a configuration that has already produced real cross-backend evidence, rather
than inventing an unvalidated one. What it adds is an explicit, recorded simulator-level
reset perturbation modeled after a failure observed while developing the parity harness.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .spec import CaptureSpec

# Pinned so an upstream retune of the stock asset cannot silently change the experiment.
CART_DAMPING = 10.0
POLE_DAMPING = 0.0
ENV_SPACING = 3.0
PHYSICS_DT = 1.0 / 120.0


@dataclass
class CaptureResult:
    """What a completed capture produced."""

    bundle_path: Path
    captured_steps: int
    declared_steps: int
    run_status: str
    checksums: dict[str, str]


def _software_versions() -> dict[str, str]:
    """Collect the version identity of everything that can change a trajectory."""
    import importlib.metadata as md

    out: dict[str, str] = {"python": platform.python_version()}
    for pkg in (
        "isaacsim", "isaaclab", "isaaclab_physx", "isaaclab_newton", "newton", "torch", "numpy",
        "warp-lang",
    ):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            continue
    try:
        import torch

        out["cuda"] = str(torch.version.cuda)
    except Exception:
        pass
    try:
        import omni.kit.app

        out["kit"] = str(omni.kit.app.get_app().get_build_version())
    except Exception:
        pass
    return out


def _hardware_metadata() -> dict[str, Any]:
    """Collect the declared device and driver without requiring an exact public GPU name."""
    import os
    import subprocess

    out: dict[str, Any] = {"platform": platform.platform()}
    try:
        import torch

        out["gpu"] = os.environ.get("IVF_HARDWARE_LABEL") or (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        out["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception:
        out["gpu"] = os.environ.get("IVF_HARDWARE_LABEL", "unavailable")
    driver = os.environ.get("IVF_DRIVER_LABEL")
    if not driver:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                driver = result.stdout.splitlines()[0].strip()
        except Exception:
            driver = None
    out["driver"] = driver or "unavailable"
    return out


def build_cartpole(spec: CaptureSpec):
    """Construct the cart-pole scene. Isaac Lab must already be running."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_assets import CARTPOLE_CFG

    robot_cfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.init_state.joint_pos = {
        "slider_to_cart": 0.0,
        "cart_to_pole": spec.initial_pole_angle,
    }
    robot_cfg.actuators["cart_actuator"].stiffness = 0.0
    robot_cfg.actuators["cart_actuator"].damping = CART_DAMPING * spec.defect.damping_scale
    robot_cfg.actuators["pole_actuator"].stiffness = 0.0
    robot_cfg.actuators["pole_actuator"].damping = POLE_DAMPING

    @configclass
    class CartpoleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = robot_cfg

    return InteractiveScene(CartpoleSceneCfg(num_envs=spec.num_envs, env_spacing=ENV_SPACING))


def apply_reset(scene, spec: CaptureSpec) -> dict[str, Any]:
    """Write the declared initial state onto the articulation.

    The configuration asks for both position and velocity to be written. When the
    test-only ``drop_reset_velocity`` perturbation is active, a zero velocity vector is
    written instead. The capture metadata records requested and applied values; the
    oracles establish the resulting trajectory and event consequences independently.
    """
    import torch

    robot = scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    pole_index = robot.joint_names.index("cart_to_pole")
    # Spread the initial angle across environments from a seeded generator. Without it
    # every clone is the same trajectory, so 16 environments carry one environment's
    # worth of information and any paired statistic computed over them is a fiction.
    rng = np.random.default_rng(spec.seed)
    offsets = rng.uniform(
        -spec.initial_pole_angle_spread, spec.initial_pole_angle_spread, size=spec.num_envs
    )
    angles = torch.as_tensor(
        spec.initial_pole_angle + offsets, dtype=joint_pos.dtype, device=joint_pos.device
    )
    joint_pos[:, pole_index] = angles
    joint_vel[:, pole_index] = spec.initial_pole_velocity

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    applied_joint_vel = torch.zeros_like(joint_vel) if spec.defect.drop_reset_velocity else joint_vel
    robot.write_joint_state_to_sim(joint_pos, applied_joint_vel)
    scene.reset()
    return {
        "joint_names": list(robot.joint_names),
        "requested_joint_pos": joint_pos.detach().cpu().tolist(),
        "requested_joint_vel": joint_vel.detach().cpu().tolist(),
        "applied_joint_pos": joint_pos.detach().cpu().tolist(),
        "applied_joint_vel": applied_joint_vel.detach().cpu().tolist(),
    }


def roll_out(sim, scene, spec: CaptureSpec) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Step the simulation and capture the declared signals.

    Capture happens after the physics step and after the scene update, so every array
    describes the same post-step instant. The hook name is recorded in the contract, so
    a consumer never has to infer whether a sample is pre- or post-step.
    """
    import torch

    robot = scene["robot"]
    n_joints = robot.data.joint_pos.shape[1]
    joint_pos = np.zeros((spec.steps, spec.num_envs, n_joints), dtype=np.float32)
    joint_vel = np.zeros_like(joint_pos)
    root_pos = np.zeros((spec.steps, spec.num_envs, 3), dtype=np.float32)
    root_quat = np.zeros((spec.steps, spec.num_envs, 4), dtype=np.float32)
    actions = np.zeros((spec.steps, spec.num_envs, n_joints), dtype=np.float32)

    captured = 0
    for step in range(spec.steps):
        # Passive workload: the commanded effort is identically zero every step, which
        # is written explicitly rather than skipped so the action stream is a recorded
        # fact rather than an assumption.
        effort = torch.zeros_like(robot.data.joint_pos)
        robot.set_joint_effort_target(effort)
        actions[step] = effort.detach().cpu().numpy()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        joint_pos[step] = robot.data.joint_pos.detach().cpu().numpy()
        joint_vel[step] = robot.data.joint_vel.detach().cpu().numpy()
        root_pos[step] = robot.data.root_link_pos_w.detach().cpu().numpy()
        root_quat[step] = robot.data.root_link_quat_w.detach().cpu().numpy()
        captured = step + 1

    pole_index = robot.joint_names.index("cart_to_pole")
    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_link_pos_w": root_pos,
        "root_link_quat_w": root_quat,
        "pole_angle": joint_pos[:, :, pole_index : pole_index + 1].copy(),
        "pole_velocity": joint_vel[:, :, pole_index : pole_index + 1].copy(),
        "abs_pole_angle": np.abs(joint_pos[:, :, pole_index : pole_index + 1]).copy(),
    }
    return arrays, actions, captured


def build_contract(spec: CaptureSpec, arrays: dict[str, np.ndarray], *, captured_steps: int,
                   joint_names: list[str], solver_settings: dict[str, Any],
                   initial_state: dict[str, Any], run_status: str) -> dict[str, Any]:
    """Assemble the ``trajectory_bundle/v1`` capture contract for this run."""
    import hashlib

    config_digest = hashlib.sha256(
        json.dumps(spec.config_identity(), sort_keys=True).encode()
    ).hexdigest()
    initial_state_digest = hashlib.sha256(
        json.dumps(
            {
                "pole_angle": spec.initial_pole_angle,
                "pole_angle_spread": spec.initial_pole_angle_spread,
                "pole_velocity": spec.initial_pole_velocity,
                "seed": spec.seed,
                "num_envs": spec.num_envs,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    units = {
        "joint_pos": "rad and m (revolute and prismatic joints in one array)",
        "joint_vel": "rad/s and m/s",
        "root_link_pos_w": "m",
        "root_link_quat_w": "dimensionless",
        "pole_angle": "rad",
        "pole_velocity": "rad/s",
        "abs_pole_angle": "rad",
    }
    frames = {
        "joint_pos": "joint", "joint_vel": "joint",
        "root_link_pos_w": "world", "root_link_quat_w": "world",
        "pole_angle": "joint", "pole_velocity": "joint", "abs_pole_angle": "joint",
    }
    return {
        "schema": "trajectory_bundle/v1",
        "run_status": run_status,
        "declared_steps": spec.steps,
        "captured_steps": captured_steps,
        "task": {
            "id": spec.task,
            "variant": "stock CARTPOLE_CFG, pinned actuator gains, passive",
            "config_digest_sha256": config_digest,
            "joint_names": joint_names,
        },
        "backend": {
            "id": spec.backend,
            "solver_settings": solver_settings,
            "features": ["articulation", "revolute_joint", "prismatic_joint"],
        },
        "software": _software_versions(),
        "hardware": _hardware_metadata(),
        "seed": {
            "value": spec.seed,
            "env_ids": list(range(spec.num_envs)),
            "env_order": "InteractiveScene clone index, ascending",
        },
        "timing": {
            "physics_dt": PHYSICS_DT,
            "control_dt": PHYSICS_DT,
            "decimation": 1,
            "action_applied": "before_physics_step",
            "capture_hook": "post_step_post_update",
        },
        "frames": {
            "convention": "world_z_up_right_handed",
            "length_unit": "m",
            "angle_unit": "rad",
            "note": "root arrays are world-frame and include the per-environment origin",
        },
        "quaternion": {"layout": "wxyz", "scalar_first": True, "normalized": True,
                       "hemisphere": "unconstrained"},
        "reset": {
            "semantics": "writes_pose_and_velocity",
            "initial_state_digest": initial_state_digest,
            "initial_state": initial_state,
            "applied_before_step": 0,
            "randomized_fields": (
            ["cart_to_pole initial position (uniform, seeded)"]
            if spec.initial_pole_angle_spread else []
        ),
        },
        "termination": {
            "declared": True,
            "signal": "abs_pole_angle",
            "condition": f"abs_pole_angle > {spec.termination_angle}",
            "on_termination": "no_auto_reset (the rollout continues so trajectories stay aligned)",
        },
        "arrays": {
            name: {
                "shape": list(arr.shape),
                "dtype": arr.dtype.name,
                "unit": units.get(name, "unknown"),
                "frame": frames.get(name, "unknown"),
                "semantics": f"captured at post_step_post_update, joint order {joint_names}",
            }
            for name, arr in arrays.items()
        },
        "defect_injected": spec.defect.to_jsonable(),
    }


def write_bundle(output: Path, spec: CaptureSpec, arrays: dict[str, np.ndarray],
                 actions: np.ndarray, contract: dict[str, Any], *, run_status: str) -> CaptureResult:
    """Write the payload, then the checksums, then the completion marker.

    The order is the contract. An interruption at any point leaves a bundle that IVF
    refuses with ``IVF-BUNDLE-INCOMPLETE`` rather than one that reads as a short but
    successful run.
    """
    from ivf.bundle import finalize_v1

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trajectories.npz", **arrays, __actions__=actions)
    metadata = {
        "scenario_name": spec.task,
        "backend": spec.backend,
        "device": spec.device,
        "seed": spec.seed,
        "physics_dt": PHYSICS_DT,
        "horizon": contract["captured_steps"],
        "num_envs": spec.num_envs,
        "capture_spec": spec.to_jsonable(),
        "capture_contract": contract,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = finalize_v1(output, run_status=run_status)
    return CaptureResult(
        bundle_path=output, captured_steps=contract["captured_steps"],
        declared_steps=spec.steps, run_status=run_status, checksums=checksums,
    )


def run_capture(spec: CaptureSpec, output: Path) -> CaptureResult:
    """Boot Isaac Lab, run the workload, and write a ``trajectory_bundle/v1``.

    Must be called only after :func:`parity_capture.cli.launch_app` has started Kit.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass  # noqa: F401 - ensures the config system is live

    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=spec.device)
    if spec.backend == "newton":
        try:
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

            sim_cfg.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
        except Exception as exc:  # pragma: no cover - depends on the installed backend
            raise RuntimeError(f"backend 'newton' requested but not constructible: {exc}") from exc

    sim = sim_utils.SimulationContext(sim_cfg)
    scene = build_cartpole(spec)
    sim.reset()

    solver_settings: dict[str, Any] = {}
    try:
        physics_cfg = sim.cfg.physics
        solver_settings = json.loads(
            json.dumps({"manager": type(physics_cfg).__name__, **physics_cfg.to_dict()},
                       default=str, sort_keys=True)
        )
    except Exception as exc:  # provenance is best effort, but its absence is recorded
        solver_settings = {"unavailable": str(exc)}

    initial_state = apply_reset(scene, spec)
    arrays, actions, captured = roll_out(sim, scene, spec)

    run_status = "completed" if captured == spec.steps else "partial"
    contract = build_contract(
        spec, arrays, captured_steps=captured,
        joint_names=list(scene["robot"].joint_names),
        solver_settings=solver_settings, initial_state=initial_state, run_status=run_status,
    )
    result = write_bundle(Path(output), spec, arrays, actions, contract, run_status=run_status)
    print(f"[parity-capture] captured {captured}/{spec.steps} steps, status={run_status}",
          file=sys.stderr, flush=True)
    return result
