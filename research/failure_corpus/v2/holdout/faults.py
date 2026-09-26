"""Sealed holdout fault families for the v2 physics-backend migration corpus.

Each family models a bug (or a legitimate, harmless difference) that a team could
introduce while porting a PhysX-trained Isaac Lab locomotion setup to the
Newton / MuJoCo-Warp backend. Families are applied to the candidate run only.

Conventions verified against the Isaac Lab 10.2.0 source tree:
  * robot data fields are ``ProxyArray`` objects; ``.torch`` gives a torch view.
  * quaternions are stored (x, y, z, w) (``isaaclab.utils.math.quat_apply``).
  * the policy observation is [lin_vel(3), ang_vel(3), gravity(3), cmd(3),
    joint_pos(nj), joint_vel(nj), last_action(nj)].

No Isaac Lab / torch import happens at module import time; everything is lazy.
Hooks never modify their inputs in place (inputs may be inference tensors).
"""

from __future__ import annotations

import re
import weakref

# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------

_STATE_WEAK: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_STATE_ID: dict = {}


def _state(u, family: str) -> dict:
    """Per-env, per-family mutable state (a fresh env gets fresh state)."""
    try:
        per_env = _STATE_WEAK.setdefault(u, {})
    except TypeError:
        # unhashable / non-weakrefable env: key by id, but guard against id reuse
        entry = _STATE_ID.get(id(u))
        if entry is None or entry[0] is not u:
            entry = (u, {})
            _STATE_ID[id(u)] = entry
        per_env = entry[1]
    return per_env.setdefault(family, {})


def _T(x):
    """ProxyArray / warp array / tensor -> torch tensor (zero-copy where possible)."""
    import torch

    if isinstance(x, torch.Tensor):
        return x
    tt = getattr(x, "torch", None)
    if tt is not None:
        return tt
    import warp as wp  # pragma: no cover - only reached with raw warp arrays

    return wp.to_torch(x)


def _robot(u):
    return u.scene["robot"]


def _layout(obs):
    """Slices of the flat-velocity policy observation, or None if the size is unexpected."""
    d = int(obs.shape[-1]) - 12
    if d <= 0 or d % 3:
        return None
    nj = d // 3
    return {
        "lin": slice(0, 3),
        "ang": slice(3, 6),
        "grav": slice(6, 9),
        "cmd": slice(9, 12),
        "jpos": slice(12, 12 + nj),
        "jvel": slice(12 + nj, 12 + 2 * nj),
        "act": slice(12 + 2 * nj, 12 + 3 * nj),
        "nj": nj,
    }


def _replace(obs, sl: slice, new):
    """Return a copy of ``obs`` with columns ``sl`` replaced (no in-place writes)."""
    import torch

    return torch.cat([obs[:, : sl.start], new.to(obs.dtype), obs[:, sl.stop :]], dim=1)


def _quat_apply(q, v):
    """Rotate v by q, q in (x, y, z, w). Same formula as isaaclab.utils.math.quat_apply."""
    xyz = q[:, :3]
    t = 2.0 * xyz.cross(v, dim=-1)
    return v + q[:, 3:4] * t + xyz.cross(t, dim=-1)


def _quat_apply_inverse(q, v):
    """Rotate v by q^-1, q in (x, y, z, w). Same formula as isaaclab.utils.math.quat_apply_inverse."""
    xyz = q[:, :3]
    t = 2.0 * xyz.cross(v, dim=-1)
    return v - q[:, 3:4] * t + xyz.cross(t, dim=-1)


def _root_quat_xyzw(u):
    return _T(_robot(u).data.root_link_quat_w).float()


def _newton_cfgs(cfg) -> list:
    """NewtonCfg objects reachable from cfg.sim.physics (resolved or still a preset)."""
    phys = getattr(getattr(cfg, "sim", None), "physics", None)
    if phys is None:
        return []
    if hasattr(phys, "solver_cfg"):
        return [phys]
    out = []
    for name in ("newton_mjwarp",):
        c = getattr(phys, name, None)
        if c is not None and hasattr(c, "solver_cfg"):
            out.append(c)
    return out


def _action_scale(u, params) -> float:
    try:
        s = u.cfg.actions.joint_pos.scale
        if isinstance(s, (int, float)):
            return float(s)
    except Exception:
        pass
    return float(params.get("action_scale", 0.5))


def _episode_start_mask(u, n, device):
    """True for envs whose episode just (re)started, so there is no previous sample."""
    import torch

    buf = getattr(u, "episode_length_buf", None)
    if buf is None:
        return torch.zeros(n, dtype=torch.bool, device=device)
    return (_T(buf) == 0).to(device)


# ----------------------------------------------------------------------------------------
# joint_body_ordering
# ----------------------------------------------------------------------------------------


class ResetJointOffsetDofIndexShift:
    """Reset joint offsets are scattered with a qpos-style index into a qvel-style array.

    For a floating-base robot the generalized position has 7 root coordinates and the
    generalized velocity has 6, so a port that computes joint slots from one layout and
    writes into the other lands every joint value ``shift`` slots away. Here the per-joint
    reset offsets (requested pose minus default pose) are rolled by ``shift`` joints in
    native order before being written. The magnitudes are unchanged and look like normal
    reset randomization.
    """

    def transform_reset(self, state, u, params):
        import torch

        k = int(params.get("shift", 1))
        jpos = state["joint_pos"]
        default = _T(_robot(u).data.default_joint_pos).to(jpos.dtype)
        off = jpos - default
        new = default + torch.roll(off, shifts=k, dims=1)
        return {"joint_pos": new, "joint_vel": state["joint_vel"], "root_vel": state["root_vel"]}


# ----------------------------------------------------------------------------------------
# obs_action_ordering
# ----------------------------------------------------------------------------------------


class ProjectedGravityQuatWxyzMisread:
    """Projected gravity computed by a helper that still assumes (w, x, y, z) quaternions.

    The candidate data path returns (x, y, z, w). The ported helper reads component 0 as
    the scalar, so it rotates gravity by the wrong quaternion. At zero yaw the misread
    quaternion is a 180 degree turn about z and the result is still (0, 0, -1), so the
    bug is invisible in a standing robot facing +x and grows with heading.
    """

    def transform_obs(self, obs, u, params):
        import torch

        lay = _layout(obs)
        if lay is None:
            return obs
        q = _root_quat_xyzw(u).to(obs.device)
        # believed (w, x, y, z) = stored (x, y, z, w); re-express that belief as xyzw
        q_believed = q[:, [1, 2, 3, 0]]
        g = torch.tensor([0.0, 0.0, -1.0], device=obs.device).expand(q.shape[0], 3)
        g_b = _quat_apply_inverse(q_believed, g)
        return _replace(obs, lay["grav"], g_b)


class BaseLinVelWorldFrame:
    """Base linear velocity observation fed in the world frame instead of the base frame.

    MuJoCo's free-joint qvel stores the linear part in the world frame (the angular part is
    body-local). A port that reads the root linear velocity MuJoCo-style skips the rotation
    into the base frame. The values stay plausible (same magnitude, same vertical part);
    only the heading-dependent x/y mixing is wrong.
    """

    def transform_obs(self, obs, u, params):
        lay = _layout(obs)
        if lay is None:
            return obs
        q = _root_quat_xyzw(u).to(obs.device)
        v_b = obs[:, lay["lin"]].float()
        v_w = _quat_apply(q, v_b)
        return _replace(obs, lay["lin"], v_w)


# ----------------------------------------------------------------------------------------
# timestep_decimation
# ----------------------------------------------------------------------------------------


class ActionOneStepTransportDelay:
    """The candidate loop applies the action computed at the previous control step.

    A double-buffered command path (policy writes buffer A while the simulator consumes
    buffer B) adds one control period (20 ms at 50 Hz) of transport latency. The first
    step applies a zero action (default pose target).
    """

    def transform_action(self, a, u, params):
        import torch

        st = _state(u, "action_one_step_transport_delay")
        k = int(params.get("delay_steps", 1))
        buf = st.get("buf")
        if buf is None or len(buf) != k or buf[0].shape != a.shape:
            buf = [torch.zeros_like(a) for _ in range(k)]
        out = buf[0]
        st["buf"] = buf[1:] + [a.detach().clone()]
        return out.to(a.dtype)


class JointVelFiniteDiffPhysicsDt:
    """Joint velocity observation rebuilt by finite differences with the wrong period.

    The candidate data path did not expose joint velocities to the observation builder,
    so the port differentiates successive joint-position observations. It divides by the
    physics step (sim dt) instead of the control period (sim dt * decimation), inflating
    the joint-velocity observation by the decimation factor, and it inherits position
    noise divided by dt.
    """

    def transform_obs(self, obs, u, params):
        import torch

        lay = _layout(obs)
        if lay is None:
            return obs
        st = _state(u, "joint_vel_finite_diff_physics_dt")
        dt = float(params.get("dt", getattr(u, "physics_dt", 0.005)))
        jpos = obs[:, lay["jpos"]].detach().clone()
        prev = st.get("prev")
        if prev is None or prev.shape != jpos.shape:
            vel = torch.zeros_like(jpos)
        else:
            vel = (jpos - prev.to(jpos.device)) / dt
            fresh = _episode_start_mask(u, jpos.shape[0], jpos.device)
            vel = torch.where(fresh[:, None], torch.zeros_like(vel), vel)
        st["prev"] = jpos
        return _replace(obs, lay["jvel"], vel)


# ----------------------------------------------------------------------------------------
# reset_randomization
# ----------------------------------------------------------------------------------------


class ResetRootTwistAngularLinearOrder:
    """Reset root velocity assembled in Warp's (angular, linear) order.

    Newton's public spatial vectors are (linear, angular); Warp's native spatial_vector is
    (angular, linear). The reset path packs the requested twist in the Warp order, so the
    requested linear velocity is applied as an angular rate and vice versa.
    """

    def transform_reset(self, state, u, params):
        import torch

        rv = state["root_vel"]
        new = torch.cat([rv[:, 3:6], rv[:, 0:3]], dim=1)
        return {"joint_pos": state["joint_pos"], "joint_vel": state["joint_vel"], "root_vel": new}


class SpawnHeightFromAssetKeyframe:
    """Root spawn height taken from the converted asset's home keyframe, not the task's init_state.

    The candidate's asset conversion carried the vendor 'home' keyframe base height, which
    sits higher than the training init_state. Every reset starts with the robot above its
    standing height and a short drop, in both the open-loop capture and closed-loop resets.
    """

    def cfg(self, env_cfg, platform, params):
        dz = float(params["dz"])
        init = env_cfg.scene.robot.init_state
        x, y, z = tuple(init.pos)
        init.pos = (x, y, z + dz)


# ----------------------------------------------------------------------------------------
# actuator_solver_config
# ----------------------------------------------------------------------------------------


class ActuatorModelSubstitution:
    """The candidate uses a different actuator model class than the one the policy was trained with.

    A port that cannot reproduce the training actuator on the new backend swaps it for the
    closest available model with the same nominal gains: the Go2 DC-motor model (with its
    torque-speed saturation) becomes a solver-side implicit PD drive, the ANYmal-D LSTM
    actuator network becomes an analytic DC motor, and the humanoid implicit leg drives
    become explicit PD torque sources evaluated at the physics rate.
    """

    def cfg(self, env_cfg, platform, params):
        from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg

        acts = env_cfg.scene.robot.actuators
        for group, spec in params["groups"].items():
            old = acts.get(group)
            if old is None:
                continue
            common = dict(joint_names_expr=old.joint_names_expr, armature=old.armature, friction=old.friction)
            to = spec["to"]
            if to == "implicit":
                lim = old.effort_limit if old.effort_limit is not None else old.effort_limit_sim
                acts[group] = ImplicitActuatorCfg(
                    stiffness=old.stiffness, damping=old.damping, effort_limit_sim=lim, **common
                )
            elif to == "dc_motor":
                acts[group] = DCMotorCfg(
                    stiffness=spec["stiffness"],
                    damping=spec["damping"],
                    saturation_effort=spec["saturation_effort"],
                    effort_limit=spec["effort_limit"],
                    velocity_limit=spec["velocity_limit"],
                    **common,
                )
            elif to == "ideal_pd":
                lim = old.effort_limit_sim if old.effort_limit_sim is not None else old.effort_limit
                acts[group] = IdealPDActuatorCfg(
                    stiffness=old.stiffness, damping=old.damping, effort_limit=lim, **common
                )


# ----------------------------------------------------------------------------------------
# ineffective_stale_config
# ----------------------------------------------------------------------------------------


class MjwarpPresetOverridesLost:
    """The tuned MJWarp solver preset is silently replaced by MJWarpSolverCfg defaults.

    A config refactor rebuilds the solver config from the class and copies only the
    buffer sizes (njmax / nconmax) and the contact-pipeline switch, so the per-robot
    tuning (implicitfast integrator, elliptic cone / impratio for ANYmal-D) is lost and
    the defaults (Euler integrator, pyramidal cone, impratio 1) apply. Nothing errors.
    """

    def cfg(self, env_cfg, platform, params):
        for nc in _newton_cfgs(env_cfg):
            old = nc.solver_cfg
            if "MJWarp" not in type(old).__name__:
                continue
            nc.solver_cfg = type(old)(
                njmax=old.njmax, nconmax=old.nconmax, use_mujoco_contacts=old.use_mujoco_contacts
            )


class FrictionRandomizationNoop:
    """The robot-material event does not reach the candidate's shapes.

    The training config sets robot shape friction through a startup material event. On the
    candidate the event is registered against the wrong backend path and never applies, so
    robot shapes keep the importer default friction instead of the configured value.
    Emulated by removing the event from the candidate config.
    """

    def cfg(self, env_cfg, platform, params):
        if getattr(env_cfg.events, "physics_material", None) is not None:
            env_cfg.events.physics_material = None


# ----------------------------------------------------------------------------------------
# checkpoint_schema
# ----------------------------------------------------------------------------------------


class LeggedGymObsScalesApplied:
    """Observation scaling from a legged_gym-style deploy config applied to an Isaac Lab policy.

    Isaac Lab velocity tasks feed unscaled observations. The candidate's deploy wrapper was
    reused from a legged_gym lineage that multiplies lin_vel by 2, ang_vel by 0.25, the
    command by (2, 2, 0.25) and dof_vel by 0.05 before the policy sees them.
    """

    def transform_obs(self, obs, u, params):
        import torch

        lay = _layout(obs)
        if lay is None:
            return obs
        s = params
        scale = torch.ones(obs.shape[-1], device=obs.device, dtype=obs.dtype)
        scale[lay["lin"]] = float(s["lin_vel"])
        scale[lay["ang"]] = float(s["ang_vel"])
        scale[lay["cmd"]] = torch.tensor(s["commands"], device=obs.device, dtype=obs.dtype)
        scale[lay["jpos"]] = float(s["dof_pos"])
        scale[lay["jvel"]] = float(s["dof_vel"])
        return obs * scale


class StaleDefaultPoseActionOffset:
    """Action offset built from a stale default pose.

    The deploy side adds its own ``default_angles`` to ``scale * action``. Those angles
    came from an older asset keyframe that differs from the training init pose on a few
    joints, so the commanded targets are shifted by (stale - training) on those joints.
    Emulated in raw-action space as a constant offset of delta / action_scale.
    """

    def transform_action(self, a, u, params):
        import torch

        st = _state(u, "stale_default_pose_action_offset")
        off = st.get("off")
        if off is None or off.shape[-1] != a.shape[-1]:
            names = list(_robot(u).joint_names)
            scale = _action_scale(u, params)
            vals = [0.0] * len(names)
            for pattern, delta in params["offsets"]:
                rx = re.compile(pattern)
                for i, n in enumerate(names):
                    if rx.fullmatch(n):
                        vals[i] = float(delta) / scale
            off = torch.tensor(vals, dtype=a.dtype, device=a.device)
            st["off"] = off
        return a + off.to(a.device, a.dtype)


# ----------------------------------------------------------------------------------------
# termination_metric
# ----------------------------------------------------------------------------------------


class ContactThresholdImpulseUnits:
    """Illegal-contact threshold re-derived for impulse units but compared against force.

    The port treated the candidate's contact reading as a per-substep impulse (N*s) and
    converted the 1 N force threshold by dividing by the physics dt, while the sensor
    actually reports force. The base-contact threshold becomes threshold / dt (200 N at
    dt = 0.005), so many base impacts no longer end the episode and the fall count drops.
    """

    def cfg(self, env_cfg, platform, params):
        term = getattr(env_cfg.terminations, "base_contact", None)
        if term is None:
            return
        dt = float(params.get("dt", env_cfg.sim.dt))
        thr = float(term.params.get("threshold", 1.0))
        term.params["threshold"] = thr / dt


class TimeoutCountedAsTermination:
    """Episode time limit flagged as a termination instead of a truncation.

    The candidate's termination config loses the ``time_out`` flag, so reaching the
    episode length is reported as ``terminated`` rather than ``truncated``. Physics is
    unchanged; every survivor that reaches the time limit is counted as a failure.
    """

    def cfg(self, env_cfg, platform, params):
        term = getattr(env_cfg.terminations, "time_out", None)
        if term is not None:
            term.time_out = False


# ----------------------------------------------------------------------------------------
# benign
# ----------------------------------------------------------------------------------------


class MjwarpSolverIterationHeadroom:
    """Benign: more solver / line-search iteration headroom. The solver stops at its tolerance first."""

    def cfg(self, env_cfg, platform, params):
        for nc in _newton_cfgs(env_cfg):
            sc = nc.solver_cfg
            if "MJWarp" not in type(sc).__name__:
                continue
            sc.iterations = int(sc.iterations * float(params["factor"]))
            sc.ls_iterations = int(sc.ls_iterations * float(params["factor"]))


class EnvSpacingWidened:
    """Benign: wider spacing between independent envs on a flat plane (no inter-env interaction)."""

    def cfg(self, env_cfg, platform, params):
        env_cfg.scene.env_spacing = float(params["env_spacing"])


class ActionClipWide:
    """Benign: rsl_rl-style action clip at +/-100, far outside the policy's output range."""

    def transform_action(self, a, u, params):
        c = float(params["clip"])
        return a.clamp(-c, c)


class DebugVisualizationOff:
    """Benign: command debug markers disabled (headless run, purely visual)."""

    def cfg(self, env_cfg, platform, params):
        bv = getattr(env_cfg.commands, "base_velocity", None)
        if bv is not None and hasattr(bv, "debug_vis"):
            bv.debug_vis = False


FAMILIES: dict[str, object] = {
    # defects
    "reset_joint_offset_dof_index_shift": ResetJointOffsetDofIndexShift(),
    "projected_gravity_quat_wxyz_misread": ProjectedGravityQuatWxyzMisread(),
    "base_lin_vel_world_frame": BaseLinVelWorldFrame(),
    "action_one_step_transport_delay": ActionOneStepTransportDelay(),
    "joint_vel_finite_diff_physics_dt": JointVelFiniteDiffPhysicsDt(),
    "reset_root_twist_angular_linear_order": ResetRootTwistAngularLinearOrder(),
    "spawn_height_from_asset_keyframe": SpawnHeightFromAssetKeyframe(),
    "actuator_model_substitution": ActuatorModelSubstitution(),
    "mjwarp_preset_overrides_lost": MjwarpPresetOverridesLost(),
    "friction_randomization_noop": FrictionRandomizationNoop(),
    "legged_gym_obs_scales_applied": LeggedGymObsScalesApplied(),
    "stale_default_pose_action_offset": StaleDefaultPoseActionOffset(),
    "contact_threshold_impulse_units": ContactThresholdImpulseUnits(),
    "timeout_counted_as_termination": TimeoutCountedAsTermination(),
    # benign
    "mjwarp_solver_iteration_headroom": MjwarpSolverIterationHeadroom(),
    "env_spacing_widened": EnvSpacingWidened(),
    "action_clip_wide": ActionClipWide(),
    "debug_visualization_off": DebugVisualizationOff(),
}
