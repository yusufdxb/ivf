"""v4 holdout fault plugin: PhysX -> Newton/MuJoCo-Warp migration faults and benign controls.

Every family is applied to the candidate (Newton ``newton_mjwarp``) run only. Each hook receives the
case ``params`` dict as its last argument. The module imports without Isaac Lab; Isaac Lab symbols are
imported lazily inside the ``cfg`` hooks that need them.

Defect families (``defect: true``)
    mirrored_actuator_gear_sign        joint_body_ordering
    root_velocity_link_point           joint_body_ordering
    command_rotated_by_heading         obs_action_ordering
    joint_vel_prelaunch_snapshot       timestep_decimation
    reset_offset_scale_conflation      reset_randomization
    reset_world_index_shift            reset_randomization
    torque_limit_unclamped             actuator_solver_config
    ctrlrange_target_clamp             actuator_solver_config
    training_noise_stale_cfg           ineffective_stale_config
    bf16_io_policy_export              checkpoint_schema
    contact_termination_inert          termination_metric
    success_threshold_relaxed          termination_metric

Benign families (``defect: false``)
    command_arrows_hidden              visualization only
    viewer_follow_camera               viewport camera only
    diagnostic_obs_group               unused, noise-free observation group
    origin_marker_debug_draw           debug drawing only
    sensor_period_restated             numerically identical re-specification
"""

from __future__ import annotations

import re

import torch

# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def _t(x):
    """Return a torch view of an Isaac Lab data field (ProxyArray, warp array or tensor)."""
    if isinstance(x, torch.Tensor):
        return x
    tv = getattr(x, "torch", None)
    if isinstance(tv, torch.Tensor):
        return tv
    import warp as wp  # noqa: PLC0415

    return wp.to_torch(x)


def _robot(u):
    return u.scene["robot"]


def _num_joints(u) -> int:
    return int(_robot(u).num_joints)


def _slices(u):
    """Column slices of the concatenated policy observation (fixed term order, see PLUGIN_API)."""
    nj = _num_joints(u)
    return {
        "lin": slice(0, 3),
        "ang": slice(3, 6),
        "grav": slice(6, 9),
        "cmd": slice(9, 12),
        "jp": slice(12, 12 + nj),
        "jv": slice(12 + nj, 12 + 2 * nj),
        "act": slice(12 + 2 * nj, 12 + 3 * nj),
        "rest": slice(12 + 3 * nj, None),
    }


def _state(u, key, factory):
    """Per-env-instance mutable state stored on the unwrapped env (deterministic, no globals)."""
    name = "_holdout_v4_" + key
    st = u.__dict__.get(name)
    if st is None:
        st = factory()
        u.__dict__[name] = st
    return st


def _joint_position_term(u):
    am = u.action_manager
    names = list(am.active_terms)
    name = "joint_pos" if "joint_pos" in names else names[0]
    return am.get_term(name)


def _as_matrix(v, like):
    """Broadcast a float / (N, nj) tensor to ``like``'s shape and device."""
    if isinstance(v, torch.Tensor):
        return v.to(device=like.device, dtype=like.dtype).expand_as(like)
    return torch.full_like(like, float(v))


def _joint_ids(u, regex: str) -> list[int]:
    pat = re.compile(regex)
    return [i for i, n in enumerate(_robot(u).joint_names) if pat.fullmatch(n)]


# ----------------------------------------------------------------------------------------
# defect families
# ----------------------------------------------------------------------------------------


class MirroredActuatorGearSign:
    """joint_body_ordering: the converted model drives the left-side abduction joints through an
    actuator whose transmission gear is -1 (mirrored axis convention). A position servo with gear g
    tracks ctrl / g, so those joints track the negated position target. Sensing is unaffected."""

    def transform_action(self, a_native, u, params):
        ids = _state(u, "gear_ids", lambda: _joint_ids(u, params["joint_regex"]))
        if not ids:
            return a_native
        term = _joint_position_term(u)
        a = a_native.clone()
        scale = _as_matrix(term._scale, a)
        offset = _as_matrix(term._offset, a)
        idx = torch.as_tensor(ids, device=a.device, dtype=torch.long)
        s = scale[:, idx]
        s = torch.where(s.abs() < 1e-9, torch.ones_like(s), s)
        # processed target q* = offset + s*a  ->  joint tracks -q*;  solve offset + s*a' = -(offset + s*a)
        a[:, idx] = -a[:, idx] - 2.0 * offset[:, idx] / s
        return a


class RootVelocityLinkPoint:
    """joint_body_ordering: the port reports the base linear velocity of the root link origin
    (actor frame) where the policy was trained on the root centre-of-mass velocity."""

    def transform_obs(self, obs_policy, u, params):
        d = _robot(u).data
        v_com = _t(d.root_com_lin_vel_b).to(obs_policy.dtype)
        v_link = _t(d.root_link_lin_vel_b).to(obs_policy.dtype)
        out = obs_policy.clone()
        out[:, 0:3] = out[:, 0:3] - v_com + v_link
        return out


class CommandRotatedByHeading:
    """obs_action_ordering: the command adapter rotates the planar velocity command by the measured
    base yaw (world-frame command) before it reaches a policy that expects base-frame commands."""

    def transform_obs(self, obs_policy, u, params):
        yaw = _t(_robot(u).data.heading_w).to(obs_policy.dtype)
        c, s = torch.cos(yaw), torch.sin(yaw)
        sl = _slices(u)["cmd"]
        out = obs_policy.clone()
        vx, vy = obs_policy[:, sl.start], obs_policy[:, sl.start + 1]
        out[:, sl.start] = c * vx - s * vy
        out[:, sl.start + 1] = s * vx + c * vy
        return out


class JointVelPrelaunchSnapshot:
    """timestep_decimation: with the decimation loop folded into one captured graph launch, the port
    reads joint velocities from the snapshot taken before the launch, i.e. ``lag_steps`` control
    periods old. Joint positions and all other terms are current."""

    def transform_obs(self, obs_policy, u, params):
        lag = int(params.get("lag_steps", 1))
        sl = _slices(u)["jv"]
        hist = _state(u, "jv_hist", list)
        cur = obs_policy[:, sl].clone()
        hist.append(cur)
        if len(hist) > lag + 1:
            del hist[0]
        out = obs_policy.clone()
        out[:, sl] = hist[0]
        return out


class ResetOffsetScaleConflation:
    """reset_randomization: the Newton reset path conflates ``reset_joints_by_offset`` and
    ``reset_joints_by_scale`` semantics. Open loop: the requested additive joint offset is applied as
    a multiplicative fraction of the default pose (q = q0 * (1 + offset)). Closed loop: the task's
    scale range is rewritten as an offset range by subtracting one (valid only for q0 = 1 rad)."""

    def cfg(self, env_cfg, platform, params):
        ev = getattr(getattr(env_cfg, "events", None), "reset_robot_joints", None)
        if ev is None or getattr(ev.func, "__name__", "") != "reset_joints_by_scale":
            return
        from isaaclab.envs import mdp as base_mdp  # noqa: PLC0415

        lo, hi = ev.params["position_range"]
        ev.func = base_mdp.reset_joints_by_offset
        ev.params["position_range"] = (float(lo) - 1.0, float(hi) - 1.0)

    def transform_reset(self, state, u, params):
        d = _robot(u).data
        q0 = _t(d.default_joint_pos).to(state["joint_pos"].dtype)
        off = state["joint_pos"] - q0
        q = q0 + q0 * off
        lim = _t(d.soft_joint_pos_limits).to(q.dtype)
        q = torch.maximum(torch.minimum(q, lim[..., 1]), lim[..., 0])
        return {"joint_pos": q, "joint_vel": state["joint_vel"], "root_vel": state["root_vel"]}


class ResetWorldIndexShift:
    """reset_randomization: the per-env reset buffer is scattered with an off-by-``shift`` world
    index, so env i receives the sample drawn for env i-shift. Aggregate statistics are unchanged."""

    def transform_reset(self, state, u, params):
        k = int(params.get("shift", 1))
        return {key: torch.roll(val, shifts=k, dims=0) for key, val in state.items()}


class TorqueLimitUnclamped:
    """actuator_solver_config: the converted actuators are emitted with force clamping disabled
    (MJCF forcelimited=false semantics), so neither the solver-side effort limit nor the explicit
    actuator model's torque saturation is enforced."""

    def cfg(self, env_cfg, platform, params):
        big = float(params.get("limit", 1.0e9))
        for act in env_cfg.scene.robot.actuators.values():
            implicit = "Implicit" in type(act).__name__
            act.effort_limit_sim = big
            if not implicit:
                if getattr(act, "effort_limit", None) is not None:
                    act.effort_limit = big
                if getattr(act, "saturation_effort", None) is not None:
                    act.saturation_effort = big


class CtrlrangeTargetClamp:
    """actuator_solver_config: the converted position actuators carry ctrlrange = joint range with
    ctrllimited on, so PD targets outside the joint range are clamped before the servo sees them."""

    def transform_action(self, a_native, u, params):
        term = _joint_position_term(u)
        a = a_native
        scale = _as_matrix(term._scale, a)
        offset = _as_matrix(term._offset, a)
        lim = _t(_robot(u).data.joint_pos_limits).to(a.dtype)
        margin = float(params.get("margin", 0.0))
        target = offset + scale * a
        clamped = torch.maximum(torch.minimum(target, lim[..., 1] - margin), lim[..., 0] + margin)
        s = torch.where(scale.abs() < 1e-9, torch.ones_like(scale), scale)
        return torch.where(scale.abs() < 1e-9, a, (clamped - offset) / s)


class TrainingNoiseStaleCfg:
    """ineffective_stale_config: the evaluation override that disables observation corruption is
    applied to a stale copy of the cfg; the candidate policy sees the training-time uniform noise."""

    def transform_obs(self, obs_policy, u, params):
        g = _state(u, "noise_gen", lambda: torch.Generator(device=obs_policy.device).manual_seed(int(params["seed"])))
        sl = _slices(u)
        out = obs_policy.clone()
        for key in ("lin", "ang", "grav", "jp", "jv"):
            amp = float(params.get(key, 0.0))
            if amp <= 0.0:
                continue
            block = out[:, sl[key]]
            noise = (torch.rand(block.shape, generator=g, device=block.device, dtype=block.dtype) * 2.0 - 1.0) * amp
            out[:, sl[key]] = block + noise
        amp = float(params.get("height_scan", 0.0))
        rest = out[:, sl["rest"]]
        if amp > 0.0 and rest.shape[1] > 0:
            noise = (torch.rand(rest.shape, generator=g, device=rest.device, dtype=rest.dtype) * 2.0 - 1.0) * amp
            out[:, sl["rest"]] = torch.clamp(rest + noise, -1.0, 1.0)
        return out


class Bf16IoPolicyExport:
    """checkpoint_schema: the policy is re-exported for the new runtime with bfloat16 I/O tensors
    (keep_io_types off); observations are rounded to bf16 on the way in and actions on the way out."""

    @staticmethod
    def _q(x, params):
        dt = getattr(torch, params.get("dtype", "bfloat16"))
        return x.to(dt).to(x.dtype)

    def transform_obs(self, obs_policy, u, params):
        return self._q(obs_policy, params)

    def transform_action(self, a_native, u, params):
        return self._q(a_native, params)


class ContactTerminationInert:
    """termination_metric: the illegal-contact termination reads a contact force buffer that the
    backend never fills, so it never fires. The cfg (and any recorded termination spec) is unchanged;
    only the runtime manager entry is swapped."""

    def post_build(self, u, params):
        tm = getattr(u, "termination_manager", None)
        if tm is None:
            return
        for name in list(tm.active_terms):
            term_cfg = tm.get_term_cfg(name)
            fn = term_cfg.func
            if getattr(fn, "__name__", "") != "illegal_contact" or getattr(term_cfg, "time_out", False):
                continue

            def _zero_contact(env, *args, **kwargs):
                return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

            _zero_contact.__name__ = fn.__name__
            _zero_contact.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
            _zero_contact.__module__ = getattr(fn, "__module__", __name__)
            tm.set_term_cfg(name, term_cfg.replace(func=_zero_contact))


class SuccessThresholdRelaxed:
    """termination_metric: the candidate branch carries a backend-tolerance edit to the episode
    success criterion (the biped yaw relaxation leaks to quadrupeds and the XY threshold is loosened),
    inflating Metrics/success_rate without any change to the dynamics."""

    def cfg(self, env_cfg, platform, params):
        cmd = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
        if cmd is None:
            return
        if hasattr(cmd, "vel_xy_success_threshold"):
            cmd.vel_xy_success_threshold = float(params["vel_xy_success_threshold"])
        if hasattr(cmd, "vel_yaw_success_threshold"):
            cmd.vel_yaw_success_threshold = float(params["vel_yaw_success_threshold"])


# ----------------------------------------------------------------------------------------
# benign families (cfg-only; no runtime hooks)
# ----------------------------------------------------------------------------------------


class CommandArrowsHidden:
    """Benign: hide the command/velocity debug arrows for headless batch runs."""

    def cfg(self, env_cfg, platform, params):
        cmd = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
        if cmd is not None:
            cmd.debug_vis = bool(params.get("debug_vis", False))


class ViewerFollowCamera:
    """Benign: re-frame the viewport camera to follow the robot of env 0."""

    def cfg(self, env_cfg, platform, params):
        v = env_cfg.viewer
        v.origin_type = "asset_root"
        v.asset_name = "robot"
        v.env_index = 0
        v.eye = tuple(float(x) for x in params["eye"])
        v.lookat = tuple(float(x) for x in params["lookat"])
        v.resolution = tuple(int(x) for x in params["resolution"])


class DiagnosticObsGroup:
    """Benign: add a noise-free ``diagnostics`` observation group (base height, applied joint effort)
    that no policy consumes. It draws no random numbers and writes nothing to the simulator."""

    def cfg(self, env_cfg, platform, params):
        from isaaclab.envs import mdp as base_mdp  # noqa: PLC0415
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg  # noqa: PLC0415
        # Built as an instance with attributes (ObservationManager iterates the group's __dict__), so
        # no dataclass has to be declared inside this dynamically loaded, unregistered module.
        group = ObservationGroupCfg(concatenate_terms=True, enable_corruption=False)
        group.base_height = ObservationTermCfg(func=base_mdp.base_pos_z)
        group.applied_effort = ObservationTermCfg(func=base_mdp.joint_effort)
        env_cfg.observations.diagnostics = group


class OriginMarkerDebugDraw:
    """Benign: draw terrain/env-origin frame markers and request contact-sensor debug drawing."""

    def cfg(self, env_cfg, platform, params):
        terrain = getattr(env_cfg.scene, "terrain", None)
        if terrain is not None:
            terrain.debug_vis = True
        cs = getattr(env_cfg.scene, "contact_forces", None)
        if cs is not None and hasattr(cs, "debug_vis"):
            cs.debug_vis = True


class SensorPeriodRestated:
    """Benign equivalent refactor: sensor update periods are re-derived from the simulation settings
    (contact sensor = physics dt, height scanner = control period). A value is written only when the
    re-derived number is identical to the one already configured, so the resolved cfg is unchanged."""

    def cfg(self, env_cfg, platform, params):
        dt = env_cfg.sim.dt
        cs = getattr(env_cfg.scene, "contact_forces", None)
        if cs is not None and getattr(cs, "update_period", None) == dt:
            cs.update_period = dt * 1
        hs = getattr(env_cfg.scene, "height_scanner", None)
        period = env_cfg.decimation * dt
        if hs is not None and getattr(hs, "update_period", None) == period:
            hs.update_period = period


FAMILIES: dict[str, object] = {
    # defects
    "mirrored_actuator_gear_sign": MirroredActuatorGearSign(),
    "root_velocity_link_point": RootVelocityLinkPoint(),
    "command_rotated_by_heading": CommandRotatedByHeading(),
    "joint_vel_prelaunch_snapshot": JointVelPrelaunchSnapshot(),
    "reset_offset_scale_conflation": ResetOffsetScaleConflation(),
    "reset_world_index_shift": ResetWorldIndexShift(),
    "torque_limit_unclamped": TorqueLimitUnclamped(),
    "ctrlrange_target_clamp": CtrlrangeTargetClamp(),
    "training_noise_stale_cfg": TrainingNoiseStaleCfg(),
    "bf16_io_policy_export": Bf16IoPolicyExport(),
    "contact_termination_inert": ContactTerminationInert(),
    "success_threshold_relaxed": SuccessThresholdRelaxed(),
    # benign controls
    "command_arrows_hidden": CommandArrowsHidden(),
    "viewer_follow_camera": ViewerFollowCamera(),
    "diagnostic_obs_group": DiagnosticObsGroup(),
    "origin_marker_debug_draw": OriginMarkerDebugDraw(),
    "sensor_period_restated": SensorPeriodRestated(),
}
