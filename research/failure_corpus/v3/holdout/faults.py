"""v3 holdout fault plugin: PhysX -> Newton/MuJoCo-Warp migration faults and benign controls.

Every family is an object exposing a subset of the hooks documented in PLUGIN_API.md:
``cfg(env_cfg, platform, params)``, ``post_build(u, params)``,
``transform_action(a_native, u, params)``, ``transform_obs(obs_policy, u, params)`` and
``transform_reset(state, u, params)``. All hooks are applied to the candidate (Newton) run only.

Defect families model a migration bug as it would be written by an engineer porting a
PhysX-trained locomotion stack to Newton/MJWarp. Benign families only touch visualization,
rendering, an unused observation group, or re-specify values with numerically identical ones.

Isaac Lab APIs used here were checked against the Isaac Lab source tree
(``isaaclab_newton.assets.articulation.Articulation`` write/set methods, ``ProxyArray.torch``,
``NewtonCfg`` / ``NewtonShapeCfg`` / ``MJWarpSolverCfg``, ``EventTermCfg``/``TerminationTermCfg``
``replace``, ``ActionManager.action``, ``JointAction._scale``).
"""

from __future__ import annotations

import functools

import torch

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

_BASE_OBS = 12  # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)


def _t(x):
    """Return a torch view of an Isaac Lab data field (torch tensor, ProxyArray or warp array)."""
    if isinstance(x, torch.Tensor):
        return x
    tv = getattr(x, "torch", None)
    if isinstance(tv, torch.Tensor):
        return tv
    try:  # plain warp array
        import warp as wp

        if isinstance(x, wp.array):
            return wp.to_torch(x)
    except Exception:  # pragma: no cover - warp absent in unit tests
        pass
    return torch.as_tensor(x)


def _num_joints_from_obs(obs: torch.Tensor) -> int | None:
    """Policy obs = 12 base dims + joint_pos(nj) + joint_vel(nj) + actions(nj)."""
    extra = obs.shape[-1] - _BASE_OBS
    if extra <= 0 or extra % 3 != 0:
        return None
    return extra // 3


def _newton_cfgs(env_cfg) -> list:
    """Return the NewtonCfg object(s) that will drive the candidate simulation.

    The candidate cfg is normally already resolved to ``NewtonCfg``; if a PresetCfg is still
    present, its ``newton_mjwarp`` alternative is the one the preset resolver will pick.
    """
    phys = getattr(env_cfg.sim, "physics", None)
    if phys is None:
        return []
    if type(phys).__name__ == "NewtonCfg":
        return [phys]
    alt = getattr(phys, "newton_mjwarp", None)
    if alt is not None and type(alt).__name__ == "NewtonCfg":
        return [alt]
    return []


def _func_name(func) -> str:
    if isinstance(func, str):
        return str(func).split(":")[-1].split(".")[-1]
    return getattr(func, "__name__", type(func).__name__)


def _resolve_callable(func):
    if isinstance(func, str):
        from isaaclab.utils.string import string_to_callable

        return string_to_callable(str(func))
    return func


def _env_id_tensor(env, env_ids) -> torch.Tensor:
    n = env.num_envs
    if env_ids is None:
        return torch.arange(n, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(n, device=env.device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=env.device).long().flatten()


def _robot(u):
    return u.scene["robot"]


# --------------------------------------------------------------------------------------
# DEFECT 1 (actuator_solver_config): PhysX contactOffset ported into Newton shape margin
# --------------------------------------------------------------------------------------


class ContactOffsetAsShapeMargin:
    """The PhysX collision ``contactOffset`` (a detection band that produces no force) is ported
    into Newton's ``default_shape_cfg.margin``, which Newton/MuJoCo treat as geometric
    inflation of every collision shape. Feet and ground are inflated, so the robot stands
    and walks a few centimetres above the true contact surface."""

    def cfg(self, env_cfg, platform, params):
        for nc in _newton_cfgs(env_cfg):
            nc.default_shape_cfg = nc.default_shape_cfg.replace(margin=float(params["margin_m"]))


# --------------------------------------------------------------------------------------
# DEFECT 2 (actuator_solver_config): PhysX iteration counts ported onto MJWarp iteration budget
# --------------------------------------------------------------------------------------


class PhysxIterationCountsAsMJWarpBudget:
    """The articulation's PhysX ``solver_position_iteration_count`` (4 or 8 TGS iterations) is
    copied onto MJWarp's ``iterations`` / ``ls_iterations`` (defaults 100 / 50, preset values
    otherwise untouched). The Newton-method constraint solve stops long before convergence,
    so contact and limit impulses are systematically under-resolved without any error."""

    def cfg(self, env_cfg, platform, params):
        for nc in _newton_cfgs(env_cfg):
            sc = nc.solver_cfg
            if sc is None or type(sc).__name__ != "MJWarpSolverCfg":
                continue
            nc.solver_cfg = sc.replace(
                iterations=int(params["iterations"]), ls_iterations=int(params["ls_iterations"])
            )


# --------------------------------------------------------------------------------------
# DEFECT 3 (actuator_solver_config): MuJoCo-authored frictionloss reaches the candidate only
# --------------------------------------------------------------------------------------


class MjcFrictionlossCandidateOnly:
    """The robot asset used for the migration was round-tripped through the vendor MJCF, which
    carries a default-class ``frictionloss``. PhysX ignores ``mjc:*`` attributes, while the
    MJWarp import path resolves them into ``Model.joint_friction`` (an absolute N*m dry
    friction). Only the candidate gets joint Coulomb friction on every DOF."""

    def post_build(self, u, params):
        robot = _robot(u)
        robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=float(params["frictionloss_nm"]))


# --------------------------------------------------------------------------------------
# DEFECT 4 (ineffective_stale_config): disabled (limited="false") ranges become active limits
# --------------------------------------------------------------------------------------


class DisabledJointRangeEnforced:
    """The converted MJCF keeps a nominal, *disabled* (``limited="false"``) joint range around the
    default pose. The Newton importer turns that finite range into active joint limits, so the
    candidate enforces a much narrower range than the PhysX asset. The disable flag is silently
    ineffective; nothing errors, strides are just clipped at the stale range."""

    def post_build(self, u, params):
        robot = _robot(u)
        lim = _t(robot.data.joint_pos_limits).clone().float()  # (E, J, 2)
        dflt = _t(robot.data.default_joint_pos).float()  # (E, J)
        h = float(params["half_range_rad"])
        lo = torch.maximum(lim[..., 0], dflt - h)
        hi = torch.minimum(lim[..., 1], dflt + h)
        # never produce an empty range
        hi = torch.maximum(hi, lo + 1.0e-3)
        new = torch.stack([lo, hi], dim=-1).contiguous()
        robot.write_joint_position_limit_to_sim_index(limits=new, warn_limit_violation=False)


# --------------------------------------------------------------------------------------
# DEFECT 5 (joint_body_ordering): per-link mass table keyed by sorted names applied by index
# --------------------------------------------------------------------------------------


def _name_sorted_permutation(names: list[str]) -> list[int]:
    return sorted(range(len(names)), key=lambda i: names[i])


class LinkMassTableNameSortedOrder:
    """Identified per-link masses were exported from the PhysX run as a table sorted by link name
    and written back positionally into Newton's body order. PhysX orders bodies by breadth-first
    parsing, Newton follows the USD/MJCF tree, and the sorted table matches neither, so link
    masses are shuffled among the limbs. Total mass and the root body are unchanged."""

    def post_build(self, u, params):
        robot = _robot(u)
        names = list(robot.body_names)
        m = _t(robot.data.body_mass).clone().float()  # (E, B)
        start = 1 if params.get("keep_root", True) else 0
        idx = list(range(start, len(names)))
        if len(idx) < 2:
            return
        perm = _name_sorted_permutation([names[i] for i in idx])
        src = torch.tensor([idx[p] for p in perm], device=m.device, dtype=torch.long)
        dst = torch.tensor(idx, device=m.device, dtype=torch.long)
        new = m.clone()
        new[:, dst] = m[:, src]
        robot.set_masses_index(masses=new.contiguous())


# --------------------------------------------------------------------------------------
# DEFECT 6 (obs_action_ordering): previous-action slot fed the scaled (processed) action
# --------------------------------------------------------------------------------------


class LastActionSlotScaled:
    """The candidate's observation adapter fills the ``actions`` slot with the scaled joint-target
    offset (``raw * scale``, the processed-action convention of many MuJoCo deploy loops)
    instead of the raw policy output the PhysX policy was trained on."""

    @staticmethod
    def _scale(u, params) -> float:
        try:
            term = u.action_manager.get_term("joint_pos")
            s = getattr(term, "_scale", None)
            if isinstance(s, (int, float)):
                return float(s)
        except Exception:
            pass
        return float(params["action_scale"])

    def transform_obs(self, obs_policy, u, params):
        nj = _num_joints_from_obs(obs_policy)
        if nj is None:
            return obs_policy
        out = obs_policy.clone()
        a0 = _BASE_OBS + 2 * nj
        out[:, a0 : a0 + nj] = out[:, a0 : a0 + nj] * self._scale(u, params)
        return out


# --------------------------------------------------------------------------------------
# DEFECT 7 (obs_action_ordering): Optional-annotated term hoisted to the front of the group
# --------------------------------------------------------------------------------------


class OptionalTermHoistedToFront:
    """In the candidate's observation config one term was annotated ``ObsTerm | None`` (so the
    Newton variant can disable it). The configclass then moves that term to the front of the
    concatenated policy group, shifting every term that preceded it."""

    _SLICES = {
        "base_lin_vel": (0, 3),
        "base_ang_vel": (3, 6),
        "projected_gravity": (6, 9),
        "velocity_commands": (9, 12),
    }

    def transform_obs(self, obs_policy, u, params):
        if _num_joints_from_obs(obs_policy) is None:
            return obs_policy
        a, b = self._SLICES[params.get("term", "velocity_commands")]
        return torch.cat([obs_policy[:, a:b], obs_policy[:, :a], obs_policy[:, b:]], dim=1)


# --------------------------------------------------------------------------------------
# DEFECT 8 (timestep_decimation): policy decimation counted at the wrong rate
# --------------------------------------------------------------------------------------


class PolicyDecimationHeldAcrossSteps:
    """The candidate control loop derives its policy decimation from a different physics rate
    (e.g. ``control_decimation`` written for a 500 Hz sim reused at 200 Hz), so the policy is
    only re-queried every ``hold`` control steps and its previous output is re-applied in
    between. The env still steps at 50 Hz, but the policy effectively runs at 50/hold Hz."""

    def transform_action(self, a_native, u, params):
        k = int(params["hold"])
        if k <= 1:
            return a_native
        if int(u.common_step_counter) % k == 0:
            return a_native
        prev = u.action_manager.action
        if prev is None or prev.shape != a_native.shape:
            return a_native
        return prev.clone().to(a_native.dtype)


# --------------------------------------------------------------------------------------
# DEFECT 9 (reset_randomization): reset selector count collapses to one sample
# --------------------------------------------------------------------------------------


def _broadcast_root(env, ids, asset_name):
    robot = env.scene[asset_name]
    pose = _t(robot.data.root_link_pose_w)[ids].clone()
    origins = env.scene.env_origins[ids]
    rel = pose[0, :3] - origins[0]
    pose[:, :3] = origins + rel
    pose[:, 3:7] = pose[0:1, 3:7]
    vel = _t(robot.data.root_com_vel_w)[ids].clone()
    vel[:] = vel[0:1].clone()
    robot.write_root_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=vel.contiguous(), env_ids=ids)


def _broadcast_joints(env, ids, asset_name):
    robot = env.scene[asset_name]
    q = _t(robot.data.joint_pos)[ids].clone()
    qd = _t(robot.data.joint_vel)[ids].clone()
    q[:] = q[0:1].clone()
    qd[:] = qd[0:1].clone()
    robot.write_joint_state_to_sim_index(position=q.contiguous(), velocity=qd.contiguous(), env_ids=ids)


def _make_broadcast_reset(orig, kind):
    @functools.wraps(orig)
    def _wrapped(env, env_ids, *args, **kwargs):
        orig(env, env_ids, *args, **kwargs)
        ids = _env_id_tensor(env, env_ids)
        if ids.numel() < 2:
            return
        asset_cfg = kwargs.get("asset_cfg")
        name = getattr(asset_cfg, "name", None) or "robot"
        if kind == "root":
            _broadcast_root(env, ids, name)
        else:
            _broadcast_joints(env, ids, name)

    _wrapped._ivf_holdout_broadcast = True
    return _wrapped


class ResetSampleBroadcast:
    """A custom reset callback written for index tensors receives the new slice/mask selector,
    derives its sample count as one, and the single randomized reset sample (the first selected
    environment's) is broadcast to every environment reset in that call. Resets still happen
    and look randomized per episode, but the across-environment reset distribution collapses."""

    def cfg(self, env_cfg, platform, params):
        events = getattr(env_cfg, "events", None)
        if events is None:
            return
        for name, term in list(events.__dict__.items()):
            if term is None or getattr(term, "mode", None) != "reset" or not hasattr(term, "func"):
                continue
            fname = _func_name(term.func)
            kind = "root" if fname.startswith("reset_root") else ("joint" if fname.startswith("reset_joints") else None)
            if kind is None:
                continue
            func = _resolve_callable(term.func)
            if getattr(func, "_ivf_holdout_broadcast", False):
                continue
            setattr(events, name, term.replace(func=_make_broadcast_reset(func, kind)))

    def transform_reset(self, state, u, params):
        out = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] > 1:
                out[k] = v[0:1].expand_as(v).clone()
            else:
                out[k] = v
        return out


# --------------------------------------------------------------------------------------
# DEFECT 10 (checkpoint_schema): raw actions clipped to a unit Box action space
# --------------------------------------------------------------------------------------


class RawActionClippedToUnitBox:
    """The PhysX checkpoint was trained through the RSL-RL wrapper with ``clip_actions=None``
    (unbounded Gaussian actions). The candidate runner rebuilds the policy from an agent config
    whose action space is ``Box(-1, 1)`` and clips raw actions to it before they reach the env."""

    def transform_action(self, a_native, u, params):
        c = float(params["clip"])
        return torch.clamp(a_native, -c, c)


# --------------------------------------------------------------------------------------
# DEFECT 11 (checkpoint_schema): stochastic Gaussian head used at evaluation time
# --------------------------------------------------------------------------------------


class StochasticActorHeadAtEval:
    """After the rsl-rl >= 5 config migration the actor keeps a ``GaussianDistributionCfg``;
    the candidate evaluation calls the sampling path instead of the deterministic mean, so every
    action carries i.i.d. Gaussian exploration noise with the checkpoint's learned std."""

    def transform_action(self, a_native, u, params):
        std = float(params["std"])
        gen = torch.Generator(device=a_native.device)
        gen.manual_seed(int(params["seed"]) * 1_000_003 + int(u.common_step_counter))
        noise = torch.randn(a_native.shape, generator=gen, device=a_native.device, dtype=a_native.dtype)
        return a_native + std * noise


# --------------------------------------------------------------------------------------
# DEFECT 12 (termination_metric): illegal-contact termination reads only the newest sample
# --------------------------------------------------------------------------------------


def illegal_contact_latest_sample(env, threshold: float, sensor_cfg) -> torch.Tensor:
    """Port of ``illegal_contact`` that reads ``net_forces_w`` (newest physics sample) instead of the
    ``net_forces_w_history`` window, so contacts that begin and end inside the decimation window
    are never seen by the termination."""
    sensor = env.scene.sensors[sensor_cfg.name]
    f = _t(sensor.data.net_forces_w)
    return torch.any(torch.linalg.norm(f[:, sensor_cfg.body_ids], dim=-1) > threshold, dim=1)


class TerminationReadsLatestContactOnly:
    """The candidate's contact termination was rewritten against the Newton contact sensor using
    its instantaneous ``net_forces_w`` field; the history window over the decimation substeps is
    dropped. Brief base/body impacts that resolve mid-window no longer terminate episodes."""

    def cfg(self, env_cfg, platform, params):
        terms = getattr(env_cfg, "terminations", None)
        if terms is None:
            return
        for name, term in list(terms.__dict__.items()):
            if term is None or not hasattr(term, "func"):
                continue
            if _func_name(term.func) == "illegal_contact":
                setattr(terms, name, term.replace(func=illegal_contact_latest_sample))


# --------------------------------------------------------------------------------------
# BENIGN controls
# --------------------------------------------------------------------------------------


class BenignViewerCameraReframe:
    """Move the (headless-unused) viewport camera and change its resolution. Rendering only."""

    def cfg(self, env_cfg, platform, params):
        env_cfg.viewer = env_cfg.viewer.replace(
            eye=tuple(float(x) for x in params["eye"]),
            lookat=tuple(float(x) for x in params["lookat"]),
            resolution=tuple(int(x) for x in params["resolution"]),
        )


class BenignCommandArrowDebugVisOff:
    """Turn off the velocity-command arrow markers. Visualization only; command sampling unchanged."""

    def cfg(self, env_cfg, platform, params):
        cmds = getattr(env_cfg, "commands", None)
        bv = getattr(cmds, "base_velocity", None) if cmds is not None else None
        if bv is not None:
            cmds.base_velocity = bv.replace(debug_vis=bool(params["debug_vis"]))


class BenignUnusedDebugObsGroup:
    """Add a separate, noise-free observation group for logging base height and orientation. It
    is not concatenated into, and not fed to, the policy group and reads state only."""

    def cfg(self, env_cfg, platform, params):
        from isaaclab.envs import mdp
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg

        grp = ObservationGroupCfg()
        grp.base_height = ObservationTermCfg(func=mdp.base_pos_z)
        grp.base_quat = ObservationTermCfg(func=mdp.root_quat_w)
        grp.enable_corruption = False
        grp.concatenate_terms = True
        setattr(env_cfg.observations, str(params["group_name"]), grp)


class BenignIdenticalRespecification:
    """Re-specify gravity, physics dt and decimation from their own current values through a
    different code path (explicit float/int conversion). The resolved config is identical."""

    def cfg(self, env_cfg, platform, params):
        env_cfg.sim.gravity = tuple(float(g) for g in env_cfg.sim.gravity)
        env_cfg.sim.dt = float(env_cfg.sim.dt)
        env_cfg.decimation = int(env_cfg.decimation)


class BenignTerrainVisualMaterialSwap:
    """Replace the ground's visual (MDL) material with a flat preview-surface colour. The physics
    material of the terrain is a different field and is untouched."""

    def cfg(self, env_cfg, platform, params):
        import isaaclab.sim as sim_utils

        terrain = getattr(env_cfg.scene, "terrain", None)
        if terrain is None:
            return
        env_cfg.scene.terrain = terrain.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(float(c) for c in params["diffuse_color"]),
                roughness=float(params["roughness"]),
            )
        )


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------

FAMILIES: dict[str, object] = {
    "contact_offset_as_shape_margin": ContactOffsetAsShapeMargin(),
    "physx_iteration_counts_as_mjwarp_budget": PhysxIterationCountsAsMJWarpBudget(),
    "mjc_frictionloss_candidate_only": MjcFrictionlossCandidateOnly(),
    "disabled_joint_range_enforced": DisabledJointRangeEnforced(),
    "link_mass_table_name_sorted_order": LinkMassTableNameSortedOrder(),
    "last_action_slot_scaled": LastActionSlotScaled(),
    "optional_term_hoisted_to_front": OptionalTermHoistedToFront(),
    "policy_decimation_held_across_steps": PolicyDecimationHeldAcrossSteps(),
    "reset_sample_broadcast": ResetSampleBroadcast(),
    "raw_action_clipped_to_unit_box": RawActionClippedToUnitBox(),
    "stochastic_actor_head_at_eval": StochasticActorHeadAtEval(),
    "termination_reads_latest_contact_only": TerminationReadsLatestContactOnly(),
    "benign_viewer_camera_reframe": BenignViewerCameraReframe(),
    "benign_command_arrow_debug_vis_off": BenignCommandArrowDebugVisOff(),
    "benign_unused_debug_obs_group": BenignUnusedDebugObsGroup(),
    "benign_identical_respecification": BenignIdenticalRespecification(),
    "benign_terrain_visual_material_swap": BenignTerrainVisualMaterialSwap(),
}
