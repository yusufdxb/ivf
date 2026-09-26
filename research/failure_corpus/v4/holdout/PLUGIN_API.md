# Fault plugin API for the v4 holdout

You are writing fault families for a physics-backend migration study. A capture producer
runs PhysX-trained locomotion policies on Isaac Lab velocity tasks, on PhysX (baseline)
and Newton/MuJoCo-Warp (candidate). Your faults are applied to the **candidate** run only.
Downstream validators judge whether a candidate experiment is valid. You must not look at
how any validator works: do not read `src/ivf/`, `research/failure_corpus/v2/eval/`,
`research/failure_corpus/v2/`, `research/failure_corpus/v3/` (entire directory),
`research/failure_corpus/v4/eval/`, `research/failure_corpus/v4/results/`,
`research/failure_corpus/v4/corpus/`, `research/failure_corpus/REPORT.md`,
`research/failure_corpus/results/`, `research/failure_corpus/corpus/`, `docs/`, or `tests/`.
In particular do not read any previous holdout's fault module, case list, labels, or results.

## What to deliver

1. `research/failure_corpus/v4/holdout/faults.py` defining `FAMILIES: dict[str, object]`.
   Each value is an object (a class instance or a module-like namespace) with any subset
   of these methods. Each receives the case's `params` dict as its last argument.

   | hook | signature | when it runs |
   |---|---|---|
   | `cfg` | `cfg(env_cfg, platform, params) -> None` | mutate the Isaac Lab env cfg before the env is built (after the standard controlled protocol) |
   | `post_build` | `post_build(u, params) -> None` | once, right after `gym.make`; `u` is the unwrapped `ManagerBasedRLEnv` |
   | `transform_action` | `transform_action(a_native, u, params) -> Tensor` | every control step; `a_native` is the `(num_envs, num_joints)` action in the simulator's native joint order, about to be passed to `env.step` |
   | `transform_obs` | `transform_obs(obs_policy, u, params) -> Tensor` | every control step; the `(num_envs, obs_dim)` observation the policy is fed (closed loop) and that is recorded |
   | `transform_reset` | `transform_reset(state, u, params) -> dict` | once per open-loop run; `state` has `joint_pos`, `joint_vel` (native order, `(num_envs, num_joints)`) and `root_vel` (`(num_envs, 6)`, world-frame linear then angular) that are about to be written to the simulator; return the dict to write |

   Hooks must be deterministic given `params` and the env's own seeded state, must not
   crash, and must not change the number of envs, the episode length, or the control
   horizon. Use torch operations on the given device.

2. `research/failure_corpus/v4/holdout/fault_cases.json`: a list of family entries:
   ```json
   {"family": "name", "defect": true, "category": "<one of the 8 below or 'none'>",
    "params": {"go2": {...}, "g1": {...}, "h1": {...}, "anymal_d": {...}, "cassie": {...}, "spot": {...}, "go2_rough": {...}},
    "mechanism": "one paragraph: what goes wrong in a real migration",
    "sources": [{"url": "...", "quote": "verbatim <= 40 words, fetched"}]}
   ```
   Also for benign families: `"contract_preservation": "<argument>"` and
   `"static_cfg_diff": {"<platform>": [<list of dotted cfg paths that differ from the clean
   Newton cfg>]}` produced by actually loading the configs (see Benign controls below).

   Categories: `joint_body_ordering`, `obs_action_ordering`, `timestep_decimation`,
   `reset_randomization`, `actuator_solver_config`, `ineffective_stale_config`,
   `checkpoint_schema`, `termination_metric`. Use `"defect": false, "category": "none"` for
   **benign** variants (see the strict rules below).
   Parameters may differ per platform; omit a platform only if the family cannot apply.

## Environment facts

- Platforms and tasks: `go2` Isaac-Velocity-Flat-UnitreeGo2, `g1` Isaac-Velocity-Flat-G1,
  `h1` Isaac-Velocity-Flat-H1, `anymal_d` Isaac-Velocity-Flat-AnymalD, `cassie`
  Isaac-Velocity-Flat-Cassie, `spot` Isaac-Velocity-Flat-Spot, `go2_rough`
  Isaac-Velocity-Rough-UnitreeGo2 (rough terrain; its policy observation also has a
  `height_scan` term). Isaac Lab source: `~/Projects/isaac-sim-contrib/IsaacLab`
  (Isaac Lab 10.2.0, Newton 1.4.0.dev0, MuJoCo-Warp 3.8.0.3). Read it freely.
- Control at 50 Hz (sim dt 0.005, decimation 4). Open-loop capture: 16 envs, 250 control
  steps, replaying a fixed recorded action stream, no auto-reset. Closed-loop evaluation:
  64 envs, 1000 steps, the policy acting on the (possibly transformed) observations with
  the task's own terminations and resets.
- The candidate is built with the `newton_mjwarp` preset. The producer already remaps
  joint order between the policy's training order and the simulator's native order, so a
  correct candidate needs no further remap.
- Observation terms (policy group, concatenated): base_lin_vel(3), base_ang_vel(3),
  projected_gravity(3), velocity_commands(3), joint_pos(nj), joint_vel(nj), actions(nj).

## Rules

- Draw on **new** documented migration failures, not these already-used sources: Isaac Lab
  PRs/issues #6485, #6649, #6765, #6850, #6913, #7097, #7103, #7202, #7236, #7520, #7574,
  #7601, #7607, #7612, #7613, #7786, #7992; newton #911, #3698; mujoco_warp #1635, #1677;
  unitree_rl_gym #32, #47; unitree_rl_lab #31, #145; HOVER #38; Deploy_Tienkung #8;
  UniLab #579; the Isaac Lab sim-to-sim and joint/body-ordering doc pages. Also avoid
  re-implementing these already-used perturbations: joint-order interface permutation,
  capture in native joint order, swapping two observation terms, changing dt/decimation,
  base-mass randomization on one side, dropping reset velocity or reset joint offsets,
  armature change, MJWarp nconmax/njmax, actuator stiffness/damping scaling, wrong physics
  preset path, action-scale change, termination pointing at a wrong body.
- Aim for 10 to 14 defect families across as many categories as possible, and 4 to 6
  benign families that satisfy the strict benign rules below. Prefer failures that are **silent** (no crash, plausible behavior).
- Quote sources only from pages you actually fetched.
- Test that `faults.py` imports and that each hook runs on a tiny fake tensor where
  possible without Isaac Lab; you cannot run the simulator.

## Benign controls (strict)

A benign family must be **demonstrably contract-preserving**: it must not change anything
the experiment is defined by or measures. Allowed kinds of change: visualization, debug
drawing, rendering, logging, video, UI; adding an observation group or sensor that is not
fed to the policy and does not alter physics; re-specifying a config value with a
numerically identical value through a different path; equivalent refactors whose resolved
config is identical. Not allowed as benign: anything that changes world-frame positions,
environment layout or spacing, physics or solver settings (including iteration counts and
tolerances), actuation, observations the policy sees, action processing, resets,
randomization, terminations, rewards, or time limits.

For each benign family you must provide static evidence: load the clean Newton env cfg
and the benign-modified env cfg for every platform (no simulator), apply your `cfg` hook,
flatten both with `to_dict()`, and list every dotted path that differs in
`static_cfg_diff`. Every listed path must be a visualization/logging/rendering/unused-group
path. A benign family whose runtime hooks (`transform_*`, `post_build`) change any tensor
the policy or simulator consumes is not benign. Provide 4 to 6 benign families.

## Exclusions for v4

Do not re-implement any perturbation used in earlier rounds (by mechanism, not just name):
joint-order interface permutation or native-order capture; observation-term swap;
dt/decimation change; one-sided base-mass randomization; dropped reset velocity or reset
joint offsets; reset DOF-index shift; angular/linear twist order at reset; spawn height
from asset keyframe; armature change; MJWarp nconmax/njmax; actuator stiffness/damping
scaling; actuator model substitution; wrong physics preset path; MJWarp preset overrides
lost; friction randomization removed; action-scale change; stale default-pose action
offset; one-step action delay; finite-difference joint velocity; quaternion wxyz/xyzw
misread of gravity; base velocity in world frame; legged_gym observation scales; contact
threshold unit error; timeout counted as termination; termination pointing at a wrong body;
PhysX contact offset copied into shape margin; PhysX iteration counts used as MJWarp
iterations; joint friction loss on one side only; disabled joint range enforced; link-mass
table written in name-sorted order; last-action observation slot scaled; optional
observation term hoisted to the front; policy re-queried every other step with held action;
one reset sample broadcast to all envs; raw actions clipped to a unit box; stochastic actor
head at evaluation; termination reading only the newest contact sample.
