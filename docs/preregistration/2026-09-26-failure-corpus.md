# Pre-registration: IVF against a cross-backend failure corpus

Registered 2026-09-26, before any corpus case was captured or evaluated. Work lives in
`research/failure_corpus/`. The object under test is IVF frozen at
`d64f74049a7e2b64bba5289f5627f136d71bbe4f` (see
`research/failure_corpus/provenance/FROZEN_IVF.md`). IVF code is not modified by this study.

Pilot activity before registration, disclosed: (1) a feasibility probe that built the
GO2, G1, H1 and ANYmal-D flat velocity envs on PhysX and Newton/MJWarp and stepped them with
zero actions; (2) a check that the pre-PR-#7103 Isaac Lab smoke-test preset path
instantiates `PhysxManager` instead of `NewtonMJWarpManager` in this checkout (runtime
evidence, not config inspection); (3) training of one PhysX policy per platform, which is
instrument preparation. No IVF verdict and no baseline decision had been computed on any
corpus case.

## 1. Claim under test

> On cross-backend (PhysX to Newton/MJWarp) locomotion experiments carrying an externally
> documented silent validity defect, IVF with calibrated tolerances flags (non-`PASS`) at
> least 0.20 more of the defective cases than the strongest conventional sim-to-sim
> evaluation (native smoke validation OR closed-loop policy-performance comparison OR
> open-loop trajectory RMSE, thresholds calibrated on the same clean pairs), at a clean-case
> false-positive rate no more than 0.10 above that conventional composite, and assigns the
> documented root-cause category to at least 50% of the defects it flags.

Comparator: the conventional composite (B1 OR B2 OR B3 below). Primary analysis runs on the
sealed holdout only.

## 2. Arms

| Arm | Purpose |
|---|---|
| **IVF-cal (treatment)** | Frozen IVF, fixed manifest template, tolerances = 1.25 x worst value on clean calibration pairs |
| IVF-strict | Frozen IVF, tolerances from IVF's documented flagship method (one-control-period displacement of the baseline). Secondary |
| B0 accept-all | No-treatment floor |
| B1 native | Isaac Lab native smoke semantics: env builds, steps without exception, obs/reward finite (`_check_valid_tensor`) |
| B2 performance | Conventional sim-to-sim: PhysX-trained policy run closed-loop on the candidate; flag if |delta| in mean return, fall terminations, or velocity-tracking error vs the PhysX reference exceeds 1.25 x the worst clean calibration delta |
| B3 trajectory | Open-loop replay RMSE of `joint_pos` and `root_link_pos_w` vs the reference; same 1.25 x calibration rule |
| B4 config-diff | Flag if the resolved env cfg digest (physics subtree removed) differs. Conventional engineering practice, reported separately |
| Conventional composite | B1 OR B2 OR B3 (primary comparator) |
| Sham (dose-matched) | Random flagger at each arm's realized flag rate; expected recall equals the flag rate. Reported next to every recall |
| Positive control | (a) same-backend condition (PhysX vs PhysX+fault), where backend noise is absent; (b) `ivf calibrate` synthetic taxonomy, 17/17 expected; (c) gross faults (decimation change, action-scale change) must be flagged by IVF-cal and B3 in the same-backend condition |
| IVF ablations | IVF re-run with manifests that keep only / drop one component group: validity-only, trajectory oracles, event+decision oracles, invariants, `policy_obs` oracle, `joint_pos_target` oracle, statistical oracle |

## 3. Corpus

Each family is a controlled perturbation reproducing an externally documented failure
(sources and verbatim quotes in `research/failure_corpus/corpus/external_cases.json`). A
family is applied to the candidate only. Dev variant parameters are used on GO2/G1/H1 seed 3;
holdout uses ANYmal-D (unseen platform, dev variants, seeds 3 and 4) plus GO2/G1/H1 seed 4
with the holdout variant.

| Family | Category | Dev variant | Holdout variant | Sources |
|---|---|---|---|---|
| joint_order_obs_action | joint_body_ordering | per-limb (depth-first) permutation of actions and joint obs | left/right swap | IsaacLab sim-to-sim doc, PR #6913, issue #6485, unitree_rl_lab #145 |
| obs_term_swap | obs_action_ordering | swap `base_ang_vel` and `projected_gravity` blocks | swap `base_lin_vel` and `velocity_commands` | unitree_rl_gym #32 |
| timestep_dt_decimation | timestep_decimation | decimation 4 to 2, dt unchanged | dt 0.005 to 0.01, decimation 4 to 2 | Deploy_Tienkung #8, HOVER #38 |
| randomization_asymmetry | reset_randomization | startup base-mass randomization active on candidate only, range (0.8, 1.25) | range (0.9, 1.1) | IsaacLab #7786, #7097, PR #7992 |
| reset_velocity_dropped | reset_randomization | requested root velocity written as zero | joint offsets ignored at reset | IsaacLab #7236; upstream parity-harness reset defect |
| armature_dropped | actuator_solver_config | all actuator armature set to 0 | same | IsaacLab PR #7612, #7607; unitree_rl_lab #31; unitree_rl_gym #47 |
| contact_capacity | actuator_solver_config (Newton only) | nconmax 2, njmax 8 | nconmax 3, njmax 12 | IsaacLab PR #6850; Isaac Sim / MuJoCo Warp docs |
| actuator_gain_scale | actuator_solver_config | damping x 0 | stiffness and damping x 0.7 | newton #3698; unitree_rl_gym #47 |
| preset_not_applied | ineffective_stale_config (cross only) | candidate built via pre-#7103 smoke-test path | same | IsaacLab PR #7103 (verified locally) |
| action_scale | checkpoint_schema | action scale x 2 | action scale x 0.5 | UniLab #579 |
| termination_body_mismatch | termination_metric | base-contact termination points at body index 1 | body index 2 | IsaacLab body-ordering doc, PR #6913 |

Clean (defect-absent) cases: Newton clean; Newton benign solver variant (nconmax and njmax
doubled, more capacity); PhysX A/A rerun (same-backend clean).

Conditions: **cross** (baseline PhysX clean, candidate Newton + fault) is primary; **same**
(baseline PhysX clean, candidate PhysX + fault) is the positive-control condition, used for
families that are backend-agnostic.

Separately reported, not pooled: the shipped IVF synthetic taxonomy (17 classes, 3 seeds)
with baselines B1/B3 computed on the same synthetic signals; and two IVF-internal historical
cases (shipped cart-pole reset defect; shipped cart-pole bundles declaring quaternion
layout `wxyz` while containing `xyzw` data).

## 4. Protocol

Producer (`research/failure_corpus/capture/corpus_capture.py`) writes `trajectory_bundle/v1`
directly. Every contract field is read from the live env at runtime. Labels are written
outside the bundle. Open-loop capture: 16 envs, 250 control steps, fixed per-env command
schedule, deterministic requested reset state, no auto-reset, actions = the PhysX clean
reference policy stream for that seed. Closed-loop: 64 envs, 1000 control steps, the
env's own terminations and resets, same fixed commands. PLAY-style controlled config for
all jobs (obs noise, pushes, random startup mass/COM off). Identity digest
(`config_digest_sha256`) covers the backend-independent task identity, mirroring the shipped
cart-pole producer. A pre-registered producer ablation re-seals bundles with the full
resolved-cfg digest to test a richer identity.

IVF manifest template (fixed now): invariants `finite_state`, `unit_quaternion`;
trajectory oracles on `joint_pos`, `joint_vel`, `joint_pos_target`, `root_link_pos_w`,
`root_link_quat_w` (geodesic), `base_height`, `policy_obs` (absolute, `second_largest`);
event oracles on fall (`base_height` below a per-platform threshold) and on
`illegal_contact` above 0.5; decision oracle on survival over the horizon; statistical
equivalence on horizon-mean `base_height`; metamorphic `observation_definition_stability`
and `action_replay_consumption`. Controls: the flagship's `require_same` list;
`solver_specific_parameters` allowed to differ.

Calibration: seeds 0, 1, 2 on all four platforms, clean pairs only (PhysX reference vs
Newton clean for cross; vs PhysX rerun for same). No fault case is used for any threshold.
Floor 1e-6 for any tolerance that calibrates to zero.

Definitions. Flag = any verdict other than `PASS`. Recall = flagged defect cases / defect
cases. False-positive rate = flagged clean cases / clean cases. False-acceptance rate =
`PASS` on defect cases / defect cases. Localization: IVF output mapped to a category by the
fixed rule in `research/failure_corpus/eval/localize.py` (validity reason codes first, else
the divergence classification of the earliest failing oracle); correct if equal to the
case's documented category. Comparator for localization: majority-category guess.
Secondary labels: a defect is **consequential** if its closed-loop outcome differs from the
clean Newton case by more than the B2 threshold.

## 5. Confound answers

- A1 onset: report the fraction of IVF detections with first violation at step 0 or 1.
- A2 exposure: every arm sees identical horizons and env counts.
- A3 seeds: the unit is the case. Per-family n is small (2 to 6); no per-family significance
  claim will be made.
- A4 bundling: cross-backend noise and the fault co-vary in the cross condition; the same
  condition and the clean Newton case isolate each.
- A6 outcome: closed-loop consequence recorded for every case; reported next to detection.
- A8 ceiling: IVF-strict is expected near 100% flag rate on clean cross pairs (the flagship
  already FAILs a clean PhysX/Newton pair); reported, not hidden.
- A9 selection: template, thresholds and mapping fixed before evaluation. Dev results may
  expose harness bugs; every change after dev is logged as a deviation, and the holdout
  label file hash is committed before holdout capture.
- A10 leakage: calibration seeds (0 to 2) are disjoint from evaluation seeds (3, 4).

## 6. Analysis plan

Primary (holdout): recall difference IVF-cal minus conventional composite on the same
cases, exact McNemar test on discordant pairs, 95% CI by case bootstrap stratified by family
(10,000 resamples). FPR difference on clean holdout cases with the same method.
Proportions carry Wilson 95% intervals. Secondary and exploratory: dev split, IVF-strict,
B4, ablations, same-backend condition, synthetic stratum. Multiplicity: only the two primary
endpoints are confirmatory; everything else is labelled exploratory.

## 7. Kill criteria (numeric)

1. Detection claim dead if the 95% CI of (IVF-cal recall minus composite recall) on the
   holdout includes 0, or if IVF-cal flags fewer than 50% of holdout defects that the
   composite accepts.
2. Operating-point claim dead if IVF-cal holdout FPR exceeds composite FPR by more than
   0.10.
3. Localization claim dead if IVF-cal top-1 category accuracy on flagged holdout defects is
   below 0.50, or its Wilson lower bound does not exceed the majority-guess accuracy.

Any dead claim is reported as a negative result in the report, with the same prominence as
a positive one.

## 8. Stopping rule

The corpus is fixed at the families above. If a family cannot be executed (crash, API
absent), it is reported as not executed with the error, not replaced by a different
family after results are seen.
