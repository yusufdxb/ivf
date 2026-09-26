# Does IVF catch cross-backend validity failures that conventional evaluation accepts?

> **Archived round record.** Results below stand as recorded for this round. The authoritative research conclusion is [`v4/REPORT_V4.md`](v4/REPORT_V4.md); where this report differs, v4 governs.

Research report, 2026-09-26. Branch `research/failure-corpus-eval`, local only.

**Answer: no, not on this corpus.** The claim under test was

> IVF detects and localizes silent validity failures in cross-backend robotics
> experiments that conventional sim-to-sim evaluation can otherwise accept.

On the sealed holdout, IVF's apparent recall advantage is entirely a flag-everything
effect. Once the oracles that also flag clean experiments are removed, IVF matches the
conventional composite's false-positive rate and catches fewer defects than it (28/60 vs
31/60). Localization is below a majority-class guess. Two of the three pre-registered
kill criteria are met. The evidence does **not** support a publishable positive claim.
It does support a narrower, honest negative/diagnostic paper or technical report (section 8).

Everything below is recomputable from `research/failure_corpus/results/` with the
commands in section 9. The pre-registration is
[`docs/preregistration/2026-09-26-failure-corpus.md`](../../docs/preregistration/2026-09-26-failure-corpus.md)
(commit `85f1322`, before any capture). Deviations D1 to D8 are in
[`DEVIATIONS.md`](DEVIATIONS.md). D1 to D7 were made before any fault case was
evaluated; D8 is a labelled post-hoc ablation.

## 1. What was frozen

IVF `d64f740` (`src/ivf` tree `4932658`), 265 tests pass, `ivf calibrate` 17/17. The
harness refuses to run if `src/ivf` differs from that tree. IVF code was not modified.
Simulator: Isaac Sim 6.0.0.1, Isaac Lab 10.2.0 (July 2026 `develop` snapshot), PhysX
2.8.1, Newton 1.4.0.dev0 with MuJoCo-Warp 3.8.0.3. Full provenance:
[`provenance/FROZEN_IVF.md`](provenance/FROZEN_IVF.md). One PhysX-trained velocity policy
per platform (GO2, G1, H1, ANYmal-D); all reach 985 to 994 of 1000 steps in training.

## 2. Corpus and provenance

Thirty-one externally documented cases were collected first, before any perturbation was
built, each with a URL and a verbatim quote checked against the source
([`corpus/external_cases.json`](corpus/external_cases.json)): 26 failures that happened
to someone and 5 official known-limitation notes. They were turned into 13 perturbation
families, applied to the candidate only.

| Family | Category | Where the case comes from | Native here? |
|---|---|---|---|
| joint_order_interface | joint/body ordering | Isaac Lab sim-to-sim doc; PR #6913; issue #6485; unitree_rl_lab #145 | **yes** (finding F1) |
| capture_not_canonicalized | joint/body ordering | same, applied to the recorder | **yes** (F1) |
| obs_term_swap | obs/action ordering | unitree_rl_gym #32 | no |
| timestep_dt_decimation | timestep/decimation | Deploy_Tienkung #8; HOVER #38 | no |
| randomization_asymmetry | reset/randomization | Isaac Lab #7786, #7097; PR #7992 | in-tree for COM |
| reset_velocity_dropped / reset_joint_offsets_ignored | reset/randomization | Isaac Lab #7236; IVF's own upstream reset defect | no |
| armature_mismatch | actuator/solver | Isaac Lab PR #7612, #7607; unitree_rl_lab #31 | **yes** for GO2 (in-tree preset) |
| contact_capacity | actuator/solver | Isaac Lab PR #6850; MuJoCo-Warp and Isaac Sim docs | no |
| actuator_gain_scale | actuator/solver | newton #3698; unitree_rl_gym #47 | no |
| preset_not_applied | ineffective/stale config | Isaac Lab PR #7103 | **yes** (F2) |
| action_scale | checkpoint/schema | UniLab #579 | no |
| termination_body_mismatch | termination/metric | Isaac Lab body-ordering doc; PR #6913 | no |

Clean cases: Newton with remap applied, Newton with doubled contact capacity (benign
solver change), and PhysX reruns. Conditions: **cross** (PhysX baseline vs Newton
candidate, primary) and **same** (PhysX vs PhysX, positive control).

| Split | Platforms, seeds | Defect | Clean |
|---|---|---|---|
| calibration (thresholds only) | all 4, seeds 0 to 2 | 0 | 24 |
| dev (developmental) | GO2, G1, H1, seed 3 | 60 | 9 |
| **holdout (primary)** | ANYmal-D unseen seeds 3 and 4; GO2/G1/H1 seed 4 with new variants; clean seeds 5 and 6 on all | 100 | 31 |

Holdout capture ids are opaque hashes. The label file hash was committed (`8bf94d1`)
before holdout capture and re-verified before scoring. All 252 captures ran to
completion with no exception.

Separately reported: IVF's own 17-class synthetic taxonomy, and an IVF-internal case
(F3).

## 3. Arms

- **IVF-cal (treatment):** fixed template (18 oracles, the flagship's 14 controls),
  tolerances = 1.25 x the worst value IVF itself reported on clean calibration pairs.
- **IVF-strict:** tolerances by IVF's flagship method (one control period of motion).
- **B1 native:** Isaac Lab's own smoke semantics (no exception, finite obs/reward).
- **B2 performance:** the PhysX policy run closed-loop on the candidate; return, falls,
  tilt and tracking vs the PhysX reference; the 1.25 x calibration rule.
- **B3 trajectory:** open-loop RMSE on joint position and base position. **B3all** uses
  IVF's full signal set (D7).
- **Composite (primary comparator):** B1 or B2 or B3.
- **Sham:** a random flagger at the same overall flag rate.

## 4. Results on the holdout, cross-backend (60 defect, 18 clean)

| Arm | Recall | Clean FPR | Recall minus dose-matched sham |
|---|---|---|---|
| B1 native | 0/60 | 0/18 | 0.00 |
| B2 performance | 26/60 | 1/18 | +0.09 |
| B3 trajectory | 18/60 | 0/18 | |
| **Composite** | **31/60** | **1/18** | **+0.11** |
| **IVF-cal (registered)** | **53/60** | **14/18** | **+0.02** |
| IVF-cal, event/decision oracles removed (D8) | 28/60 | 1/18 | |
| IVF-strict | 60/60 | 18/18 | 0.00 |
| B4 config diff | 55/60 | 18/18 | |

Pre-registered endpoints:

1. **Recall difference, IVF-cal minus composite:** +0.37, 95% CI [0.27, 0.47], McNemar
   p = 2.7e-5. IVF-cal flags 25 of the 29 defects the composite accepts. *The literal
   criterion passes, but the dose check (A7) removes it:* IVF-cal flags 86% of all cross
   cases, and its recall exceeds a random flagger at that rate by only 0.02.
2. **FPR difference:** +0.72 (14/18 vs 1/18) against a 0.10 limit. **Kill criterion
   met.** All 14 false positives are `IVF-ORACLE-EVENT-COUNT-MISMATCH`. In open-loop
   replay, humanoids and ANYmal-D fall on one backend but not the other even when both
   experiments are clean, and IVF's event oracle fails on any occurrence mismatch
   regardless of tolerance.
3. **Localization:** 8/53 correct (0.15, Wilson [0.08, 0.27]), below the 0.50 bar and
   below the majority-category guess (0.26). **Kill criterion met.**

At matched specificity (D8), IVF vs composite is 5 IVF-only against 8 composite-only
detections (McNemar p = 0.58, difference CI [-0.10, 0.00]). IVF's 5 unique catches are all
`preset_not_applied`, detected only through the self-comparison guard (F2). The
composite's 8 unique catches are observation-order swaps and termination mismatches,
which only closed-loop evaluation exposes.

**Per platform.** GO2 is the only platform where registered IVF-cal is discriminative
(0/4 clean flagged). There, IVF and the composite are identical: 7/12 recall each, 0/4
FPR. On G1, H1 and ANYmal-D, IVF-cal flags 100% of clean cases.

**Same-backend positive control (40 defect, 13 clean).** Registered IVF-cal flags 40/40
defects and 13/13 clean cases. Every clean flag is `IVF-EXPERIMENT-SELF-COMPARISON`:
PhysX reruns are bit-identical and IVF refuses them by design. With that refusal counted
as not flagged, IVF gives 33/40 and 0/13, **case-for-case identical to the composite**.
The pipeline works; the positive control passes for both arms.

**Dev split** (developmental) shows the same pattern: IVF-cal 31/36 at 4/6 FPR; composite
22/36 at 0/6; without event oracles 20/36 at 0/6.

## 5. Failures IVF misses

These families were missed by IVF at every non-degenerate operating point, and by every
baseline too, on the cross-backend holdout:

| Family | Why IVF cannot see it | Missing capability |
|---|---|---|
| randomization_asymmetry | the effect is inside the calibrated cross-backend envelope; no control covers event/randomization config | declared-vs-effective randomization control (record applied masses and COM) |
| reset_velocity_dropped, reset_joint_offsets_ignored | the initial-state digest hashes the *requested* state, and the applied-state difference sits under the chaotic cross-backend envelope (caught 5/5 in the same-backend condition) | a reference-free, per-subject check that applied equals requested state |
| contact_capacity (MJWarp `nconmax` overflow) | no backend-health telemetry; the effect is small in flat walking | backend health probe (overflow counters) in the capture contract |
| termination_body_mismatch | termination config is not a control; the signal only fires when contact happens | termination/sensor semantics digest as a control |
| preset_not_applied | caught only when PhysX is bit-deterministic (F2) | declared-vs-runtime backend identity check (IVF's documented blind spot #2) |
| actuator_gain_scale on ANYmal-D | a physical no-op: the actuator network ignores stiffness and damping | config-effectiveness probe; no comparison can see it |

Structural limits, measured here:

- **Open-loop replay is the wrong instrument for cross-backend locomotion.** Clean
  PhysX vs Newton pairs diverge by up to 1.9 rad in joint position and 4.2 m in base
  position within 5 s. Calibrated tolerances therefore swallow most real defects, and the
  event oracle flags clean pairs. Newton open-loop replay is also not reproducible run to
  run in this stack: a fault with no physical effect in open loop still differed from
  Newton clean by 0.05 to 0.97 rad.
- **The classifier cannot localize cross-backend.** Rule 1 ("a recorded configuration
  difference explains the divergence") fires on every cross-backend pair, because solver
  settings always differ. Of 53 holdout detections, 18 mapped to the solver category and
  12 to termination (fall-event mismatches), largely independent of the true cause; all
  10 joint-order detections were called solver differences. The vocabulary has no joint-ordering, checkpoint or
  termination category.
- **No joint-order control.** `capture_not_canonicalized` (arrays in a different joint
  order, names truthfully recorded) is flagged only as a numerical divergence, never as
  an invalid comparison.

## 6. Ablation: which IVF components contribute (holdout cross)

| IVF variant | Recall | FPR |
|---|---|---|
| full | 53/60 | 14/18 |
| event + decision oracles only | 49/60 | 14/18 |
| trajectory oracles only | 28/60 | 1/18 |
| without event + decision (D8) | 28/60 | 1/18 |
| joint_pos and joint_vel oracles only | 24/60 | 0/18 |
| statistical oracle only | 17/60 | 0/18 |
| validity layer only | 10/60 | 0/18 |
| without the 14 declared controls | 53/60 | 14/18 |
| without the `policy_obs` oracle | 53/60 | 14/18 |
| without the `joint_pos_target` oracle | 53/60 | 14/18 |

The event/decision oracles account for nearly all recall and all false positives. The
declared-control layer adds nothing measurable (removing it changes no flag decision;
4 verdict labels change, for example `INVALID_EXPERIMENT` to `FAIL`); its 10
detections are 5 timestep cases (also caught by B2) and 5 self-comparison refusals. The
trajectory oracles perform like simple RMSE on the same signals (28/60 at 1/18 vs B3all
21/60 at 0/18 and composite 31/60 at 1/18).

## 7. Findings outside the pre-registered metrics

- **F1. Native joint-order mismatch.** In this Isaac Lab checkout, Newton/MJWarp orders
  joints per limb (depth first) and PhysX breadth first, on all four platforms, with no
  remap option. A PhysX-trained GO2 policy run on Newton through the stock path had 4.4
  base-contact terminations per env and return -3.4. With a name-based remap it had 0 and
  35.0 (PhysX: 0 and 38.5). This is the documented sim-to-sim failure, occurring by
  default. Closed-loop evaluation catches it loudly. IVF flags it, but classifies it as a
  solver difference.
- **F2. The native Newton smoke test silently ran PhysX.** Before Isaac Lab PR #7103
  (2026-08-17), `test_environments_newton.py` applied the Newton preset after the config
  was already resolved, so the "Newton" test instantiated `PhysxManager` (verified at
  runtime in this checkout). Native validation accepts it; B2/B3 accept it (the physics
  is identical); IVF refuses it only because deterministic PhysX makes the arrays
  bit-identical. That is an accident of determinism, not a backend check.
- **F3. IVF's own evidence carries a quaternion mislabel.** The shipped cart-pole bundles
  store the identity orientation as `[0,0,0,1]`, which is `(x,y,z,w)` per the Isaac Lab 3
  API, while declaring layout `wxyz`. Read as declared, that is a 180-degree yaw. IVF
  passed it (V-15 pass, unit-norm pass), because both subjects carry the same mislabel.
  When one side was relabelled correctly, IVF returned `INVALID_EXPERIMENT` but cannot
  say which declaration is true (`results/internal_quaternion_layout.json`). IVF was not
  changed.
- **F4. Contract friction.** A geodesic tolerance in `rad` on a quaternion declared
  `dimensionless` is refused (D6). A bit-identical independent rerun is refused as
  self-comparison (D2).
- **Synthetic taxonomy (IVF's home ground):** IVF 45/48, B1 or B3all 42/48, both 0/3
  FPR. The only difference is `corrupted_metadata`. IVF's classification matched the
  category in 15/33 flagged trials.

## 8. Does the evidence support a publishable claim?

**The target claim: no.** At matched specificity IVF does not detect more than
conventional sim-to-sim evaluation (28 vs 31 of 60), it detects nothing that evaluation
misses except through a determinism artifact, and it does not localize (0.15). Most
cross-backend defects that conventional evaluation accepts (29/60) are accepted by IVF
too at any usable operating point. Native validation (B1) caught 0 of 60.

**What is defensible:**

1. An empirical negative result with a real corpus: externally sourced, frozen tool,
   sealed holdout, unseen platform. It shows that pairwise open-loop trajectory
   comparison across physics backends is dominated by legitimate chaotic divergence, so
   acceptance tools built on it cannot separate migration defects from backend
   differences for legged locomotion.
2. The corpus and harness themselves: 13 families tied to 31 documented cases, a
   producer with explicit joint-order canonicalization, and two native failures found in
   a current Isaac Lab stack (F1, F2).
3. A capability list (section 5) naming what a validator would need: reference-free
   per-subject checks (applied vs requested state, declared vs runtime backend,
   randomization and termination semantics digests, backend health counters, a joint-order
   control), and closed-loop evaluation kept as a first-class signal rather than replaced.

Appropriate venue: a workshop paper or technical report framed as "why cross-backend
acceptance testing fails, and what it needs". Not a claim that IVF works. Per the
critical rule, IVF was not changed to chase these cases. The next step would be building
the section 5 capabilities, then re-running this frozen holdout unchanged.

**Limits of this study.** One Isaac Lab snapshot, one workstation GPU, flat terrain, one
policy per platform, 5 s open-loop horizon, 16 open-loop and 64 closed-loop envs, and 5
holdout cases per family. The consequence labels depend on thresholds that are near zero
in the deterministic same-backend condition. The perturbations reproduce documented
failure *mechanisms*; except F1, F2 and the GO2 armature split, they are not literal
replays of the original incidents.

## 9. Reproduce

```bash
# captures (GPU, Isaac Lab interpreter); data lives outside the repo
research/failure_corpus/capture/train_policies.sh            # optional; policies are committed
python research/failure_corpus/corpus/make_plans.py
$ISAACLAB_PYTHON research/failure_corpus/capture/corpus_capture.py --plan research/failure_corpus/plans/<split>__<platform>.json --out $IVF_CORPUS_DATA/captures/<split>
# evaluation (CPU)
uv run --frozen python research/failure_corpus/eval/run_ivf.py --split calibration --arm permissive
uv run --frozen python research/failure_corpus/eval/calibrate_thresholds.py
uv run --frozen python research/failure_corpus/eval/run_ivf.py --split holdout --arm cal --ablations full minus_event_decision ...
uv run --frozen python research/failure_corpus/eval/score.py --split holdout
uv run --frozen python research/failure_corpus/eval/synthetic.py
```

Raw bundles (about 300 MB) are not committed. Per-case IVF summaries, thresholds,
`cases.csv` and `scores.json` are, and `results/capture_manifest.json` records every
bundle's finalized root hash.
