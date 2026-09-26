# Deviations from the pre-registration

Every change to the registered design, in order. D1 to D5 were made after a smoke run on
**clean GO2 calibration captures only** (seeds 0 to 2) and before any fault case was
captured or any IVF or baseline decision was computed on any case.

## D1. Fall event uses tilt, not base height

Registered: fall = `base_height` below a per-platform threshold (GO2 0.15 m).
Observed on the clean PhysX reference: the trained GO2 policy walks upright (roll and pitch
near zero, vx about 0.45 m/s) in a low crouch with base height about 0.14 m, so an absolute
height threshold labelled 15 of 16 healthy walking envs as fallen.
Change: new recorded signal `upright` = minus the z component of gravity in the base frame
(1 upright, 0 horizontal). Fall event and survival decision = `upright` below 0.5 (tilt
beyond 60 degrees). `base_height` stays as a trajectory signal only.

## D2. Producer fills the v1 `capture` block

Observed: PhysX A/A reruns are bit-identical, so bundles without a capture id were
byte-identical and IVF refused them as self-comparison (`IVF-EXPERIMENT-SELF-COMPARISON`).
The v1 contract carries `capture.capture_id` and `capture.created_utc`, which a faithful
producer fills. Change: the producer writes a UUID and UTC timestamp.
**Correction (found while testing the harness, still before any fault case):** this does
not change IVF's behavior. IVF's self-comparison guard (`V-19`) hashes the signal arrays,
not the bundle, so a bit-identical independent PhysX rerun is still refused as
`IVF-EXPERIMENT-SELF-COMPARISON`. Under the registered template every same-backend clean
case on deterministic PhysX is therefore flagged by IVF. This is reported as measured,
with a sensitivity analysis that counts the self-comparison refusal as "not flagged".

## D3. Native joint-order mismatch; "clean" redefined

Observed and verified at runtime: in this Isaac Lab checkout Newton/MJWarp enumerates joints
per limb, depth first (`FL_hip, FL_thigh, FL_calf, FR_hip, ...`) while PhysX enumerates
breadth first (`FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ...`). This holds on GO2, G1, H1
and ANYmal-D (`research/failure_corpus/policies/*/joint_orders.json`). `robot.joint_names`
agrees with the data on each backend; the orders simply differ, and the checkout has no
ordering remap option. A PhysX-trained GO2 policy run on Newton through the stock path
collapsed (4.4 base-contact terminations per env, return -3.4). With a name-based remap it
walked (0 terminations, return 35.0 vs 38.5 on PhysX).

This is the externally documented failure (Isaac Lab sim-to-sim doc, PR #6913, issue #6485)
occurring natively, so the registered "clean Newton" candidate was itself defective.
Change: every job remaps canonical (PhysX training) order to the native order at the policy
interface and records joint arrays in canonical order, declaring it in `task.joint_names`.
The joint-order family becomes `joint_order_interface`: cross dev = native no-remap (the
stock behavior of this checkout), same dev = the Newton enumeration applied on PhysX,
holdout = a left/right-swapped remap table.

## D4. New family `capture_not_canonicalized` (joint_body_ordering, cross only)

Physics correct (remap applied), but joint arrays are recorded in Newton's native order
with truthful native names. The comparison with a PhysX-order baseline is then invalid
index-wise. Tests whether IVF notices incomparable joint orders. Added because D3 showed
the situation arises by default in this checkout.

Also under D3/D4: the registered `armature_dropped` family is renamed `armature_mismatch`.
Only GO2 has a backend-conditioned armature preset here (PhysX 0.0, Newton 0.02), so the
controlled protocol pins GO2 armature to its training value on both backends, and the fault
sets armature to 0.02 where training used 0 (GO2, ANYmal-D) and to 0 where training used a
nonzero value (G1, H1). Setting 0 on ANYmal-D would have been a no-op.

## D5. More clean holdout cases

Registered holdout had 10 clean cross cases and 5 clean same cases. Added clean-only
seeds 5 and 6 on all four platforms (Newton clean and PhysX rerun per seed), so the holdout
FPR estimate rests on 18 clean cross and 13 clean same cases.

## D6. Quaternion geodesic tolerance declared `dimensionless`

Found while testing the harness on clean calibration pairs. IVF's unit contract compares
the tolerance unit with the signal's declared unit. The root quaternion is (correctly)
declared `dimensionless`, while the geodesic metric reports radians, so a tolerance in
`rad` is refused with `IVF-CONTROL-TOLERANCE-UNIT-MISMATCH` and the whole case becomes
`INVALID_EXPERIMENT`. There is no way to state "geodesic radians over a dimensionless
quaternion" in the contract. The template declares the geodesic tolerance as
`dimensionless` with the scope text naming radians. Recorded as an IVF usability finding.

Label hashes after D1 to D5 are in `corpus/LABEL_HASHES.txt` and were committed before any
dev or holdout capture.

## D7. Added comparator: simple RMSE over IVF's own signal set (exploratory)

Registered B3 compares only `joint_pos` and `root_link_pos_w`. IVF's template also watches
`joint_pos_target`, `policy_obs` and others, so an IVF advantage over B3 could come from
the signal set rather than from IVF. Added `B3all` (RMSE on all seven IVF trajectory
signals, same 1.25 x calibration rule) and `composite_all` (B1 or B2 or B3all). Both are
labelled exploratory. The primary comparator stays the registered composite. Also added,
as a labelled sensitivity analysis, IVF-cal with the V-19 self-comparison refusal counted
as not flagged (see D2 correction). Added before any fault case was evaluated.

## D8. Post-hoc exploratory ablation `minus_event_decision` (after holdout scoring)

Added after the holdout was scored, because every IVF-cal false positive on a clean case
came from `IVF-ORACLE-EVENT-COUNT-MISMATCH`. It removes the fall/contact event oracles and
the survival decision. It is post hoc and exploratory, and it does not change any
registered result. Also added after scoring: the dose-matched sham computation and the
per-platform breakdown, both reporting-only.

## v2 notes

- The v1 corpus was re-declared development data after inspection (v2/OVERLAP.md).
- IVF changes (`1a7cced`) were made against v1 mechanisms, then frozen (`48861c1`) before
  any v2 development evaluation. The independent fault author's completion summary
  (family names and mechanisms) was seen before that evaluation; IVF source has not
  changed since the freeze.
- The producer's armature pinning was generalized (copy from the resolved PhysX config)
  while the development recapture was running; it yields identical values on the four
  existing platforms and was needed for Cassie.
- Known producer instrumentation bug found in the holdout analysis: the recorded
  termination threshold and contact flag use a constant 1.0 instead of the live
  termination config. Not fixed; reported.

## v3 notes

- IVF source frozen at tree `becf39d` (commit `582c9e7`) before the v3 fault author started;
  unchanged since.
- **Producer fix after the v3 faults were sealed (clean calibration data only).** The first
  v3 calibration showed clean PhysX and Newton disagreeing on per-body friction for G1, H1,
  ANYmal-D and Cassie (relative 1.8), because the two backends assign collision shapes to
  links differently (PhysX reported no collision shape on bodies where Newton has one).
  That would have calibrated the effective-model tolerance to 2.25 and disabled the check.
  The producer now records, per environment, the mean/min/max static friction and
  restitution over all robot collision shapes (Newton restricted to shapes with the
  collide flag). Verified: clean G1 PhysX vs Newton identical (0.8 / 0.0); the GO2
  positive control (material event removed) still reads 1.0 vs 0.8. The earlier
  calibration captures were set aside, not used.
- Thresholds are frozen per robot before that robot's holdout cases are captured (five
  robots first, Spot after its training finishes).
- **Spot producer fixes (clean calibration data only, after the v3 faults were sealed).**
  (1) The base-body lookup hardcoded `pelvis` for platforms other than GO2 and ANYmal-D;
  Spot's base is `body` (all 9 Spot calibration jobs raised). (2) Spot's stock config
  applies sampled material randomization (static friction 0.3 to 1.0) as "PhysX-only
  startup randomization", so stock Newton Spot silently drops it; IVF flagged all 3 clean
  Spot pairs (`IVF-CONTROL-RANDOMIZATION-MISMATCH`, realized friction 0.69 vs 1.0). This is
  a real in-tree backend-conditioned randomization difference, reported as a finding. To
  keep "clean" like-for-like, the controlled protocol now disables startup material
  randomization when its ranges sample (as it already did for mass/COM); fixed materials
  (the other five robots) are unchanged. After this, 18/18 clean cross pairs pass.
- **Infrastructure retry during v3 holdout capture (before any scoring).** One Spot run (a
  benign control, `benign_command_arrow_debug_vis_off`, seed 14) hit a GPU out-of-memory
  error late in a long capture process, and the following CUDA error killed the process
  before three further benign Spot runs started. Those four runs were re-captured
  unchanged; the failed attempt is kept outside the repo. No holdout case had been
  evaluated at that point.

## v4 notes (final confirmatory round)

- IVF source unchanged since the v3 freeze (`becf39d`). The v4 IVF arm is the eight
  configuration/state checks only, frozen in `v4/eval/v4.py` before the fault author started.
- Rough GO2 was trained on PhysX with 1024 envs (not 4096) to avoid starving another
  session's GPU job. Its first run was stopped by me for that reason; the stop command also
  matched its own shell (a known `pkill -f` pitfall), without affecting the other job.
- Rough GO2 clean calibration seed 2 failed on PhysX (`Failed to get DOF velocities from
  backend`) and was re-captured unchanged; clean calibration data only.
- The thresholds command was made to skip calibration cases not yet captured; flat-robot
  thresholds equal the v3 values exactly.
- **Infrastructure interruption during v4 holdout capture (before any scoring).** The first
  capture pass was stopped by the host (Claude Code reaped the background process because
  system memory was critically low while the session was idle) at 220 of 266 runs. Resumed
  from the frozen state (`cda56f1`: faults, labels, IVF tree `becf39d`, thresholds and
  pre-registration unchanged, all hashes re-verified). Only unfinished runs were captured;
  completed runs were skipped by the producer. Unfinished runs were identified by capture id
  without opening labels or outcomes: one G1 run whose capture raised
  `Failed to get DOF velocities from backend` (the same transient PhysX read failure seen and
  cleared on retry in Rough GO2 calibration), one partial H1 run, one partial Spot run, six
  unstarted Spot runs, and 38 unstarted Rough GO2 runs. The failed and partial directories were
  moved aside, not deleted, and re-captured as infrastructure retries. The resume ran under a
  memory watchdog that stops the capture if available memory falls below 4 GiB.
