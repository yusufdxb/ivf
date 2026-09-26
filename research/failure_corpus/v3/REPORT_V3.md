# Technical report: IVF experimental-validity checks added to conventional sim-to-sim validation

> **Archived round record.** Results below stand as recorded for this round. The authoritative research conclusion is [`../v4/REPORT_V4.md`](../v4/REPORT_V4.md); where this report differs, v4 governs.

2026-09-26. Branch `research/failure-corpus-eval`, local only.
Pre-registration: [`docs/preregistration/2026-09-26-v3-holdout.md`](../../../docs/preregistration/2026-09-26-v3-holdout.md).
Earlier rounds: [v1](../REPORT.md), [v2](../v2/REPORT_V2.md). Deviations: [`../DEVIATIONS.md`](../DEVIATIONS.md).

## Result

**The pre-registered claim was not reproduced.** The claim was:

> Adding IVF experimental-state/configuration checks to conventional sim-to-sim validation
> reduces silent false acceptance without increasing false alarms.

On a fresh, independently authored holdout (144 defects, 72 clean cases, six robots
including one never used before), the union **did** reduce false acceptance: from 0.354
to 0.104, a drop of 0.25 (95% CI [0.22, 0.28]). But it added **5** false alarms over
conventional validation, and the rule allowed at most 1. Under the decision rule fixed
before the holdout was written, development stops here and this is the negative report.

| Arm | Defect recall | False acceptance | False alarms |
|---|---|---|---|
| Conventional (native OR closed-loop performance OR trajectory RMSE) | 93/144 (0.646) | 0.354 | 1/72 |
| IVF | 92/144 (0.639) | 0.361 | 5/72 |
| **Union** | **129/144 (0.896)** | **0.104** | **6/72** |

| Criterion (pre-registered) | Result | Met? |
|---|---|---|
| Union false acceptance at least 0.10 below conventional, CI excluding 0 | 0.25 [0.22, 0.28] | yes |
| Union adds at most 1 false alarm | adds 5 | **no** |
| Decision | not reproduced: negative report | |

## Where the false alarms came from

All 5 IVF false alarms are the same robot and seed: **Spot, seed 13**. They include the
plain Newton clean case and four benign controls, and every one fails the same oracle,
`traj_root_link_pos_w`. That is the open-loop trajectory comparison from v1, not one of
the experimental-state/configuration checks. On that seed, clean PhysX and Newton Spot
diverged in base position beyond the envelope calibrated from three clean seeds. On the
other seed (14), Spot raised no false alarm, and the other five robots raised none at all.

The experiment-input checks themselves (reset realization, backend identity, solver
conformance and saturation, joint order, policy interface, effective model, randomization,
termination semantics, timing) produced **0 false alarms on 72 clean cases**.

This diagnosis does not change the result: the frozen IVF includes the trajectory oracle,
and the pre-registered rule applies to IVF as frozen.

## Overlap (144 defects)

| | Count |
|---|---|
| both | 56 |
| IVF only | 36 |
| conventional only | 37 |
| neither | 15 |

| Category | both | IVF only | conventional only | neither |
|---|---|---|---|---|
| actuator/solver config | 2 | 11 | 9 | 14 |
| checkpoint/schema | 22 | 2 | 0 | 0 |
| ineffective/stale config | 5 | 1 | 5 | 1 |
| joint/body ordering | 12 | 0 | 0 | 0 |
| obs/action ordering | 1 | 0 | 23 | 0 |
| reset/randomization | 1 | 11 | 0 | 0 |
| termination/metric | 1 | 11 | 0 | 0 |
| timestep/decimation | 12 | 0 | 0 | 0 |

**What only IVF caught (36).**

| Family | Caught by | Count |
|---|---|---|
| reset sample broadcast to all envs | reset realization | 11 |
| termination reads only the newest contact sample | live termination semantics | 11 |
| PhysX iteration counts used as the MJWarp solver budget | solver conformance | 10 |
| other | trajectory divergence | 4 |

The last two checks were added in v3.

**What only conventional validation caught (37).** Closed-loop effects: the last-action
observation slot scaled, an observation term hoisted to the front, a joint-range
enforcement change, and contact-offset margin inflation.

**What neither caught (15).** Mostly actuator/solver effects too small to move closed-loop
performance: contact offset as shape margin, and joint friction loss present only in the
candidate. IVF records neither shape margin nor joint friction loss.

## Attribution (secondary)

- Coverage (flagged defects with a named cause): 48/92 = 0.52.
- Precision (named cause matches the author's label): 24/48 = 0.50.
- Top-1 over all detections: 24/92 = 0.26, against a majority-category guess of 0.14.

Localization remains weak and is not a claim.

## Exploratory, post hoc (cannot change the decision)

Using only IVF's experiment-input checks, computed from the same scored outputs, with no
recapture:

| Arm | Recall | False alarms |
|---|---|---|
| input checks alone | 48/144 | 0/72 |
| conventional + input checks | 125/144 | 1/72 |

The second arm would reduce false acceptance by about 0.22 (CI [0.19, 0.24]) at the same
false-alarm count as conventional validation. This subset was chosen after seeing the
result, so it is a hypothesis for a future pre-registered test, not a finding.

## Other findings from this round

- **Spot's stock config applies material randomization only on PhysX.** Its sampled
  static friction range (0.3 to 1.0) is labelled "PhysX-only startup randomization", so a
  stock Newton Spot experiment silently drops it. IVF flagged all 3 clean Spot calibration
  pairs (`IVF-CONTROL-RANDOMIZATION-MISMATCH`; realized friction 0.69 vs 1.0). This is the
  same class as Isaac Lab issue #7786 (base-COM randomization disabled on Newton). For the
  holdout, sampled startup material randomization was disabled on both backends so that
  "clean" was like-for-like.
- **Every robot tested has a different joint order on Newton and PhysX in this Isaac Lab
  checkout.** Spot and Cassie join GO2, G1, H1 and ANYmal-D.
- **Some readback values have to be read carefully.** PhysX and Newton assign collision
  shapes to links differently, so realized friction is compared as a per-env distribution
  over collision shapes, not per body. MuJoCo-Warp's `opt.timestep` reads as MuJoCo's
  default until the first step, so substeps come from Newton's manager.

## Integrity notes

- IVF source was frozen (`becf39d`) before the fault author started, and never changed.
- The holdout faults and labels were sealed and verified by hash before scoring. They were
  scored once.
- The producer received three fixes after sealing, each found on clean calibration data
  only, each logged: material readback, Spot base body, and Spot's sampled material
  randomization.
- Four benign Spot runs were re-captured after an out-of-memory crash, before any scoring.
- The fault author was an agent of the same model family, separated by instructions and
  sealed files, not an independent human. It disclosed that two of its families resemble
  earlier mechanisms; both were kept as delivered.

## What this supports, stated plainly

- In v2 and v3, IVF's experimental-state/configuration checks caught failures that
  closed-loop sim-to-sim evaluation accepted: reset realization, termination semantics,
  solver configuration, realized actuator models. In both rounds every IVF false alarm came
  from a trajectory oracle, not from these checks. In v3 they raised 0 false alarms on 72
  clean cases. (v1 had no such checks.)
- IVF as frozen, including its open-loop trajectory oracles, raised false alarms on one
  robot/seed. The pre-registered combined claim therefore failed.
- IVF does not replace conventional validation: each arm alone accepts about 35% of defects.
- A future round would have to pre-register the input-checks-only configuration and test
  it on a new holdout. Per the decision rule, that is not done here.
