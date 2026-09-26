# Final confirmatory evaluation (v4): IVF configuration/state checks + conventional validation

> **Status: authoritative final research conclusion.** Research development stopped 2026-09-26; archived as tag `research-archive/failure-corpus-v4-final` (commit `89398cb`). Index: [`../README.md`](../README.md).

2026-09-26. Branch `research/failure-corpus-eval`, local only. This is the last round.
Pre-registration: [`docs/preregistration/2026-09-26-v4-confirmatory.md`](../../../docs/preregistration/2026-09-26-v4-confirmatory.md).
Earlier rounds: [v1](../REPORT.md), [v2](../v2/REPORT_V2.md), [v3](../v3/REPORT_V3.md). Deviations: [`../DEVIATIONS.md`](../DEVIATIONS.md).

## Result

**The pre-registered claim failed.** Per the decision rule fixed before the holdout was
written, IVF research development stops, and the project is kept as a negative technical
report and a tool.

> Claim: Adding IVF configuration/state verification checks to conventional sim-to-sim
> validation reduces false acceptance without materially increasing false alarms.

| Criterion (all required) | Result | Met? |
|---|---|---|
| False-acceptance reduction ≥ 0.10 | 0.548 → 0.423, reduction **0.125** | yes |
| 95% CI excludes 0 (family-clustered bootstrap, 12 families) | **[0.000, 0.304]** | **no** |
| Union adds at most 1 false alarm | adds **0** | yes |

The point estimate clears the bar and the union adds no false alarms, but the whole gain
comes from 2 of the 12 fault families. When families are resampled, as pre-registered, the
reduction is consistent with zero.

## Numbers (168 defects in 12 families, 84 clean cases, 7 robot/tasks, seeds 15 and 16, scored once)

| Arm | Recall | False acceptance | False alarms |
|---|---|---|---|
| Conventional (native OR closed-loop performance OR open-loop trajectory RMSE) | 76/168 (0.452) | 0.548 | 12/84 |
| IVF config/state checks only | 28/168 (0.167) | 0.833 | **0/84** |
| Union | 97/168 (0.577) | 0.423 | 12/84 |

Per-arm Wilson intervals, which ignore clustering: conventional false-alarm rate [0.08,
0.23]; IVF [0.00, 0.04].

### Uncertainty (pre-registered)

| Analysis | 95% CI for the reduction | Clusters |
|---|---|---|
| Family-clustered bootstrap (**primary**) | [0.000, 0.304] | 12 families |
| Robot/task-clustered bootstrap (sensitivity) | [0.089, 0.155] | 7 robots/tasks |
| Two-way family × robot (sensitivity) | [0.000, 0.310] | |

The robot-clustered interval excludes 0 only because every robot sees the same two reset
families. The primary analysis treats family as the unit, since cases within a family share
one mechanism, and it does not exclude 0.

### Overlap

| Category | both | IVF only | conventional only | neither |
|---|---|---|---|---|
| actuator/solver config | 0 | 0 | 10 | 18 |
| checkpoint/schema | 0 | 0 | 2 | 12 |
| ineffective/stale config | 0 | 0 | 9 | 5 |
| joint/body ordering | 0 | 0 | 17 | 11 |
| obs/action ordering | 0 | 0 | 14 | 0 |
| reset/randomization | 7 | 21 | 0 | 0 |
| termination/metric | 0 | 0 | 8 | 20 |
| timestep/decimation | 0 | 0 | 9 | 5 |
| **total** | **7** | **21** | **69** | **71** |

- **IVF-only catches (21): all reset/randomization.** They come from two families,
  `reset_offset_scale_conflation` and `reset_world_index_shift` (14/14 each). The
  reset-realization check (V-24) flagged all 28 IVF detections. The randomization check
  (V-27) also flagged 12 of them.
- **IVF caught nothing in the other 10 families.** Most of those faults act on tensors at
  run time or on quantities IVF does not read back:
  - action clamping to the joint range;
  - bf16 policy input/output rounding;
  - observation noise left on;
  - a stale joint-velocity observation;
  - a mirrored gear sign;
  - velocity taken at the root link;
  - a rotated command;
  - unclamped torque limits (effort limits are not in the recorded model);
  - an inert contact termination swapped at run time (the live config is unchanged);
  - a relaxed success metric.
- **Conventional-only catches (69)** cover every other category.
- **Neither (71)** is large this round. Most of these faults either have small effects in
  closed loop or change quantities neither arm observes.

### False alarms

IVF raised none on 84 clean cases, consistent with v3. All 12 conventional false alarms
come from the closed-loop performance comparison, 6 each on G1 and on Rough GO2, and
include the plain Newton clean case. The union therefore adds none.

## What four rounds support

1. **IVF does not replace conventional sim-to-sim validation.** In every round, closed-loop
   evaluation catches failures IVF cannot see.
2. **IVF's configuration/state checks are orthogonal and quiet.** They raised 0 false
   alarms on 84 clean cases here (pre-registered) and 0 on 72 in v3 (computed post hoc). When a failure's mechanism is a declared input
   they record, they catch it:
   - v2 and v3: reset realization, realized actuator models, solver configuration, and
     termination semantics;
   - v4: reset realization and randomization.
3. **Their benefit depends on the failure mix.** On the v3 holdout, adding them cut false
   acceptance by 0.25. On this holdout the benefit came from 2 of 12 families, and under
   the pre-registered family-clustered analysis it is not distinguishable from zero. The
   claim that adding them reliably reduces false acceptance is **not supported**.
4. **Localization** stayed weak throughout and was never a claim.

## Integrity

- **Frozen objects:** IVF source (`becf39d`), the eight-check IVF arm, the conventional
  baseline, thresholds, the success criteria and the clustered analysis were all fixed and
  committed before the fault author started (`112f203`) or before any capture (`cda56f1`).
- **Holdout author:** a new independent author, told not to read IVF, the harnesses, any
  earlier holdout's faults or labels, or any report. 12 defect and 5 benign families. All 5
  benign static diffs were visualization or unused-group paths, so none were dropped.
- **Seals:** faults and labels were sealed by hash (`be90f07`) and re-verified before
  scoring.
- **Infrastructure events, all before scoring and all logged:**
  - a host-initiated stop for low system memory (220/266 captured);
  - a resume that filled only unfinished runs;
  - a G1 PhysX read failure and a partial H1 run, both retried;
  - a Rough GO2 GPU-memory exhaustion, retried in bounded fresh processes.
  All 266 runs completed; none were excluded. Unfinished runs were identified by capture
  ID only.
- **Scoring:** exactly once. Nothing was tuned or rescored afterwards.
- **Independence limit:** the fault authors were agents of the same model family, separated
  by instructions and sealed files, not independent humans.

## Decision

**Fail.** IVF research development stops. The project stays as:

- a negative technical report (v1 to v4);
- the tool, whose configuration/state checks are a quiet, orthogonal supplement but not a
  demonstrated reliable improvement;
- the corpus and harness;
- the documented findings:
  - PhysX and Newton joint orders differ on every robot tested;
  - the pre-#7103 Newton smoke test silently ran PhysX;
  - Spot's stock config applies material randomization on PhysX only;
  - IVF's own cart-pole evidence mislabels its quaternion layout.
