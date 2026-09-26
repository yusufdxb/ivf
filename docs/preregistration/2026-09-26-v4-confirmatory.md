# Pre-registration: final confirmatory evaluation (v4)

Registered 2026-09-26, before the v4 fault author was started and before any v4 holdout
capture. This is the last round: development stops after it regardless of outcome.

## Claim (frozen)

> Adding IVF configuration/state verification checks to conventional sim-to-sim validation
> reduces false acceptance without materially increasing false alarms.

The hypothesis comes from v3 (post-hoc observation that IVF's input checks raised no false
alarms while its trajectory oracle did). The v3 holdout is not rescored or reused.

## Frozen objects

- IVF source: `src/ivf` tree `becf39d2b1c1264d1b6330ca66b79390640985c9` (unchanged since the
  v3 freeze). `research/failure_corpus/v4/FROZEN_V4.txt`.
- **IVF arm** (`research/failure_corpus/v4/eval/v4.py`): a case is flagged if and only if at
  least one of these checks fails: V-24 reset state actually applied, V-25 runtime backend,
  V-28 solver settings vs intended preset, V-22 realized actuator/model/material
  parameters, V-27 randomization terms, V-20 joint ordering, V-21 policy-input semantics
  (observation layout, joint order, action scale), V-23 live termination configuration.
  No trajectory, event, decision, statistical or invariant oracle contributes. Realized-value
  tolerance 1e-4 (all clean v3 calibration pairs agreed exactly). Intended presets:
  `research/failure_corpus/v4/intended_presets.json`.
- **Conventional arm:** native smoke OR closed-loop performance OR open-loop trajectory RMSE,
  thresholds 1.25 x worst clean cross-backend calibration value (seeds 0 to 2), unchanged
  from v3.
- **Union:** conventional OR IVF.
- Producer: `research/failure_corpus/v4/capture/corpus_capture_v4.py` (v3 producer plus the
  Rough GO2 task; recording logic unchanged).

## Holdout

- New independent fault author; may not read IVF, any harness, any earlier holdout's faults,
  labels or results, or any report. Earlier mechanisms are excluded by name. New fault
  implementations only.
- Benign controls must be demonstrably contract-preserving (static config diff limited to
  visualization, logging, rendering, or unused observation groups); any benign family that
  fails this is dropped before sealing.
- Robots/tasks: GO2, G1, H1, ANYmal-D, Cassie, Spot (flat) and **Rough-terrain GO2** (new
  task). Seeds 15 and 16 (never used). Per (robot, seed): Newton clean plus every family.
- Faults and labels sealed by hash before capture. Scored exactly once. Nothing is tuned
  after scoring.

## Success criteria (all three required)

1. False acceptance reduced by at least 0.10: FA(conventional) minus FA(union) >= 0.10 on
   the defect cases.
2. The 95% CI for that reduction excludes 0.
3. The union adds at most 1 false alarm over conventional on the clean cases.

## Statistical analysis (fixed now)

Defect cases are not independent: each fault family appears on up to 7 robots/tasks x 2
seeds, and cases of one family share a mechanism. Therefore:

- **Primary CI:** percentile interval from a **cluster bootstrap over fault families**
  (10,000 resamples; each resample draws families with replacement and keeps all their
  robot/seed cases). The reduction is computed paired within the resampled cases.
- **Sensitivity (reported, not decisive):** cluster bootstrap over robots/tasks, and a
  two-way bootstrap that resamples families and robots independently (cells kept with
  multiplicity).
- False alarms are counted, not modelled: criterion 3 is a count.
- Descriptive per-arm rates carry Wilson intervals, labelled as ignoring clustering.

## Reported

Recall, false acceptance and false alarms for conventional, IVF and the union; the both /
IVF-only / conventional-only / neither table overall and by category; categories and IVF
checks behind unique catches; the clustered uncertainty above; per robot and per family.

## Decision

- **Pass:** stop development; prepare the paper around IVF as orthogonal configuration/state
  verification that complements behavioral validation.
- **Fail:** stop IVF research development; keep the project as a negative technical report
  and a tool.
