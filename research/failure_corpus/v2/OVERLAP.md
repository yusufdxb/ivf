# Detection overlap on the 60 cross-backend defects (v1 holdout, now development data)

Computed from `research/failure_corpus/results/holdout/cases.csv` at commit `c59eb34`, with
no re-run. "Conventional" is the registered composite (native smoke OR closed-loop
performance OR open-loop trajectory RMSE). From this point the 60-case corpus is
**development data**: its failures have been inspected, so it is no longer a holdout.

## Registered IVF-cal (flags 14/18 clean cases)

| | Count |
|---|---|
| both | 28 |
| IVF only | 25 |
| conventional only | 3 |
| neither | 4 |

20 of the 25 IVF-only catches are `IVF-ORACLE-EVENT-COUNT-MISMATCH`, the same signal that
fires on 14 of 18 clean cases, so they are not evidence of detection. The other 5 are
`preset_not_applied` via the self-comparison guard.

## IVF at matched false-alarm rate (event/decision oracles removed; FPR 1/18, same as conventional)

| Category | both | IVF only | conventional only | neither |
|---|---|---|---|---|
| actuator/solver config | 3 | 0 | 1 | 11 |
| checkpoint/schema | 5 | 0 | 0 | 0 |
| ineffective/stale config | 0 | 5 | 0 | 0 |
| joint/body ordering | 10 | 0 | 0 | 0 |
| obs/action ordering | 0 | 0 | 5 | 0 |
| reset/randomization | 0 | 0 | 0 | 10 |
| termination/metric | 0 | 0 | 2 | 3 |
| timestep/decimation | 5 | 0 | 0 | 0 |
| **total** | **23** | **5** | **8** | **24** |

## Decision

- **Not (1).** IVF's only orthogonal detections are the 5 `preset_not_applied` cases, and
  they depend on PhysX being bit-deterministic, not on a check designed for the failure.
- **(2) holds.** On the 36 cases either arm catches, IVF catches 28 and conventional 31;
  IVF's trajectory evidence overlaps conventional trajectory and performance evidence.
- **(3) holds.** Localization was 8/53, below a majority-category guess.

The 24 "neither" cases (reset/randomization, actuator/solver parameters, termination
semantics) and the preset case are all failures where the *physics comparison cannot
separate the defect from legitimate backend difference*, while the experiment's declared
inputs are wrong. That is the gap an experimental-validity layer should fill, and it
justifies implementing the Section 5 capabilities as general checks.
