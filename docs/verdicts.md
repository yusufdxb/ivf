# Reading a verdict

## The six verdicts

| Verdict | Exit | Means | What to do |
|---|---|---|---|
| `PASS` | 0 | every declared control held and every acceptance criterion was met | accept, for this workload and these criteria |
| `FAIL` | 1 | at least one criterion failed on adequate evidence | read the first divergence and its classification |
| `INCONCLUSIVE` | 2 | the run completed but the evidence cannot decide | add samples, or widen a margin *with a rationale* |
| `UNSUPPORTED` | 3 | the combination needs a backend or runtime feature that is not available | fix the environment; nothing was learned about the change |
| `INVALID_EXPERIMENT` | 4 | a control was violated, so no comparison would be interpretable | fix the experiment, not the code under test |
| `ERROR` | 5 | infrastructure failed before a verdict was possible | this is an IVF or environment bug |

Usage errors exit `64`, so a broken command line is never mistaken for a failed
experiment.

Two invariants hold everywhere:

* **Infrastructure failure never becomes scientific failure.** A crashed run, an
  unreadable bundle, or a missing runtime is `ERROR` or `UNSUPPORTED`, never `FAIL`.
* **An uninterpretable experiment outranks its own results.** `INVALID_EXPERIMENT` is
  decided before oracles are consulted, and the oracle numbers are recorded but decide
  nothing.

## Precedence

```
INVALID_EXPERIMENT  >  FAIL  >  UNSUPPORTED  >  INCONCLUSIVE  >  PASS
```

`ERROR` sits outside the ladder entirely.

`FAIL` outranks `UNSUPPORTED` deliberately: if one oracle could not run because a feature
is missing but another genuinely failed on real evidence, the real failure is the
actionable result. A run whose oracles all `skipped` is `INCONCLUSIVE`, never `PASS`,
a run in which nothing was checked provides no reassurance.

## Reason codes

Every non-pass verdict carries stable, greppable codes. They are part of the public
contract; renaming one is a breaking change, and `ivf compare` diffs them across runs.

### Experiment validity
```
IVF-CONTROL-ASSET-MISMATCH                  IVF-CONTROL-TASK-VARIANT-MISMATCH
IVF-CONTROL-ACTION-SEQUENCE-MISMATCH        IVF-CONTROL-WARMUP-UNDECLARED
IVF-CONTROL-INITIAL-STATE-MISMATCH          IVF-CONTROL-HORIZON-MISMATCH
IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH IVF-PROTOCOL-PARTIAL-RUN
IVF-CONTROL-CONTROL-FREQUENCY-MISMATCH      IVF-PROTOCOL-STALE-ARTIFACT
IVF-CONTROL-SEED-MISMATCH                   IVF-PROTOCOL-TOLERANCE-RATIONALE-MISSING
IVF-CONTROL-ENV-COUNT-MISMATCH              IVF-CONTROL-SOLVER-PRESENTED-AS-EQUIVALENT
```

### Oracles
```
IVF-ORACLE-INVARIANT-VIOLATION    IVF-ORACLE-EVENT-COUNT-MISMATCH
IVF-ORACLE-NON_EQUIVALENT         IVF-ORACLE-DECISION-DISAGREEMENT
IVF-ORACLE-EVENT-TIMING-DELTA     IVF-ORACLE-METAMORPHIC-VIOLATION
IVF-ORACLE-ULP-EXCEEDED
```

### Evidence sufficiency
```
IVF-SAMPLE-INSUFFICIENT      IVF-EQUIVALENCE-UNDECIDED      IVF-EFFECT-BELOW-MEANINGFUL
```

### Capability and infrastructure
```
IVF-BACKEND-FEATURE-UNSUPPORTED   IVF-RUNTIME-EXECUTION-ERROR
IVF-RUNTIME-UNAVAILABLE           IVF-EVIDENCE-WRITE-ERROR
IVF-ORACLE-UNKNOWN                IVF-EVIDENCE-CHECKSUM-MISMATCH
                                  IVF-EVIDENCE-SCHEMA-INCOMPATIBLE
```

The full glosses live in `ivf.verdicts.REASON_CODES` and are printed in the report.

## Validity check statuses

Distinct from oracle statuses, and the third one is the point:

| Status | Means |
|---|---|
| `pass` | the control was checked and held |
| `fail` | the control was checked and was violated → `INVALID_EXPERIMENT` |
| `unverifiable` | **a subject does not record what the control needs.** Not a match. Not a violation. IVF cannot tell |
| `not_applicable` | the manifest declared this property may differ |

`unverifiable` is what stops absence of evidence from reading as evidence of a match. The
cross-backend example in this repository reports the solver-settings control as
`unverifiable` because the released bundles record no solver configuration at all, a real
gap in the evidence, surfaced rather than assumed away.

## What a FAIL tells you

The evidence bundle's `divergence.jsonl` carries, per failing oracle:

* the first step at which the subjects differ at all, above float noise
* the first step at which the declared tolerance was exceeded
* the first step at which a semantic event disagreed
* the affected environment IDs and signal components
* both subjects' values at the first violating step
* the surrounding error window, plotted in the report
* the action in effect at that step
* configuration differences between the subjects
* a rule-based classification, a confidence, the basis for it, and its limitations

### Classifications

`setup_mismatch` · `asset_mismatch` · `reset_mismatch` · `cloning_mismatch` ·
`coordinate_frame_mismatch` · `quaternion_convention_mismatch` · `action_mismatch` ·
`unsupported_feature` · `numerical_drift` · `contact_model_difference` ·
`solver_parameter_difference` · `sensor_semantic_difference` · `task_level_regression` ·
`unknown`

The classifier is deterministic and rule-based. There is no model and no LLM: a
classification you cannot re-derive from the record by hand is not evidence, and a
debugging aid that hallucinates is worse than none.

`unknown` with `confidence: low` is a legitimate and common answer. Detection and
localization are much stronger than classification, and the record says so. Read
`classification_basis` before acting on a classification, and `limitations` before
trusting it.

## What a PASS does not tell you

A `PASS` means the declared criteria were met, on this workload, at this sample size,
against these tolerances. It does not mean either subject is physically correct, and it
does not generalize to quantities you did not check. Every oracle records this in its own
`limitations` field, and the report prints them.
