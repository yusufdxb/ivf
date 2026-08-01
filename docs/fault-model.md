# Fault model and limitations

What IVF is designed to catch, what it measurably catches, and what it does not see.

The last category is the important one. A validator that only advertises its strengths is
a validator whose weaknesses you will discover in production.

## The taxonomy

`ivf.faults` ships 17 versioned fault classes (`ivf.faults/v1`) across two injection
surfaces:

**Generation** mutates the parameters of a run *before* it is integrated, so the resulting
trajectory is entirely self-consistent. This models simulator-level fault shapes without
claiming a simulator discovered them. Evidence metadata records the injected fixture as
ground truth for auditability. The validity and oracle logic does not consume that label;
selected controlled declarations and the commanded action stream remain identical, so
the trajectory oracle still has to establish the consequence.

**Trace** mutates recorded arrays or metadata after the fact, emulating a recorder,
serialization or plumbing defect.

| Fault | Category | Surface | Expected | Responsible layer |
|---|---|---|---|---|
| `none` (negative control) | control | generation | not detected | none |
| `ignored_reset_velocity` | reset | generation | detected | trajectory equivalence |
| `wrong_env_origin` | frame | trace | detected | trajectory equivalence |
| `incorrect_clone_count` | cloning | trace | detected | experiment validity |
| `quaternion_ordering` | convention | trace | detected | trajectory equivalence |
| `dropped_action` | action | generation | detected | trajectory equivalence |
| `shifted_action_timing` | action | generation | detected above a measured severity | trajectory equivalence |
| `observation_field_swap` | observation | trace | detected | trajectory equivalence |
| `stale_sensor_state` | sensor | trace | detected | trajectory equivalence |
| `incorrect_timestep` | timing | generation | detected | experiment validity |
| `unit_scaling` | units | generation | detected | trajectory equivalence |
| `altered_friction` | dynamics | generation | detected | trajectory equivalence |
| `altered_mass` | dynamics | generation | detected | trajectory equivalence |
| `silent_nan` | numerical | trace | detected | invariant |
| `truncated_rollout` | protocol | trace | detected | experiment validity |
| `corrupted_metadata` | metadata | trace | detected | experiment validity |
| `unsupported_feature_misreported` | capability | trace | **not detected** | none |

## Measured detectability

```bash
uv run ivf calibrate --seeds 11 23 47
```

Injects each class through the real `validate` path and measures what happened. The
current matrix is committed at [`detectability-matrix.md`](detectability-matrix.md).

**17/17 fault classes behave exactly as declared. 0 false positives across 51 trials.**

The negative control is as load-bearing as the positives: a validator aggressive enough to
flag everything is as useless as one that flags nothing, and `none` firing even once fails
calibration.

## What IVF does not see

### 1. Faults outside the taxonomy

**This is the biggest limit and it cannot be fixed by more testing of the same kind.** The
taxonomy is author-generated. A perfect matrix establishes that IVF detects the fault
classes its authors enumerated. It does not establish detection of fault classes nobody
thought of, and no amount of blinding changes that: blinding the evaluator is not the same
as discovering unknown unknowns.

Treat every detection number in this repository as scoped to the enumerated set.

### 2. A capability that is misreported

`unsupported_feature_misreported` is declared undetectable and measured undetectable. IVF
trusts each subject's own declaration of what it supports. A run that advertises a feature
it does not have is invisible to every comparison oracle, because comparison cannot tell
you a subject is lying about itself. Closing this needs an independent capability probe in
`ivf doctor`, not another oracle.

### 3. Identical wrongness, except where an invariant catches it

A defect in a shared layer that affects both subjects the same way agrees perfectly with
itself. Every pairwise oracle passes. The only defense is the reference-free invariants
(`finite_state`, `unit_quaternion`, `bounded`) and, where the workload has a closed form,
an analytic check. That defense is thin, and it is the weakest claim in the stack.

### 4. Faults below their minimum severity

Every detectable class declares the smallest parameterization at which detection is
claimed, and one of them was measured rather than assumed:

> `shifted_action_timing`: at drive amplitude 1.5 a one-step delay produces 1.78e-3 rad of
> second-largest error against a 2e-3 rad budget and is **not** detected. The crossing is
> near amplitude 1.68.

The signal from a one-step action delay is the command's slope times one step, so a
low-frequency or low-amplitude command hides it below any operationally sane budget. This
is a property of the physics, not a bug, and the calibration workload's amplitude was
raised to 2.5 to sit above the measured crossing rather than the tolerance being loosened
to hide the miss.

Other declared limits:

* `ignored_reset_velocity`, undetectable when the declared initial velocity is zero
* `dropped_action`, undetectable in a passive scenario; there is no action to drop
* `quaternion_ordering`, undetectable for a pure-identity trajectory, which is invariant
  under the component swap
* `stale_sensor_state`, undetectable on a signal genuinely constant over the window
* `altered_friction`, at small deltas over short horizons this is below the noise of any
  defensible tolerance

### 5. Classification, as opposed to detection

Detection and first-step localization are strong. **Classification is much weaker**, and
the divergence record says so per finding: every classification carries a `confidence`
(`low`/`medium`/`high`, never a fabricated probability), the `classification_basis` it
keyed on, and its `limitations`.

Known confusions:

* a constant per-environment offset is classified `coordinate_frame_mismatch`, but a
  legitimate frame convention change looks identical; IVF cannot separate them without an
  explicit frame declaration
* a divergence present from step 0 is classified `reset_mismatch`, but a first-step
  actuation difference manifests identically
* smooth growth is classified `numerical_drift`, but accumulated float error and a small
  genuine dynamics difference look the same over a long horizon
* a discontinuity is classified `sensor_semantic_difference`, but a genuine contact event
  produces the same shape

`unknown` with low confidence is a legitimate and frequent answer.

### 6. Anything about correctness

IVF reports divergence between two subjects. Neither is a reference. `FAIL` establishes
that they differ, not which one is wrong. `PASS` establishes that they agree within the
declared criteria, not that either is physically right.

### 7. Policy transfer

Whether a measured divergence predicts closed-loop policy transfer failure is
**unestablished**. Nothing in this repository measures it, and no wording here should be
read as implying it.

## Where the fixture stops being physics

The synthetic reference system is a damped driven pendulum integrated with semi-implicit
Euler. It is not a physics claim and not a model of anything. It is a fixture chosen
because it is the smallest system exhibiting every property the oracles need: a continuous
trajectory, a discrete event, a downstream decision, and an orientation signal. It was
reproduced on the RC software and CPU environment. IVF makes no bit-identity claim across
arbitrary platforms or dependency versions.

Detectability measured on it transfers to a real workload only insofar as the real
workload has comparable signal magnitudes and tolerances. Reusing the 2 mrad illustrative
budget makes the synthetic examples internally comparable; it does not prove detection
on an unexecuted real workload.
