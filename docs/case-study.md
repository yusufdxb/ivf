# Case study: a silent reset defect, end to end

This is the whole loop (fail, localize, fix, pass) on evidence shipped in this
repository, followed by the same tooling applied to real cross-backend data where the
honest answer is more complicated.

Every number below is copied from `validation/evidence/`. Reproduce it:

```bash
uv run ivf reproduce validation/evidence/synthetic-reset-velocity-defect-20260731T150031Z-3cdc747d
```

## The defect

The candidate's reset writes joint position and drops joint velocity. This is a real Isaac
Lab defect class, and it is nasty because it is **invisible in provenance**: the asset
declaration, the initial-state declaration, the timestep and the commanded action stream
are byte-identical between the two subjects. Nothing structural is wrong. The robot simply
starts from a state nobody asked for.

A conventional test suite does not see it. The environment builds, steps, terminates and
resets. The trajectory is smooth and physically plausible. It is just the wrong trajectory.

## 1. Fail

```console
$ ivf validate validation/examples/synthetic_fault.yaml
FAIL  synthetic-reset-velocity-defect
  IVF-ORACLE-NON_EQUIVALENT
  [        pass] finite_state
  [        pass] unit_quaternion
  [        fail] pole_angle_agreement: worst second_largest absolute error 9.207e-02 rad
                 exceeds 2.000e-03 rad; first violation at step 1
  [        fail] pole_orientation_agreement: worst second_largest geodesic error 9.207e-02 rad
                 exceeds 2.000e-03 rad; first violation at step 1
  [        pass] termination_timing: worst timing delta 0 steps <= 1
  [        pass] episode_survival: agreement 1.000 >= 1 over 24 envs
  [        fail] mean_angle_equivalence: mean paired difference -9.559e-03 rad,
                 95% CI [-9.792e-03, -9.326e-03] outside ±0.005 rad (Cohen's dz -16.04)
  [        pass] action_replay: both subjects consumed the same action stream
$ echo $?
1
```

Three things worth noticing.

**The experiment was valid.** All thirteen validity checks passed, so the failure is a
statement about the subjects and not about the setup. Had the two runs used different
assets or timesteps, this would have been `INVALID_EXPERIMENT` and the numbers would have
decided nothing.

**Two independent oracles agree on the magnitude.** The joint-space error and the
orientation geodesic angle both report 9.207e-02 rad. They are computed by different code
paths on different signals, and their agreement is a consistency check on the tooling
itself, not just on the subjects. (This is not decoration: an early build had the geodesic
metric off by a factor of two, and it was this disagreement that exposed it.)

**The downstream decision did not change.** Termination timing and episode survival both
passed: on this workload the defect moves the trajectory without moving the outcome. IVF
reports that rather than collapsing everything into one number, and it changes what you do
with the finding: this is a correctness bug, not an outage.

## 2. Localize

From `divergence.jsonl`:

| | |
|---|---|
| first numerical difference | step **0** |
| first tolerance violation | step **1** |
| affected environments | all **24** |
| affected components | `[0]` |
| classification | **`reset_mismatch`** |
| confidence | `medium` |
| basis | *"tolerance exceeded at step 1, before dynamics could accumulate"* |
| limitations | *"Cannot separate an initial-state difference from a first-step actuation difference; both manifest at step 0. Check the action stream digest and the initial-state digest to disambiguate."* |

That is an actionable finding rather than a bisect. The difference exists before dynamics
could produce it, in every environment simultaneously, so it is a start-state problem and
not an accumulation problem. And the record tells you how to discriminate the one
alternative it cannot rule out: the action-stream digest is identical, which the
`action_replay` oracle confirms independently, and that leaves the initial state.

The report renders the error window around step 1 as an inline SVG with the tolerance
marked, so the shape (a jump, not a ramp) is visible at a glance.

## 3. Fix

`validation/examples/synthetic_fixed.yaml` is the same manifest with the injected defect
removed from the candidate. Every oracle, every tolerance, every seed is unchanged. That is
what makes the pair a fair before/after: the only thing that moved is the subject.

## 4. Pass

```console
$ ivf validate validation/examples/synthetic_fixed.yaml
PASS  synthetic-reset-velocity-fixed
  [        pass] pole_angle_agreement: worst second_largest absolute error 0.000e+00 rad <= 2.000e-03 rad
  [        pass] pole_orientation_agreement: worst second_largest geodesic error 0.000e+00 rad <= 2.000e-03 rad
  [        pass] mean_angle_equivalence: mean paired difference +0.000e+00 rad,
                 95% CI [+0.000e+00, +0.000e+00] inside ±0.005 rad
  ... 8 passed, 0 failed
$ echo $?
0
```

This run is also the **negative control for the entire validator**. If it ever fails, the
tolerances are too tight and every failure IVF reports elsewhere is suspect.

## 5. Confirm the fix in CI

```bash
uv run ivf compare validation/evidence/<fixed-run> validation/evidence/<defect-run>
```

Because these are two *different* manifests, `ivf compare` refuses to treat them as
comparable and says so before printing anything:

```
NOT COMPARABLE
  - different experiments: 'synthetic-reset-velocity-fixed' vs 'synthetic-reset-velocity-defect'

The differences below are reported for information only; they are not
evidence about the subjects under test.
```

That is the correct behaviour and it is worth internalizing: a verdict difference across
two different contracts is not evidence about the code. For a real regression gate, run the
*same* manifest twice and compare those, which is exactly what `ivf reproduce` does; on
the shipped bundles it reports `No material differences.`

---

## The same tooling on real cross-backend data

```bash
uv run ivf validate validation/examples/cartpole_cross_backend.yaml
```

PhysX versus Newton/MJWarp on a passive cart-pole, 480 steps, 4 environments, from bundles
captured on a GPU workstation on 2026-07-12 and vendored under `validation/bundles/`. One
run, three different honest answers.

**`FAIL` on pointwise agreement.** Worst second-largest joint-position error **3.395**
against a 1e-2 budget, first violation at step 17. Two integrators of a passive
articulation genuinely diverge, and widening the tolerance until it passed would be the
dishonest move.

**`INCONCLUSIVE` on the equivalence question.** Four paired environments cannot support an
equivalence claim, so the statistical oracle returns `IVF-SAMPLE-INSUFFICIENT` rather than
a confident-looking interval. Before the method floor was added, this same oracle produced
a CI of `[+0.582, +0.582]` with Cohen's dz of 5.9e6: an interval that collapsed to a point
because four nearly-identical differences have no spread. Arithmetic that looks certain is
not evidence.

**`unverifiable` on the solver control.** The manifest declares that solver parameters may
legitimately differ between backends. But these bundles record no solver configuration at
all, so IVF cannot say *what* differs:

> *"the manifest declares solver parameters may differ, but at least one subject records no
> solver configuration at all. IVF therefore cannot report what differs, so a
> solver-parameter explanation for any divergence below is a hypothesis rather than a
> finding."*

That is a real gap in the released evidence, found by the pre-existing parity work's own
adversarial audit, and now surfaced automatically on every run that touches those bundles
instead of living in a report nobody rereads.

**What this run does not say.** It does not say Newton is wrong. It does not say PhysX is
right. It does not say a policy trained on one will fail on the other. It says these two
backends disagree on this workload by this much, starting at this step, and that the
available evidence cannot attribute the disagreement to a solver difference because nobody
recorded the solvers.

That is the whole product: a defensible answer, with its limits attached, that someone else
can check.
