# IVF: Isaac Validation Framework

You are about to upgrade Isaac Lab, switch a physics backend, change an environment
config, or accept a PR that touches the simulation path. Your tests still pass. Your
robot still walks in the viewport.

**Does your policy still behave the same way, and what evidence would you show someone
who asked?**

IVF answers that question. It runs a declared comparison, applies acceptance criteria
you wrote down *before* seeing the data, and seals the result into an evidence bundle
that another engineer can verify on a laptop with no GPU six months later.

It does **not** claim that PhysX or Newton is correct. Neither is a reference. IVF
reports, quantifies, localizes and classifies divergence, and refuses to publish a
comparison it cannot justify.

---

## What a run looks like

```console
$ ivf validate validation/examples/synthetic_fault.yaml
FAIL  synthetic-reset-velocity-defect  (synthetic-reset-velocity-defect-20260731T150031Z-3cdc747d)
  IVF-ORACLE-NON_EQUIVALENT
  [        pass] finite_state: invariant 'finite_state' held over 134400 elements across 4 signal(s)
  [        pass] unit_quaternion: invariant 'unit_quaternion' held over 76800 elements across 1 signal(s)
  [        fail] pole_angle_agreement: pole_angle: worst second_largest absolute error 9.207e-02 rad
                 exceeds 2.000e-03 rad; first violation at step 1
  [        fail] pole_orientation_agreement: pole_quat: worst second_largest geodesic error 9.207e-02 rad
                 exceeds 2.000e-03 rad; first violation at step 1
  [        pass] termination_timing: event 'termination': same occurrence pattern across 24 environments,
                 worst timing delta 0 steps <= 1
  [        pass] episode_survival: decision 'episode_success': agreement 1.000 >= 1 over 24 envs
  [        fail] mean_angle_equivalence: pole_angle: mean paired difference -9.559e-03 rad,
                 95% CI [-9.792e-03, -9.326e-03] lies entirely outside ±0.005 rad (Cohen's dz -16.04)
  [        pass] action_replay: both subjects consumed the same action stream (cf981e84f287c172…)

evidence: validation/evidence/synthetic-reset-velocity-defect-20260731T150031Z-3cdc747d
report:   validation/evidence/synthetic-reset-velocity-defect-20260731T150031Z-3cdc747d/report.html
rerun:    ivf validate validation/examples/synthetic_fault.yaml --results-root ivf-results
$ echo $?
1
```

The evidence bundle records that the first divergence appeared **at step 1**, in **all
24 environments**, and classifies it as `reset_mismatch` with the basis stated:
*"the difference is present from step 0 and already 100% of its eventual magnitude, so
it did not accumulate"*. That is enough to send someone to the reset path instead of to
a bisect.

Note what did **not** fail. Termination timing and episode survival both passed: this
defect changes the trajectory without changing the downstream decision on this workload.
IVF says so rather than rolling everything into one number.

---

## Five minutes

```bash
git clone <this repo> && cd ivf
uv sync
uv run ivf doctor
uv run ivf validate validation/examples/synthetic_fault.yaml     # exits 1: the defect
uv run ivf validate validation/examples/synthetic_fixed.yaml     # exits 0: the fix
uv run ivf report ivf-results/<run-id>
```

No GPU. No Isaac Lab. No Isaac Sim. The synthetic reference workload runs in numpy in
about a second, and it is a real end-to-end run of the same code path a live experiment
uses, not a mock.

Already have evidence? The repository ships three finalized bundles under
`validation/evidence/`, including a real PhysX-vs-Newton cart-pole comparison captured
on a GPU workstation. Open any `report.html`, or:

```bash
uv run ivf reproduce validation/evidence/<run-id> --verify-only    # checksums
uv run ivf compare validation/evidence/<a> validation/evidence/<b>
```

---

## The five commands

| Command | What it does | Needs a GPU? |
|---|---|---|
| `ivf doctor` | reports what this machine can validate, and fails clearly when it cannot | no |
| `ivf validate <manifest>` | runs a declared experiment, seals an evidence bundle, returns a typed verdict | depends on the manifest |
| `ivf report <run>` | shows or re-renders the static HTML report | no |
| `ivf compare <a> <b>` | diffs two evidence bundles: verdicts, criteria, versions, hardware | no |
| `ivf reproduce <run>` | verifies checksums and re-runs when the runtime allows | no, to verify |

Exit codes are typed so CI can branch on the *kind* of outcome:
`0` pass · `1` fail · `2` inconclusive · `3` unsupported · `4` invalid experiment ·
`5` error · `64` usage error.

An infrastructure failure never becomes a scientific failure.

---

## What makes a verdict trustworthy

**Tolerances carry their reasoning.** A bare number is rejected by the schema. Every
threshold declares a unit, a scope, a rationale, an aggregation rule, a minimum sample
count, and whether it is exact, numerical, statistical, event-based, or an engineering
judgement.

**The experiment is checked before the data is.** Thirteen validity checks run before
any oracle. If the two subjects did not share the asset, the action stream, the
timestep, or the environment count, the verdict is `INVALID_EXPERIMENT` and the numbers
are reported but decide nothing. A missing input is reported as *unverifiable*, never as
a match.

**Statistics are equivalence tests, not significance tests.** With enough environments
any difference is detectable, so the question IVF asks is whether the difference is
small enough not to matter, against a declared margin and a declared minimum meaningful
effect. An interval straddling the margin returns `INCONCLUSIVE`, because failing to
prove equivalence is not proof of equivalence.

**Detection claims are measured, not asserted.** `ivf calibrate` injects a 17-class
fault taxonomy through the real validate path and produces a detectability matrix.
Current result: **17/17 fault classes behave exactly as declared, 0 false positives on
51 trials**, including the rows that declare a fault to be *undetectable* and say why.
See [`docs/detectability-matrix.md`](docs/detectability-matrix.md).

**Evidence is sealed.** Every bundle carries per-file SHA-256 digests sealed by a
`SEAL.json`, and its files are made read-only. Editing, adding or removing a file after
the fact is detected by `ivf reproduce --verify-only`.

---

## What IVF is not

* Not a performance benchmark. Isaac Lab already ships benchmarking, recorders and
  output formatters; IVF consumes those rather than reimplementing them.
* Not an engine-certification suite. It reports divergence between two subjects and
  never designates one as correct.
* Not a predictor of closed-loop policy transfer. That is unestablished and stays so.
* Not a service. The report is one static HTML file with no JavaScript and no external
  assets.

Its honest limits, including the fault classes it is known not to catch, are documented
in [`docs/fault-model.md`](docs/fault-model.md).

---

## Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | five minutes, no GPU |
| [Validation contracts](docs/concepts.md) | the ideas the whole tool rests on |
| [Writing a manifest](docs/manifests.md) | every field, with worked tolerances |
| [Adding an oracle](docs/oracles.md) | the extension point |
| [Reading a verdict](docs/verdicts.md) | verdicts, reason codes, what to do next |
| [Compatibility](docs/compatibility.md) | which Isaac Lab versions, and on what evidence |
| [Evidence schema](docs/evidence-schema.md) | every file in a bundle |
| [Fault model and limits](docs/fault-model.md) | what IVF does not see |
| [Case study](docs/case-study.md) | fail → localize → fix → pass, end to end |
| [Contributing](CONTRIBUTING.md) | tests, markers, review expectations |
| [Current-state audit](docs/engineering/current-state-audit.md) | how this repository came to exist |

## Relationship to `isaaclab_contrib.parity`

IVF is the acceptance layer. The capture side and the cross-backend statistical core
live upstream in `isaaclab_contrib.parity`, which owns scenario execution, trajectory
bundles, model-discrepancy floors and A/A tolerance calibration. IVF reads those
trajectory bundles with numpy and never imports Isaac Lab, which is why the offline half
of this tool works anywhere. Nothing here duplicates the comparison mathematics that
lives upstream.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
