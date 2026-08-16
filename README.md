# IVF: Isaac Validation Framework

**A command-line acceptance checker for simulator experiments: you write down what
"unchanged behavior" means in a YAML file, IVF runs the comparison and hands back a
tamper-evident PASS or FAIL bundle you can re-verify a year later.** It is for robotics
and simulation engineers who change a physics backend, an engine version, or a reset
path and need something stronger than a green test suite.

## What problem this solves

You upgrade Isaac Lab, or swap PhysX for Newton/MJWarp, or touch the reset path. Your
unit tests still pass, because none of them was written to catch a pole angle that drifts
3.5 rad by step 29 in one environment out of sixteen. The usual fallback, eyeballing a
plot or diffing arrays by hand, produces no record anyone can audit later.

IVF makes the acceptance criteria explicit before the run: which signals, which
tolerances and their units, which controls must hold, how many samples are the minimum.
It checks that the experiment was even valid (same action stream, same task, same reset
semantics) before letting any result decide a verdict, then seals the manifest, signals,
reasoning, provenance, and per-file SHA-256 digests into one evidence directory.

## See it work (CPU only, no GPU, no Isaac Sim)

```console
$ uv sync --frozen
$ uv run ivf validate validation/examples/synthetic_fixed.yaml
PASS  synthetic-reset-velocity-fixed  (synthetic-reset-velocity-fixed-20260812T013122Z-c64ac825)
  [        pass] finite_state: invariant 'finite_state' held over 134400 elements across 4 signal(s)
  [        pass] pole_angle_agreement: pole_angle: worst second_largest absolute error 0.000e+00 rad <= 2.000e-03 rad
  [        pass] termination_timing: event 'termination': same occurrence pattern across 24 environments, worst timing delta 0 steps <= 1
  [        pass] episode_survival: decision 'episode_success': agreement 1.000 >= 1 over 24 envs
  [        pass] action_replay: both subjects consumed the same action stream (cf981e84f287c172…)

evidence: ivf-results/synthetic-reset-velocity-fixed-20260812T013122Z-c64ac825
report:   ivf-results/synthetic-reset-velocity-fixed-20260812T013122Z-c64ac825/report.html
```

(Three passing oracle lines elided for length; exit code `0`.)

The same binary re-verifies the shipped real-capture bundle without re-running anything:

```console
$ uv run ivf reproduce artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --verify-only
integrity verified against included seal  cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 (13 files)
recorded verdict  FAIL  cartpole-v1-physx-vs-newton
```

For a guided audit of that bundle, follow
[`docs/review-in-five-minutes.md`](docs/review-in-five-minutes.md). CPU-only users can
validate every shipped manifest, compare bundles, verify seals, and open the static HTML
reports. A simulator is needed only to produce new real captures.

## Status and verified numbers

| Item | Value |
|---|---|
| Test suite | 265 passed, 5 skipped (`uv run pytest -q`, CPU-only run; the skips are the simulator-backed tests) |
| Flagship case | real PhysX versus Newton/MJWarp cart-pole captures, recorded verdict `FAIL`, 5 of 9 oracles pass and 4 fail |
| Experiment validity | 22 checks recorded: 18 pass, 3 unverifiable, 1 not applicable, 0 failed |
| Evidence integrity | 13 files re-verified from `SEAL.json` by `ivf reproduce --verify-only` |
| CI coverage | CPU only. Simulator-backed tests skip on hosted runners, and a skip is not a pass |
| Hardware verification | one workstation with a single NVIDIA GPU, Isaac Sim 6.0.0.1, Isaac Lab 10.2.0. No multi-GPU or cross-hardware determinism claim |
| Scope of the real evidence | one passive cart-pole workload, one recorded Isaac stack, PhysX plus one Newton/MJWarp preset |
| Release | `0.1.0rc2` |

IVF is not a performance benchmark. It does not measure throughput and does not designate
a reference engine. It checks a predeclared behavioral acceptance contract and preserves
the data, reasoning, provenance, and checksums needed to audit the verdict.

## Flagship result: PhysX versus Newton/MJWarp

The release case compares real PhysX and Newton/MJWarp cart-pole captures through the
same strict bundle and oracle path. Its validity record contains 22 checks, zero failed
checks, and three controls explicitly left unverifiable. It verified the action sequence, frequency, task
identity, reset semantics, coordinate and quaternion conventions, and other declared
controls. Backend solver settings were recorded and allowed to differ.

The typed result is `FAIL`: trajectory and event-timing contracts exceeded their
declared budgets. The recorded oracle lines localize where, for example:

```text
fail  pole_angle_agreement: worst second_largest absolute error 3.533e+00 rad exceeds
      2.880e-02 rad; first violation at step 29
fail  termination_timing: event 'termination': worst timing delta 2 steps exceeds the
      declared 1 steps
pass  episode_survival: decision 'episode_survives_400_steps': agreement 1.000 >= 1 over 16 envs
```

The final survive-or-terminate decision still agreed for every paired environment, with
agreement `1.0`. This result says that the two captures do not satisfy this workload's
acceptance contract. It does not say which backend is physically correct.

- [Flagship case study](docs/case-study.md)
- [Sealed evidence bundle](artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912)
- [Static HTML report](artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912/report.html)
- [Experiment manifest](validation/examples/cartpole_physx_vs_newton_v1.yaml)

## Simulator capture workflow

New capture requires the recorded Isaac Sim 6.0.0.1 and Isaac Lab 10.2.0 stack, an
NVIDIA GPU, and the `parity-capture` adapter installed in the Isaac Lab interpreter.
Set `ISAACLAB_PYTHON` to that interpreter. Removing an inherited `PYTHONPATH` is required
because ROS and simulator environments commonly export incompatible Python paths.

```bash
export ISAACLAB_PYTHON=/path/to/isaaclab/python
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  "$ISAACLAB_PYTHON" -m parity_capture.cli doctor

env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  IVF_CAPTURE_PYTHON="$ISAACLAB_PYTHON" \
  IVF_HARDWARE_LABEL='NVIDIA GPU' \
  uv run pytest -q -rs -m "gpu and isaaclab and physx"
```

Newton/MJWarp capture is an optional second path using
`validation/capture/cartpole_newton.yaml`. The complete install, capture, strict-load,
and validation procedure is in
[`docs/reproduction/independent-rc1-reproduction.md`](docs/reproduction/independent-rc1-reproduction.md).

## Commands and exit codes

| Command | Purpose | GPU required? |
|---|---|---|
| `ivf doctor` | report available validation capabilities | no |
| `ivf validate <manifest>` | run a declared experiment and seal the verdict | only for a live runtime subject |
| `ivf report <bundle>` | show or re-render the static HTML report | no |
| `ivf compare <a> <b>` | compare two evidence bundles | no |
| `ivf reproduce <bundle>` | verify checksums and optionally re-run | no for verification |
| `ivf calibrate` | measure the declared synthetic fault-detectability matrix | no |

A completed run returns exactly one typed verdict:

| Verdict | Meaning | Exit code |
|---|---|---|
| `PASS` | every decision-bearing criterion passed | `0` |
| `FAIL` | at least one decision-bearing criterion failed | `1` |
| `INCONCLUSIVE` | the experiment was valid, but evidence was insufficient for pass or fail | `2` |
| `UNSUPPORTED` | the requested runtime or subject is not implemented or unavailable | `3` |
| `INVALID_EXPERIMENT` | required comparison controls were not established | `4` |
| `ERROR` | infrastructure failed before a scientific verdict could be produced | `5` |

A usage error exits `64`. An infrastructure failure does not become a scientific failure.

## Why the evidence is auditable

- Tolerances include units, scope, source measurements, formulas, assumptions,
  aggregation rules, minimum samples, uncertainty, interpretation, and limitations.
  Drift tests derive every shipped threshold from the provenance registry.
- Experiment validity is checked before oracle results can decide a verdict. Missing
  controls are reported as unverifiable, never treated as matches.
- Equivalence criteria use declared margins. Failure to establish equivalence can return
  `INCONCLUSIVE` rather than being mislabeled as a difference.
- Each bundle contains per-file SHA-256 digests sealed by `SEAL.json`. Editing, adding,
  or removing a file is detected by `ivf reproduce --verify-only`.

  IVF evidence bundles include checksums and an integrity seal that detect accidental
  corruption, incomplete transfer, and uncoordinated modification. The seal is not a
  digital signature and does not establish authenticity against an actor who can modify
  and reseal the entire bundle. Verification answers "is this bundle internally
  consistent", not "who produced it".
- Divergence records identify the first violating step, affected environments, signal,
  tolerance, and event-level consequence where applicable.

## Limits

IVF does not:

- determine which backend is physically correct;
- prove universal backend parity or cross-hardware determinism;
- predict learned-policy transfer or sim-to-real behavior;
- turn unverified controls into matches;
- replace simulator performance benchmarking, capture tooling, or domain validation.

The RC1 simulator evidence covers one passive cart-pole workload, one recorded Isaac
stack, PhysX, and one Newton/MJWarp preset. A pass or fail applies only to the declared
signals, controls, workload, data, and acceptance budgets.

## Documentation

| Document | Purpose |
|---|---|
| [Five-minute review](docs/review-in-five-minutes.md) | inspect the flagship contract, verdict, seal, and first divergence |
| [Quickstart](docs/quickstart.md) | run the CPU workflow |
| [Flagship case study](docs/case-study.md) | understand the real PhysX-versus-Newton result |
| [Independent RC1 reproduction](docs/reproduction/independent-rc1-reproduction.md) | clean CPU and optional simulator tracks |
| [RC1 release notes](docs/releases/v0.1.0-rc1.md) | verified scope and limitations |
| [RC1 scientific equivalence](docs/releases/v0.1.0-rc1-equivalence.json) | machine-readable history-rewrite and content comparison |
| [Validation contracts](docs/concepts.md) | acceptance model and validity rules |
| [Manifest reference](docs/manifests.md) | declare inputs, controls, and tolerances |
| [Verdicts and reason codes](docs/verdicts.md) | interpret typed outcomes |
| [Compatibility](docs/compatibility.md) | verified software and hardware boundary |
| [Capture boundary](docs/capture-boundary.md) | strict `trajectory_bundle/v1` contract |
| [Evidence schema](docs/evidence-schema.md) | files and checksums in a sealed bundle |
| [Fault model](docs/fault-model.md) | detectable, undetectable, and out-of-scope failures |
| [Claims and evidence](docs/engineering/claims-and-evidence.md) | public proof boundary and reproduction commands |
| [Tolerance provenance](docs/engineering/tolerance-provenance.md) | machine-derived acceptance budgets |

## Relationship to `isaaclab_contrib.parity`

IVF is the acceptance layer. Upstream `isaaclab_contrib.parity` owns scenario execution,
trajectory bundles, model-discrepancy floors, and A/A tolerance calibration. This
repository contains one narrow `parity-capture` adapter for the release cart-pole case.
IVF core reads trajectory bundles with numpy and does not import Isaac Lab. It does not
duplicate the upstream floor or kappa mathematics.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
