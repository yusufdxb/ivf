# IVF: Isaac Validation Framework

IVF is an offline acceptance and evidence layer for simulator experiments. It validates
capture integrity, checks declared behavioral contracts, localizes meaningful divergence,
and produces sealed, reproducible verdicts.

Use IVF when a simulator backend, version, configuration, or reset path changes and a
passing test suite is not enough to answer: did the declared workload remain within its
acceptance contract, and what evidence supports that decision?

IVF accepts a versioned experiment manifest plus either generated synthetic signals or
captured trajectory bundles. Real captures use the strict `trajectory_bundle/v1`
boundary. A completed run returns one typed verdict:

| Verdict | Meaning |
|---|---|
| `PASS` | every decision-bearing criterion passed |
| `FAIL` | at least one decision-bearing criterion failed |
| `INCONCLUSIVE` | the experiment was valid, but evidence was insufficient for pass or fail |
| `UNSUPPORTED` | the requested runtime or subject is not implemented or unavailable |
| `INVALID_EXPERIMENT` | required comparison controls were not established |
| `ERROR` | infrastructure failed before a scientific verdict could be produced |

IVF is not a performance benchmark. It does not measure throughput or designate a
reference engine. It checks a predeclared behavioral acceptance contract and preserves
the data, reasoning, provenance, and checksums needed to audit the verdict.

## Fastest meaningful demonstration

The CPU path needs Python and UV, but no GPU, Isaac Lab, Isaac Sim, or CUDA:

```bash
uv sync --frozen
uv run ivf doctor
uv run ivf validate validation/examples/synthetic_fixed.yaml
uv run ivf reproduce artifacts/evidence/cartpole-v1-physx-vs-newton-20260810T172728Z-f190cf5b --verify-only
```

The synthetic manifest exercises validation, oracles, a typed `PASS`, evidence sealing,
and reporting. The final command independently verifies every file in the shipped
flagship evidence bundle. For a short guided audit, follow
[`docs/review-in-five-minutes.md`](docs/review-in-five-minutes.md).

CPU-only users can also validate all shipped manifests, compare bundles, verify seals,
and open static reports. The simulator is needed only to produce new real captures.

## Flagship result: PhysX versus Newton/MJWarp

The release case compares real PhysX and Newton/MJWarp cart-pole captures through the
same strict bundle and oracle path. Its validity record contains 22 checks, zero failed
checks, and three controls explicitly left unverifiable. It verified the action sequence, frequency, task
identity, reset semantics, coordinate and quaternion conventions, and other declared
controls. Backend solver settings were recorded and allowed to differ.

The typed result is `FAIL`: trajectory and event-timing contracts exceeded their
declared budgets. The final survive-or-terminate decision still agreed for every paired
environment, with agreement `1.0`. This result says that the two captures do not satisfy
this workload's acceptance contract. It does not say which backend is physically correct.

- [Flagship case study](docs/case-study.md)
- [Sealed evidence bundle](artifacts/evidence/cartpole-v1-physx-vs-newton-20260810T172728Z-f190cf5b)
- [Static HTML report](artifacts/evidence/cartpole-v1-physx-vs-newton-20260810T172728Z-f190cf5b/report.html)
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

Exit codes are `0` pass, `1` fail, `2` inconclusive, `3` unsupported,
`4` invalid experiment, `5` error, and `64` usage error. An infrastructure failure does
not become a scientific failure.

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
