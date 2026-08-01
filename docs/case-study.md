# Case study: reset semantics and cross-backend acceptance

This repository ships two deliberately different reset demonstrations and one real
cross-backend result. Their provenance matters as much as their verdicts.

## CPU synthetic fixture

`validation/examples/synthetic_fault.yaml` uses an explicit generation-level
`ignored_reset_velocity` fixture. The generated metadata records the fixture. No captured
array is edited after generation, and the verdict code does not consume the fixture label.

```bash
uv run ivf validate validation/examples/synthetic_fault.yaml
uv run ivf validate validation/examples/synthetic_fixed.yaml
```

The perturbed manifest returns `FAIL`. Its first numerical difference is at step 0, its
pointwise angle and orientation criteria fail, and its horizon-mean equivalence criterion
fails. The 1.2 rad event is never reached by either subject, so termination timing and the
survive or terminate decision are negative controls in this fixture. They are not evidence
that the perturbation is operationally benign outside this workload.

The fixed manifest returns `PASS` with the same workload, seeds, oracles and tolerances.
It is the offline negative control for the validator.

## Real PhysX reset perturbation

`validation/examples/cartpole_reset_defect.yaml` compares two real Isaac Lab PhysX
captures. The candidate capture enables a test-only simulator perturbation:

```yaml
defect:
  drop_reset_velocity: true
```

The adapter requests a pole joint velocity of `0.5 rad/s` and, only with that switch,
writes an explicit zero joint-velocity vector before stepping PhysX. The baseline writes
the requested value. Both bundles record requested and applied reset state, the switch,
software and hardware metadata, trajectories and checksums.

This is an **explicit simulator-level test perturbation modeled after a previously
observed reset-semantics failure**. It is not a restored historical implementation and it
is not an upstream Isaac Lab product defect. Upstream commit
`9aaa389f24231fadcca9f871af3eccb79e738b20` fixed a related parity-harness failure in which
a configured rigid-body root velocity of `1.0 m/s` was not written after reset. The IVF
case uses joint angular velocity and deliberately writes zero, so it is modeled after that
failure rather than literally reproducing it.

```bash
uv run ivf validate validation/examples/cartpole_reset_defect.yaml
```

The captured result returns `FAIL`:

| Observation | Verified result |
|---|---:|
| first pole angular-rate tolerance violation | step 0 |
| first pole-angle tolerance violation | step 6 |
| affected environments | all 16 |
| worst pole-angle error | 0.5537 rad |
| worst pole angular-rate error | 1.471 rad/s |
| worst termination timing delta | 9 steps |
| paired survive or terminate agreement | 1.0 |
| reason codes | `IVF-ORACLE-NON_EQUIVALENT`, `IVF-ORACLE-EVENT-TIMING-DELTA` |

The step-0 rate signature is the direct reset consequence. Angle error then accumulates
through integration, and threshold crossings move by 6 to 9 samples. The final
survive-or-terminate decision remains equal on this 400-step workload, which does not
erase the trajectory and event failures.

The switch defaults to false, accepts only a YAML boolean and is absent from nominal and
corrected manifests. `tests/test_capture_spec.py` enforces that guard. Full provenance is
in `docs/engineering/reset-defect-provenance.md`.

## Corrected and benign PhysX controls

```bash
uv run ivf validate validation/examples/cartpole_corrected.yaml
uv run ivf validate validation/examples/cartpole_benign_difference.yaml
```

The corrected capture returns `PASS` with zero differences. The benign capture also
returns `PASS`: its worst pole-angle difference is `0.001450 rad`, its worst pole-rate
difference is `0.004199 rad/s`, event timing is unchanged, and its horizon-mean shift is
about `7.82e-05 rad`. These are results against declared budgets, not claims of physical
correctness.

## Real PhysX versus Newton v1 result

The release workflow also captured the same passive cart-pole through real PhysX and
Newton/MJWarp execution and loaded both through strict `trajectory_bundle/v1` validation:

```bash
uv run ivf validate validation/examples/cartpole_physx_vs_newton_v1.yaml
```

The experiment is valid for the controls the bundle contract can establish. It verifies
normalized task configuration, requested initial-state distribution, task identity,
the explicit zero-effort action stream, observation schema, timestep, control frequency, seed and environment
ordering, reset and action timing semantics, horizon and completion. Solver settings are
recorded and explicitly allowed to differ.

Three controls are explicitly unverifiable: binary asset identity, realized backend
initial state, and backend-internal state. They are not treated as matches.

The result returns `FAIL`:

| Observation | Verified result |
|---|---:|
| worst second-largest pole-angle error | 3.533347845 rad |
| first pole-angle violation | step 29 |
| worst second-largest pole-rate error | 12.99448848 rad/s |
| first pole-rate violation | step 13 |
| worst and median threshold timing delta | 2 steps |
| first event disagreement | step 35 |
| paired survive or terminate agreement | 1.0 |
| mean paired pole-angle difference | -0.18621337 rad |
| 95% confidence interval | [-0.18767944, -0.18477211] rad |
| Cohen's dz | -60.0368 |
| reason codes | `IVF-ORACLE-NON_EQUIVALENT`, `IVF-ORACLE-EVENT-TIMING-DELTA` |

This says the two recorded backends exceeded this workload's declared acceptance
criteria. It does not identify either backend as correct, prove cross-hardware
determinism, establish general PhysX/Newton equivalence, predict learned-policy transfer,
or predict sim-to-real behavior.

The older `validation/examples/cartpole_cross_backend.yaml` remains an offline legacy
bundle example. It returns `INCONCLUSIVE` because its pre-v1 evidence cannot verify enough
controls for a stronger cross-backend claim.
