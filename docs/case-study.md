# Case study: real PhysX versus Newton/MJWarp acceptance

This case study covers the flagship v0.1.0-rc1 experiment. Both inputs came from real
simulator execution and entered IVF through strict `trajectory_bundle/v1` loading. The
result is a typed `FAIL` against a declared contract, not a judgment that either backend
is physically correct.

## 1. Decision being evaluated

The decision was whether PhysX and Newton/MJWarp preserved the declared behavior of one
passive cart-pole workload closely enough to satisfy its trajectory, timing, statistical,
and semantic acceptance criteria.

The decision was not which backend is more accurate, faster, or more suitable for a
different task.

## 2. Experiment contract

The baseline is the committed PhysX `trajectory_bundle/v1` capture. The candidate is the
committed Newton/MJWarp `trajectory_bundle/v1` capture. Each contains 16 paired
environments and 400 control steps at 120 Hz with an explicit zero-effort action tensor.

The contract is declared in
[`validation/examples/cartpole_physx_vs_newton_v1.yaml`](../validation/examples/cartpole_physx_vs_newton_v1.yaml).
It requires finite state and unit quaternions, bounded pole-angle and pole-rate errors,
bounded termination timing, paired survive-or-terminate agreement, mean-angle
equivalence, and identical replayed actions.

Before the oracles could decide a verdict, the validity record evaluated 22 checks.
No validity check failed. Three controls remained explicitly unverifiable.

## 3. Verified controls

The evidence verified:

- normalized task configuration and task identity;
- requested initial-state distribution;
- the exact action sequence and action timing;
- observation names, definitions, shapes, and units;
- simulation timestep, control timestep, and 120 Hz control frequency;
- seed schedule and environment ordering;
- reset semantics;
- coordinate-frame and quaternion conventions;
- environment count, horizon, warm-up, and completion.

These checks establish that the recorded comparison met its declared experimental
contract. They do not establish that either simulator is a physical reference.

## 4. Unverifiable controls

The bundle contract could not verify:

- binary asset identity;
- realized backend initial state beyond the recorded requested and applied reset data;
- backend-internal state not exposed through the bundle.

The validity record labels all three `unverifiable`. IVF does not treat them as matches.

## 5. Allowed backend-specific differences

Solver-specific configuration was required to be declared, but was allowed to differ.
PhysX used its recorded PhysX settings. Newton used
`NewtonCfg(solver_cfg=MJWarpSolverCfg())`. Package identities, solver parameters, and
backend metadata are preserved in the captures and evidence.

Allowing these settings to differ is necessary for a meaningful cross-backend
experiment. It is not a claim that the settings are equivalent.

## 6. Numerical results

| Metric | Verified result |
|---|---:|
| worst second-largest pole-angle error | 3.533347845 rad |
| first pole-angle tolerance violation | step 29 |
| worst second-largest pole-rate error | 12.99448848 rad/s |
| first pole-rate tolerance violation | step 13 |
| mean paired pole-angle difference | -0.18621337 rad |
| 95% confidence interval | [-0.18767944, -0.18477211] rad |
| Cohen's dz | -60.0368 |

The pole-angle and pole-rate trajectories exceeded their declared budgets. The paired
mean-angle confidence interval was also entirely outside the declared equivalence
margin.

## 7. Event-level results

| Event or decision | Verified result |
|---|---:|
| worst termination timing delta | 2 steps |
| median termination timing delta | 2 steps |
| first event disagreement | step 35 |
| paired survive-or-terminate agreement | 1.0 |

Termination timing failed its contract, while the final binary survival decision was
preserved for all paired environments. IVF reports both facts rather than collapsing
them into one score.

## 8. Final typed verdict

The final verdict is `FAIL` with reason codes:

- `IVF-ORACLE-NON_EQUIVALENT`
- `IVF-ORACLE-EVENT-TIMING-DELTA`

The experiment itself was valid. The fail follows from decision-bearing trajectory,
statistical, and timing criteria, not from an infrastructure error or an invalid
comparison.

## 9. Evidence and reproduction commands

- [Sealed evidence bundle](../artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912)
- [Static HTML report](../artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912/report.html)
- [PhysX capture](../artifacts/cartpole-physx-baseline)
- [Newton/MJWarp capture](../artifacts/cartpole-newton-baseline)

```bash
uv sync --frozen
uv run ivf reproduce artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --verify-only
uv run ivf report artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --print-verdict
uv run ivf validate validation/examples/cartpole_physx_vs_newton_v1.yaml
```

The final validation command is expected to exit `1` because the recorded scientific
verdict is `FAIL`. That exit is successful completion of the acceptance workflow.

## 10. Limitations

- Neither backend is thereby proven physically correct.
- The result covers one passive cart-pole workload and one recorded software and hardware
  environment.
- It does not establish universal PhysX/Newton parity or cross-hardware determinism.
- The three unverifiable controls bound the strength of the comparison.
- The result does not predict learned-policy transfer or sim-to-real behavior.
- Statistical intervals describe the recorded paired environments and declared method,
  not a population-wide property of either backend.
