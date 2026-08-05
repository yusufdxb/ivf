# Writing a manifest

A manifest is the validation contract: what is being compared, what must be identical for
the comparison to mean anything, and what would count as a failure. It is parsed into
typed objects, so a mistake fails at load with a path-qualified message rather than
producing a confident, meaningless report.

Start by copying `validation/examples/synthetic_fault.yaml` and editing it.

## Top level

```yaml
schema_version: ivf.validation/v1     # required, exact match
name: cartpole-backend-migration      # required, [A-Za-z][A-Za-z0-9_.-]*
description: >                        # optional, but it ends up in the report
  Why this comparison exists and what a reader should conclude from it.
subjects: {...}                       # required, exactly baseline and candidate
workload: {...}                       # required
controls: {...}                       # optional, but a manifest without controls
oracles: [...]                        # required, at least one
verdict_policy: {...}                 # optional
```

Unknown top-level keys are rejected. A typo in `verdict_policy` must not be silently
dropped.

## `subjects`

Exactly two: `baseline` and `candidate`.

```yaml
subjects:
  baseline:
    kind: parity_bundle
    label: PhysX                      # shown in the report
    path: artifacts/cartpole-physx-baseline
    bundle_sha256: 0947a726333914db45727c138f17152995aec5bda3c6d862e1501e4f7e4a0794
  candidate:
    kind: parity_bundle
    label: Newton / MJWarp
    path: artifacts/cartpole-newton-baseline
    bundle_sha256: 86ac7101ce268106b963a45301559476184a6085a0e3439cba7b4d59b2744565
```

| `kind` | Parameters | Needs |
|---|---|---|
| `synthetic` | `system`, plus that system's parameters; optional `fault`, `fault_params` | nothing |
| `parity_bundle` | `path` to an `isaaclab_contrib.parity` trajectory bundle; v1 captures should also set `bundle_sha256` to the finalized root in `COMPLETE` | nothing (numpy only) |
| `isaaclab` | backend/preset | reserved in core; returns `UNSUPPORTED` |

To run a live workload, generate a trajectory bundle with `isaaclab_contrib.parity` or
the repository's narrow `parity-capture` cart-pole adapter, then point a
`parity_bundle` subject at it. IVF core remains offline and does not import Isaac Lab.
For v1, a mismatched `bundle_sha256` returns `INVALID_EXPERIMENT` with
`IVF-BUNDLE-CHECKSUM-MISMATCH` before any oracle runs. The paired capture specs generate
these locks automatically.

## `workload`

```yaml
workload:
  task: Isaac-Cartpole
  num_envs: 64
  steps: 2000
  warmup_steps: 100        # declare it even if it is 0
  seeds: [11, 23, 47, 71, 97]
```

`warmup_steps` is optional but its absence is *reported*: without it, early transient
behaviour is included in every oracle, and IVF records a `V-12` check as `unverifiable`
rather than assuming you meant zero.

`seeds` matter for the statistical oracles. For `synthetic` subjects, multiple seeds are
run and concatenated along the environment axis, keeping the pairing intact because both
subjects walk the same seed list in the same order. A single seed gives one draw from the
initial-condition distribution, and an interval computed from it describes that one draw.
Recorded bundles carry their own seed; the manifest's list does not apply to them.

## `controls`

```yaml
controls:
  require_same:
    - asset_identity
    - action_sequence
    - initial_state_distribution
    - observation_definition
    - control_frequency
    - num_envs
    - horizon
    - task_variant
  allow_different:
    - solver_specific_parameters
  unsupported_or_unverifiable:
    - asset_binary_identity
    - initial_state_realization
    - backend_internal_state
```

Known controls: `asset_identity`, `action_sequence`, `initial_state_distribution`,
`observation_definition`, `control_frequency`, `seeds`, `num_envs`, `horizon`,
`task_variant`, `solver_specific_parameters`, `frame_convention`,
`quaternion_convention`, `reset_semantics`, `environment_ordering`, `action_timing`,
`asset_binary_identity`, `initial_state_realization`, and `backend_internal_state`. An
unknown name is rejected: silently ignoring a control you asked for would be the worst
possible failure.

`allow_different` is a claim about interpretation, not a way to silence a check. Declaring
`solver_specific_parameters` as allowed to differ means "if these subjects diverge, a
solver difference is an admissible explanation", and if neither subject records its
solver settings, IVF still reports that it cannot tell you *what* differs.

`unsupported_or_unverifiable` names a scientifically relevant property that the current
bundle contract cannot establish. It is shown in the validity report and is not counted
as a match. The three backend-internal controls in the example are accepted only in this
partition, so a manifest cannot accidentally require them and have the request ignored.

A violated control produces `INVALID_EXPERIMENT`. The physics timestep is checked
unconditionally, whether you declared it or not, because step-by-step comparison of runs
at different control frequencies is meaningless rather than merely loose.

## `oracles`

Each entry is `type`, an optional `name` (defaults to the type, must be unique), the
type's parameters, and usually a `tolerance`.

```yaml
oracles:
  - type: invariant
    name: finite_state
    check: finite_state

  - type: trajectory_equivalence
    name: pole_angle_agreement
    signal: pole_angle
    metric: absolute            # absolute | relative | geodesic
    tolerance: {...}
```

See [Adding an oracle](oracles.md) for the full list of types and their parameters.

## Tolerances

This is the part people get wrong, so the schema is strict. A bare number is rejected.

```yaml
tolerance:
  value: 2.0e-3
  unit: rad
  scope: per-step absolute angle error, reduced over environments
  rationale: >
    Two milliradians is the predeclared engineering budget for this illustrative
    synthetic fixture. Its nominal trajectories do not cross the 1.2 rad event threshold,
    so this value is not presented as a termination-derived physical budget.
  aggregation: second_largest
  min_samples: 800
  kind: engineering
```

| Field | Meaning |
|---|---|
| `value` | the threshold magnitude, non-negative |
| `unit` | `rad`, `m`, `ulp`, `steps`, `fraction`, `dimensionless`… recorded verbatim |
| `scope` | prose: what the threshold applies to. It is what a reviewer reads first |
| `rationale` | why this number and not another. Under 10 characters is rejected |
| `aggregation` | `max`, `mean`, `p95`, `second_largest`, `final`, `any`, `all` |
| `min_samples` | below this the oracle returns `inconclusive`, not a guess |
| `kind` | `exact`, `numerical`, `statistical`, `event`, `engineering` |
| `minimum_meaningful_effect` | **required** for `kind: statistical` |

### Choosing `aggregation`

`mean` dilutes a single diverging environment by the environment count. `max` gates on the
single worst bifurcation tail, which in a chaotic regime can be noise. `second_largest`
trims one tail event, so at least two environments must exceed the threshold. It is the
default choice for reductions over environments when that tradeoff is intended.

### Choosing `kind`

* `exact`, bitwise identity is a real contract here (replayed actions, a same-seed rerun)
* `numerical`, a floating-point budget derived from precision, not from the workload
* `statistical`, an equivalence margin subject to sampling uncertainty
* `event`, a discrete timing budget, in `steps`
* `engineering`, a judgement about what this workload can absorb. Not a physical claim,
  and never presented as one

### Writing a rationale that survives review

Derive measured quantities explicitly when a value claims measured provenance. When a
threshold is engineering judgement, label it that way without inventing a physical story.
Good rationales look like:

* *the measured rate is X rad/s and the control frequency is Y Hz, so one sampled period
  corresponds to X/Y rad*
* *this is N float32 ULP at the characteristic magnitude of this signal, which rounding
  alone cannot reach*
* *10% of the pole length, the level at which a consumer would notice a different swing*

Bad rationales look like *"this is what passes"*. If you cannot write the good version,
that is information: you do not yet know what the comparison is for.

### Statistical tolerances

`minimum_meaningful_effect` is mandatory and load-bearing. With enough environments any
difference becomes resolvable, and without a declared floor a 0.1 mrad difference becomes
a CI failure the day someone adds environments. When the interval lies outside the margin
but the observed effect is below this floor, IVF passes the oracle and records
`IVF-EFFECT-BELOW-MEANINGFUL`.

There is also a floor the manifest cannot lower: below 8 paired samples the percentile
bootstrap describes the observed points rather than the population, so the oracle returns
`INCONCLUSIVE` regardless of what `min_samples` says.

## `verdict_policy`

```yaml
verdict_policy:
  invalid_experiment_on_control_violation: true   # default
  inconclusive_on_insufficient_samples: true      # default
  unsupported_is_failure: false                   # default
```

Leave the defaults unless you have a reason. `unsupported_is_failure: true` is occasionally
right for a CI job that must not silently skip a feature it depends on.

## Checking your work

```bash
uv run ivf validate my_manifest.yaml
```

A malformed manifest exits `64` with a path-qualified message. The resolved form, with
every default materialized, is written to the evidence bundle as
`manifest.resolved.json`, and hashed: two manifests that mean the same thing produce the
same digest regardless of key order or formatting.
