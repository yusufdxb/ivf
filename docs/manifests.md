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
    path: validation/bundles/cartpole_passive_physx_cuda0_s0_29075be3
  candidate:
    kind: parity_bundle
    label: Newton / MJWarp
    path: validation/bundles/cartpole_passive_newton-mjwarp_cuda0_s0_30d7a160
```

| `kind` | Parameters | Needs |
|---|---|---|
| `synthetic` | `system`, plus that system's parameters; optional `fault`, `fault_params` | nothing |
| `parity_bundle` | `path` to an `isaaclab_contrib.parity` trajectory bundle | nothing (numpy only) |
| `isaaclab` | backend/preset | Isaac Lab; **not implemented in this build**, returns `UNSUPPORTED` |

To run a live workload today, generate a trajectory bundle with
`isaaclab_contrib.parity` and point a `parity_bundle` subject at it.

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
```

Known controls: `asset_identity`, `action_sequence`, `initial_state_distribution`,
`observation_definition`, `control_frequency`, `seeds`, `num_envs`, `horizon`,
`task_variant`, `solver_specific_parameters`. An unknown name is rejected: silently
ignoring a control you asked for would be the worst possible failure.

`allow_different` is a claim about interpretation, not a way to silence a check. Declaring
`solver_specific_parameters` as allowed to differ means "if these subjects diverge, a
solver difference is an admissible explanation", and if neither subject records its
solver settings, IVF still reports that it cannot tell you *what* differs.

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
    2 mrad is roughly the angular resolution below which the downstream termination
    decision in this workload is unaffected: the termination threshold is 1.2 rad and the
    trajectory crosses it at about 3 rad/s, so 2 mrad corresponds to under one control
    step of timing shift.
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
single worst bifurcation tail, which in a chaotic regime is noise. `second_largest` keeps
single-environment sensitivity while trimming one tail event, and is the default choice
for reductions over environments.

### Choosing `kind`

* `exact`, bitwise identity is a real contract here (replayed actions, a same-seed rerun)
* `numerical`, a floating-point budget derived from precision, not from the workload
* `statistical`, an equivalence margin subject to sampling uncertainty
* `event`, a discrete timing budget, in `steps`
* `engineering`, a judgement about what this workload can absorb. Not a physical claim,
  and never presented as one

### Writing a rationale that survives review

Derive the number from something outside the experiment. Good rationales look like:

* *the downstream decision threshold is X and the signal crosses it at Y, so a budget of
  X/Y corresponds to under one control step*
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
