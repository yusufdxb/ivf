# Oracles

An oracle answers one question about one pair of subjects. It never decides the run
verdict; the runner does that by rolling outcomes up under the manifest's policy. That
split is what lets an oracle honestly return "I cannot tell" instead of guessing.

## Shipped oracles

### `invariant`

Properties that must hold with no reference run. The cheapest and most trustworthy checks
IVF has, and the only defense against a defect that affects both subjects identically: a
shared-layer bug agrees perfectly with itself and passes every pairwise comparison.

```yaml
- type: invariant
  name: finite_state
  check: finite_state          # finite_state | unit_quaternion | bounded
  signals: [pole_angle]        # optional; defaults to all shared signals
  # bounded also takes lower / upper
```

### `trajectory_equivalence`

Pointwise agreement against a numerical or engineering budget. Produces a full divergence
record on failure, not just a number.

```yaml
- type: trajectory_equivalence
  signal: pole_angle
  metric: absolute             # absolute | relative | geodesic
  tolerance: {...}
```

`geodesic` requires a 4-vector signal and returns the rotation angle between two
orientations, handling the quaternion double cover (`q` and `-q` are the same rotation, and
a componentwise metric reports up to 2 rad for them). It is computed from the chord as
`4*arcsin(d/2)` rather than `2*arccos(|q_a . q_b|)`: the two agree in exact arithmetic, but
arccos is ill-conditioned exactly where this oracle spends its time, turning float64
rounding on identical rotations into ~3e-8 rad of spurious angle.

### `exact_equivalence`

Bitwise identity. Appropriate only where identity is a real contract: replayed actions,
recorded metadata, a same-backend same-seed rerun. Using it across backends is a category
error.

### `ulp_equivalence`

Distance in units in the last place, against a declared ULP budget.

```yaml
- type: ulp_equivalence
  signal: root_link_pos_w
  precision: float32           # float32 | float64
  tolerance: {value: 1, unit: ulp, kind: numerical, ...}
```

The right yardstick for "the same computation on the same hardware". The wrong yardstick
across backends, where the computations differ by design. A ULP claim is scoped to a
precision and a device and does not transfer.

### `trajectory_envelope`

Membership of the candidate in a band derived from a baseline *ensemble*. Returns
`skipped` on a single-environment baseline rather than manufacturing a degenerate band
from one point.

### `event_equivalence`

When and whether a semantic event fires. Reports two failures separately because they mean
different things: an event firing in one subject and not the other is a behavioural change
(`IVF-ORACLE-EVENT-COUNT-MISMATCH`), while an event firing in both at different times is a
timing delta in steps (`IVF-ORACLE-EVENT-TIMING-DELTA`).

```yaml
- type: event_equivalence
  event: termination
  signal: abs_pole_angle
  condition: above             # above | below
  threshold: 1.2
  tolerance: {value: 1, unit: steps, kind: event, ...}   # unit must be 'steps'
```

### `decision_equivalence`

The oracle that answers "does this change alter what I would *do*". A binary decision per
environment, derived from an event, compared for agreement.

```yaml
- type: decision_equivalence
  decision: episode_success
  signal: abs_pole_angle
  condition: above
  threshold: 1.2
  polarity: survived           # survived | triggered
  tolerance: {value: 1.0, unit: fraction, ...}           # unit must be 'fraction'
```

With `n` environments the finest resolvable agreement rate below 1.0 is `1 - 1/n`; asking
for a finer one returns `INCONCLUSIVE` rather than a number the sample cannot support.

### `statistical_equivalence`

Paired equivalence of a per-environment summary against a declared margin. See
[Validation contracts](concepts.md) for why this is an equivalence test rather than a
significance test.

```yaml
- type: statistical_equivalence
  signal: pole_angle
  summary: mean                # mean | final | max | abs_max | rms
  tolerance:
    kind: statistical
    minimum_meaningful_effect: 1.0e-3    # required
    ...
```

Intervals are percentile bootstrap on the paired per-environment differences: paired
because both subjects ran the same seeds, bootstrap because it assumes no distribution and
needs no SciPy. The resample seed is recorded, so the interval is reproducible. When a
manifest declares several statistical oracles, the family-wise false-alarm budget is
Šidák-split across them.

### `metamorphic`

Relations that hold without anyone knowing the correct output.

```yaml
- type: metamorphic
  relation: env_permutation_invariance
  signal: pole_angle
  tolerance: {...}
```

| Relation | Asserts | Valid when |
|---|---|---|
| `env_permutation_invariance` | per-environment outcomes do not depend on environment ordering | environments are independent. False when they interact (shared ground contact across origins, global randomization coupling). IVF cannot verify this precondition |
| `action_replay_consumption` | a replayed run consumed exactly the recorded action stream | a source records actions or their hash. Returns `skipped`, not `pass`, otherwise |
| `observation_definition_stability` | output formatting did not change the observation definition | always |
| `quaternion_double_cover` | the pose metric treats `q` and `-q` as identical | the signal is a 4-vector |

### `required_features`

Verifies the environment provides what the manifest depends on. Returns `unsupported`,
never `fail`: an absent backend capability is a property of the machine, not a defect in
the change under test. It trusts each subject's own feature declaration, and the fault
taxonomy records that gap explicitly rather than hiding it.

## Adding an oracle

Three steps. Nothing in the runner changes.

**1. Write the function.** It takes an `OracleContext` and returns an `OracleOutcome`.

```python
# src/ivf/oracles/energy.py
from __future__ import annotations

import numpy as np

from ..divergence import localize
from .base import OracleContext, OracleOutcome, aggregate, register


@register("energy_conservation")
def energy_conservation(ctx: OracleContext) -> OracleOutcome:
    """Require the candidate's total energy drift to match the baseline's.

    Passive scenes should not gain energy. This compares drift rather than absolute
    energy, so it does not need the two subjects to agree on the potential's zero.
    """
    signal = str(ctx.spec.params["signal"])
    tol = ctx.tolerance
    baseline, candidate = ctx.signal_pair(signal)

    base_drift = baseline[-1] - baseline[0]
    cand_drift = candidate[-1] - candidate[0]
    error = np.abs(cand_drift - base_drift)
    worst = aggregate(error, tol.aggregation)
    metrics = {"worst_drift_difference": worst, "n_samples": int(baseline.shape[1])}

    if baseline.shape[1] < tol.min_samples:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="inconclusive",
            summary=f"{signal}: {baseline.shape[1]} environments is below the declared minimum",
            reason_codes=["IVF-SAMPLE-INSUFFICIENT"], metrics=metrics,
            tolerance=tol.to_jsonable(),
        )
    if worst <= tol.value:
        return OracleOutcome(
            name=ctx.spec.name, type=ctx.spec.type, status="pass",
            summary=f"{signal}: energy drift agrees to {worst:.3e} {tol.unit}",
            metrics=metrics, tolerance=tol.to_jsonable(),
            limitations="Agreement in drift does not establish either subject conserves energy.",
        )
    return OracleOutcome(
        name=ctx.spec.name, type=ctx.spec.type, status="fail",
        summary=f"{signal}: energy drift differs by {worst:.3e} {tol.unit}",
        reason_codes=["IVF-ORACLE-NON_EQUIVALENT"], metrics=metrics,
        tolerance=tol.to_jsonable(),
    )
```

**2. Import it** in `src/ivf/oracles/__init__.py` so registration happens.

**3. Test all three outcomes** in `tests/test_oracles.py`: the pass, the fail, and the
case where it must decline to answer.

### Rules an oracle must follow

* **Never turn missing data into a pass.** Too few samples is `inconclusive` with
  `IVF-SAMPLE-INSUFFICIENT`. A precondition that does not hold is `skipped`.
* **Never turn a missing capability into a failure.** That is `unsupported`.
* **Emit only registered reason codes.** A test walks the source and fails on any
  `IVF-` string not in `ivf.verdicts.REASON_CODES`.
* **Put everything a reader needs to disagree into `metrics`.** Effect sizes, intervals,
  observed maxima, sample counts.
* **State limitations.** `limitations` is what the result does *not* establish, and it is
  rendered in the report.
* **Localize failures.** Call `ivf.divergence.localize` so the report can name the first
  step, the environments and the components rather than only a magnitude.
* **Do not catch broadly.** An oracle that raises is turned into an `inconclusive` outcome
  by the runner and labelled as an IVF defect, which is correct; swallowing it yourself
  turns your bug into evidence about the subjects.
