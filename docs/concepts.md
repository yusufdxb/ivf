# Validation contracts

Everything in IVF follows from one observation: **most simulation regressions are not
caught by a test failing, they are caught by a person noticing a robot looks wrong.**
The gap is not test coverage. It is that nobody wrote down, in advance, what "the same
behaviour" was supposed to mean.

A validation contract is that statement, written before the data exists.

## A contract has four parts

**Subjects.** The two things being compared, and where their signals come from.

**Controls.** The properties that must be identical for the comparison to mean anything:
the asset, the action stream, the initial-state distribution, the observation definition,
the control frequency. A control is *declared*, not inferred. IVF checks exactly what you
declared, and it tells you when it could not check something you asked for.

**Oracles.** The acceptance criteria. Each answers one question about the pair, and each
carries a fully qualified tolerance: a value, a unit, a scope, a rationale, an
aggregation rule, a minimum sample count, and a kind.

**A verdict policy.** How the individual answers roll up into one decision.

The manifest is that contract, in YAML, hashed so a run can be traced back to it.

## Why controls come first

The most damaging thing a validation tool can do is publish a confident comparison of two
things that were never comparable. It is worse than reporting nothing, because someone
will act on it.

So the validity layer runs before any oracle and can veto the entire run. If the timesteps
differ, the verdict is `INVALID_EXPERIMENT`, and the numbers are recorded but decide
nothing. Comparing two trajectories at different control frequencies step-by-step is not a
loose comparison, it is a meaningless one.

A third status matters as much as pass and fail: **unverifiable**. When a subject does not
record the metadata a control needs, IVF says so explicitly. Absence of evidence is never
treated as evidence of a match. This is not hypothetical: the cross-backend bundles
shipped in this repository record no solver settings at all, and the example manifest that
uses them reports exactly that.

## Why tolerances carry their reasoning

`tolerance: 0.002` is rejected by the schema. Over what? Reduced how across environments?
Below how many samples is it meaningless? Is it a physical claim, a floating-point budget,
or someone's judgement about what the workload can absorb?

Every one of those changes what the number means, and none of them is recoverable from the
number. So the schema demands them, and the report prints the rationale next to any
criterion that failed. The person reading your regression is then arguing with your
reasoning rather than guessing at it.

The `kind` field is the honesty marker. An `engineering` tolerance is a judgement about a
workload, and it is never presented as a physical claim.

## Why the aggregation rule is explicit

Reducing a per-environment error curve to one number is a real decision with real
consequences:

* `mean` dilutes a single diverging environment by the environment count. With 16
  environments, one broken robot is a 16× smaller signal.
* `max` gates on the single worst bifurcation tail, which in a chaotic regime is noise.
* `second_largest` keeps single-environment sensitivity while trimming one tail event.

The pre-existing parity work settled on `second_largest` for environment reductions after
exactly this argument. IVF does not pick for you; it makes you pick, and records it.

## Why the statistics are equivalence tests

With enough environments, every difference is statistically detectable. A significance
test therefore answers a question nobody asked, and answers it "yes" as soon as you buy a
bigger GPU.

The question that matters is whether the difference is small enough not to matter. So the
decision rule is a confidence interval against a declared equivalence margin:

* interval inside the margin → equivalent
* interval outside, effect at or above the declared minimum meaningful effect → a real
  regression
* interval outside, effect below the minimum meaningful effect → real, but not
  operationally meaningful; reported with `IVF-EFFECT-BELOW-MEANINGFUL`
* interval straddling the margin → `INCONCLUSIVE`

That last case is the one this design exists for. **Failing to prove equivalence is not
proof of equivalence**, and a tool that reports it as a pass is lying by omission.

## Why detection has to be measured

A validator that has never been shown a defect it failed to catch is a validator whose
detection claims are decoration.

`ivf calibrate` injects a versioned fault taxonomy through the real `validate` path and
produces a detectability matrix: per fault, whether detection was expected, which layer
caught it, the measured detection rate over several seeds, and the false-positive result
from the negative control. Rows that declare a fault *undetectable* are asserted to not
fire, which is what keeps an honest limitation from quietly becoming a detection claim.

The matrix's own limitation is stated in its header: the taxonomy is author-generated, so
a perfect matrix establishes that IVF catches the faults its authors enumerated, not that
it catches faults nobody thought of.

## Why evidence is sealed

A verdict you cannot re-derive is an opinion. An evidence bundle is JSON, YAML, plain text
and `.npz`, readable with numpy and nothing else, checksummed per file, sealed, and made
read-only. Six months later on a different machine, `ivf reproduce --verify-only` will
tell you whether what you are reading is what was produced.

## What IVF deliberately does not decide

* **Which subject is correct.** Neither backend is a reference. Exceeding a tolerance
  establishes that they differ.
* **Whether agreement means correctness.** Two implementations can be identically wrong;
  only the reference-free invariants catch that, and they are the weakest claim in the
  stack.
* **Whether a divergence predicts policy transfer failure.** Unestablished, and it stays
  that way until someone measures it.
