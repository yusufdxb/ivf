# Contributing

## Setup

```bash
uv sync
env -u PYTHONPATH uv run pytest -q      # a leaked ROS PYTHONPATH breaks collection
uv run ruff check .
```

The whole CPU suite runs in about three seconds. There is no reason to skip it.

## Test markers

```
unit             isolated logic, no simulator
integration      drives the CLI or a full validate run end to end
gpu              requires a CUDA device
isaaclab         requires an importable Isaac Lab
physx / newton   requires that backend
slow             more than a couple of seconds
fault_injection  exercises the fault taxonomy / detectability matrix
reproduction     verifies evidence checksums and reproduction
```

```bash
uv run pytest -m "unit"                    # fastest loop
uv run pytest -m "fault_injection"         # the detectability matrix
uv run pytest -m "not slow"                # pre-commit
uv run pytest -m "gpu or isaaclab"         # on a workstation with the stack
```

## Rules that are not negotiable

**Missing Isaac Lab is not a pass.** A test that needs a runtime must be marked and skip
with a message naming what is missing. Silently passing because an import failed is how a
validation tool ends up validating nothing.

**No silent broad exception handling.** The runner deliberately converts an oracle
exception into an `inconclusive` outcome labelled as an IVF defect. Do not do this inside
an oracle: swallowing your own bug turns it into evidence about the subjects.

**Numerical tests document why their tolerance is valid.** In a docstring or a comment,
next to the assertion. `assert x < 1e-6` with no explanation is not reviewable.

**Anchor metrics to ground truth, not only to self-consistency.** A metric can be
symmetric, well-conditioned, double-cover-correct and still off by a constant factor. At
least one test per metric must compare against a value derived independently. This is not
hypothetical: it is how the geodesic angle's factor-of-two error was found.

**Tests do not depend on order** and do not write into the working tree. Use the
`results_root` fixture.

**Randomized tests print and persist their seed** so a failure is reproducible.

**Generated artifacts are not committed** unless they are an intentional fixture. The
current intentional fixtures are `validation/bundles/` (real PhysX/Newton captures) and
`validation/evidence/` (three sealed bundles the docs cite). Regenerating the latter
changes run ids, so update the docs that name them in the same commit.

## Adding a reason code

Add it to `ivf.verdicts.REASON_CODES` with a one-line gloss. A test walks the source and
fails on any `IVF-` string that is not registered, so a typo cannot ship. Reason codes are
a public contract: `ivf compare` diffs them across runs and CI jobs grep for them, so
renaming one is a breaking change.

## Adding an oracle

See [docs/oracles.md](docs/oracles.md). Test all three outcomes: pass, fail, and the case
where the oracle must decline to answer.

## Adding a fault class

Add a `FaultSpec` to `ivf.faults.TAXONOMY` with an honest `expected_detectable`, a
`minimum_severity`, and `limitations` when it is not universally detectable. Implement it
on the `generation` or `trace` surface, then run:

```bash
uv run ivf calibrate --seeds 11 23 47 --output docs/detectability-matrix.md
```

The campaign asserts your declaration against measured behaviour, in both directions. If
you declare a fault detectable and it is not, the matrix fails, and the correct response
is usually to fix the *declaration* and record the measured severity, not to loosen a
tolerance until the fault shows up. That is what happened with `shifted_action_timing`,
whose real detection threshold is now recorded as a measured number.

## What a review will ask

* Does a failure produce a localized, classified divergence record, or only a magnitude?
* Does every new tolerance have a rationale derived from something outside the experiment?
* Can an oracle return "I cannot tell", and does it?
* Does any new claim exceed the evidence in this repository?
* Does a `PASS` still state what it does not establish?

## Commit style

Small, logically isolated changes. Run the narrowest relevant check first, then the whole
suite. Do not claim a change works without having executed it.
