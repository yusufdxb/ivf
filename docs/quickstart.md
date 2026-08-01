# Quickstart

Five minutes, on any machine. No GPU, no Isaac Lab, no Isaac Sim.

## Install

```bash
git clone <this repo> && cd ivf
uv sync
```

`uv sync` installs from the committed lockfile, so you get the exact dependency versions
this repository was tested against. Without `uv`, `pip install -e ".[dev]"` works too but
resolves dependencies fresh.

## 1. Ask what this machine can do

```bash
uv run ivf doctor
```

It reports three capability levels. `offline` (read, verify, compare and render evidence)
and `synthetic` (also execute the numpy reference workloads) need nothing but Python and
numpy. `isaaclab` needs the simulator stack and a GPU, and on most machines it will
correctly report `none`. That is not an error; two thirds of IVF works without it.

To make a CI job fail early when a capability is missing:

```bash
uv run ivf doctor --require synthetic     # exits nonzero and names the failing probe
```

## 2. Run an experiment that should fail

```bash
uv run ivf validate validation/examples/synthetic_fault.yaml
```

The candidate carries a silently injected defect: a reset that writes position but drops
velocity. Nothing in its recorded provenance reveals it. IVF exits `1`, names the failing
criteria, and localizes the first divergence to step 1 across all 24 environments,
classifying it as `reset_mismatch`.

## 3. Run the same experiment with the defect removed

```bash
uv run ivf validate validation/examples/synthetic_fixed.yaml
```

Exits `0`. The manifest is identical apart from the injected defect: same oracles, same
tolerances, same seeds. This is the negative control for the whole tool. If it ever fails,
the tolerances are too tight and every failure IVF reports is suspect.

## 4. Look at the report

```bash
uv run ivf report ivf-results/<run-id>
```

Prints the path to a single self-contained HTML file: verdict, reason codes, validity
checks, per-oracle results with their tolerance rationales, effect sizes with intervals,
the first divergence with a plotted error window, limitations, and the commands to
reproduce it.

## 5. Compare two runs

```bash
uv run ivf compare ivf-results/<baseline> ivf-results/<candidate>
```

Works entirely offline. It diffs verdicts, reason codes, manifests, library versions,
hardware, validity checks, oracle statuses and metrics, and refuses to compare two
bundles that are not comparable rather than printing a misleading diff.

## 6. Verify evidence someone else produced

```bash
uv run ivf reproduce validation/evidence/<run-id> --verify-only
```

Recomputes every file digest against the sealed `CHECKSUMS.sha256`. Drop `--verify-only`
to also re-execute the manifest and diff the two runs.

### If `uv run pytest` cannot import `ivf`

On a host that exports a global `PYTHONPATH` (a sourced ROS 2 workspace is the usual
cause), that path leaks into the project environment and pytest loads plugins from a
different Python's site-packages. Run the suite hermetically instead:

```bash
env -u PYTHONPATH uv run pytest -q
```

Nothing in IVF depends on an inherited `PYTHONPATH`.

## Real cross-backend data

`validation/bundles/` contains six genuine PhysX and Newton/MJWarp trajectory bundles
captured on a GPU workstation on 2026-07-12, and `artifacts/` contains four
`trajectory_bundle/v1` captures produced by a live Isaac Lab run together with their
sealed verdicts. Nothing about reading any of them needs a simulator:

```bash
uv run ivf validate validation/examples/cartpole_cross_backend.yaml
```

This one is worth reading closely. It ends in `FAIL` on the pointwise joint-position
oracle, returns `INCONCLUSIVE` on the statistical oracle because four environments cannot
support an equivalence claim, and reports the solver-settings control as **unverifiable**
because the released bundles record no solver configuration at all. Three different
honest answers in one run.

## Next

* [Validation contracts](concepts.md) for why it is shaped this way
* [Writing a manifest](manifests.md) to describe your own comparison
* [Case study](case-study.md) for the full fail → localize → fix → pass loop
