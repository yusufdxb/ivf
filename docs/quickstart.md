# Quickstart

Five minutes, on any machine. No GPU, no Isaac Lab, no Isaac Sim.

## Install

```bash
git clone https://github.com/yusufdxb/ivf.git && cd ivf
uv sync --frozen
```

`uv sync --frozen` installs from the committed lockfile without updating it, so you get the exact dependency versions
this repository was tested against. Without `uv`, `python -m pip install ".[dev]"` works
but resolves dependencies fresh and is not the locked reproduction path.

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

The candidate carries an explicit generation-level reset perturbation. Evidence metadata
records the fixture label for auditability, but the verdict logic does not consume it.
IVF exits `1`, names the failing criteria, and localizes the divergence as
`reset_mismatch` from the trajectories.

## 3. Run the same experiment with the defect removed

```bash
uv run ivf validate validation/examples/synthetic_fixed.yaml
```

Exits `0`. The manifest is identical apart from the injected defect: same oracles, same
tolerances, same seeds. This is the negative control for the whole tool. If it ever fails,
the tolerances are too tight and every failure IVF reports is suspect.

## 4. Look at the report

```bash
uv run ivf report artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --print-verdict
```

Prints the path to a single self-contained HTML file: verdict, reason codes, validity
checks, per-oracle results with their tolerance rationales, effect sizes with intervals,
the first divergence with a plotted error window, limitations, and the commands to
reproduce it.

## 5. Compare two runs

```bash
review_results="$(mktemp -d)"
uv run ivf --results-root "$review_results" validate validation/examples/synthetic_fixed.yaml
review_candidate="$(find "$review_results" -mindepth 1 -maxdepth 1 -type d -print -quit)"
uv run ivf compare validation/evidence/synthetic-reset-velocity-fixed-20260801T050934Z-c64ac825 "$review_candidate"
```

This creates one temporary rerun of the same fixed manifest and compares it with the
shipped fixed evidence. It works entirely offline. The command diffs verdicts, reason
codes, manifests, library versions, hardware, validity checks, oracle statuses, and
metrics. It refuses to compare two bundles that are not comparable rather than printing
a misleading diff.

## 6. Verify evidence someone else produced

```bash
uv run ivf reproduce artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --verify-only
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

`validation/bundles/` contains six legacy PhysX and Newton/MJWarp trajectory bundles
captured on a GPU workstation on 2026-07-12. `artifacts/` contains real
`trajectory_bundle/v1` captures from PhysX and Newton together with sealed IVF evidence.
Nothing about reading any of them needs a simulator:

```bash
uv run ivf validate validation/examples/cartpole_physx_vs_newton_v1.yaml
```

The experiment is valid under its declared controls and ends in `FAIL`: pole angle and
pole angular rate exceed their predeclared budgets, termination is two steps earlier on
Newton, and the 400-step survive/terminate decision remains identical. Solver settings
are recorded and explicitly allowed to differ. Binary asset identity, realized backend
state, and backend-internal state remain unverifiable and are listed as such.

## Next

* [Validation contracts](concepts.md) for why it is shaped this way
* [Writing a manifest](manifests.md) to describe your own comparison
* [Five-minute review](review-in-five-minutes.md) for the flagship evidence path
* [Case study](case-study.md) for the real PhysX-versus-Newton acceptance result
