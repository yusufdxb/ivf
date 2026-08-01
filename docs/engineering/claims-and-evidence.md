# Claims and evidence

This table is the public proof boundary for v0.1.0rc1. A command returning a typed
non-pass verdict is evidence that the validator completed honestly, not evidence that the
subjects agreed.

| Claim | Evidence | Supported scope | Known limitation | Reproduction command |
|---|---|---|---|---|
| IVF installs and executes without a simulator | clean wheel installation and RC CPU transcript | Python 3.12 on the recorded Linux host | Python 3.10 and 3.11 are CI targets but were not locally re-executed for this RC closure | `env -u PYTHONPATH uv run pytest` |
| Every shipped acceptance tolerance has machine-verifiable provenance | `validation/provenance/tolerances.yaml` and drift tests | shipped example manifests | engineering budgets remain workload-specific judgement | `env -u PYTHONPATH uv run pytest tests/test_tolerance_provenance.py` |
| Strict v1 bundles reject incomplete, inconsistent or tampered capture evidence | bundle contract tests and sealed real captures | `trajectory_bundle/v1` | does not prove a simulator produced physically correct data | `env -u PYTHONPATH uv run pytest tests/test_bundle_v1.py` |
| The synthetic reset fixture is detected and localized | failing synthetic evidence bundle | declared damped-pendulum workload and seeds | explicit generation fixture, not simulator evidence | `env -u PYTHONPATH uv run ivf validate validation/examples/synthetic_fault.yaml` |
| The synthetic corrected control passes | passing synthetic evidence bundle | same synthetic workload and criteria | a pass covers only declared signals and tolerances | `env -u PYTHONPATH uv run ivf validate validation/examples/synthetic_fixed.yaml` |
| The real PhysX reset perturbation is detected | v1 PhysX captures, failing evidence, and perturbation provenance | one passive cart-pole workload on the recorded runtime | explicit simulator-level test perturbation modeled after, not identical to, a historical harness failure | `env -u PYTHONPATH uv run ivf validate validation/examples/cartpole_reset_defect.yaml` |
| The real corrected and benign PhysX controls pass | sealed v1 evidence for both comparisons | one passive cart-pole workload on the recorded runtime | does not establish cross-hardware determinism | `env -u PYTHONPATH uv run ivf validate validation/examples/cartpole_corrected.yaml` |
| A real Newton execution traverses the complete v1 path | Newton capture bundle, simulator-marked test and sealed cross-backend evidence | Newton/MJWarp cart-pole path on the recorded Isaac stack | exactly one task and backend preset were exercised | `IVF_CAPTURE_PYTHON=<isaac-python> env -u PYTHONPATH uv run pytest -m "gpu and isaaclab and newton"` |
| The recorded PhysX and Newton captures exceed the declared cross-backend criteria | valid experiment record, oracle metrics and `FAIL` evidence | captured 16-environment, 400-step passive cart-pole experiment | binary asset identity, realized initial state and backend-internal state are unverifiable | `env -u PYTHONPATH uv run ivf validate validation/examples/cartpole_physx_vs_newton_v1.yaml` |
| Shipped evidence is tamper-evident and independently verifiable offline | per-file checksums and `SEAL.json` in every evidence directory | committed evidence file contents | local read-only file modes are not a portable security boundary | `for bundle in validation/evidence/* artifacts/evidence/*; do env -u PYTHONPATH uv run ivf reproduce "$bundle" --verify-only; done` |
| Upstream parity mathematics remains green | upstream offline compare test and full parity suite in the RC transcript | recorded upstream commit | upstream suite is offline and does not repeat simulator capture | see `docs/reproduction/ivf-v0.1.0-rc1.md` |

Exact commit IDs, package hashes, evidence roots, exit codes and test totals are recorded
in `docs/reproduction/ivf-v0.1.0-rc1.md`.
