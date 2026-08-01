# IVF v0.1.0rc1 reproduction record

Recorded on 2026-08-01. This record was assembled from command output, including
non-zero expected verdict exits. The implementation and evidence candidate reproduced in
the clean run was:

```text
commit 66fb3c776eb7e4051defc351b60970d42d617814
tree   51330534560e4b072dd4ac3d056a129d0ae2ee6e
branch detached
```

The commit containing this record necessarily has a different commit hash. A Git commit
cannot contain its own hash. The release handoff records the literal final hash and a
second clean run against that exact checkout. The auditable identity inside this file is
therefore the reproduced source-and-evidence commit above, plus the final `git rev-parse
HEAD` output in the release handoff.

## Clean environment

- Worktree: a new detached Git worktree, separate from the development checkout
- OS: Ubuntu 22.04.5 LTS, Linux 6.8.0-136-generic x86_64
- Python: CPython 3.12.13
- IVF: 0.1.0rc1
- numpy: 2.5.1
- PyYAML: 6.0.3
- pytest: 9.1.1
- Ruff: 0.16.1
- `PYTHONPATH`: removed for every command
- `PYTHONHASHSEED`: 0 for execution checks
- hardware label in shareable evidence: `NVIDIA Blackwell consumer GPU`
- driver label: 570.211.01
- installation: wheel, non-editable; import resolved from the clean virtual
  environment's `site-packages`

The locked setup and build commands were:

```bash
git worktree add --detach <new-repository-worktree> 66fb3c776eb7e4051defc351b60970d42d617814
env -u PYTHONPATH uv venv .venv-rc1 --python 3.12
env -u PYTHONPATH UV_PROJECT_ENVIRONMENT=.venv-rc1 uv sync --frozen --no-install-project
env -u PYTHONPATH uv build
env -u PYTHONPATH uv pip install --python .venv-rc1/bin/python --no-deps \
  dist/ivf-0.1.0rc1-py3-none-any.whl
```

`uv build` returned 0 and emitted one setuptools deprecation warning about the TOML table
form of `project.license`; the package still built successfully. Package digests from
that run were:

```text
d8696bde59a7b909916d9c1fc753db3e061ca068dcf7655b644ca4c906051f52  ivf-0.1.0rc1.tar.gz
2ebb67a66d62feac79923bb5bef90ec220b0b588f32fcfd53a199394cdda3f29  ivf-0.1.0rc1-py3-none-any.whl
```

These are artifact-instance hashes, not a claim that archives built without a fixed
`SOURCE_DATE_EPOCH` are byte-identical across build times.

## Lint and tests

```text
ruff check .
exit 0: All checks passed

pytest -q -rs
exit 0: 206 passed, 5 skipped
```

All five skips named the missing external simulator capture command in the wheel-only
environment: four PhysX tests require `IVF_CAPTURE_PYTHON`, and one Newton test requires
the same Isaac Lab interpreter. They were executed separately on the recorded simulator
stack and passed.

Marker results:

| Command | Exit | Result |
|---|---:|---|
| `pytest -m unit` | 0 | 188 passed, 23 deselected |
| `pytest -m integration -rs` | 0 | 24 passed, 5 simulator skips, 182 deselected |
| `pytest -m fault_injection` | 0 | 7 passed, 204 deselected |
| `pytest -m reproduction` | 0 | 9 passed, 202 deselected |
| `pytest -m "gpu and isaaclab and physx"` | 0 | 5 passed, 206 deselected |
| `pytest -m "gpu and isaaclab and newton"` | 0 | 1 passed, 210 deselected |

The two simulator marker commands used the exact Isaac Lab interpreter through
`IVF_CAPTURE_PYTHON`, accepted the Kit EULA, removed `PYTHONPATH`, and applied the coarse
public hardware label.

## Documented CPU workflows

Every command used the installed wheel and returned the typed exit expected by its
manifest:

| Command | Exit | Verdict | Reason codes |
|---|---:|---|---|
| `ivf doctor` | 0 | offline and synthetic available | none |
| `ivf validate validation/examples/synthetic_fault.yaml` | 1 | `FAIL` | `IVF-ORACLE-NON_EQUIVALENT` |
| `ivf validate validation/examples/synthetic_fixed.yaml` | 0 | `PASS` | none |
| `ivf validate validation/examples/cartpole_reset_defect.yaml` | 1 | `FAIL` | `IVF-ORACLE-NON_EQUIVALENT`, `IVF-ORACLE-EVENT-TIMING-DELTA` |
| `ivf validate validation/examples/cartpole_corrected.yaml` | 0 | `PASS` | none |
| `ivf validate validation/examples/cartpole_benign_difference.yaml` | 0 | `PASS` | none |
| `ivf validate validation/examples/cartpole_physx_vs_newton_v1.yaml` | 1 | `FAIL` | `IVF-ORACLE-NON_EQUIVALENT`, `IVF-ORACLE-EVENT-TIMING-DELTA` |
| `ivf validate validation/examples/cartpole_cross_backend.yaml` | 2 | `INCONCLUSIVE` | `IVF-SAMPLE-INSUFFICIENT` |

Two controlled executions of `synthetic_fixed.yaml` were compared:

```text
ivf compare <first-fixed-evidence> <second-fixed-evidence>
exit 0
verdict PASS == PASS
No material differences.
```

The generated `reproduce.sh` from the shipped passing synthetic evidence also executed
successfully and produced `PASS`.

## Shipped evidence verification

`ivf reproduce <bundle> --verify-only` returned 0 for all seven committed evidence
bundles. The evidence-root value is the `checksums_sha256` field in `SEAL.json`.

| Experiment | Verdict | Evidence root SHA-256 |
|---|---|---|
| legacy PhysX versus Newton | `INCONCLUSIVE` | `e6655ed33a3720c553b6b38acb2d942669f8ac01a3c8e00b3adcb43be8ef2479` |
| synthetic reset fixture | `FAIL` | `cad710da363f05136c03936c847150097ac0d07e597b734436cbcef3efa05e0a` |
| synthetic corrected | `PASS` | `3872adb41cd38729c8f27d69cfe671d7829a239264bd2ab21800cb693de1eaec` |
| PhysX benign difference | `PASS` | `7122c817bd2559f9eabbf645906a815a56fe3d4b8265b5e45197c4c52130180b` |
| PhysX corrected | `PASS` | `c2d6e4c5f20b7e43c65f85640ba16a8cd16b894a4a9c7d1831569f1c87076c01` |
| PhysX reset perturbation | `FAIL` | `ca3ebcf2b4800808180e01b1ef85cf2307f4bc98dbc174ee14949d6364399157` |
| PhysX versus Newton v1 | `FAIL` | `8c72dba08d50230aab22e99bb13cbd34a4810858e2456437f99e0df25da01e49` |

The strict raw capture roots from each `COMPLETE` marker are:

| Capture | Root SHA-256 |
|---|---|
| Newton baseline | `86ac7101ce268106b963a45301559476184a6085a0e3439cba7b4d59b2744565` |
| PhysX baseline | `0947a726333914db45727c138f17152995aec5bda3c6d862e1501e4f7e4a0794` |
| PhysX benign | `21dfab878b225cba9802ce5376ec7701437d5d0e08acf84efdd6881317a21b77` |
| PhysX corrected | `3ad826ba9aa1fa758d01f7f55977a053fa5117dbff4a88596613dad88af00536` |
| PhysX reset perturbation | `3452705e34b358582227a8fe4a9cfd9f040e5f6ed8bf6e8dda983673c4b76fea` |

## Real simulator runtime and Newton path

Runtime observed from the real command output and installed metadata:

| Component | Value |
|---|---|
| Isaac Sim | 6.0.0.1 |
| Isaac Lab | 10.2.0 |
| Isaac Lab PhysX package | 2.8.1 |
| Isaac Lab Newton package | 1.7.0 |
| Newton | 1.4.0.dev0 |
| Kit | 110.1.1+production.305458.6312fa25.gl |
| PyTorch | 2.11.0+cu128 |
| CUDA | PyTorch 12.8; Warp toolkit 12.9 with driver API 12.8 |
| Warp | 1.15.0.dev20260626 |
| driver | 570.211.01 |
| GPU reported by runtime | NVIDIA Blackwell consumer GPU |
| Newton preset | `NewtonCfg(solver_cfg=MJWarpSolverCfg())` |

Exact capture command:

```bash
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  IVF_HARDWARE_LABEL='NVIDIA Blackwell consumer GPU' \
  IVF_DRIVER_LABEL='570.211.01' \
  <isaac-lab-python> -m parity_capture.cli \
  validation/capture/cartpole_newton.yaml \
  --output artifacts/cartpole-newton-baseline
```

It returned 0, captured 400 of 400 steps, and wrote a completed strict v1 bundle. The
bundle contains normalized task configuration, backend and solver settings, requested and
applied reset state, software and hardware metadata, seven trajectory signals, and an
explicit `(400, 16, 2)` zero-effort action tensor. Strict loading, the Newton simulator
test, IVF execution, typed `FAIL`, evidence sealing and offline verification all passed.

Relevant runtime warnings were protobuf duplicate descriptor registration, MaterialX
availability, deferred USD-stage subscription, Newton shape-color replacement
deprecation, muted USD diagnostics, and the expected fixed-root warning that root center
of-mass velocity was unavailable and set to zero. None interrupted capture. The last
warning limits any interpretation of fixed-root COM velocity; that signal is not an
acceptance signal in the v1 manifest.

## Cross-backend result

The v1 experiment was valid. It verified task and normalized configuration identity,
requested initial-state distribution, the explicit action stream, observation definition,
physics and control timestep, seed and environment order, task variant, frame and
quaternion conventions, reset semantics, action timing, environment count, horizon,
warm-up and completion. Solver-specific parameters were recorded and allowed to differ.

Binary asset identity, realized backend initial state and backend-internal state were
explicitly unverifiable. They were not treated as matches.

| Metric | Result |
|---|---:|
| worst second-largest pole-angle error | 3.533347845 rad |
| first angle violation | step 29 |
| worst second-largest pole-rate error | 12.99448848 rad/s |
| first rate violation | step 13 |
| worst and median event timing delta | 2 steps |
| first event disagreement | step 35 |
| survive or terminate agreement | 1.0 |
| mean paired angle difference | -0.18621337 rad |
| 95% confidence interval | [-0.18767944, -0.18477211] rad |
| Cohen's dz | -60.0368 |

Verdict: `FAIL`, with `IVF-ORACLE-NON_EQUIVALENT` and
`IVF-ORACLE-EVENT-TIMING-DELTA`.

This result does not prove either backend physically correct, generalize beyond the
recorded workload and runtime, establish cross-hardware determinism, or predict
learned-policy transfer or sim-to-real behavior.

## Upstream parity regression

The upstream checkout was
`290149083ec5060111ba6fd26b1213c55d9be59c` on
`yusufdxb/feature-parity-harness` and remained clean.

| Command | Exit | Result |
|---|---:|---|
| `pytest source/isaaclab_contrib/test/parity/test_parity_offline_compare.py` | 0 | 12 passed |
| `pytest source/isaaclab_contrib/test/parity` | 0 | 133 passed |

## Limitations

- The clean local reproduction is one Linux/Python host, not a cross-platform or
  cross-hardware determinism study.
- The simulator closure is one passive cart-pole workload and one Newton/MJWarp preset.
- The reset case is an explicit test-only perturbation modeled after a historical harness
  failure, not a simulator-discovered product defect and not a literal restoration.
- Statistical intervals describe the recorded paired environments and declared method;
  they do not establish population-wide backend behavior.
- The build warning about the deprecated setuptools license-table form is non-fatal but
  remains for a future package-metadata maintenance release.
