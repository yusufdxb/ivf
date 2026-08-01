# Independent v0.1.0-rc1 reproduction

This guide assumes Linux, Git, Python 3.12, UV, and no existing IVF checkout. The CPU
track is required for independent review. The simulator track is optional and requires a
supported NVIDIA GPU and Isaac installation.

The release tag resolves to the public release commit. Record the resolved commit in the
[reviewer result template](reviewer-result-template.md) before running tests.

## Track A: required CPU-only review

### 1. Clone and resolve the release

```bash
git clone https://github.com/yusufdxb/ivf.git
cd ivf
git checkout --detach v0.1.0-rc1
git status --short
git rev-parse HEAD
```

`git status --short` must be empty. The release page must name the same commit printed by
`git rev-parse HEAD`.

### 2. Install locked dependencies without installing IVF editable

```bash
env -u PYTHONPATH uv venv .venv-review --python 3.12
env -u PYTHONPATH UV_PROJECT_ENVIRONMENT=.venv-review uv sync --frozen --no-install-project
env -u PYTHONPATH uv build
env -u PYTHONPATH uv pip install --python .venv-review/bin/python --no-deps dist/ivf-0.1.0rc1-py3-none-any.whl
env -u PYTHONPATH .venv-review/bin/python -c "import ivf; print(ivf.__version__, ivf.__file__)"
```

The import path must be inside `.venv-review/lib/`, not the repository `src/` directory.
No inherited `PYTHONPATH` is used.

### 3. Run lint and the CPU suite

```bash
env -u PYTHONPATH .venv-review/bin/ruff check .
env -u PYTHONPATH .venv-review/bin/pytest -q -rs
```

Simulator-marked tests may skip because the external Isaac Lab interpreter is not
configured. Record every skip and its reported reason.

### 4. Exercise validation and evidence verification

```bash
env -u PYTHONPATH .venv-review/bin/ivf doctor
env -u PYTHONPATH .venv-review/bin/ivf validate validation/examples/synthetic_fixed.yaml
env -u PYTHONPATH .venv-review/bin/ivf reproduce artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --verify-only
env -u PYTHONPATH .venv-review/bin/ivf report artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --print-verdict
```

Expected results are synthetic `PASS`, flagship integrity `ok`, and recorded flagship
`FAIL`. The latter is a completed scientific verdict, not a command failure.

Verify every shipped bundle:

```bash
for bundle in artifacts/evidence/* validation/evidence/*; do
  env -u PYTHONPATH .venv-review/bin/ivf reproduce "$bundle" --verify-only
done
```

### 5. Inspect the static HTML report

```bash
python3 -m http.server 8000
```

Open
`http://127.0.0.1:8000/artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912/report.html`
and then stop the server with Ctrl-C. Confirm the typed verdict, reason codes, 22 validity
checks with zero failed, three unverifiable controls, numerical tables, and divergence plots.

## Track B: optional simulator-backed review

The recorded supported runtime is Isaac Sim 6.0.0.1 with Isaac Lab 10.2.0. The verified
packages include Isaac Lab PhysX 2.8.1, Isaac Lab Newton 1.7.0, Newton 1.4.0.dev0,
PyTorch 2.11.0+cu128, and the Newton MJWarp preset. Other versions are unverified.

Set the Isaac Lab interpreter explicitly. Do not source a workspace that injects another
Python installation. Every command removes `PYTHONPATH` to prevent ROS or host packages
from contaminating the simulator process.

```bash
export ISAACLAB_PYTHON=/path/to/isaaclab/python
env -u PYTHONPATH "$ISAACLAB_PYTHON" -m pip install . ./adapters/parity_capture
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  "$ISAACLAB_PYTHON" -m parity_capture.cli doctor
```

`OMNI_KIT_ACCEPT_EULA=YES` records non-interactive acceptance of the Kit EULA. Set a
coarse public hardware label before creating evidence intended for sharing.

### PhysX capture: required for the simulator track

```bash
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  IVF_HARDWARE_LABEL='NVIDIA GPU' \
  "$ISAACLAB_PYTHON" -m parity_capture.cli \
  validation/capture/cartpole_physx.yaml \
  --output review-artifacts/cartpole-physx

env -u PYTHONPATH .venv-review/bin/python -c "from pathlib import Path; from ivf.bundle import load_v1; load_v1(Path('review-artifacts/cartpole-physx')); print('strict v1 load ok')"

env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  IVF_CAPTURE_PYTHON="$ISAACLAB_PYTHON" \
  IVF_HARDWARE_LABEL='NVIDIA GPU' \
  .venv-review/bin/pytest -q -rs -m "gpu and isaaclab and physx"
```

The direct capture must finish all 400 steps, write its completion marker last, and pass
strict v1 loading. The marker subset genuinely launches PhysX and is expected to pass
five tests on the recorded stack.

### Newton/MJWarp capture: optional

```bash
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  IVF_HARDWARE_LABEL='NVIDIA GPU' \
  "$ISAACLAB_PYTHON" -m parity_capture.cli \
  validation/capture/cartpole_newton.yaml \
  --output review-artifacts/cartpole-newton

env -u PYTHONPATH .venv-review/bin/python -c "from pathlib import Path; from ivf.bundle import load_v1; load_v1(Path('review-artifacts/cartpole-newton')); print('strict v1 load ok')"

env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  IVF_CAPTURE_PYTHON="$ISAACLAB_PYTHON" \
  IVF_HARDWARE_LABEL='NVIDIA GPU' \
  .venv-review/bin/pytest -q -rs -m "gpu and isaaclab and newton"
```

The Newton marker subset genuinely launches Newton/MJWarp and is expected to pass one
test on the recorded stack. If Newton is unavailable, record the full error and classify
it as unsupported runtime, environment failure, or implementation defect. Do not replace
it with a mock result.

### Validate the shipped cross-backend evidence

```bash
env -u PYTHONPATH .venv-review/bin/ivf reproduce artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --verify-only
env -u PYTHONPATH .venv-review/bin/ivf validate validation/examples/cartpole_physx_vs_newton_v1.yaml
```

The first command must report integrity `ok`. The second is expected to return `FAIL`
with exit code `1`, `IVF-ORACLE-NON_EQUIVALENT`, and
`IVF-ORACLE-EVENT-TIMING-DELTA`.

The committed evidence can be verified across machines, but bit-identical simulator
trajectories or rebuilt archives are not expected across hardware, drivers, operating
systems, or build times.
