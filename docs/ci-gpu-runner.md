# CI and the GPU path

## What CI covers today

Continuous integration for this repository is **CPU-only**. The `cpu`, `workflow`, and
`package` jobs run on GitHub-hosted Ubuntu runners, which have no GPU, no CUDA device,
and no Isaac Sim or Isaac Lab installation.

Everything those jobs exercise runs on synthetic data or on trajectory bundles that were
captured earlier and committed as artifacts. That is a real and useful amount of
coverage: it proves the manifest schema, the oracles, the verdict logic, the evidence
sealing, and the checksum reproduction all behave. It proves nothing whatsoever about
whether IVF works against a live simulator.

State this plainly, because the distinction is easy to lose:

> **A green CI run is not evidence that the simulator path works.** On every hosted
> runner the simulator-backed tests skip, and a skip is not a pass.

To keep that from silently degrading, the `cpu` job asserts the skip. It runs the two
simulator modules and fails if the expected skip reasons are absent. If someone ever
makes those tests report `passed` on a CPU runner, CI goes red rather than quietly
inflating what the suite covers.

## The `simulator` job

`.github/workflows/ci.yml` defines a `simulator` job that runs the real chain: boot Kit,
capture PhysX, capture Newton, ingest, run the oracles, seal, and verify. It is gated:

```yaml
if: vars.IVF_GPU_RUNNER != ''
runs-on: ${{ vars.IVF_GPU_RUNNER }}
```

With no `IVF_GPU_RUNNER` repository variable set, the job does not execute at all. It is
absent from the check list rather than present and green. That is deliberate: an
unexercised path should look unexercised.

The job is **not** a required check and nothing depends on it, so its absence never
blocks a merge and its absence is never mistaken for coverage.

### The anti-green-skip gate

The failure mode this job exists to avoid is subtle. Every simulator test skips when
Isaac Lab is missing, which is correct behavior everywhere else. On a job whose whole
purpose is to run the simulator, a skip and a pass are indistinguishable from the
outside: a self-hosted runner that lost its Isaac Lab install after a driver upgrade
would keep reporting green while covering nothing.

`IVF_REQUIRE_SIMULATOR=1`, which the job sets, converts every missing prerequisite into
a test failure naming the prerequisite. The mechanism lives in `tests/simulator_gate.py`
and is itself tested on CPU by `tests/test_simulator_gate.py`, including a guard that
fails if a new bare `pytest.skip` appears in either simulator module.

## What a GPU runner would require

This is what someone provisioning the runner has to supply. None of it can be installed
by `uv sync`, which is why the IVF core deliberately does not depend on Isaac Lab.

| Requirement | Detail |
|---|---|
| Runner type | Self-hosted. GitHub-hosted runners have no GPU. |
| GPU | A CUDA device that Isaac Sim supports, with a driver new enough for the shipped CUDA build. |
| VRAM | Enough for a 16-environment cart-pole scene under both backends. The shipped workload is small; the floor is set by Kit itself, not by the scene. |
| Isaac Sim | A licensed install. `OMNI_KIT_ACCEPT_EULA=YES` must be set, and accepting the EULA is a decision the runner's operator makes, not one CI can make for them. |
| Isaac Lab | Installed, with `isaaclab_physx` for the PhysX backend and `isaaclab_newton` (plus `newton`, `warp-lang`, `mujoco-warp`) for the Newton backend. Both are needed: the cross-backend comparison is the point. |
| `ivf-parity-capture` | Installed **into the Isaac Lab interpreter**, from `adapters/parity_capture`. It is a separate distribution precisely so IVF core stays laptop-installable. |
| Disk | Isaac Sim is multi-gigabyte, and the first run populates shader, USD, and Warp kernel caches. Use a persistent runner; an ephemeral one pays the cold-cache cost every time. |
| Time | With warm caches one capture is single-digit seconds. With cold caches the first capture is several times that, dominated by Kit startup and kernel compilation, not by the rollout. |

### Repository variables to set

| Variable | Purpose |
|---|---|
| `IVF_GPU_RUNNER` | The self-hosted runner label. Setting it is what turns the job on. |
| `IVF_CAPTURE_PYTHON` | Absolute path to the Isaac Lab interpreter that has `ivf-parity-capture` installed. |
| `IVF_HARDWARE_LABEL` | A generic hardware label, for example `NVIDIA (Blackwell) consumer GPU`. Evidence bundles record this string verbatim, so it must not name the exact GPU model or its VRAM. The job refuses to run if it is unset. |

The job checks the last two before it captures anything, so a misconfigured runner fails
on a one-second configuration step rather than after a Kit boot.

## Running the GPU path by hand

The same chain the `simulator` job runs, on a workstation that already has the runtime:

```bash
env -u PYTHONPATH \
  IVF_REQUIRE_SIMULATOR=1 \
  IVF_CAPTURE_PYTHON=/path/to/isaaclab/env/bin/python \
  OMNI_KIT_ACCEPT_EULA=YES \
  IVF_HARDWARE_LABEL="NVIDIA (Blackwell) consumer GPU" \
  uv run pytest tests/test_simulator_end_to_end.py \
                tests/test_simulator_newton_end_to_end.py -v
```

Drop `IVF_REQUIRE_SIMULATOR` and the same command degrades to honest skips on a machine
without the runtime.

## Why the job never runs from a pull request

This repository is public and the simulator job runs on a self-hosted runner. A workflow
that let a pull request reach that runner would let anyone who can open a PR execute code
on the machine holding the Isaac Sim installation. The job is therefore gated on
`github.event_name != 'pull_request'`, so it runs only from `workflow_dispatch` and from
pushes to `main`, both of which require write access to the repository.

Exercise the GPU path deliberately:

```bash
gh workflow run ci.yml --ref <branch>
```

A pull request will always show this job as skipped. That is by design and is not evidence
about the simulator either way.
