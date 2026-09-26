# Frozen IVF under evaluation

Recorded 2026-09-26, before any corpus case was evaluated.

The object under test is IVF exactly as it exists at the commit below. Nothing under
`src/ivf/`, `adapters/`, `validation/`, or `uv.lock` may change on this branch. The
evaluation harness (`research/failure_corpus/eval/`) refuses to run if the `src/ivf`
tree hash differs from the value recorded here.

| Item | Value |
|---|---|
| Repository | public IVF repository, `main` |
| Commit | `d64f74049a7e2b64bba5289f5627f136d71bbe4f` |
| `src/ivf` tree hash | `4932658076919e65c87ecb0e358b3fd3f27336c3` |
| `adapters` tree hash | `555c96ff55f96064a4dc7f6384f13c11cf7f6018` |
| `uv.lock` sha256 | `2de96e4f3e877f34ebb932fee8395626d8e4f9a68936d8a031cb8649810113f6` |
| Package version | `ivf 0.1.0rc2` |
| uv | 0.11.6 |
| Host Python for IVF | 3.10.12 via `uv run --frozen` |

## Behavioral baseline at freeze

Executed on this commit, output observed:

- `env -u PYTHONPATH uv run --frozen pytest -q`: **265 passed, 5 skipped** (skips are the
  simulator-backed tests).
- `uv run --frozen ruff check .`: All checks passed.
- `uv run --frozen ivf calibrate --seeds 11 23 47`: 17/17 fault classes behave as declared,
  0 false positives. Full output: [`ivf_calibrate_frozen.txt`](ivf_calibrate_frozen.txt).

## Simulator stack used to produce corpus captures

| Component | Version |
|---|---|
| Isaac Sim | 6.0.0.1 |
| Isaac Lab | 10.2.0 (`isaaclab_tasks` 8.1.7, `isaaclab_rl` 0.7.0) |
| Isaac Lab source checkout | `290149083ec5060111ba6fd26b1213c55d9be59c`, clean |
| isaaclab_physx | 2.8.1 |
| isaaclab_newton | 1.7.0 |
| newton | 1.4.0.dev0 |
| mujoco-warp | 3.8.0.3 |
| warp-lang | 1.15.0.dev20260626 |
| torch | 2.11.0+cu128 |
| rsl-rl-lib | 5.4.1 |
| NVIDIA driver | 570.211.01 |
| GPU | one NVIDIA Blackwell consumer GPU (exact model withheld) |
| OS kernel | Linux 6.8 |

The Isaac Lab checkout is a July 2026 snapshot of `develop` plus an unrelated feature
branch. It predates upstream PR #7103 (2026-08-17); see case `EXT-PRESET-NOT-APPLIED`.
