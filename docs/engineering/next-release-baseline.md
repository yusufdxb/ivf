# Next-release baseline: verified repository state

Everything below was read from the repositories themselves at the start of this cycle.
Nothing here is carried over from a previous report.

## Public repository

Fresh clone of `https://github.com/yusufdxb/ivf`:

| Property | Value |
|---|---|
| branch | `main` |
| commit | `3aa56ac0ea1c82f0f6a7c67e3275d95dffe5f59d` |
| tree | `6c31233bd1d60fd493ee2cc7ec2a1ad5ab271d0c` |
| tag `v0.1.0-rc1` | annotated tag `b2e65189090aadda0d29e76d06702c1bb45f45ec` |
| tag points at | `3aa56ac0ea1c82f0f6a7c67e3275d95dffe5f59d` |
| tagged | 2026-08-01, "IVF v0.1.0-rc1" |
| `git fsck --full` | clean |
| working tree | clean |
| history | 12 commits, sanitized |

The tag is immutable and was not moved, deleted, recreated, or force-updated.

## Local successor repository

`~/Projects/ivf`:

| Property | Value |
|---|---|
| branch | `ci/gpu-simulator-gate` |
| commit | `3bd11dc9577de66f056338bb564ac7c3fb670cba` |
| tree | `8fe0eb220977ff90d0a4f90e3130ac5712555e22` |
| working tree | clean, no staged or unstaged diff |
| remotes | none |
| tags | none |
| history | 16 commits |

## Upstream parity repository

`~/Projects/isaac-sim-contrib/IsaacLab`:

| Property | Value |
|---|---|
| branch | `yusufdxb/feature-parity-harness` |
| commit | `290149083ec5060111ba6fd26b1213c55d9be59c` |

Not modified, not pushed, and no maintainer was contacted.

## History relationship

The two histories are **disjoint**. `git merge-base main private-local/ci/gpu-simulator-gate`
returns nothing: there is no common ancestor at all.

The private branch is a parallel history whose commits carry the same subjects as the
public ones but different object ids, which is what sanitization by history rewrite
produces. `git log --left-right --cherry-pick` pairs eleven commits by patch id and leaves:

* public-only: `3aa56ac` "Finalize sanitized public RC1 provenance";
* private-only: `a8a2d3b` "Remove stale pre-RC evidence bundles", plus four commits of
  later work (`efc0901`, `0bdd434`, `1f8b6c0`, `3bd11dc`).

**Consequence for this cycle:** nothing could be merged. A merge would have grafted an
unrelated root onto public history and reintroduced the content sanitization removed. Every
change was transplanted file by file after review. See
[private-to-public-change-map.md](private-to-public-change-map.md).

## What the local branch did and did not contain

The local branch was expected to carry audit remediation. It does not. Its four
post-RC1 commits contain capture-handoff hardening and a GPU CI gate. Checked directly:

* `artifacts/cartpole-physx-corrected/trajectories.npz` has blob id
  `f414da007388eb2cfe240a81f1cf1e256925965c` on **both** branches, identical to the
  baseline's blob on both. The corrected case was not recaptured on the private branch.
* No unit or frame enforcement exists in `src/ivf/signals.py` or the oracle layer on
  either branch.
* `src/ivf/cli.py` is untouched by the private branch, so the CLI error contract is
  unfixed there.

Every audit finding was therefore reproduced against public RC1 and fixed from scratch on
this branch, not inherited. See [../audits/v0.1.0-rc1-remediation.md](../audits/v0.1.0-rc1-remediation.md).

## Capture environment

The software stack the shipped bundles declare was confirmed to exist and to be the one
that produced them:

| Package | Declared in bundles | Present in the capture environment |
|---|---|---|
| isaacsim | 6.0.0.1 | 6.0.0.1 |
| isaaclab | 10.2.0 | 10.2.0 |
| isaaclab_physx | 2.8.1 | 2.8.1 |
| isaaclab_newton | 1.7.0 | 1.7.0 |
| newton | 1.4.0.dev0 | 1.4.0.dev0 |
| warp-lang | 1.15.0.dev20260626 | 1.15.0.dev20260626 |
| torch | 2.11.0+cu128 | 2.11.0+cu128 |
| numpy | 2.4.4 | 2.4.4 |
| python | 3.12.13 | 3.12.13 |

A second Isaac installation on the same workstation carries a different stack and is not
the capture environment. The distinction matters only because checking the wrong one
would suggest the bundles declare a stack that does not exist.
