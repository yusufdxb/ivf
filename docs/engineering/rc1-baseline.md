# v0.1.0-rc1 repository baseline

Recorded on 2026-07-31 before release-candidate changes.

## IVF

- Repository: `ivf` checkout
- Branch: `main`
- HEAD: `f3e47d5648495d95eb26fbcbbb5cc8cb7b5d0f70`
- Working tree: clean (`git status --short`, `git diff`, and `git diff --cached` produced no output)
- Required history: `729a8db`, `85fca6e`, and `f3e47d5` are present.

Recent history at the baseline:

```text
f3e47d5 Document the hermetic test command for hosts with a global PYTHONPATH
85fca6e Make a plain uv sync install the test tools
729a8db Add the trajectory_bundle/v1 capture boundary and a real Isaac Lab capture path
51bf7ac Remove em dashes from docs and report output
ead2382 Add IVF: an auditable acceptance layer for Isaac Lab changes
```

## Isaac Lab upstream parity repository

The enclosing `isaac-sim-contrib` directory is not itself a Git repository. The
upstream parity work is in its nested Isaac Lab clone.

- Repository: nested `IsaacLab` checkout
- Branch: `yusufdxb/feature-parity-harness`
- HEAD: `290149083ec5060111ba6fd26b1213c55d9be59c`
- Working tree: clean (`git status --short`, `git diff`, and `git diff --cached` produced no output)
- Required history: `e3f6de3` and `2901490` are present.

Recent history at the baseline:

```text
2901490 Record the report-only default for calibration-deficit policies
e3f6de3 Add a kit-less offline parity bundle comparison entry point
7eb8572 Record member-bootstrap SD alongside the kappa jackknife SE
8c44352 Validate bundles on load and reject non-identifier quantity names
dabf33a Record kappa jackknife SE as a calibration regime diagnostic
b9e4379 Tolerate unknown metadata fields when loading bundles
f522ca5 Demote gating on kappa calibration deficit; harden hash canonicalization
718562c Trim parity exports to user-facing surface
9e89ad5 Add regression scenarios for wrench, pose-write, and actuator contracts
25be5c4 Add capture integrity report and payload corruption detection
```

## Sibling Isaac Sim clone

The container also holds a separate, clean Isaac Sim clone. It is not the
repository containing the parity commits, but it is recorded to remove path
ambiguity.

- Repository: sibling `IsaacSim` checkout
- Branch: `main`
- HEAD: `987015050efebfd0cd5d3736ae47fffe5adee308`
- Working tree: clean (`git status --short`, `git diff`, and `git diff --cached` produced no output)

No pre-existing dirty or untracked files were present in any of these Git
working trees. No unrelated modification therefore needs to be preserved at
this baseline.
