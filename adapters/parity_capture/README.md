# parity-capture

Runs one Isaac Lab workload and writes a `trajectory_bundle/v1` capture for IVF.

This package lives **outside IVF core** and IVF core does not depend on it. That is the
point: `ivf` stays installable on a laptop with no GPU and no simulator, while this
adapter is installed into an environment that already has Isaac Lab.

## Install

Into the environment that has Isaac Lab:

```bash
<isaaclab-python> -m pip install -e /path/to/ivf -e /path/to/ivf/adapters/parity_capture
```

## Use

```bash
parity-capture doctor          # check prerequisites without booting Kit

parity-capture validation/capture/cartpole_physx.yaml \
  --output artifacts/cartpole-physx
```

The command boots the real simulator, builds the workload, runs the rollout, captures the
declared signals, checksums the payload, and writes the completion marker last. It then
prints the IVF command to validate the result.

There is no mock mode and no dry run. A capture tool whose happy path can be exercised
without a simulator will eventually be trusted without one.

## Scope

**One workload, supported completely:** `cartpole_passive`, the stock `CARTPOLE_CFG` with
pinned actuator gains, released off-centre, passive. The construction mirrors the proven
`isaaclab_contrib.parity` scenario, so it inherits a configuration that has already
produced real cross-backend evidence.

Adding a second workload is a deliberate act, not a plugin registration. Every task needs
its own declared reset and termination semantics, and a generic loader would let someone
add a task without declaring them.

Both `physx` and `newton` have been executed on the stated compatibility runtime. The
Newton path constructs `NewtonCfg(solver_cfg=MJWarpSolverCfg())`. This proves the narrow
capture chain for this workload only; it does not imply general Newton support or backend
equivalence.

## Test-only defects

The `defect:` block injects a deliberate defect into the simulator run so IVF can be shown
catching something a simulator actually produced, rather than a mutated array:

| Field | Effect |
|---|---|
| `drop_reset_velocity` | Writes zero joint velocity instead of the requested value before simulation. The bundle records requested and applied values plus the active switch; IVF oracles independently establish the trajectory and event consequences |
| `damping_scale` | Multiplies cart-actuator damping. Near 1 it is the benign-difference knob: a real configuration change producing a real, small numerical difference |

Any active defect prints a loud banner to stderr before the capture starts. A defective
bundle nobody can tell is defective is a trap.

The reset case is an explicit simulator-level test perturbation modeled after a
reset-semantics failure observed while developing the parity harness and fixed in
upstream commit `9aaa389`. It is not a literal reproduction of that historical case and
is not described as an Isaac Lab product defect.

## Provenance redaction

`IVF_HARDWARE_LABEL` replaces the exact GPU model in the recorded provenance with a
coarser label. Captures are meant to be shared, and an exact device model is sometimes
more identifying than a team wants to publish.
