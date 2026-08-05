# The capture boundary: `trajectory_bundle/v1`

Everything IVF judges arrives through this boundary. It is the contract between whatever
produced a rollout and whatever decides if that rollout is acceptable.

## Reconciliation, not replacement

v1 is a **strict profile layered on the existing parity bundle**, not a new format. The
physical layout is unchanged:

```
<bundle>/
  metadata.json        parity RunMetadata, plus a "capture_contract" block (v1)
  trajectories.npz     (steps, envs, dim) arrays, plus __actions__
  integrity.json       parity capture-integrity report, optional
  CHECKSUMS.sha256     sha256 of every other file          <- added by v1
  COMPLETE             completion marker, written LAST     <- added by v1
```

Consequences of that choice:

* every parity bundle already on disk stays readable, and the six pre-v1 captures
  vendored under `validation/bundles/` still load;
* the capture side did not have to be rewritten to adopt v1;
* routing is automatic. `ivf.bundle.is_v1` keys on the completion marker (or a declared
  contract), so a producer that upgrades gets strictness immediately without every
  consumer's manifest being edited.

A pre-v1 bundle is not rejected. It loads through the legacy reader, and the controls it
cannot substantiate are reported as **unverifiable** rather than assumed to hold.

## Normative directory and storage rules

The directory is the bundle. `metadata.json` and `trajectories.npz` are required payload
files. `CHECKSUMS.sha256` and `COMPLETE` are required finalization files.
`integrity.json` is optional. No other file is allowed unless it is listed in
`CHECKSUMS.sha256`. Checksum paths are relative and use `/` separators.

`trajectories.npz` is a NumPy ZIP archive. Every entry uses ZIP DEFLATE compression;
pickled object arrays are not part of v1. Signal arrays have rank three and shape
`(captured_steps, num_envs, signal_dim)`. Stored dtype must equal the declared dtype.
Each signal has a nonempty physical unit, or the literal `dimensionless`, and a nonempty
coordinate frame. The reserved `__actions__` array is required and has shape
`(captured_steps, num_envs)` or `(captured_steps, num_envs, action_dim)`. Passive control
is an explicit zero array, not an omitted action stream.

The schema name is exactly `trajectory_bundle/v1`. Unknown fields inside v1 are additive
extensions and are ignored unless another declared contract assigns them meaning. An
unknown major, including `trajectory_bundle/v2`, is rejected with
`IVF-BUNDLE-SCHEMA-UNSUPPORTED`. A reader must not guess or silently downgrade.

## Identity model

Identity is split so a capture does not decide how a later comparison uses it.

| Identity | Representation |
|---|---|
| Capture manifest | `metadata.capture_spec` is the normalized `parity.capture/v1` mapping and is covered by the bundle checksums. `task.config_digest_sha256` identifies the normalized workload configuration independent of backend, output label, and test perturbation. |
| Run | `COMPLETE.checksums_sha256`, the SHA-256 of `CHECKSUMS.sha256`, is the finalized content root. A directory name is only a label. |
| Baseline and candidate | Assigned by `ivf.validation/v1` under `subjects.baseline` and `subjects.candidate`. Each v1 subject records `path` and `bundle_sha256`; IVF checks that lock before any oracle executes. |
| Task | `task.id`, `task.variant`, and `task.config_digest_sha256`. |
| Asset | New captures record `task.asset.id` and `task.asset.source_uri`. `binary_identity` is `unverifiable` when a remote or cached asset cannot be hashed. A validation manifest must not present that as a verified binary match. |
| Backend | `backend.id`, nonempty `backend.solver_settings`, and declared features. |
| Software | `software` records installed package and engine versions. New captures also record `software.source_commits`, including the commit and dirty flag when a module comes from a Git checkout, or an explicit `unavailable` status. |
| Hardware | Nonempty `hardware`, including a driver value. A producer may use a coarse public GPU label, but uses the literal `unavailable` instead of silently dropping an unqueryable value. |

The IVF manifest itself is identified by the canonical digest written to the sealed
evidence verdict. Copying a bundle from one role to another never rewrites it; role and
hash lock belong to the validation manifest.

## Time, seeds, frames, reset, and episodes

`seed.value` is the run seed. `seed.env_ids` is the ordered per-environment schedule and
contains no duplicates. `seed.env_order` states how array axis 1 maps to those IDs. The
cart-pole producer uses one NumPy PCG64 stream for ordered initial-angle values and
records requested and applied reset state.

`timing.physics_dt` and `timing.control_dt` are positive seconds. `decimation` gives the
physics-to-control ratio. `action_applied` names when an action becomes active, and
`capture_hook` names when state is sampled. The cart-pole profile has zero warm-up steps.
Sample 0 is the first post-reset, post-physics-step state; sample `i` represents
`(i + 1) * physics_dt` seconds after reset completion. New captures spell this out as
`warmup_steps`, `warmup_semantics`, and `timestamp_convention`. The older committed v1
cart-pole captures have the same fixed profile semantics.

World arrays use the declared right-handed up-axis convention and include each
environment origin. Joint arrays use the recorded joint order. Quaternion arrays are
unit length within `1e-3`. `quaternion.layout` is `wxyz` or `xyzw`, and `scalar_first`
must agree with it. Quaternion hemisphere is unconstrained, so downstream rotation
metrics must be sign-invariant when appropriate.

`reset.semantics` states whether pose and velocity were written and includes an initial
state digest. `termination.declared` is required even when false. When true it names the
signal, condition, and reset behavior. The cart-pole profile does not auto reset after
termination, so environment order and trajectory alignment remain fixed.

## Why the completion marker is written last

`COMPLETE` is written after the payload and after the checksum file. A capture that was
killed, ran out of disk, or crashed before finalization therefore has no marker.

This is the difference between *detecting* a truncated run and *inferring* one. Without
the marker, a 200-step payload from a 400-step run is indistinguishable from a 200-step
run that went fine, and the consumer is left guessing from data that looks entirely
normal. With it, the refusal is structural.

The marker means the write was finalized, not that simulation succeeded.
`capture_contract.run_status` and `COMPLETE.run_status` agree:

* `completed`: `captured_steps == declared_steps`, with no error block;
* `partial`: `captured_steps < declared_steps`, declared honestly;
* `failed`: zero or more steps, plus `error.type`, `error.message`, and
  `error.failed_step`.

A finalized partial or failed record can be inspected, but IVF marks it incomplete and
returns `INVALID_EXPERIMENT` before scientific comparison. The shipped capture CLI is
stricter on an exception: it exits nonzero and never writes `COMPLETE`, so a crash cannot
resemble a successful bundle.

## What the contract declares, and why each item earns its place

Every declaration below is a question whose wrong answer silently invalidates a
comparison rather than making it fail loudly.

| Declaration | Why it is mandatory |
|---|---|
| `schema` | A future format must be refused, not half-parsed |
| `run_status`, `declared_steps`, `captured_steps` | Makes truncation self-declaring; `completed` with fewer captured than declared steps is refused |
| `task.id`, `task.config_digest_sha256` | Two captures of different experiments must not be compared as one |
| `backend.id`, `backend.solver_settings` | The released pre-v1 bundles recorded no solver settings, which made three of six families un-comparable. v1 makes the field structural |
| `software`, `hardware` | Runtime or hardware changes are plausible explanations for divergence and cannot be reconstructed later |
| `seed.value`, `seed.env_ids`, `seed.env_order` | Environment ordering is what makes a paired comparison meaningful, so it is declared rather than assumed |
| `timing.physics_dt`, `control_dt`, `action_applied`, `capture_hook` | Step-by-step comparison of runs at different control rates is meaningless, not merely loose; and a consumer must never have to guess whether a sample is pre- or post-step |
| `frames.convention` | A frame difference and a bug look identical in the data |
| `quaternion.layout` + `scalar_first` | `wxyz` read as `xyzw` produces a plausible, wrong rotation. There is no safe default, so an undeclared or self-contradictory pair is refused |
| `reset.semantics` | `writes_pose_only` is a legitimate declaration. Declaring it is what lets a reader distinguish that case from the defect where velocity was dropped by accident |
| `termination.declared` | `false` is a valid answer. Requiring it stops silence being read as "never happened" |
| `arrays[*].shape/dtype/unit/frame` | Units and frames are not recoverable from the numbers; shape and dtype are cross-checked against the payload |
| reserved `__actions__` | Exact actions, leading dimensions, and action timing must be explicit |
| `CHECKSUMS.sha256` | A bundle that changed after capture is not the bundle that was captured |

## What is refused, and with which code

A contract violation raises `BundleContractError` at load time, which the runner turns
into **`INVALID_EXPERIMENT`**, not `ERROR` and not `FAIL`. The capture is uninterpretable;
that is a statement about the evidence, not about the infrastructure and not about the
subject under test. Nothing reaches the oracle layer.

| Failure | Reason code |
|---|---|
| No completion marker | `IVF-BUNDLE-INCOMPLETE` |
| Digest mismatch, missing file, or an unrecorded extra file | `IVF-BUNDLE-CHECKSUM-MISMATCH` |
| No checksum file at all | `IVF-BUNDLE-CHECKSUM-MISSING` |
| Quaternion layout absent, unknown, or contradicting `scalar_first`; non-unit quaternions | `IVF-BUNDLE-QUATERNION-AMBIGUOUS` |
| Declared shape disagrees with payload, `env_ids`, or `captured_steps`; undeclared array | `IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT` |
| Declared dtype disagrees with payload | `IVF-BUNDLE-ARRAY-DTYPE-INCONSISTENT` |
| Short capture claiming `completed` | `IVF-BUNDLE-PARTIAL-UNDECLARED` |
| A required declaration is missing | `IVF-BUNDLE-CONTRACT-INCOMPLETE` |
| No contract block | `IVF-BUNDLE-CONTRACT-MISSING` |
| Unsupported schema | `IVF-BUNDLE-SCHEMA-UNSUPPORTED` |
| Unknown frame convention / reset semantics / missing termination declaration | `IVF-BUNDLE-FRAME-UNDECLARED`, `IVF-BUNDLE-RESET-UNDECLARED`, `IVF-BUNDLE-TERMINATION-UNDECLARED` |

Unreadable JSON or NPZ after valid transport checks is an operational read failure and
maps to `ERROR` with `IVF-RUNTIME-EXECUTION-ERROR`. A readable bundle that violates its
scientific contract maps to `INVALID_EXPERIMENT` with a stable bundle code above.

`tests/test_bundle_v1.py` asserts each of these as a refusal, and asserts end to end that
a broken capture produces zero oracle outcomes.

## Producing one

```bash
parity-capture validation/capture/cartpole_physx.yaml --output artifacts/cartpole-physx-baseline
parity-capture validation/capture/cartpole_newton.yaml --output artifacts/cartpole-newton-baseline
```

See [the capture adapter](../adapters/parity_capture/README.md). The writer is
`parity_capture.bundle_writer.finalize_v1`, a standard-library implementation of this
file specification. It does not import IVF. The consumer has its own strict validator,
so agreement occurs through bytes on disk rather than a shared Python object. These
commands must run inside the recorded Isaac Lab environment. The RC workflow executed
both PhysX and Newton/MJWarp; this is support for the one declared cart-pole path, not a
general backend plugin interface.

For a no-edit local PhysX workflow, use the paired capture specs. Each successful command
updates one generated IVF manifest with the captured path and finalized root hash, then
prints the exact validation command:

```bash
parity-capture validation/capture/cartpole_physx_baseline.yaml \
  --output artifacts/generated/cartpole-baseline
parity-capture validation/capture/cartpole_physx_candidate.yaml \
  --output artifacts/generated/cartpole-candidate
uv run ivf validate artifacts/generated/cartpole_real.yaml
```

No Python editing, array copying, metadata construction, or path/hash editing is part of
this workflow.

## Reading one

```python
from ivf.bundle import load_v1

bundle = load_v1("artifacts/cartpole-physx-baseline")   # raises on any violation
print(bundle.contract.run_status, bundle.contract.quaternion["layout"])
```

`load_v1` validates everything before returning. There is no partially valid result: a
caller gets an interpretable bundle or an explanation.
