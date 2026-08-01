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

## Why the completion marker is written last

`COMPLETE` is written after the payload and after the checksum file. A capture that was
killed, ran out of disk, or crashed mid-rollout therefore has no marker.

This is the difference between *detecting* a truncated run and *inferring* one. Without
the marker, a 200-step payload from a 400-step run is indistinguishable from a 200-step
run that went fine, and the consumer is left guessing from data that looks entirely
normal. With it, the refusal is structural.

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

`tests/test_bundle_v1.py` asserts each of these as a refusal, and asserts end to end that
a broken capture produces zero oracle outcomes.

## Producing one

```bash
parity-capture validation/capture/cartpole_physx.yaml --output artifacts/cartpole-physx-baseline
parity-capture validation/capture/cartpole_newton.yaml --output artifacts/cartpole-newton-baseline
```

See [the capture adapter](../adapters/parity_capture/README.md). The writer is
`ivf.bundle.finalize_v1`, which is the only supported way to seal a bundle: it checksums
the payload and stamps the marker in the required order. These commands must run inside
the recorded Isaac Lab environment. The RC workflow executed both PhysX and
Newton/MJWarp; this is support for the one declared cart-pole path, not a general backend
plugin interface.

## Reading one

```python
from ivf.bundle import load_v1

bundle = load_v1("artifacts/cartpole-physx-baseline")   # raises on any violation
print(bundle.contract.run_status, bundle.contract.quaternion["layout"])
```

`load_v1` validates everything before returning. There is no partially valid result: a
caller gets an interpretable bundle or an explanation.
