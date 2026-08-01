# Reset-perturbation provenance

## Classification

The flagship PhysX case is an **explicit simulator-level test perturbation**. It is not
a restored historical implementation, a configuration defect, post-capture array
corruption, or a simulator-discovered defect.

The switch was introduced in public IVF commit
`8a7bb15cbf1f6fb69cfccdbf2a4a566629f2b991` and is enabled only by
`validation/capture/cartpole_physx_reset_defect.yaml`:

```yaml
defect:
  drop_reset_velocity: true
```

`parity_capture.capture.apply_reset` constructs the requested pole velocity of
`0.5 rad/s`. With the switch active it writes an explicit zero joint-velocity vector
with the requested joint positions before PhysX steps. The trajectory arrays are then
captured from real simulator execution and are never modified after capture.

The bundle records all of the following:

- the active perturbation switch;
- requested joint positions and velocities;
- applied joint positions and velocities;
- normalized capture specification;
- capture timing, actions, software, hardware label and checksums.

IVF validity and oracle decisions do not consume the perturbation bookkeeping field.
The field is retained so an auditor can distinguish a deliberate demonstration from a
discovered defect. The oracles independently establish its numerical and semantic
consequences.

## Relationship to the historical failure

The perturbation is modeled after a reset-semantics failure observed while developing
the upstream parity harness and fixed in Isaac Lab branch commit
`9aaa389f24231fadcca9f871af3eccb79e738b20`.

Before that fix, the parity child runner called `sim.reset()` without explicitly writing
configured velocities. A calibration rigid body declared a root linear velocity of
`1.0 m/s`, but live PhysX retained zero velocity. The fix writes configured root and
joint pose and velocity explicitly before capture.

The IVF case is not a literal reproduction:

| | Historical harness failure | IVF release demonstration |
|---|---|---|
| Affected state | rigid-body root linear velocity | cart-pole joint angular velocity |
| Requested value | 1.0 m/s | 0.5 rad/s |
| Mechanism | missing harness write after reset | explicit test switch writes zero velocity |
| Status | fixed by `9aaa389` | active only in one declared capture manifest |

It must not be described as an upstream Isaac Lab product defect or as a defect IVF
discovered.

## Nominal contamination guard

- `drop_reset_velocity` defaults to `false`.
- Nominal and corrected capture manifests omit the `defect` block.
- The parser accepts only YAML booleans. The string `"false"` is rejected rather than
  being truthy.
- An active switch prints a prominent warning before Kit launches.
- `tests/test_capture_spec.py` proves omitted and explicit `false` values stay inactive,
  explicit `true` activates, and quoted `"false"` is refused.
- Simulator tests use separate manifests and output directories. Capture refuses to
  overwrite any non-empty directory.

## Observed signature on the RC capture

The real PhysX comparison produced:

- pole angular-rate difference at the first captured post-reset step, index `0`;
- velocity tolerance violation at index `0` across environments `0-15`;
- pole-angle tolerance violation at index `6` across all 16 environments;
- threshold-crossing delays of `6-9` steps, median `7`, worst `9`;
- unchanged 400-step survive or terminate decision in all 16 environments;
- final verdict `FAIL` with `IVF-ORACLE-NON_EQUIVALENT` and
  `IVF-ORACLE-EVENT-TIMING-DELTA`.

The committed source manifest reproduces the result offline:

```bash
uv run ivf validate validation/examples/cartpole_reset_defect.yaml
```

The exact sealed evidence identifier and root hash are recorded in
`docs/reproduction/ivf-v0.1.0-rc1.md` after final-HEAD reproduction.
