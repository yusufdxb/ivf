# Changelog

## 0.1.0rc1

First release candidate of the Isaac Validation Framework.

- Offline acceptance execution with typed verdicts and deterministic defect localization.
- Strict, checksummed `trajectory_bundle/v1` ingestion and evidence sealing.
- Real Isaac Lab cart-pole capture through PhysX and Newton/MJWarp.
- Machine-readable provenance for every shipped acceptance tolerance.
- Explicit provenance and default-off guard for the simulator reset perturbation.
- CPU-reproducible synthetic, corrected, benign-difference, reset-defect and
  cross-backend examples.

The release makes no claim of physical correctness, general backend equivalence,
cross-hardware determinism, learned-policy transfer or sim-to-real prediction.
