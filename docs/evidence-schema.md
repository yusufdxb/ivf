# Evidence schema `ivf.evidence/v1`

An evidence bundle is one directory. Everything in it is JSON, YAML, plain text or
`.npz`: readable with numpy and the standard library, without IVF and without Isaac Lab.

```
ivf-results/<run-id>/
├── manifest.original.yaml      the file the user wrote, byte for byte
├── manifest.resolved.json      defaults materialized; this is what was hashed
├── verdict.json                verdict, reason codes, identity, timings
├── validity.json               every experiment-validity check
├── oracles.json                every oracle outcome, with metrics and tolerances
├── divergence.jsonl            one localization record per failing oracle
├── provenance.json             versions, hardware, env vars, exact command
├── signals/
│   ├── baseline.npz            (steps, envs, dim) arrays per signal
│   ├── baseline.metadata.json  the source's provenance
│   ├── candidate.npz
│   └── candidate.metadata.json
├── report.html                 self-contained static report
├── reproduce.sh                the exact rerun command
├── CHECKSUMS.sha256            SHA-256 of every file above
└── SEAL.json                   schema version + digest of CHECKSUMS.sha256
```

The run id is `<manifest name>-<UTC timestamp>-<first 8 of the manifest digest>`, with a
`-NN` suffix if two runs of the same manifest land in the same second. The digest makes it
obvious at a glance when two runs came from the same declared experiment.

## `verdict.json`

```json
{
  "verdict": "FAIL",
  "exit_code": 1,
  "reason_codes": ["IVF-ORACLE-NON_EQUIVALENT"],
  "run_id": "synthetic-reset-velocity-defect-20260731T150031Z-3cdc747d",
  "experiment": "synthetic-reset-velocity-defect",
  "manifest_digest_sha256": "3cdc747d...",
  "ivf_version": "0.1.0",
  "evidence_schema_version": "ivf.evidence/v1",
  "created_utc": "2026-07-31T15:00:31.7+00:00",
  "wall_time_s": 0.41,
  "alpha": 0.05,
  "seed": 0,
  "error": null,
  "counts": {"pass": 5, "fail": 3, "inconclusive": 0, "unsupported": 0, "skipped": 0}
}
```

`error` is non-null only for `ERROR` runs, and carries the traceback.

## `validity.json`

`{"valid": bool, "n_checks": int, "n_failed": int, "n_unverifiable": int, "checks": [...]}`
where each check is `check_id`, `name`, `status` (`pass` | `fail` | `unverifiable` |
`not_applicable`), `detail`, `reason_code`, `baseline_value`, `candidate_value`.

| id | Check |
|---|---|
| V-01 | asset identity matches |
| V-02 | action sequence matches |
| V-03 | initial-state distribution matches |
| V-04 | observation definition matches |
| V-05 | control frequency matches |
| V-06 | seed matches |
| V-07 | environment count matches |
| V-08 | captured horizon matches |
| V-09 | task variant matches |
| V-10 | solver settings match or are recorded |
| V-11 | each run is complete |
| V-12 | warm-up is declared |
| V-13 | physics timestep matches (always checked) |

## `oracles.json`

A list of outcomes: `name`, `type`, `status` (`pass` | `fail` | `inconclusive` |
`unsupported` | `skipped`), `summary`, `reason_codes`, `metrics`, `tolerance`,
`divergence`, `limitations`.

`metrics` is oracle-specific and is where effect sizes, confidence intervals, sample
counts and observed maxima live. Non-finite floats serialize as `null` rather than as
invalid JSON.

`tolerance` is the fully qualified declaration, rationale included, so a reader six months
later can argue with the criterion and not just the number.

## `divergence.jsonl`

One JSON object per line, per failing oracle:

```json
{
  "signal": "pole_angle",
  "oracle": "pole_angle_agreement",
  "first_numerical_difference_step": 0,
  "first_tolerance_violation_step": 1,
  "first_event_disagreement_step": null,
  "affected_env_ids": [0, 1, 2, "..."],
  "affected_components": [0],
  "baseline_values": [0.3364], "candidate_values": [0.3344],
  "window": {"start_step": 0, "end_step": 10, "env_id": 7, "error": [...], "threshold": 0.002},
  "action_at_violation": [0.0],
  "contact_state": null,
  "config_differences": {},
  "classification": "reset_mismatch",
  "confidence": "medium",
  "classification_basis": "tolerance exceeded at step 1, before dynamics could accumulate",
  "limitations": "Cannot separate an initial-state difference from a first-step actuation difference..."
}
```

## `provenance.json`

IVF version and evidence schema, the manifest digest and source path, creation time, the
exact command and working directory, Python version and executable, hostname, user,
platform, the environment variables that affect execution, and the full `ivf doctor`
report as it was at run time.

## `signals/*.npz`

`np.savez_compressed` of `{signal_name: (steps, envs, dim) float64}`. Read it with numpy
alone:

```python
import numpy as np
with np.load("signals/baseline.npz") as payload:
    angle = payload["pole_angle"]
```

`signals/<role>.metadata.json` carries what the source recorded: asset identity, physics
dt, seed, solver settings, backend, device, library versions, hashes.

## `CHECKSUMS.sha256` and `SEAL.json`

`CHECKSUMS.sha256` is `<sha256>  <relative path>` per line, `sha256sum -c`-compatible,
covering every file except itself and the seal. `SEAL.json` records the schema version,
the run id, the file count, and the SHA-256 of the checksum file.

On finalization every file is made read-only. Verification checks the seal, then every
recorded digest, then looks for files that appeared afterwards, a tampered bundle usually
gains or loses a file rather than editing one in place.

```bash
uv run ivf reproduce <bundle> --verify-only
```

## Guarantees

* **Immutable after finalization.** Finalizing twice is refused.
* **Self-describing.** Nothing requires IVF to interpret.
* **Version-checked.** A major-version mismatch is refused with a clear message rather
  than half-parsed. There is no silent migration.
* **Written even on failure.** An `ERROR` run still produces a sealed bundle, because an
  infrastructure failure with no record of what was attempted is the hardest kind to debug.
