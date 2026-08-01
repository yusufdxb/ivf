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
  "ivf_version": "0.1.0rc1",
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

Strict v1 comparisons add declared checks for frame convention, quaternion convention,
reset semantics, environment ordering and action timing. Controls that the current bundle
cannot prove are placed in `unsupported_or_unverifiable`; they are never silently counted
as matches.

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

IVF version and evidence schema, the manifest digest and portable source path, creation
time, the exact command, Python version, platform, the names of environment variables that
affect execution, and the `ivf doctor` report as it was at run time. Public evidence does
not record a username, hostname, absolute checkout path, Python executable path or raw
`PYTHONPATH` value. Hardware labels may be supplied explicitly so a release artifact can
record a useful class without disclosing a private machine identifier.

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

Finalization attempts to make every local file read-only. That is a best-effort accidental
edit guard, not a portable security boundary: Git and archive formats need not preserve
mode bits. Verification checks the seal, then every recorded digest, then looks for files
that appeared afterwards. Portable integrity comes from the hashes.

```bash
uv run ivf reproduce <bundle> --verify-only
```

## Guarantees

* **Tamper-evident after finalization.** Finalizing twice is refused, and any content or
  file-set change is detected by verification. Filesystem read-only modes are local only.
* **Self-describing.** Nothing requires IVF to interpret.
* **Version-checked.** A major-version mismatch is refused with a clear message rather
  than half-parsed. There is no silent migration.
* **Written even on failure.** An `ERROR` run still produces a sealed bundle, because an
  infrastructure failure with no record of what was attempted is the hardest kind to debug.
