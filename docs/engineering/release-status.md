# Release status

**Current published release:** `v0.1.0-rc1` (commit `3aa56ac`). That tag is immutable and
has not been moved.

**Candidate branch:** `audit-hardening-and-gpu-gate`, built on public `main`.

## What changed since RC1

An independent audit of RC1 returned "keep as rc1" with six load-bearing findings. All six
were reproduced against the tag and closed. See
[../audits/v0.1.0-rc1-remediation.md](../audits/v0.1.0-rc1-remediation.md) for the exact
command that reproduced each one.

| Finding | Status |
|---|---|
| H1 corrected-case provenance | closed: relabelled as a declared A/A identity check; benign-difference is now the primary engineering negative control |
| H2 unit and frame enforcement | closed: automatic per-signal checks before any oracle runs |
| M1 seal threat model | closed: wording corrected, boundary documented and tested |
| M2 CLI operational errors | closed: typed ERROR, exit 5, no default traceback |
| M3 input identity and self-comparison | closed: identical content refused by default |
| M6 release-equivalence claims | closed: public aggregates recompute from a checked-in script |

Every tolerance value is byte-identical to the tag. The flagship PhysX-versus-Newton
result is unchanged and still `FAIL`.

## Verification state

| Check | Result |
|---|---|
| `ruff check .` | clean |
| `pytest -q -rs` | 257 passed, 5 skipped |
| wheel build and clean non-editable install | passes, imports from `site-packages` |
| all shipped evidence bundles verify | yes, 7 of 7 |
| upstream parity suite | 133 passed |
| release-equivalence recomputation | 7 of 7 groups match |
| real PhysX simulator tests | 5 passed, author's workstation only |
| real Newton simulator tests | 1 passed, author's workstation only |

## What has not been verified

* **The GPU simulator CI job has never run on GitHub.** It is gated on the repository
  variable `IVF_GPU_RUNNER`, which is unset, so the job does not execute. A green CI run
  is **not** evidence that the simulator path works. See [../ci-gpu-runner.md](../ci-gpu-runner.md).
* **No external reproduction has completed.**
* **No second independent audit has been performed.**
* **Cross-machine determinism is not claimed.** The corrected case rests on bit-identical
  re-execution observed on one machine.

Those three gaps are why this is a release *candidate*.
