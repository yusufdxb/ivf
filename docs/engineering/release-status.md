# Release status

**Current published release:** `v0.1.0-rc1` (commit `3aa56ac`). That tag is immutable and
has not been moved.

**Merged into `main`:** the audit remediation and the gated simulator CI job, as
[PR #1](https://github.com/yusufdxb/ivf/pull/1). `v0.1.0-rc2` is prepared but not tagged.

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
| real PhysX simulator tests | 5 passed, on a self-hosted runner in public CI |
| real Newton simulator tests | 1 passed, on a self-hosted runner in public CI |
| simulator job, `IVF_GPU_RUNNER` unset | skipped, does not block |
| simulator job, `IVF_GPU_RUNNER` set | executed and passed |

## What has not been verified

* **No external reproduction has completed.**
* **No second independent audit has been performed.**
* **Cross-machine determinism is not claimed.** The corrected case rests on bit-identical
  re-execution observed on one machine.

Those gaps are why this is a release *candidate*.

The simulator job has now been exercised publicly in both modes, on one self-hosted
runner. That establishes the gate works; it does not establish that the GPU path works on
arbitrary hardware.
