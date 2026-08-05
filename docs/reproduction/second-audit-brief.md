# Second independent audit: reviewer brief

You are auditing a release candidate that has already failed one audit. That audit returned
`KEEP AS v0.1.0-rc1` with six load-bearing findings. All six are claimed closed. Your job is
to decide whether they are, and whether anything was broken in closing them.

Work only from what is in this repository. Do not ask the author how something is meant to
behave: if the answer is not in the repository, that is itself a finding.

## What you are given

* the candidate commit on branch `audit-hardening-and-gpu-gate`;
* [`README.md`](../../README.md);
* [`independent-rc1-reproduction.md`](independent-rc1-reproduction.md);
* [`reviewer-result-template.md`](reviewer-result-template.md), which is where your results go;
* [`../audits/v0.1.0-rc1-remediation.md`](../audits/v0.1.0-rc1-remediation.md), which is the
  set of claims you are testing.

You are deliberately not given the author's working notes or live guidance.

## What to execute

```bash
git clone <candidate> ivf && cd ivf && git checkout audit-hardening-and-gpu-gate
env -u PYTHONPATH uv sync --frozen
env -u PYTHONPATH uv run --frozen ruff check .
env -u PYTHONPATH uv run --frozen pytest -q -rs
env -u PYTHONPATH uv build
```

Then install the built wheel into a fresh environment and confirm `ivf` imports from
`site-packages` rather than from the source tree.

Run every command in the README and in the reproduction guide. Verify every shipped evidence
bundle:

```bash
for b in validation/evidence/* artifacts/evidence/*; do
  env -u PYTHONPATH uv run --frozen ivf reproduce "$b" --verify-only
done
```

Recompute the release record:

```bash
git worktree add /tmp/rc1 v0.1.0-rc1
env -u PYTHONPATH uv run --frozen python tools/recompute_release_equivalence.py --check --root /tmp/rc1
```

## What to attack

1. **Unit and frame enforcement.** Mutate a candidate bundle's declared unit, then its
   frame, then a tolerance unit, re-finalize it properly, and confirm the run is refused
   with no oracle outcomes. Then find a mutation that is *not* caught. There is at least
   one class: units are compared as strings, so `m` versus `cm` is refused rather than
   converted, and a unit declared wrongly but consistently on both sides is invisible.
   Decide whether the documented limitation matches what you observe.
2. **Self-comparison.** Point both subjects at the same capture and confirm refusal. Then
   try to obtain a `PASS` on identical content *without* declaring
   `experiment_mode: identity_check`. If you can, that is a critical finding.
3. **Corrected-case provenance.** Read `validation/examples/cartpole_corrected.yaml` and
   decide whether its description overstates anything. Confirm for yourself that
   `artifacts/cartpole-physx-{baseline,corrected}/trajectories.npz` are byte-identical, and
   that the repository says so plainly rather than implying a fresh independent capture.
4. **CLI error contract.** Drive every documented exit code. Confirm no expected
   operational failure returns 1, and that no traceback appears without `--debug`.
5. **Seal threat model.** Perform a coordinated edit plus reseal and confirm the bundle
   verifies. Then confirm the repository documents that as a boundary rather than a defect.
6. **Release equivalence.** Confirm the public aggregates recompute, and that nothing
   labelled `author_attested` is presented as though it had been checked.
7. **The flagship results.** Confirm the PhysX-versus-Newton case still returns `FAIL` and
   that no tolerance was loosened to make anything pass. Compare the tolerance values on
   this branch against the tag.

## What you cannot verify here, and should not credit

* **The GPU simulator job has never run on GitHub.** It is gated on a repository variable
  that is not set, so it does not execute. The simulator path was exercised only on the
  author's workstation. Treat every simulator claim as author-attested.
* **Cross-machine determinism.** The corrected case rests on bit-identical re-execution on
  one machine. Nothing establishes that any other machine reproduces those bytes.
* **Private history.** Everything under `author_attested` in the release record is
  unverifiable by construction.

## What would make you return "not ready"

Any of: a `PASS` obtainable on dimensionally incompatible signals; an undeclared
self-comparison that returns `PASS`; an operational failure that exits 1; a tolerance that
moved without a recorded rationale; a claim in the README or a case study that the evidence
does not support; a shipped evidence bundle that fails verification.

Record your result in `reviewer-result-template.md`. A finding with a reproduction command
is worth more than a finding with an opinion.
