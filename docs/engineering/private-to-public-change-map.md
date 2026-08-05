# Private-to-public change map

The private successor branch shares no ancestor with public history (see
[next-release-baseline.md](next-release-baseline.md)), so no merge was possible. Every
change was classified and then transplanted, rewritten, or skipped.

## Commits

| Private commit | Public equivalent | Action | Reason |
|---|---|---|---|
| `ead2382` … `6bd3d29` (11 commits) | present under rewritten ids `3aea32b` … `650ef11` | skip | already published; identical patch ids |
| `a8a2d3b` "Remove stale pre-RC evidence bundles" | none | skip | a deletion-only commit against private paths; the deletions are already absent publicly |
| `5a4246f` "Document simulator EULA preflight" | `efe8444` | skip | already published |
| `efc0901` "Harden the real capture handoff" | none | transplant | capture-boundary hardening; safe and load-bearing for the GPU gate |
| `0bdd434` "Fix capture source revision provenance" | none | transplant | same block |
| `1f8b6c0` "Keep capture integration on the file boundary" | none | transplant | same block |
| `3bd11dc` "Gate the simulator path in CI" | none | rewrite | transplanted with a hardened `runs-on` expression, see below |

## Files

| Path | Class | Action | Privacy | Scientific effect |
|---|---|---|---|---|
| `src/ivf/bundle.py` | audit-adjacent hardening | rewrite | none | stricter contract: `hardware` and `seed.env_order` required, `unit`/`frame` may not be `unknown`, failed runs must carry a structured error block, partial runs may not claim all steps. Hand-merged with this cycle's capture-identity block |
| `src/ivf/signals.py` | audit-adjacent | rewrite | none | hand-merged with this cycle's `signal_contract`, content digest, and identity metadata |
| `adapters/parity_capture/**` | capture adapter | transplant | reads `IVF_HARDWARE_LABEL` so a bundle never records the exact GPU model | produces the stricter contract |
| `tests/test_bundle_v1.py` | tests | rewrite | none | private version plus an `experiment_mode: identity_check` declaration on two A/A fixtures |
| `tests/test_capture_spec.py` | tests | transplant | none | none |
| `tests/test_simulator_*.py` | GPU tests | transplant | none | exercise the real simulator |
| `tests/simulator_gate.py` | GPU CI gate | transplant | none | turns a missing simulator into a failure when a job promised to run one |
| `tests/test_simulator_gate.py` | GPU CI gate | transplant | none | none |
| `.github/workflows/ci.yml` | GPU CI gate | rewrite | none | see the workflow note below |
| `docs/ci-gpu-runner.md` | GPU CI docs | transplant | none | none |
| `docs/capture-boundary.md`, `docs/manifests.md`, `docs/quickstart.md`, `docs/engineering/rc1-baseline.md`, `docs/engineering/reset-defect-provenance.md` | current documentation | transplant | none | describe the stricter contract |
| `validation/capture/cartpole_physx_{baseline,candidate}.yaml` | capture specs | transplant | none | named specs for the GPU tests |
| `validation/examples/cartpole_{benign_difference,physx_vs_newton_v1,reset_defect}.yaml` | manifests | transplant | none | add `bundle_sha256` subject locks; verdicts unchanged |
| `validation/examples/cartpole_corrected.yaml` | manifest | rewrite | none | relabelled this cycle as a declared identity check; the private version keeps the false "defect removed" description |
| `docs/engineering/current-state-audit.md` | stale process artifact | **skip** | contains an internal workspace snapshot | none |
| `docs/engineering/rc1-release-checklist.md` | internal process artifact | **skip** | internal release checklist | none |
| `docs/releases/v0.1.0-rc1-equivalence.json` | private-only provenance | **skip the private state** | the private branch deletes this file entirely | rewritten this cycle instead, see M6 |
| `docs/releases/v0.1.0-rc1.md`, `docs/reproduction/*` | release docs | rewrite | private object ids removed | none |
| `README.md` | docs | rewrite | none | merged with this cycle's seal wording |

### Workflow note

The private workflow declares:

```yaml
if: vars.IVF_GPU_RUNNER != ''
runs-on: ${{ vars.IVF_GPU_RUNNER }}
```

Transplanted as:

```yaml
if: ${{ vars.IVF_GPU_RUNNER != '' }}
runs-on: ${{ vars.IVF_GPU_RUNNER || 'ubuntu-latest' }}
```

The fallback is never used, because the job is skipped whenever the variable is unset. It
is there so that an unset variable produces a skipped job rather than an invalid workflow,
which would take the CPU jobs down with it.

## Content deliberately not brought across

* private absolute paths, workstation names, and usernames;
* the exact GPU model and its VRAM;
* simulator logs and temporary worktree paths;
* build directories and scratch evidence generated only for local validation;
* private commit and tree object ids in prose and in the equivalence record. The one
  producer reference inside seven sealed bundles remains, because it participates in the
  seal, and is now labelled an author attestation where it is described.
