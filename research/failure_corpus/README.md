# IVF research evaluation: index and archive

**Status: research development stopped (2026-09-26).**
**Authoritative conclusion: [`v4/REPORT_V4.md`](v4/REPORT_V4.md).**
**Archive tag: `research-archive/failure-corpus-v4-final` (commit `89398cb`).**

Everything in this directory is preserved as it was when each result was produced:
pre-registrations, sealed holdouts, hashes, deviations, per-case results, and every
negative result. Nothing here is to be edited to change a result. Earlier round reports
stand as records of their round; where they differ from v4, v4 governs.

## Final conclusion (from v4)

- IVF's configuration/state checks are conservative: 0/84 false alarms on clean
  cross-backend runs.
- They caught 21 defects that conventional validation missed. All 21 were reset/randomization
  failures, from 2 of the 12 fault families.
- Adding them to conventional validation lowered false acceptance from 0.548 to 0.423 with
  no added false alarms. The pre-registered family-level 95% CI for that reduction was
  [0.000, 0.304], so the broad confirmatory claim **failed**.
- IVF missed most dynamic/runtime fault families. It complements closed-loop sim-to-sim
  validation and does not replace it.

## Rounds

| Round | Question | Outcome | Report | Pre-registration |
|---|---|---|---|---|
| v1 | Does IVF catch cross-backend failures that conventional evaluation accepts? | No: at matched false alarms IVF was a weaker subset; localization below a majority guess | [`REPORT.md`](REPORT.md) | [`2026-09-26-failure-corpus.md`](../../docs/preregistration/2026-09-26-failure-corpus.md) |
| v2 | Revised IVF (experiment-input checks) on an independent holdout | IVF alone worse than conventional; union cut false acceptance 0.36 to 0.20 at the same false alarms | [`v2/REPORT_V2.md`](v2/REPORT_V2.md) | [`2026-09-26-v2-holdout.md`](../../docs/preregistration/2026-09-26-v2-holdout.md) |
| v3 | Three gaps closed; fresh holdout, new robot | Not reproduced: union cut false acceptance 0.354 to 0.104 but added 5 false alarms (trajectory oracle, one Spot seed) | [`v3/REPORT_V3.md`](v3/REPORT_V3.md) | [`2026-09-26-v3-holdout.md`](../../docs/preregistration/2026-09-26-v3-holdout.md) |
| **v4 (final)** | Config/state checks only + conventional, family-clustered CI | **Failed:** reduction 0.125, CI [0.000, 0.304], 0 added false alarms | [**`v4/REPORT_V4.md`**](v4/REPORT_V4.md) | [`2026-09-26-v4-confirmatory.md`](../../docs/preregistration/2026-09-26-v4-confirmatory.md) |

Supporting records: [`v2/OVERLAP.md`](v2/OVERLAP.md) (v1 overlap analysis),
[`DEVIATIONS.md`](DEVIATIONS.md) (every deviation and infrastructure event, all rounds),
[`provenance/FROZEN_IVF.md`](provenance/FROZEN_IVF.md) (v1 freeze).

## Frozen artifacts and seals

| Round | IVF source tree | Seals |
|---|---|---|
| v1 | `4932658076919e65c87ecb0e358b3fd3f27336c3` | [`corpus/LABEL_HASHES.txt`](corpus/LABEL_HASHES.txt) |
| v2 | `48861c1ea320fe7c8c4bc023653282f401f1fa15` ([`v2/FROZEN_IVF_V2.txt`](v2/FROZEN_IVF_V2.txt)) | [`v2/holdout/FAULTS_SHA256.txt`](v2/holdout/FAULTS_SHA256.txt), [`v2/corpus/LABEL_HASHES.txt`](v2/corpus/LABEL_HASHES.txt) |
| v3 | `becf39d2b1c1264d1b6330ca66b79390640985c9` ([`v3/FROZEN_IVF_V3.txt`](v3/FROZEN_IVF_V3.txt)) | [`v3/holdout/FAULTS_SHA256.txt`](v3/holdout/FAULTS_SHA256.txt), [`v3/corpus/LABEL_HASHES.txt`](v3/corpus/LABEL_HASHES.txt) |
| v4 | `becf39d2b1c1264d1b6330ca66b79390640985c9` ([`v4/FROZEN_V4.txt`](v4/FROZEN_V4.txt)) | [`v4/holdout/FAULTS_SHA256.txt`](v4/holdout/FAULTS_SHA256.txt), [`v4/corpus/LABEL_HASHES.txt`](v4/corpus/LABEL_HASHES.txt) |

Raw capture bundles (several hundred MB) are kept outside the repository; per-case results,
thresholds, and scores are committed under each round's `results/` directory.

## Maintenance policy

Research development is closed. Do not run another holdout, add research checks, or re-score
any holdout to change these conclusions. Changes to the tool itself should be ordinary
engineering maintenance (bug fixes, dependency updates, documentation) or fixes motivated by
real failures reported by external users. Any future research question needs its own new
pre-registration and a new holdout, and does not amend these results.
