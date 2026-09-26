# Revised IVF on an independently authored holdout

Research report, 2026-09-26. Branch `research/failure-corpus-eval`, local only.
Pre-registration: [`docs/preregistration/2026-09-26-v2-holdout.md`](../../../docs/preregistration/2026-09-26-v2-holdout.md).
v1 report (the starting point): [`../REPORT.md`](../REPORT.md).

## Answer

**Revised IVF does not reduce false acceptance on its own. The union of conventional
sim-to-sim validation and IVF does, at no cost in false alarms.**

On 140 unseen cross-backend defects and 50 clean cases:

| Arm | Defects caught | False acceptance | False alarms |
|---|---|---|---|
| Conventional sim-to-sim | 89/140 | 0.36 | 14/50 |
| Revised IVF | 68/140 | **0.51** | 10/50 |
| **Conventional + IVF (union)** | **112/140** | **0.20** | **14/50** |

IVF alone accepts more defects than conventional validation, by 0.15 (95% CI [0.09, 0.21]). The
union accepts 0.16 fewer than conventional (CI [0.12, 0.21]) and 0.31 fewer than IVF
(CI [0.28, 0.35]), with the same false-alarm count as conventional alone: every IVF false
alarm is also a conventional false alarm. The contribution is therefore reframed as
**orthogonal experimental-validity checks that complement conventional validation**, not
as a validator that replaces or beats it.

## 1. Overlap on the v1 corpus, and the decision that followed

From the committed v1 results (`c59eb34`), 60 cross-backend defects, at matched false-alarm
rate (1/18): both 23, IVF only 5, conventional only 8, neither 24. The 5 IVF-only catches
were all one artifact (a byte-identical-rerun guard). Decision: IVF was **mostly a weaker
subset of conventional validation, with poor localization** (options 2 and 3, not 1).
Full table: [`OVERLAP.md`](OVERLAP.md). The 60-case corpus became development data.

The 24 "neither" cases were failures of the experiment's declared inputs (reset state,
randomization, actuator parameters, termination semantics, backend identity) that
trajectory comparison cannot separate from legitimate backend difference. That justified
the Section 5 capabilities.

## 2. What changed in IVF (general checks only)

Commit `1a7cced`, frozen as `src/ivf` tree `48861c1` (280 tests pass, 15 new). Every new
check is opt-in by manifest and reads an optional, additive `experiment_inputs` capture
block; missing evidence is `unverifiable`, never `pass`.

| Check | Mechanism it covers |
|---|---|
| `initial_state_realization` (per subject) | simulator readback after reset vs requested state |
| `backend_identity` (per subject) | runtime physics manager vs the backend the manifest declares |
| `resource_health` (per subject) | MJWarp contact/constraint buffer saturation |
| `joint_ordering` | joint arrays recorded in the same order |
| `policy_interface` | observation layout, joint order, action scale fed to the policy |
| `effective_model_parameters` | masses, gains, armature read back from the simulator |
| `termination_semantics` | termination bodies and thresholds |
| event oracle `max_occurrence_mismatch_fraction` | declared, calibratable allowance (default 0, unchanged) |
| `ivf.attribution` | cause named only from failed input evidence; otherwise `unattributed_divergence` |

The producer (`v2/capture/corpus_capture_v2.py`) records that block from the live
simulator, and accepts externally written fault plugins through fixed hooks.

**Development result (in-sample, not evidence of generalization):** on the recaptured
60-case corpus, IVF caught 55/60 at 1/18 false alarms (conventional 32/60 at 1/18) and
attributed all 55 correctly. All 5 misses were physically absent faults: MJWarp silently
raises too-small buffers (`nconmax` 2 to 16 on GO2), and ANYmal-D's actuator network
ignores gain changes.

## 3. The holdout

- **Faults:** 14 defect and 4 benign families written by a separate agent that was told
  not to read IVF, the v2 harness, or any result. The families come from new external
  sources and exclude every v1 source and perturbation. They were sealed by hash
  (`b1e09ff`) and never edited.
- **Unseen conditions:** seeds 9 and 10 (never used) and a new platform, Cassie, on which
  nothing had run. Cassie was calibrated on clean seeds 0 to 2 only.
- **Freeze:** IVF and thresholds were frozen before the holdout was captured (`48861c1`,
  `1ba2fce`), and the holdout was evaluated once.
- **Disclosure:** the fault author's completion summary (family names and mechanisms) was
  visible to me before development evaluation. IVF source was frozen at that moment and
  has not changed since. Thresholds were then derived only from clean calibration data.

### Pre-registered criteria

| Criterion | Result | Status |
|---|---|---|
| IVF reduces false acceptance by at least 0.15 at comparable FPR | -0.15 (IVF worse), CI [-0.21, -0.09]; FPR diff -0.08 | **dead** |
| Union at least 0.10 below conventional, CI excluding 0 | 0.16, CI [0.12, 0.21], same FPR | **met: reframe** |
| Attribution top-1 at least 0.5 on flagged defects | 20/68 = 0.29 | **dead** |

### Overlap by category (140 defects)

| Category | both | IVF only | conventional only | neither |
|---|---|---|---|---|
| actuator/solver config | 6 | 4 | 0 | 0 |
| checkpoint/schema | 10 | 2 | 6 | 2 |
| ineffective/stale config | 2 | 0 | 1 | 17 |
| joint/body ordering | 2 | 8 | 0 | 0 |
| obs/action ordering | 0 | 0 | 20 | 0 |
| reset/randomization | 5 | 9 | 6 | 0 |
| termination/metric | 0 | 0 | 11 | 9 |
| timestep/decimation | 20 | 0 | 0 | 0 |
| **total** | **45** | **23** | **44** | **28** |

**Where IVF is orthogonal (23 catches conventional accepted):** reset-state realization
(15: a DOF-index shift in reset offsets and angular/linear twist order), realized actuator
model (4: explicit PD substituted for the trained actuator), and trajectory divergence (4).
Closed-loop performance stayed inside the calibrated envelope for these cases; why (for
example, a policy recovering from a bad start) was not measured.

**Where conventional is orthogonal (44):** observation-semantics bugs that only matter when
the policy acts on them (gravity quaternion misread, velocity in world frame: 20/20
conventional, 0/20 IVF), metric-only termination miscounting (10/10 vs 0/10), and
checkpoint observation scaling. IVF's open-loop replay records these observations but
cannot judge them within cross-backend tolerances, and its `policy_interface` check
records term *names*, not their *semantics*.

### Failures neither catches (28)

| Family | Why IVF misses it | Missing capability |
|---|---|---|
| friction randomization silently not applied (10) | `effective_model` records masses and gains, not contact materials | read back material properties (friction, restitution) |
| contact-force threshold unit error (9) | **producer bug:** my producer records the termination threshold as a constant instead of reading the live termination config | record termination parameters from the live termination manager |
| MJWarp preset overrides lost (7) | the template declares solver settings *allowed to differ*, so a wrong solver config is not a defect to IVF | a per-subject "declared vs effective solver config" check against the intended preset |
| observation scales applied (2) | as above, term names match | observation semantics in the interface declaration |

### Localization

IVF declined to name a cause for 38 of 68 detections (`unattributed_divergence`) rather
than guess. Of the 30 it attributed, 20 were right (0.67). The 10 "wrong" ones are a DOF
index shift in the reset offsets, which IVF correctly reported as an unrealized initial
state; the author labelled it joint ordering. The old shape classifier scored 0/68.

### Other numbers

- Cassie (unseen platform): conventional 19/28, IVF 12/28, union 22/28, 2/10 false alarms each.
- False alarms: 10 of IVF's 10 and 10 of conventional's 14 are one "benign" family,
  `env_spacing_widened`, which shifts world-frame base positions. Excluding it: IVF 0/40,
  conventional 4/40, union 4/40.
- Native Isaac Lab validation caught 4/140 (non-finite values in two families).

## 4. What the evidence supports

- **Supported:** IVF's experiment-input checks detect a real class of silent validity
  failures, such as unrealized reset state and substituted actuator models, that
  closed-loop sim-to-sim evaluation accepts. Adding them to conventional validation cuts
  false acceptance from 0.36 to 0.20 on an independently written holdout, with no added
  false alarms.
- **Not supported:** IVF as a standalone validator (it misses closed-loop-only failures
  and accepts more defects than conventional evaluation), and IVF as a root-cause
  localizer (0.29 top-1).
- **Honest framing for a paper:** "Experimental-validity checks for cross-backend robot
  learning experiments are orthogonal to performance-based sim-to-sim evaluation; the two
  must be combined." Evidence: a v1 negative result, a revision built only from
  documented mechanisms, and an independent holdout with an unseen platform.

## 5. Limits

- One Isaac Lab snapshot, one GPU, flat terrain, one policy per platform.
- 10 cases per holdout family (5 platforms x 2 seeds); per-family numbers are descriptive.
- Detection of input failures depends on producer instrumentation; one instrumentation bug
  (the termination threshold) cost 9 catches.
- The fault author and I are the same model family, so "independent" means separated by
  instructions and sealed files, not by a different person.
- The benign families were also written by that author; one of them
  (`env_spacing_widened`) is arguably not benign for any validator that compares
  world-frame positions.

## 6. Reproduce

```bash
python research/failure_corpus/v2/corpus/make_plans_v2.py
$ISAACLAB_PYTHON research/failure_corpus/v2/capture/corpus_capture_v2.py --plan research/failure_corpus/v2/plans/v2holdout__<p>.json --out $IVF_CORPUS_DATA/captures_v2/v2holdout
uv run --frozen python research/failure_corpus/v2/eval/v2.py ivf --split v2holdout
uv run --frozen python research/failure_corpus/v2/eval/v2.py score --split v2holdout
```

Per-case results: `v2/results/v2holdout/cases.csv`, `scores.json`. Raw bundles stay
outside the repository.
