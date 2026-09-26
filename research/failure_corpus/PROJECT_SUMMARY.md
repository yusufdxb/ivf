# IVF: project summary

**What it is.** IVF is a command-line tool that audits cross-backend robotics simulation
experiments, such as a policy trained on PhysX being run on Newton/MuJoCo-Warp. It checks
that each run actually used the declared configuration and state: reset state, backend,
solver preset, model parameters, randomization, joint order, policy interface, and
termination rules. It compares trajectories under a declared contract and seals the
evidence so it can be re-verified later.

**What I tested.** Whether IVF catches silent validity failures that conventional
sim-to-sim evaluation accepts. I ran four pre-registered rounds on real PhysX and
Newton/MuJoCo-Warp captures of trained locomotion policies across seven robot/task
configurations (GO2, G1, H1, ANYmal-D, Cassie, Spot, and rough-terrain GO2), using
externally documented and independently authored fault families. Each holdout was sealed
before capture and scored once, and every deviation was logged.

**Result: a transparent negative.**

- On the final holdout, IVF's configuration/state checks raised **0/84 false alarms** and
  caught **21 defects that conventional validation missed**, all of them reset or
  randomization failures.
- Adding IVF to conventional validation lowered false acceptance from 0.548 to 0.423 with no
  added false alarms.
- The pre-registered family-level 95% CI for that improvement was **[0.000, 0.304]**. It
  includes zero because the gain came from 2 of 12 fault families, so the broad claim
  **failed**.
- IVF missed most dynamic and runtime faults. Closed-loop evaluation remains necessary.
- Earlier rounds found that IVF alone is weaker than conventional evaluation, and that its
  root-cause localization is poor.

**Findings along the way.**

- PhysX and Newton enumerate joints differently for every robot tested, and the checkout
  had no remap. Without one, a PhysX-trained policy silently receives permuted actions on
  Newton.
- Isaac Lab's Newton smoke test silently ran PhysX before an upstream fix.
- Spot's stock config applies material randomization on PhysX only.
- IVF's own shipped cart-pole evidence mislabels its quaternion layout.

**Conclusion.** IVF is useful as a conservative auditing layer that complements
closed-loop validation, not as a simulator-equivalence validator. Research development
stopped once the pre-registered decision rule failed; the tool remains available.

Details: [`README.md`](README.md) (index), [`v4/REPORT_V4.md`](v4/REPORT_V4.md) (authoritative).
