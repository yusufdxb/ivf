# IVF fault detectability matrix

- taxonomy: `ivf.faults/v1`
- base manifest: `calibration-base`
- seeds: [11, 23, 47]
- trials: 51 across 17 fault classes
- false positives on the negative control: **0**
- every row matches its declaration: **True**

> The taxonomy is author-generated. A fully consistent matrix establishes that IVF
> detects the fault classes its authors enumerated, not that it detects fault classes
> nobody thought of.

| fault | category | expected | layer | detection rate | oracles | classification | first step | consistent |
|---|---|---|---|---|---|---|---|---|
| `none` | control | no | - | 0% | - | - | - | yes |
| `ignored_reset_velocity` | reset | yes | oracle | 100% | abs_pole_angle_agreement, mean_angle_equivalence, pole_ang_vel_agreement, pole_angle_agreement, pole_orientation_agreement | reset_mismatch | 0 | yes |
| `wrong_env_origin` | frame | yes | oracle | 100% | pole_ang_vel_agreement | coordinate_frame_mismatch | 0 | yes |
| `incorrect_clone_count` | cloning | yes | validity | 100% | action_replay, observation_shape_stability | - | - | yes |
| `quaternion_ordering` | convention | yes | oracle | 100% | pole_orientation_agreement | quaternion_convention_mismatch | 0 | yes |
| `dropped_action` | action | yes | oracle | 100% | abs_pole_angle_agreement, mean_angle_equivalence, pole_ang_vel_agreement, pole_angle_agreement, pole_orientation_agreement | numerical_drift | 8 | yes |
| `shifted_action_timing` | action | yes | oracle | 100% | abs_pole_angle_agreement, pole_ang_vel_agreement, pole_angle_agreement, pole_orientation_agreement | unknown | 85 | yes |
| `observation_field_swap` | observation | yes | oracle | 100% | abs_pole_angle_agreement, pole_ang_vel_agreement | reset_mismatch | 0 | yes |
| `stale_sensor_state` | sensor | yes | oracle | 100% | pole_ang_vel_agreement | sensor_semantic_difference | 20 | yes |
| `incorrect_timestep` | timing | yes | validity | 100% | abs_pole_angle_agreement, mean_angle_equivalence, pole_ang_vel_agreement, pole_angle_agreement, pole_orientation_agreement | setup_mismatch | 5 | yes |
| `unit_scaling` | units | yes | oracle | 100% | abs_pole_angle_agreement, mean_angle_equivalence, pole_ang_vel_agreement, pole_angle_agreement, pole_orientation_agreement | reset_mismatch | 0 | yes |
| `altered_friction` | dynamics | yes | oracle | 100% | abs_pole_angle_agreement, mean_angle_equivalence, pole_ang_vel_agreement, pole_angle_agreement, pole_orientation_agreement | numerical_drift | 84, 85 | yes |
| `altered_mass` | dynamics | yes | oracle | 100% | abs_pole_angle_agreement, mean_angle_equivalence, pole_ang_vel_agreement, pole_angle_agreement, pole_orientation_agreement | numerical_drift | 5 | yes |
| `silent_nan` | numerical | yes | oracle | 100% | finite_state, pole_ang_vel_agreement | numerical_drift | 50 | yes |
| `truncated_rollout` | protocol | yes | validity | 100% | action_replay, mean_angle_equivalence | - | - | yes |
| `corrupted_metadata` | metadata | yes | validity | 100% | - | - | - | yes |
| `unsupported_feature_misreported` | capability | no | - | 0% | - | - | - | yes |

## Declared limitations

- **none** (min severity: n/a): Any detection here is a false positive and fails calibration.
- **ignored_reset_velocity** (min severity: initial velocity >= 0.05 rad/s against a 2e-3 rad tolerance): Undetectable when the declared initial velocity is already zero.
- **wrong_env_origin** (min severity: offset >= tolerance value): A per-env constant offset is indistinguishable from a genuine frame change without an explicit frame declaration; classification, not detection, is the weak part.
- **quaternion_ordering** (min severity: any non-identity rotation): Only applies to signals of last-dimension 4; a pure-identity trajectory is invariant under the swap and is honestly undetectable.
- **dropped_action** (min severity: drive amplitude >= 0.5 with a 2e-3 rad tolerance): Undetectable in a passive scenario, where there is no action to drop.
- **shifted_action_timing** (min severity: drive amplitude >= 1.7 on the calibration workload, measured: at amplitude 1.5 a one-step shift produces 1.78e-3 rad of second-largest error against a 2e-3 rad budget and is NOT detected; the crossing is near amplitude 1.68): Undetectable for a constant action stream, and undetectable for a slowly varying one: the signal is the action's slope times one step, so a low-frequency or low-amplitude command hides a one-step delay below any operationally sane budget. This limit was measured, not assumed.
- **stale_sensor_state** (min severity: hold length >= 5 steps on a moving signal): Undetectable on a signal that is genuinely constant over the window.
- **incorrect_timestep** (min severity: any difference detectable in float64): Caught as an invalid experiment, not as a physics difference: comparing runs at different control frequencies is uninterpretable, so IVF refuses rather than scores.
- **altered_friction** (min severity: damping delta >= 0.01 over a 400-step horizon): At small deltas over short horizons this is genuinely below the noise of any defensible tolerance; the matrix records the miss rather than hiding it.
- **unsupported_feature_misreported** (min severity: n/a): IVF cannot detect a lie about a capability it never independently probes. This row exists to keep that gap visible; closing it needs a capability probe in `ivf doctor`, not a comparison oracle.
