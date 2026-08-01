# Acceptance tolerance provenance

The machine-readable source of truth is
`validation/provenance/tolerances.yaml`. Every tolerance in every shipped example
manifest maps to exactly one record. `tests/test_tolerance_provenance.py` checks record
completeness, derives each value from named inputs, matches values and units to the
manifests, recomputes the cart-pole measurements from the sealed baseline capture, and
checks this generated index.

The registry distinguishes measured derivations from explicit engineering judgement.
A value with no empirical source is labeled that way rather than being given a physical
story after the fact. The records also state sample floors and limitations. Those fields
are provenance, not a claim that every engineering choice has statistical confidence.

For the load-bearing real cart-pole derivation:

```text
one-period angle budget
= measured median threshold-crossing rate / control frequency
= 3.456 rad/s / 120 Hz
= 0.0288 rad

median elapsed time to threshold
= (median post-step index + 1) / control frequency
= (37 + 1) / 120 Hz
= 0.3166667 s

pole angular-rate budget
= one-period angle budget / median elapsed time
= 0.0288 rad / 0.3166667 s
= 0.0909474 rad/s
```

The `+1` is required because capture index 0 is the first post-reset physics step. The
velocity gate uses the pole-only `pole_velocity` signal. It never norms the prismatic
cart velocity in m/s together with the revolute pole velocity in rad/s.

<!-- BEGIN GENERATED -->
| Provenance record | Value | Signal | Formula |
|---|---:|---|---|
| `synthetic_unit_quaternion_norm` | 1e-09 dimensionless | `pole_quat` | numerical_guard |
| `synthetic_pointwise_angle_budget` | 0.002 rad | `pole_angle` | declared_engineering_budget |
| `synthetic_absolute_angle_budget` | 0.002 rad | `abs_pole_angle` | signed_angle_budget |
| `synthetic_orientation_budget` | 0.002 rad | `pole_quat` | single_axis_angle_budget |
| `synthetic_angular_velocity_budget` | 0.02 rad/s | `pole_ang_vel` | angle_budget / permitted_accumulation_time |
| `synthetic_event_timing_allowance` | 1 steps | `abs_pole_angle` | one_sample_allowance |
| `synthetic_decision_agreement` | 1 fraction | `abs_pole_angle` | exact_agreement |
| `synthetic_horizon_mean_margin` | 0.005 rad | `pole_angle` | 2.5 * pointwise_angle_budget |
| `cartpole_float32_quaternion_norm` | 1e-05 dimensionless | `root_link_quat_w` | numerical_guard |
| `cartpole_one_period_angle_budget` | 0.0288 rad | `pole_angle` | measured_crossing_rate / control_frequency |
| `cartpole_median_horizon_velocity_budget` | 0.0909473684211 rad/s | `pole_velocity` | one_period_angle_budget / median_elapsed_time |
| `cartpole_event_timing_allowance` | 1 steps | `abs_pole_angle` | one_sample_allowance |
| `cartpole_decision_agreement` | 1 fraction | `abs_pole_angle` | exact_agreement |
| `cartpole_horizon_mean_margin` | 0.0144 rad | `pole_angle` | 0.5 * one_period_angle_budget |
| `legacy_cartpole_root_position_budget` | 0.01 m | `root_link_pos_w` | declared_engineering_budget |
| `legacy_cartpole_root_orientation_budget` | 0.001 rad | `root_link_quat_w` | declared_engineering_budget |
| `legacy_cartpole_root_position_rms_margin` | 0.05 m | `root_link_pos_w` | declared_engineering_budget |
| `quaternion_double_cover_numerical_guard` | 1e-09 rad | `root_link_quat_w` | numerical_guard |
<!-- END GENERATED -->
