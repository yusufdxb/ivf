# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment-input controls, the event occurrence allowance, and evidence attribution."""

from __future__ import annotations

import numpy as np
import pytest

from ivf.attribution import attribute
from ivf.manifest import ManifestError, parse_manifest
from ivf.oracles import OracleContext, get_oracle
from ivf.validity import check_experiment

from .conftest import make_signals

pytestmark = pytest.mark.unit

TOL = """
      value: {v}
      unit: dimensionless
      scope: test tolerance for a realized-value control
      rationale: a fixed test value chosen to sit between the pass and fail fixtures
      aggregation: max
      min_samples: 1
      kind: numerical"""


def manifest(controls: list[str], *, expect: tuple[str, str] | None = None, tol: float = 1e-6):
    tol_block = ""
    toleranced = [c for c in controls if c in ("effective_model_parameters", "initial_state_realization")]
    if toleranced:
        tol_block = "  tolerances:\n" + "".join(f"    {c}:{TOL.format(v=tol)}\n" for c in toleranced)
    eb = (f"    expect_backend: {expect[0]}\n", f"    expect_backend: {expect[1]}\n") if expect else ("", "")
    text = f"""
schema_version: ivf.validation/v1
name: inputs
subjects:
  baseline:
    kind: synthetic
    system: damped_pendulum
{eb[0]}  candidate:
    kind: synthetic
    system: damped_pendulum
{eb[1]}workload:
  task: damped_pendulum.v1
  num_envs: 4
  steps: 40
  warmup_steps: 0
  seeds: [3]
controls:
  require_same: {controls}
{tol_block}oracles:
  - type: invariant
    name: finite_state
    check: finite_state
"""
    return parse_manifest(text)


def inputs(**kw):
    base = {
        "runtime": {"backend": "physx", "manager": "PhysxManager"},
        "policy_interface": {"obs_layout": [["a", 3], ["b", 3]], "obs_joint_order": ["j0", "j1"],
                             "action_joint_order": ["j0", "j1"], "action_scale": 0.25},
        "effective_model": {"body_names": ["base"], "body_mass": [[10.0]], "joint_armature": [[0.0, 0.0]]},
        "termination_semantics": [{"term": "base_contact", "bodies": ["base"], "threshold": 1.0}],
        "reset_realization": {"requested": {"root_lin_vel": [[0.3, 0, 0]]},
                              "readback": {"root_lin_vel": [[0.3, 0, 0]]}},
        "resource_health": {"status": "available", "max_contact_utilization": 0.4},
    }
    base.update(kw)
    return base


def pair(a_inputs=None, b_inputs=None, a_joints=("j0", "j1"), b_joints=("j0", "j1")):
    a = make_signals("baseline", experiment_inputs=a_inputs or inputs(), joint_names=list(a_joints))
    b = make_signals("candidate", offset=0.01, experiment_inputs=b_inputs or inputs(), joint_names=list(b_joints))
    return a, b


def status(report, check_id, role=None):
    hits = [c for c in report.checks if c.check_id == check_id and (role is None or role in c.name)]
    assert hits, check_id
    return hits[0]


def test_matching_inputs_pass_every_control():
    m = manifest(["joint_ordering", "policy_interface", "effective_model_parameters", "termination_semantics",
                  "initial_state_realization", "resource_health"])
    rep = check_experiment(m, *pair())
    for cid in ("V-20", "V-21", "V-22", "V-23", "V-24", "V-26"):
        assert status(rep, cid).status == "pass", cid
    assert rep.valid


def test_joint_order_difference_invalidates():
    rep = check_experiment(manifest(["joint_ordering"]), *pair(b_joints=("j1", "j0")))
    c = status(rep, "V-20")
    assert c.status == "fail" and c.reason_code == "IVF-CONTROL-JOINT-ORDER-MISMATCH"
    assert not rep.valid


def test_policy_interface_names_the_differing_field():
    b = inputs(policy_interface={**inputs()["policy_interface"], "action_scale": 0.5})
    c = status(check_experiment(manifest(["policy_interface"]), *pair(b_inputs=b)), "V-21")
    assert c.status == "fail" and c.fields == ["policy_interface.action_scale"]


def test_effective_model_uses_the_declared_relative_tolerance():
    b = inputs(effective_model={**inputs()["effective_model"], "body_mass": [[10.5]]})
    assert status(check_experiment(manifest(["effective_model_parameters"], tol=0.1), *pair(b_inputs=b)),
                  "V-22").status == "pass"
    c = status(check_experiment(manifest(["effective_model_parameters"], tol=0.01), *pair(b_inputs=b)), "V-22")
    assert c.status == "fail" and c.fields == ["effective_model.body_mass"]


def test_realized_value_controls_require_a_tolerance():
    with pytest.raises(ManifestError, match="tolerances"):
        parse_manifest(manifest(["joint_ordering"]).source_text.replace(
            "require_same: ['joint_ordering']", "require_same: [effective_model_parameters]"))


def test_reset_realization_is_checked_per_subject_without_pairing():
    b = inputs(reset_realization={"requested": {"root_lin_vel": [[0.3, 0, 0]]},
                                  "readback": {"root_lin_vel": [[0.0, 0, 0]]}})
    rep = check_experiment(manifest(["initial_state_realization"], tol=1e-3), *pair(b_inputs=b))
    assert status(rep, "V-24", "baseline").status == "pass"
    c = status(rep, "V-24", "candidate")
    assert c.status == "fail" and c.reason_code == "IVF-PROTOCOL-INITIAL-STATE-NOT-REALIZED"


def test_backend_identity_compares_runtime_with_declaration():
    rep = check_experiment(manifest(["backend_identity"], expect=("physx", "newton")), *pair())
    assert status(rep, "V-25", "baseline").status == "pass"
    c = status(rep, "V-25", "candidate")
    assert c.status == "fail" and c.reason_code == "IVF-PROTOCOL-BACKEND-NOT-AS-DECLARED"


def test_backend_identity_without_a_declaration_is_unverifiable_not_pass():
    rep = check_experiment(manifest(["backend_identity"]), *pair())
    assert status(rep, "V-25", "candidate").status == "unverifiable"


def test_saturated_solver_buffer_invalidates_and_unavailable_is_unverifiable():
    b = inputs(resource_health={"status": "available", "max_contact_utilization": 1.0})
    a = inputs(resource_health={"status": "unavailable"})
    rep = check_experiment(manifest(["resource_health"]), *pair(a_inputs=a, b_inputs=b))
    assert status(rep, "V-26", "baseline").status == "unverifiable"
    assert status(rep, "V-26", "candidate").reason_code == "IVF-PROTOCOL-SOLVER-CAPACITY-SATURATED"


def test_missing_input_block_is_unverifiable_for_every_control():
    a = make_signals("baseline")
    b = make_signals("candidate", offset=0.01)
    m = manifest(["policy_interface", "effective_model_parameters", "termination_semantics",
                  "initial_state_realization", "resource_health", "joint_ordering"])
    rep = check_experiment(m, a, b)
    new = [c for c in rep.checks if c.check_id in ("V-20", "V-21", "V-22", "V-23", "V-24", "V-26")]
    assert new and all(c.status == "unverifiable" for c in new)
    assert rep.valid


def test_attribution_names_the_evidence_and_prefers_the_most_specific():
    b = inputs(policy_interface={**inputs()["policy_interface"], "obs_joint_order": ["j1", "j0"]},
               runtime={"backend": "physx", "manager": "PhysxManager"})
    rep = check_experiment(manifest(["policy_interface", "backend_identity"], expect=("physx", "newton")),
                           *pair(b_inputs=b))
    att = attribute(rep, [])
    assert att["primary"] == "backend_identity"
    assert att["causes"] == ["backend_identity", "policy_interface.joint_order"]


def test_divergence_without_input_evidence_is_unattributed():
    from ivf.oracles import OracleOutcome

    out = OracleOutcome(name="traj", type="trajectory_equivalence", status="fail", summary="x",
                        reason_codes=["IVF-ORACLE-NON_EQUIVALENT"])
    rep = check_experiment(manifest(["joint_ordering"]), *pair())
    assert attribute(rep, [out])["primary"] == "unattributed_divergence"


def _event_ctx(fraction):
    from ivf.manifest import OracleSpec, Tolerance

    tol = Tolerance(value=1, unit="steps", scope="s", rationale="test rationale text", aggregation="max",
                    min_samples=4, kind="event")
    params = {"signal": "x", "threshold": 0.5, "condition": "above"}
    if fraction is not None:
        params["max_occurrence_mismatch_fraction"] = fraction
    spec = OracleSpec(type="event_equivalence", name="ev", params=params, tolerance=tol)
    a = make_signals("baseline", envs=4)
    b = make_signals("candidate", envs=4)
    b.signals["x"] = a.signals["x"].copy()
    b.signals["x"][:, 0, :] = 0.0  # env 0 never crosses in the candidate
    return OracleContext(spec=spec, baseline=a, candidate=b, alpha=0.05, seed=0, manifest=None)


def test_event_occurrence_allowance_defaults_to_the_original_strict_contract():
    assert get_oracle("event_equivalence")(_event_ctx(None)).status == "fail"
    assert get_oracle("event_equivalence")(_event_ctx(0.25)).status == "pass"
    assert get_oracle("event_equivalence")(_event_ctx(0.2)).status == "fail"


def test_event_occurrence_allowance_is_bounded():
    with pytest.raises(ValueError):
        get_oracle("event_equivalence")(_event_ctx(1.5))


def test_new_reason_codes_are_registered():
    from ivf.verdicts import REASON_CODES

    for code in ("IVF-CONTROL-JOINT-ORDER-MISMATCH", "IVF-PROTOCOL-BACKEND-NOT-AS-DECLARED",
                 "IVF-PROTOCOL-SOLVER-CAPACITY-SATURATED", "IVF-PROTOCOL-INITIAL-STATE-NOT-REALIZED"):
        assert code in REASON_CODES
    assert np.isfinite(1.0)


def test_randomization_terms_must_match():
    rand = [{"term": "physics_material", "mode": "startup", "static_friction_range": [0.8, 0.8]}]
    a = inputs(randomization=rand)
    b = inputs(randomization=[])
    c = status(check_experiment(manifest(["randomization_semantics"]), *pair(a_inputs=a, b_inputs=b)), "V-27")
    assert c.status == "fail" and c.reason_code == "IVF-CONTROL-RANDOMIZATION-MISMATCH"
    assert c.fields == ["randomization.physics_material"]
    ok = status(check_experiment(manifest(["randomization_semantics"]), *pair(a_inputs=a, b_inputs=a)), "V-27")
    assert ok.status == "pass"


def test_realized_friction_difference_is_attributed_to_contact_material():
    a = inputs(effective_model={**inputs()["effective_model"], "body_static_friction": [[0.8]]})
    b = inputs(effective_model={**inputs()["effective_model"], "body_static_friction": [[1.0]]})
    rep = check_experiment(manifest(["effective_model_parameters"], tol=1e-4), *pair(a_inputs=a, b_inputs=b))
    assert status(rep, "V-22").status == "fail"
    assert attribute(rep, [])["primary"] == "model_parameters.contact_material"


def _solver_manifest(expect):
    text = manifest(["solver_conformance"]).source_text
    block = "    expect_solver: " + str(expect).replace("'", '"') + "\n"
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.strip() == "system: damped_pendulum"][1]
    lines.insert(idx + 1, block.rstrip("\n"))
    return parse_manifest("\n".join(lines))


def test_solver_conformance_compares_running_solver_with_declaration():
    m = _solver_manifest({"integrator": "implicitfast", "iterations": 100})
    good = inputs(solver_effective={"status": "available", "integrator": "IMPLICITFAST", "iterations": 100})
    bad = inputs(solver_effective={"status": "available", "integrator": "euler", "iterations": 100})
    rep = check_experiment(m, *pair(b_inputs=good))
    assert status(rep, "V-28", "candidate").status == "pass"
    assert status(rep, "V-28", "baseline").status == "unverifiable"
    c = status(check_experiment(m, *pair(b_inputs=bad)), "V-28", "candidate")
    assert c.status == "fail" and c.fields == ["solver.integrator"]
    assert attribute(check_experiment(m, *pair(b_inputs=bad)), [])["primary"] == "solver_configuration"


def test_solver_conformance_without_readback_is_unverifiable():
    m = _solver_manifest({"integrator": "implicitfast"})
    c = status(check_experiment(m, *pair(b_inputs=inputs(solver_effective={"status": "unavailable"}))),
               "V-28", "candidate")
    assert c.status == "unverifiable"
