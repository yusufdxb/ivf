# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Root-cause attribution from explicit evidence, kept separate from detection.

The divergence classifier (:mod:`ivf.divergence`) reads the *shape* of an error curve and
guesses. Across physics backends that guess is dominated by legitimate solver
differences, so it is not evidence of a cause. This module names a cause only when a
specific recorded input disagrees: a failed validity check whose differing fields are
known. A trajectory divergence with no such evidence is reported as
``unattributed_divergence`` together with the signal and step, and no cause is claimed.

The cause vocabulary is IVF's own; it describes *which evidence* disagreed, not a
diagnosis beyond it.
"""

from __future__ import annotations

from typing import Any

#: reason code -> cause, for checks whose code alone names the evidence
CODE_CAUSE = {
    "IVF-PROTOCOL-BACKEND-NOT-AS-DECLARED": "backend_identity",
    "IVF-PROTOCOL-SOLVER-CAPACITY-SATURATED": "solver_capacity",
    "IVF-PROTOCOL-INITIAL-STATE-NOT-REALIZED": "initial_state",
    "IVF-CONTROL-RANDOMIZATION-MISMATCH": "randomization",
    "IVF-PROTOCOL-SOLVER-NOT-AS-DECLARED": "solver_configuration",
    "IVF-CONTROL-INITIAL-STATE-MISMATCH": "initial_state",
    "IVF-CONTROL-RESET-SEMANTICS-MISMATCH": "initial_state",
    "IVF-CONTROL-JOINT-ORDER-MISMATCH": "joint_order",
    "IVF-CONTROL-TERMINATION-SEMANTICS-MISMATCH": "termination_semantics",
    "IVF-CONTROL-CONTROL-FREQUENCY-MISMATCH": "timing",
    "IVF-CONTROL-ACTION-TIMING-MISMATCH": "timing",
    "IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH": "observation_definition",
    "IVF-CONTROL-ACTION-SEQUENCE-MISMATCH": "action_stream",
    "IVF-CONTROL-ASSET-MISMATCH": "task_identity",
    "IVF-CONTROL-TASK-VARIANT-MISMATCH": "task_identity",
    "IVF-CONTROL-SEED-MISMATCH": "pairing",
    "IVF-CONTROL-ENV-COUNT-MISMATCH": "pairing",
    "IVF-CONTROL-HORIZON-MISMATCH": "pairing",
    "IVF-CONTROL-ENV-ORDER-MISMATCH": "pairing",
    "IVF-CONTROL-FRAME-CONVENTION-MISMATCH": "conventions",
    "IVF-CONTROL-QUATERNION-CONVENTION-MISMATCH": "conventions",
    "IVF-EXPERIMENT-SELF-COMPARISON": "identical_content",
}

#: field prefix -> cause, for checks whose differing fields refine the cause
FIELD_CAUSE = (
    ("policy_interface.obs_layout", "policy_interface.observation_layout"),
    ("policy_interface.obs_joint_order", "policy_interface.joint_order"),
    ("policy_interface.action_joint_order", "policy_interface.joint_order"),
    ("policy_interface.action_scale", "policy_interface.action_scale"),
    ("policy_interface.", "policy_interface.other"),
    ("effective_model.body_mass", "model_parameters.mass_properties"),
    ("effective_model.body_static_friction", "model_parameters.contact_material"),
    ("effective_model.body_restitution", "model_parameters.contact_material"),
    ("effective_model.body_com", "model_parameters.mass_properties"),
    ("effective_model.body_inertia", "model_parameters.mass_properties"),
    ("effective_model.", "model_parameters.actuation"),
)

#: fixed precedence: the most specific, least ambiguous evidence first
PRECEDENCE = (
    "backend_identity",
    "solver_configuration",
    "solver_capacity",
    "initial_state",
    "joint_order",
    "policy_interface.joint_order",
    "policy_interface.observation_layout",
    "policy_interface.action_scale",
    "policy_interface.other",
    "termination_semantics",
    "randomization",
    "model_parameters.contact_material",
    "model_parameters.actuation",
    "model_parameters.mass_properties",
    "timing",
    "observation_definition",
    "action_stream",
    "task_identity",
    "pairing",
    "conventions",
    "identical_content",
    "unattributed_divergence",
)


def _field_causes(fields: list[str]) -> list[str]:
    out = []
    for f in fields:
        for prefix, cause in FIELD_CAUSE:
            if f.startswith(prefix):
                out.append(cause)
                break
    return out


def attribute(validity: Any, outcomes: list[Any]) -> dict[str, Any]:
    """Return ``{"findings": [...], "primary": cause | None}`` for one run."""
    findings: list[dict[str, Any]] = []
    for check in validity.checks if validity is not None else []:
        if check.status != "fail":
            continue
        fields = list(getattr(check, "fields", []) or [])
        causes = _field_causes(fields) or [CODE_CAUSE.get(check.reason_code or "", "other_validity")]
        for cause in dict.fromkeys(causes):
            findings.append(
                {
                    "cause": cause,
                    "evidence": check.check_id,
                    "reason_code": check.reason_code,
                    "fields": [
                        f for f in fields if not _field_causes([f]) or cause in _field_causes([f])
                    ],
                    "detail": check.detail,
                }
            )
    if not findings:
        for o in outcomes:
            if o.status != "fail":
                continue
            div = o.divergence
            findings.append(
                {
                    "cause": "unattributed_divergence",
                    "evidence": o.name,
                    "reason_code": (o.reason_codes or [None])[0],
                    "signal": getattr(div, "signal", None) if div else None,
                    "first_step": (
                        div.first_tolerance_violation_step
                        if div and div.first_tolerance_violation_step is not None
                        else getattr(div, "first_event_disagreement_step", None)
                    )
                    if div
                    else None,
                    "detail": "the subjects differ, but no recorded input explains it",
                }
            )
    rank = {c: i for i, c in enumerate(PRECEDENCE)}
    findings.sort(key=lambda f: rank.get(f["cause"], len(PRECEDENCE)))
    return {
        "findings": findings,
        "primary": findings[0]["cause"] if findings else None,
        "causes": list(dict.fromkeys(f["cause"] for f in findings)),
    }
