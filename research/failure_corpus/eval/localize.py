# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Pre-registered, deterministic mapping from an IVF verdict to a root-cause category.

Fixed before any corpus case was evaluated. The mapping is deliberately generous to IVF:
every IVF vocabulary item with a plausible reading is mapped to a corpus category. Items
with no corresponding category map to ``unspecific``, which is scored as incorrect.

Order: (1) failed validity checks, in IVF's report order; (2) otherwise the failing oracle
with the earliest first tolerance violation (ties broken by manifest order), using its
divergence classification; event/decision oracles without a divergence record map by the
signal they watch.
"""

from __future__ import annotations

from typing import Any

REASON_CODE_MAP = {
    "IVF-CONTROL-CONTROL-FREQUENCY-MISMATCH": "timestep_decimation",
    "IVF-CONTROL-ACTION-TIMING-MISMATCH": "timestep_decimation",
    "IVF-CONTROL-INITIAL-STATE-MISMATCH": "reset_randomization",
    "IVF-CONTROL-RESET-SEMANTICS-MISMATCH": "reset_randomization",
    "IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH": "obs_action_ordering",
    "IVF-CONTROL-ACTION-SEQUENCE-MISMATCH": "obs_action_ordering",
    "IVF-CONTROL-SOLVER-PRESENTED-AS-EQUIVALENT": "actuator_solver_config",
}

CLASSIFICATION_MAP = {
    "reset_mismatch": "reset_randomization",
    "setup_mismatch": "timestep_decimation",
    "action_mismatch": "obs_action_ordering",
    "solver_parameter_difference": "actuator_solver_config",
    "contact_model_difference": "actuator_solver_config",
    "unsupported_feature": "ineffective_stale_config",
    "sensor_semantic_difference": "termination_metric",
}

EVENT_SIGNAL_MAP = {"illegal_contact": "termination_metric", "base_contact_force": "termination_metric"}


def localize(
    validity: dict[str, Any], oracles: list[dict[str, Any]], oracle_order: list[str]
) -> dict[str, Any]:
    """Return ``{"category", "basis", "signal"}`` from sealed ``validity.json`` and ``oracles.json``."""
    for check in validity.get("checks", []):
        if check.get("status") == "fail":
            code = check.get("reason_code") or ""
            return {
                "category": REASON_CODE_MAP.get(code, "unspecific"),
                "basis": f"validity {check.get('check_id')} {code}",
                "signal": None,
            }

    failing = [o for o in oracles if o.get("status") == "fail"]
    if not failing:
        return {"category": "none", "basis": "no failing validity check or oracle", "signal": None}

    def key(o):
        div = o.get("divergence") or {}
        step = div.get("first_tolerance_violation_step")
        if step is None:
            step = div.get("first_event_disagreement_step")
        if step is None:
            step = (o.get("metrics") or {}).get("first_disagreement_step")
        name = o.get("name")
        order = oracle_order.index(name) if name in oracle_order else len(oracle_order)
        return (step if isinstance(step, int) else 10**9, order)

    first = sorted(failing, key=key)[0]
    div = first.get("divergence") or {}
    signal = div.get("signal") or (first.get("metrics") or {}).get("signal")
    cls = div.get("classification")
    if cls in CLASSIFICATION_MAP:
        return {
            "category": CLASSIFICATION_MAP[cls],
            "basis": f"{first.get('name')} classification={cls}",
            "signal": signal,
        }
    sig = signal or first.get("name", "")
    for token, cat in EVENT_SIGNAL_MAP.items():
        if token in str(sig) or token in str(first.get("name")):
            return {"category": cat, "basis": f"{first.get('name')} event on {token}", "signal": sig}
    return {
        "category": "unspecific",
        "basis": f"{first.get('name')} classification={cls}",
        "signal": sig,
    }
