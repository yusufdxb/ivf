# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment-input controls: was the experiment that ran the experiment that was declared?

Pairwise trajectory comparison cannot separate a migration defect from a legitimate
backend difference when the backends diverge chaotically: the defect sits inside the
envelope that clean pairs already occupy. These checks do not compare trajectories. They
compare the *inputs* each subject actually ran with, as the capture recorded them at
runtime, and they check each subject against its own declaration where no pairing is
needed:

* ``joint_ordering``: both subjects record joint arrays in the same joint order.
* ``policy_interface``: the observation layout, joint order and action scale the policy
  was fed agree.
* ``effective_model_parameters``: masses, gains and armature read back from the simulator
  agree within a declared tolerance.
* ``termination_semantics``: termination terms watch the same bodies with the same
  thresholds.
* ``initial_state_realization`` (per subject): the simulator state read back after reset
  equals the requested state within a declared tolerance.
* ``backend_identity`` (per subject): the runtime physics backend equals the subject's
  declared ``expect_backend``.
* ``resource_health`` (per subject): no solver buffer reached capacity.

Every check needs the optional ``experiment_inputs`` capture block. A subject without it
makes the control ``unverifiable``, never ``pass``. Each failure carries the fields that
differ, which is what :mod:`ivf.attribution` uses to name a cause.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .manifest import Manifest
from .signals import SignalSet


def _inputs(sset: SignalSet) -> dict[str, Any]:
    return sset.metadata.get("experiment_inputs") or {}


def _check(
    check_id: str,
    name: str,
    status: str,
    detail: str,
    code: str | None = None,
    baseline: Any = None,
    candidate: Any = None,
    fields: list[str] | None = None,
):
    from .validity import ValidityCheck

    c = ValidityCheck(
        check_id, name, status, detail, code, baseline_value=baseline, candidate_value=candidate
    )
    c.fields = list(fields or [])  # attribution reads this; not part of the stable JSON contract
    return c


def _compare_mapping(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _max_rel_diff(a: Any, b: Any) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        return float("inf")
    scale = np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-9)
    return float(np.max(np.abs(x - y) / scale)) if x.size else 0.0


def check_experiment_inputs(manifest: Manifest, baseline: SignalSet, candidate: SignalSet) -> list:
    """Return validity checks for every experiment-input control the manifest requires."""
    required = set(manifest.controls.require_same)
    checks = []
    a_in, b_in = _inputs(baseline), _inputs(candidate)

    if "joint_ordering" in required:
        a, b = baseline.metadata.get("joint_names") or [], candidate.metadata.get("joint_names") or []
        if not a or not b:
            checks.append(
                _check(
                    "V-20",
                    "joint order matches",
                    "unverifiable",
                    "at least one subject does not record task.joint_names",
                )
            )
        elif list(a) != list(b):
            pairs = enumerate(zip(a, b, strict=False))
            first = next((i for i, (x, y) in pairs if x != y), min(len(a), len(b)))
            checks.append(
                _check(
                    "V-20",
                    "joint order matches",
                    "fail",
                    f"joint arrays use different orders (first difference at index {first}: "
                    f"{a[first] if first < len(a) else None!r} vs "
                    f"{b[first] if first < len(b) else None!r}); element-wise joint "
                    "comparison is undefined",
                    "IVF-CONTROL-JOINT-ORDER-MISMATCH",
                    a,
                    b,
                    ["joint_order"],
                )
            )
        else:
            checks.append(_check("V-20", "joint order matches", "pass", f"{len(a)} joints, same order"))

    if "policy_interface" in required:
        a, b = a_in.get("policy_interface"), b_in.get("policy_interface")
        if not a or not b:
            checks.append(
                _check(
                    "V-21",
                    "policy interface matches",
                    "unverifiable",
                    "at least one subject does not record experiment_inputs.policy_interface",
                )
            )
        else:
            diff = _compare_mapping(a, b)
            if diff:
                checks.append(
                    _check(
                        "V-21",
                        "policy interface matches",
                        "fail",
                        f"policy interface differs in {diff}",
                        "IVF-CONTROL-POLICY-INTERFACE-MISMATCH",
                        {k: a.get(k) for k in diff},
                        {k: b.get(k) for k in diff},
                        [f"policy_interface.{k}" for k in diff],
                    )
                )
            else:
                checks.append(
                    _check("V-21", "policy interface matches", "pass", f"fields {sorted(a)} identical")
                )

    if "effective_model_parameters" in required:
        tol = manifest.controls.tolerances["effective_model_parameters"].value
        a, b = a_in.get("effective_model"), b_in.get("effective_model")
        if not a or not b:
            checks.append(
                _check(
                    "V-22",
                    "effective model parameters match",
                    "unverifiable",
                    "at least one subject does not record experiment_inputs.effective_model",
                )
            )
        else:
            keys = sorted(k for k in set(a) | set(b) if not k.endswith("_names"))
            name_diff = [k for k in set(a) | set(b) if k.endswith("_names") and a.get(k) != b.get(k)]
            rel = {k: (_max_rel_diff(a[k], b[k]) if k in a and k in b else float("inf")) for k in keys}
            bad = sorted(k for k, v in rel.items() if not v <= tol) + sorted(name_diff)
            if bad:
                checks.append(
                    _check(
                        "V-22",
                        "effective model parameters match",
                        "fail",
                        "realized parameters differ beyond the declared relative tolerance "
                        f"{tol:g}: " + ", ".join(f"{k} {rel.get(k, float('inf')):.3g}" for k in bad),
                        "IVF-CONTROL-EFFECTIVE-MODEL-MISMATCH",
                        {k: rel.get(k) for k in bad},
                        None,
                        [f"effective_model.{k}" for k in bad],
                    )
                )
            else:
                checks.append(
                    _check(
                        "V-22",
                        "effective model parameters match",
                        "pass",
                        f"{len(keys)} parameter arrays within relative {tol:g}",
                    )
                )

    if "termination_semantics" in required:
        a, b = a_in.get("termination_semantics"), b_in.get("termination_semantics")
        if a is None or b is None:
            checks.append(
                _check(
                    "V-23",
                    "termination semantics match",
                    "unverifiable",
                    "at least one subject does not record experiment_inputs.termination_semantics",
                )
            )
        elif a != b:
            checks.append(
                _check(
                    "V-23",
                    "termination semantics match",
                    "fail",
                    "termination terms differ",
                    "IVF-CONTROL-TERMINATION-SEMANTICS-MISMATCH",
                    a,
                    b,
                    ["termination_semantics"],
                )
            )
        else:
            checks.append(
                _check("V-23", "termination semantics match", "pass", f"{len(a)} term(s) identical")
            )

    if "randomization_semantics" in required:
        a, b = a_in.get("randomization"), b_in.get("randomization")
        if a is None or b is None:
            checks.append(
                _check(
                    "V-27",
                    "randomization terms match",
                    "unverifiable",
                    "at least one subject does not record experiment_inputs.randomization",
                )
            )
        else:
            ka = {t.get("term"): t for t in a}
            kb = {t.get("term"): t for t in b}
            diff = sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))
            if diff:
                checks.append(
                    _check(
                        "V-27",
                        "randomization terms match",
                        "fail",
                        f"randomization terms differ: {diff}",
                        "IVF-CONTROL-RANDOMIZATION-MISMATCH",
                        {k: ka.get(k) for k in diff},
                        {k: kb.get(k) for k in diff},
                        [f"randomization.{k}" for k in diff],
                    )
                )
            else:
                checks.append(
                    _check("V-27", "randomization terms match", "pass", f"{len(ka)} term(s) identical")
                )

    for role, inp in (("baseline", a_in), ("candidate", b_in)):
        if "initial_state_realization" in required:
            tol = manifest.controls.tolerances["initial_state_realization"].value
            real = inp.get("reset_realization")
            if not real or "requested" not in real or "readback" not in real:
                checks.append(
                    _check(
                        "V-24",
                        f"{role} initial state realized",
                        "unverifiable",
                        f"the {role} does not record experiment_inputs.reset_realization",
                    )
                )
            else:
                req, got = real["requested"], real["readback"]
                errs = {
                    k: (
                        float(np.max(np.abs(np.asarray(req[k], float) - np.asarray(got[k], float))))
                        if k in got and np.shape(req[k]) == np.shape(got[k])
                        else float("inf")
                    )
                    for k in req
                }
                bad = sorted(k for k, v in errs.items() if not v <= tol)
                if bad:
                    checks.append(
                        _check(
                            "V-24",
                            f"{role} initial state realized",
                            "fail",
                            f"the {role}'s state read back after reset differs from the requested "
                            "state beyond "
                            f"{tol:g}: " + ", ".join(f"{k} {errs[k]:.3g}" for k in bad),
                            "IVF-PROTOCOL-INITIAL-STATE-NOT-REALIZED",
                            role,
                            errs,
                            [f"reset_realization.{k}" for k in bad],
                        )
                    )
                else:
                    checks.append(
                        _check(
                            "V-24",
                            f"{role} initial state realized",
                            "pass",
                            f"{sorted(errs)} within {tol:g}",
                        )
                    )

        if "backend_identity" in required:
            expected = (
                manifest.subjects[role].params.get("expect_backend")
                if hasattr(manifest, "subjects")
                else None
            )
            runtime = (inp.get("runtime") or {}).get("backend")
            if not expected or not runtime:
                checks.append(
                    _check(
                        "V-25",
                        f"{role} runs the declared backend",
                        "unverifiable",
                        f"missing {'subject expect_backend' if not expected else 'runtime backend record'}",
                    )
                )
            elif str(expected).lower() != str(runtime).lower():
                checks.append(
                    _check(
                        "V-25",
                        f"{role} runs the declared backend",
                        "fail",
                        f"the {role} declares backend {expected!r} but ran {runtime!r} "
                        f"({(inp.get('runtime') or {}).get('manager')})",
                        "IVF-PROTOCOL-BACKEND-NOT-AS-DECLARED",
                        expected,
                        runtime,
                        ["runtime.backend"],
                    )
                )
            else:
                checks.append(
                    _check("V-25", f"{role} runs the declared backend", "pass", f"both {runtime!r}")
                )

        if "resource_health" in required:
            health = inp.get("resource_health") or {}
            if health.get("status") != "available":
                checks.append(
                    _check(
                        "V-26",
                        f"{role} solver resources not saturated",
                        "unverifiable",
                        f"the {role}'s backend exposes no resource counters "
                        f"({health.get('status', 'not recorded')})",
                    )
                )
            else:
                util = {
                    k: float(v)
                    for k, v in health.items()
                    if k.startswith("max_") and k.endswith("_utilization")
                }
                bad = sorted(k for k, v in util.items() if v >= 1.0)
                if bad:
                    checks.append(
                        _check(
                            "V-26",
                            f"{role} solver resources not saturated",
                            "fail",
                            f"the {role} reached buffer capacity: "
                            + ", ".join(f"{k} {util[k]:.3g}" for k in bad),
                            "IVF-PROTOCOL-SOLVER-CAPACITY-SATURATED",
                            role,
                            util,
                            [f"resource_health.{k}" for k in bad],
                        )
                    )
                else:
                    checks.append(
                        _check(
                            "V-26",
                            f"{role} solver resources not saturated",
                            "pass",
                            ", ".join(f"{k} {v:.3g}" for k, v in sorted(util.items())),
                        )
                    )
    for role, inp in (("baseline", a_in), ("candidate", b_in)):
        if "solver_conformance" not in required:
            break
        expected = manifest.subjects[role].params.get("expect_solver")
        effective = inp.get("solver_effective") or {}
        if not expected:
            checks.append(
                _check(
                    "V-28",
                    f"{role} solver settings as declared",
                    "unverifiable",
                    f"the {role} declares no expect_solver",
                )
            )
            continue
        if effective.get("status") != "available":
            checks.append(
                _check(
                    "V-28",
                    f"{role} solver settings as declared",
                    "unverifiable",
                    f"the {role}'s running solver settings are not recorded "
                    f"({effective.get('reason', effective.get('status', 'absent'))})",
                )
            )
            continue
        bad = []
        for key, want in sorted(expected.items()):
            got = effective.get(key)
            if (
                isinstance(want, (int, float))
                and isinstance(got, (int, float))
                and not isinstance(want, bool)
            ):
                ok = abs(float(got) - float(want)) <= 1e-6 * max(1.0, abs(float(want)))
            else:
                ok = str(got).lower() == str(want).lower()
            if not ok:
                bad.append(key)
        if bad:
            checks.append(
                _check(
                    "V-28",
                    f"{role} solver settings as declared",
                    "fail",
                    f"the {role}'s running solver differs from the declared settings: "
                    + ", ".join(f"{k} {effective.get(k)!r} != {expected[k]!r}" for k in bad),
                    "IVF-PROTOCOL-SOLVER-NOT-AS-DECLARED",
                    {k: expected[k] for k in bad},
                    {k: effective.get(k) for k in bad},
                    [f"solver.{k}" for k in bad],
                )
            )
        else:
            checks.append(
                _check(
                    "V-28",
                    f"{role} solver settings as declared",
                    "pass",
                    f"{len(expected)} declared setting(s) match the running solver",
                )
            )
    return checks
