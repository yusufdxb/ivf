# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The experiment-validity layer: is this comparison interpretable at all?

This runs *before* any oracle and can veto the whole run. The design principle is that
publishing a confident comparison of two things that were never comparable is the most
damaging failure a validation tool can have, worse than reporting nothing. So the
layer prefers ``INVALID_EXPERIMENT`` whenever it cannot establish that the declared
controls held.

A third status matters as much as pass and fail: ``unverifiable``. When a source does
not record the metadata a control needs, IVF says so explicitly instead of treating
absence of evidence as a match. The pre-existing parity work found real bundles in the
wild with empty ``git_sha`` and empty ``solver_settings``; those runs must not silently
count as controlled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .manifest import Manifest
from .signals import SignalSet

CheckStatus = Literal["pass", "fail", "unverifiable", "not_applicable"]


@dataclass
class ValidityCheck:
    """One control or protocol check."""

    check_id: str
    name: str
    status: CheckStatus
    detail: str = ""
    reason_code: str | None = None
    baseline_value: Any = None
    candidate_value: Any = None

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "reason_code": self.reason_code,
            "baseline_value": _safe(self.baseline_value),
            "candidate_value": _safe(self.candidate_value),
        }


def _safe(value: Any) -> Any:
    """Coerce a metadata value into something JSON can hold."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


@dataclass
class ValidityReport:
    """The full set of checks and the resulting judgement."""

    checks: list[ValidityCheck] = field(default_factory=list)

    @property
    def failures(self) -> list[ValidityCheck]:
        """Checks that failed outright."""
        return [c for c in self.checks if c.status == "fail"]

    @property
    def unverifiable(self) -> list[ValidityCheck]:
        """Checks whose inputs were missing."""
        return [c for c in self.checks if c.status == "unverifiable"]

    @property
    def valid(self) -> bool:
        """Whether the experiment is interpretable."""
        return not self.failures

    def reason_codes(self) -> list[str]:
        """Stable reason codes for every failed check, in order."""
        return [c.reason_code for c in self.failures if c.reason_code]

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "valid": self.valid,
            "n_checks": len(self.checks),
            "n_failed": len(self.failures),
            "n_unverifiable": len(self.unverifiable),
            "checks": [c.to_jsonable() for c in self.checks],
        }


# Control name -> (check_id, metadata key or None, reason code, human name)
_CONTROL_CHECKS = {
    "asset_identity": ("V-01", "asset_identity", "IVF-CONTROL-ASSET-MISMATCH", "asset identity"),
    "initial_state_distribution": (
        "V-03", "initial_state_digest", "IVF-CONTROL-INITIAL-STATE-MISMATCH",
        "initial-state distribution",
    ),
    "control_frequency": (
        "V-05", "control_frequency_hz", "IVF-CONTROL-CONTROL-FREQUENCY-MISMATCH", "control frequency",
    ),
    "seeds": ("V-06", "seed", "IVF-CONTROL-SEED-MISMATCH", "seed"),
    "task_variant": ("V-09", "task_variant", "IVF-CONTROL-TASK-VARIANT-MISMATCH", "task variant"),
    "frame_convention": (
        "V-14", "frame_convention", "IVF-CONTROL-FRAME-CONVENTION-MISMATCH", "frame convention",
    ),
    "quaternion_convention": (
        "V-15", "quaternion_convention", "IVF-CONTROL-QUATERNION-CONVENTION-MISMATCH",
        "quaternion convention",
    ),
    "reset_semantics": (
        "V-16", "reset_semantics", "IVF-CONTROL-RESET-SEMANTICS-MISMATCH", "reset semantics",
    ),
    "environment_ordering": (
        "V-17", "environment_ordering", "IVF-CONTROL-ENV-ORDER-MISMATCH", "environment ordering",
    ),
    "action_timing": (
        "V-18", "action_timing", "IVF-CONTROL-ACTION-TIMING-MISMATCH", "action timing",
    ),
}


def check_experiment(
    manifest: Manifest, baseline: SignalSet, candidate: SignalSet
) -> ValidityReport:
    """Run every applicable validity check and return the report."""
    report = ValidityReport()
    required = set(manifest.controls.require_same)
    allowed = set(manifest.controls.allow_different)
    unsupported = set(manifest.controls.unsupported_or_unverifiable)

    for control in sorted(unsupported):
        report.checks.append(ValidityCheck(
            "V-UNVERIFIABLE",
            f"{control.replace('_', ' ')} is explicitly partitioned",
            "unverifiable",
            "the manifest records this control as unsupported or unverifiable; it is not "
            "treated as matched and cannot support the verdict",
        ))

    # --- declared metadata controls ---------------------------------------------------
    for control, (check_id, key, code, label) in _CONTROL_CHECKS.items():
        if control not in required:
            continue
        a, b = baseline.metadata.get(key), candidate.metadata.get(key)
        if a in (None, "", {}) or b in (None, "", {}):
            report.checks.append(ValidityCheck(
                check_id, f"{label} matches", "unverifiable",
                f"at least one subject does not record {key!r}; the control was requested but "
                "cannot be established",
                baseline_value=a, candidate_value=b,
            ))
        elif a != b:
            report.checks.append(ValidityCheck(
                check_id, f"{label} matches", "fail",
                f"baseline {a!r} != candidate {b!r}", code, baseline_value=a, candidate_value=b,
            ))
        else:
            report.checks.append(ValidityCheck(
                check_id, f"{label} matches", "pass", f"both {a!r}",
                baseline_value=a, candidate_value=b,
            ))

    # --- action sequence --------------------------------------------------------------
    if "action_sequence" in required:
        a, b = baseline.action_digest(), candidate.action_digest()
        if not a or not b:
            report.checks.append(ValidityCheck(
                "V-02", "action sequence matches", "unverifiable",
                "at least one subject records neither an action stream nor its digest",
            ))
        elif a != b:
            report.checks.append(ValidityCheck(
                "V-02", "action sequence matches", "fail",
                f"action digests differ ({a[:16]}… vs {b[:16]}…)",
                "IVF-CONTROL-ACTION-SEQUENCE-MISMATCH",
            ))
        else:
            report.checks.append(ValidityCheck(
                "V-02", "action sequence matches", "pass", f"digest {a[:16]}…",
            ))

    # --- observation definition -------------------------------------------------------
    if "observation_definition" in required:
        a, b = baseline.observation_definition(), candidate.observation_definition()
        if a == b:
            report.checks.append(ValidityCheck(
                "V-04", "observation definition matches", "pass", f"{len(a)} signal(s)",
            ))
        else:
            only_a = sorted(set(a) - set(b))
            only_b = sorted(set(b) - set(a))
            reshaped = sorted(k for k in set(a) & set(b) if a[k] != b[k])
            report.checks.append(ValidityCheck(
                "V-04", "observation definition matches", "fail",
                f"baseline-only {only_a}, candidate-only {only_b}, reshaped {reshaped}",
                "IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH",
                baseline_value=a, candidate_value=b,
            ))

    # --- environment count ------------------------------------------------------------
    if "num_envs" in required or "action_sequence" in required or "seeds" in required:
        a, b = baseline.num_envs, candidate.num_envs
        status = "pass" if a == b else "fail"
        report.checks.append(ValidityCheck(
            "V-07", "environment count matches", status,
            f"baseline {a} vs candidate {b}",
            None if status == "pass" else "IVF-CONTROL-ENV-COUNT-MISMATCH",
            baseline_value=a, candidate_value=b,
        ))

    # --- horizon ----------------------------------------------------------------------
    if "horizon" in required:
        a, b = baseline.steps, candidate.steps
        status = "pass" if a == b else "fail"
        report.checks.append(ValidityCheck(
            "V-08", "captured horizon matches", status, f"baseline {a} vs candidate {b}",
            None if status == "pass" else "IVF-CONTROL-HORIZON-MISMATCH",
            baseline_value=a, candidate_value=b,
        ))

    # --- solver parameters presented as equivalent -------------------------------------
    a_solver = baseline.metadata.get("solver_settings")
    b_solver = candidate.metadata.get("solver_settings")
    if "solver_specific_parameters" in required:
        if not a_solver or not b_solver:
            report.checks.append(ValidityCheck(
                "V-10", "solver settings match", "unverifiable",
                "at least one subject records no solver settings, so 'same solver' cannot be "
                "established. This is the exact gap that made released parity bundles "
                "un-comparable; it is reported, not assumed away.",
                baseline_value=a_solver, candidate_value=b_solver,
            ))
        elif a_solver != b_solver:
            report.checks.append(ValidityCheck(
                "V-10", "solver settings match", "fail",
                f"{a_solver!r} != {b_solver!r}", "IVF-CONTROL-SOLVER-PRESENTED-AS-EQUIVALENT",
                baseline_value=a_solver, candidate_value=b_solver,
            ))
        else:
            report.checks.append(ValidityCheck("V-10", "solver settings match", "pass", "identical"))
    elif "solver_specific_parameters" in allowed:
        if not a_solver or not b_solver:
            report.checks.append(ValidityCheck(
                "V-10", "solver settings are recorded", "unverifiable",
                "the manifest declares solver parameters may differ, but at least one subject "
                "records no solver configuration at all. IVF therefore cannot report *what* "
                "differs, so a solver-parameter explanation for any divergence below is a "
                "hypothesis rather than a finding.",
                baseline_value=a_solver, candidate_value=b_solver,
            ))
        else:
            report.checks.append(ValidityCheck(
                "V-10", "solver settings differ (declared allowed)", "not_applicable",
                "the manifest declares solver parameters may differ; any resulting divergence is "
                "reported as a solver-parameter difference rather than a defect",
                baseline_value=a_solver, candidate_value=b_solver,
            ))

    # --- protocol checks (always on) ---------------------------------------------------
    for role, sset in (("baseline", baseline), ("candidate", candidate)):
        if not sset.complete:
            report.checks.append(ValidityCheck(
                "V-11", f"{role} run is complete", "fail",
                f"the {role} run is marked incomplete (truncated or failed part-way)",
                "IVF-PROTOCOL-PARTIAL-RUN",
            ))
        elif sset.steps < manifest.workload.steps:
            report.checks.append(ValidityCheck(
                "V-11", f"{role} run is complete", "fail",
                f"the manifest declares {manifest.workload.steps} steps but the {role} evidence "
                f"has {sset.steps}",
                "IVF-PROTOCOL-PARTIAL-RUN",
                baseline_value=manifest.workload.steps, candidate_value=sset.steps,
            ))
        else:
            report.checks.append(ValidityCheck(
                "V-11", f"{role} run is complete", "pass", f"{sset.steps} steps captured",
            ))

    if manifest.workload.warmup_steps is None:
        report.checks.append(ValidityCheck(
            "V-12", "warm-up is declared", "unverifiable",
            "workload.warmup_steps is not declared, so early-transient behaviour is included in "
            "every oracle. Declare 0 explicitly if that is intended.",
            "IVF-CONTROL-WARMUP-UNDECLARED",
        ))
    else:
        report.checks.append(ValidityCheck(
            "V-12", "warm-up is declared", "pass", f"{manifest.workload.warmup_steps} steps discarded",
        ))

    # --- physics timing (always checked; a dt mismatch invalidates everything) ---------
    a_dt = baseline.metadata.get("physics_dt")
    b_dt = candidate.metadata.get("physics_dt")
    if a_dt is None or b_dt is None:
        report.checks.append(ValidityCheck(
            "V-13", "physics timestep matches", "unverifiable",
            "at least one subject does not record physics_dt",
            baseline_value=a_dt, candidate_value=b_dt,
        ))
    elif abs(float(a_dt) - float(b_dt)) > 1e-12:
        report.checks.append(ValidityCheck(
            "V-13", "physics timestep matches", "fail",
            f"baseline dt {a_dt} != candidate dt {b_dt}; trajectories at different timesteps are "
            "not step-comparable",
            "IVF-CONTROL-CONTROL-FREQUENCY-MISMATCH",
            baseline_value=a_dt, candidate_value=b_dt,
        ))
    else:
        report.checks.append(ValidityCheck(
            "V-13", "physics timestep matches", "pass", f"both {a_dt}",
        ))

    return report
