# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Offline comparison of two evidence bundles.

This is the command that answers "what changed between the run that passed last month
and the run that fails today", and it must work on a laptop with no GPU and no Isaac
Lab: it reads only JSON.

The comparison is careful about one thing above all: **whether the two bundles are
comparable at all**. Two runs of different manifests, different IVF versions with
incompatible evidence schemas, or different oracle sets do not produce a meaningful
verdict diff, and saying so is more useful than printing a diff of unrelated numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evidence import EvidenceBundle
from .verdicts import describe


@dataclass
class ComparisonReport:
    """A structured diff between a baseline and a candidate evidence bundle."""

    baseline_run_id: str
    candidate_run_id: str
    comparable: bool = True
    incomparability_reasons: list[str] = field(default_factory=list)

    verdict_change: dict[str, Any] = field(default_factory=dict)
    new_reason_codes: list[str] = field(default_factory=list)
    resolved_reason_codes: list[str] = field(default_factory=list)

    manifest_changes: dict[str, Any] = field(default_factory=dict)
    environment_changes: dict[str, Any] = field(default_factory=dict)
    version_changes: dict[str, Any] = field(default_factory=dict)
    hardware_changes: dict[str, Any] = field(default_factory=dict)

    validator_changes: dict[str, Any] = field(default_factory=dict)
    newly_failing_checks: list[str] = field(default_factory=list)
    resolved_checks: list[str] = field(default_factory=list)

    oracle_changes: list[dict[str, Any]] = field(default_factory=list)
    metric_changes: list[dict[str, Any]] = field(default_factory=list)
    integrity: dict[str, list[str]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """Whether anything material differs between the two bundles."""
        return bool(
            self.verdict_change.get("changed")
            or self.new_reason_codes
            or self.resolved_reason_codes
            or self.manifest_changes
            or self.validator_changes
            or self.newly_failing_checks
            or self.resolved_checks
            or self.oracle_changes
            or not self.comparable
        )

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "comparable": self.comparable,
            "incomparability_reasons": self.incomparability_reasons,
            "verdict_change": self.verdict_change,
            "new_reason_codes": self.new_reason_codes,
            "resolved_reason_codes": self.resolved_reason_codes,
            "manifest_changes": self.manifest_changes,
            "environment_changes": self.environment_changes,
            "version_changes": self.version_changes,
            "hardware_changes": self.hardware_changes,
            "validator_changes": self.validator_changes,
            "newly_failing_checks": self.newly_failing_checks,
            "resolved_checks": self.resolved_checks,
            "oracle_changes": self.oracle_changes,
            "metric_changes": self.metric_changes,
            "integrity": self.integrity,
        }


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted paths for a readable diff."""
    out: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(payload, list):
        out[prefix] = payload
    else:
        out[prefix] = payload
    return out


def _diff(baseline: Any, candidate: Any, *, ignore: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return the dotted-path differences between two nested mappings."""
    flat_a, flat_b = _flatten(baseline), _flatten(candidate)
    keys = sorted(set(flat_a) | set(flat_b))
    out: dict[str, Any] = {}
    for key in keys:
        if any(key.startswith(prefix) for prefix in ignore):
            continue
        a, b = flat_a.get(key, "<absent>"), flat_b.get(key, "<absent>")
        if a != b:
            out[key] = {"baseline": a, "candidate": b}
    return out


def compare_bundles(baseline_path: str | Path, candidate_path: str | Path) -> ComparisonReport:
    """Compare two evidence bundles and return a structured report."""
    a = EvidenceBundle.open(baseline_path)
    b = EvidenceBundle.open(candidate_path)
    a_verdict, b_verdict = a.verdict, b.verdict

    report = ComparisonReport(
        baseline_run_id=str(a_verdict.get("run_id", a.run_id)),
        candidate_run_id=str(b_verdict.get("run_id", b.run_id)),
    )

    # -- integrity first: a tampered bundle invalidates everything downstream -----------
    report.integrity = {"baseline": a.verify(), "candidate": b.verify()}
    for role, problems in report.integrity.items():
        if problems:
            report.comparable = False
            report.incomparability_reasons.append(
                f"{role} bundle failed integrity verification: {problems[0]}"
            )

    # -- comparability -------------------------------------------------------------------
    if a_verdict.get("experiment") != b_verdict.get("experiment"):
        report.comparable = False
        report.incomparability_reasons.append(
            f"different experiments: {a_verdict.get('experiment')!r} vs {b_verdict.get('experiment')!r}"
        )
    a_schema = str(a_verdict.get("evidence_schema_version", ""))
    b_schema = str(b_verdict.get("evidence_schema_version", ""))
    if a_schema.split(".")[0] != b_schema.split(".")[0]:
        report.comparable = False
        report.incomparability_reasons.append(
            f"incompatible evidence schemas: {a_schema!r} vs {b_schema!r}"
        )

    # -- verdict --------------------------------------------------------------------------
    report.verdict_change = {
        "baseline": a_verdict.get("verdict"),
        "candidate": b_verdict.get("verdict"),
        "changed": a_verdict.get("verdict") != b_verdict.get("verdict"),
    }
    a_codes = set(a_verdict.get("reason_codes") or [])
    b_codes = set(b_verdict.get("reason_codes") or [])
    report.new_reason_codes = sorted(b_codes - a_codes)
    report.resolved_reason_codes = sorted(a_codes - b_codes)

    # -- manifest ---------------------------------------------------------------------------
    report.manifest_changes = _diff(
        a.read_json("manifest.resolved.json"), b.read_json("manifest.resolved.json")
    )

    # -- environment, versions, hardware ------------------------------------------------------
    a_prov, b_prov = a.read_json("provenance.json"), b.read_json("provenance.json")
    report.environment_changes = _diff(
        a_prov.get("environment_variables", {}), b_prov.get("environment_variables", {})
    )
    report.version_changes = _diff(
        {"ivf": a_prov.get("ivf_version"), "python": a_prov.get("python", {}).get("version"),
         **_probe_versions(a_prov)},
        {"ivf": b_prov.get("ivf_version"), "python": b_prov.get("python", {}).get("version"),
         **_probe_versions(b_prov)},
    )
    report.hardware_changes = _diff(
        {"host": a_prov.get("host", {}).get("platform"), **_probe_hardware(a_prov)},
        {"host": b_prov.get("host", {}).get("platform"), **_probe_hardware(b_prov)},
    )

    # -- validity checks --------------------------------------------------------------------
    a_checks = {c["check_id"]: c for c in a.read_json("validity.json").get("checks", [])}
    b_checks = {c["check_id"]: c for c in b.read_json("validity.json").get("checks", [])}
    for check_id in sorted(set(a_checks) | set(b_checks)):
        old, new = a_checks.get(check_id), b_checks.get(check_id)
        if old is None or new is None:
            report.validator_changes[check_id] = {
                "baseline": None if old is None else old["status"],
                "candidate": None if new is None else new["status"],
                "note": "the set of validity checks differs between the two runs",
            }
            continue
        if old["status"] != new["status"]:
            report.validator_changes[check_id] = {
                "name": new["name"], "baseline": old["status"], "candidate": new["status"],
                "detail": new.get("detail", ""),
            }
            if new["status"] == "fail":
                report.newly_failing_checks.append(f"{check_id} {new['name']}")
            elif old["status"] == "fail":
                report.resolved_checks.append(f"{check_id} {old['name']}")

    # -- oracles ------------------------------------------------------------------------------
    a_oracles = {o["name"]: o for o in a.read_json("oracles.json")}
    b_oracles = {o["name"]: o for o in b.read_json("oracles.json")}
    for name in sorted(set(a_oracles) | set(b_oracles)):
        old, new = a_oracles.get(name), b_oracles.get(name)
        if old is None:
            report.oracle_changes.append({"oracle": name, "change": "added",
                                          "candidate_status": new["status"]})
            continue
        if new is None:
            report.oracle_changes.append({"oracle": name, "change": "removed",
                                          "baseline_status": old["status"]})
            continue
        if old["status"] != new["status"]:
            report.oracle_changes.append({
                "oracle": name, "change": "status",
                "baseline_status": old["status"], "candidate_status": new["status"],
                "baseline_summary": old["summary"], "candidate_summary": new["summary"],
            })
        if old.get("tolerance") != new.get("tolerance"):
            report.oracle_changes.append({
                "oracle": name, "change": "tolerance",
                "baseline_tolerance": old.get("tolerance"), "candidate_tolerance": new.get("tolerance"),
                "note": "the acceptance criterion itself changed; verdict differences across these "
                        "two runs are not evidence about the subjects",
            })
        for key in ("worst_aggregated_error", "worst_ulp", "agreement_rate",
                    "worst_timing_delta_steps", "mean_difference", "worst_breach"):
            old_v, new_v = (old.get("metrics") or {}).get(key), (new.get("metrics") or {}).get(key)
            if old_v is None and new_v is None:
                continue
            if old_v != new_v:
                report.metric_changes.append({
                    "oracle": name, "metric": key, "baseline": old_v, "candidate": new_v,
                    "delta": _delta(old_v, new_v),
                })
    return report


def _probe_versions(provenance: dict[str, Any]) -> dict[str, Any]:
    """Extract library versions the doctor recorded."""
    checks = {c["name"]: c for c in (provenance.get("doctor") or {}).get("checks", [])}
    return {
        name: checks.get(name, {}).get("value", "<absent>")
        for name in ("package:numpy", "isaaclab", "isaacsim", "torch", "backends")
    }


def _probe_hardware(provenance: dict[str, Any]) -> dict[str, Any]:
    """Extract hardware facts the doctor recorded."""
    checks = {c["name"]: c for c in (provenance.get("doctor") or {}).get("checks", [])}
    return {
        "gpu": checks.get("gpu", {}).get("value", "<absent>"),
        "gpu_driver": checks.get("gpu", {}).get("detail", ""),
        "determinism": checks.get("determinism", {}).get("value", ""),
    }


def _delta(old: Any, new: Any) -> Any:
    """Return a numeric delta when both values are numbers."""
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return new - old
    return None


def render_comparison_text(report: ComparisonReport) -> str:
    """Render a comparison report as plain text for the terminal."""
    lines = [
        "IVF evidence comparison",
        "=" * 70,
        f"baseline  {report.baseline_run_id}",
        f"candidate {report.candidate_run_id}",
        "",
    ]
    if not report.comparable:
        lines.append("NOT COMPARABLE")
        for reason in report.incomparability_reasons:
            lines.append(f"  - {reason}")
        lines.append("")
        lines.append("The differences below are reported for information only; they are not")
        lines.append("evidence about the subjects under test.")
        lines.append("")

    change = report.verdict_change
    arrow = "->" if change.get("changed") else "=="
    lines.append(f"verdict   {change.get('baseline')} {arrow} {change.get('candidate')}")
    lines.append("")

    if report.new_reason_codes:
        lines.append("New reason codes")
        for code in report.new_reason_codes:
            lines.append(f"  + {code}  {describe(code)}")
        lines.append("")
    if report.resolved_reason_codes:
        lines.append("Resolved reason codes")
        for code in report.resolved_reason_codes:
            lines.append(f"  - {code}  {describe(code)}")
        lines.append("")

    for title, payload in (
        ("Manifest changes", report.manifest_changes),
        ("Version changes", report.version_changes),
        ("Hardware changes", report.hardware_changes),
        ("Environment changes", report.environment_changes),
    ):
        if payload:
            lines.append(title)
            for key, value in sorted(payload.items()):
                lines.append(f"  {key}: {value['baseline']!r} -> {value['candidate']!r}")
            lines.append("")

    if report.newly_failing_checks:
        lines.append("Newly failing validity checks")
        lines.extend(f"  + {c}" for c in report.newly_failing_checks)
        lines.append("")
    if report.resolved_checks:
        lines.append("Resolved validity checks")
        lines.extend(f"  - {c}" for c in report.resolved_checks)
        lines.append("")

    if report.oracle_changes:
        lines.append("Oracle changes")
        for item in report.oracle_changes:
            if item["change"] == "status":
                lines.append(f"  {item['oracle']}: {item['baseline_status']} -> "
                             f"{item['candidate_status']}")
                lines.append(f"      was: {item['baseline_summary']}")
                lines.append(f"      now: {item['candidate_summary']}")
            elif item["change"] == "tolerance":
                lines.append(f"  {item['oracle']}: acceptance criterion changed - {item['note']}")
            else:
                lines.append(f"  {item['oracle']}: {item['change']}")
        lines.append("")

    if report.metric_changes:
        lines.append("Metric changes")
        for item in report.metric_changes:
            delta = "" if item["delta"] is None else f"  (delta {item['delta']:+.4g})"
            lines.append(f"  {item['oracle']}.{item['metric']}: {item['baseline']!r} -> "
                         f"{item['candidate']!r}{delta}")
        lines.append("")

    if not report.changed:
        lines.append("No material differences.")
    return "\n".join(lines) + "\n"
