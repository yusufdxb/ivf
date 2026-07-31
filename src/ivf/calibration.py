# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Fault-injection calibration: measuring what IVF actually detects.

Every detection claim IVF makes has to be earned. This module runs the shipped fault
taxonomy through the real ``validate`` path, one fault at a time, across several seeds,
and produces a **detectability matrix**: per fault class, whether detection was expected,
which layer was responsible, the measured detection rate, and the false-positive result
from the negative control.

Two properties make the numbers worth reading:

* The campaign uses the same code path a user runs. It does not call oracles directly,
  so a defect in the runner or the verdict roll-up shows up here.
* ``expected_detectable=False`` rows are asserted to *not* fire. A validator aggressive
  enough to flag everything is as useless as one that flags nothing, and the negative
  control is what separates the two.

What this does **not** establish: the taxonomy is author-generated, so a perfect matrix
means "IVF detects the faults its authors thought of", not "IVF detects unknown faults".
That distinction is recorded in the matrix header and must survive into any summary.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .evidence import unseal
from .faults import TAXONOMY, TAXONOMY_VERSION, FaultSpec
from .manifest import parse_manifest
from .runner import validate
from .verdicts import Verdict


@dataclass
class TrialResult:
    """One injected trial."""

    fault: str
    seed: int
    verdict: str
    reason_codes: list[str]
    detected: bool
    layer: str
    """Which layer flagged it: ``validity``, ``oracle``, ``runtime`` or ``none``."""

    responsible_oracles: list[str] = field(default_factory=list)
    first_step: int | None = None
    classification: str | None = None
    quantity: str | None = None


@dataclass
class FaultRow:
    """One row of the detectability matrix."""

    spec: FaultSpec
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        """Fraction of trials in which the fault was flagged."""
        return sum(t.detected for t in self.trials) / len(self.trials) if self.trials else float("nan")

    @property
    def layers(self) -> list[str]:
        """Distinct layers that flagged this fault."""
        return sorted({t.layer for t in self.trials if t.detected})

    @property
    def oracles(self) -> list[str]:
        """Distinct oracles that flagged this fault."""
        return sorted({o for t in self.trials for o in t.responsible_oracles})

    @property
    def classifications(self) -> list[str]:
        """Distinct divergence classifications observed."""
        return sorted({t.classification for t in self.trials if t.classification})

    @property
    def localized_steps(self) -> list[int]:
        """First-violation steps observed."""
        return sorted({t.first_step for t in self.trials if t.first_step is not None})

    @property
    def consistent(self) -> bool:
        """Whether the measured behaviour matches the declared expectation.

        Declared detectable means every trial must be flagged. Declared undetectable
        means no trial may be flagged: for the negative control that is the
        false-positive test, and for a genuinely invisible fault it keeps the honest
        limitation from silently becoming a detection claim.
        """
        if not self.trials:
            return False
        rate = self.detection_rate
        return rate == 1.0 if self.spec.expected_detectable else rate == 0.0

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "fault": self.spec.name,
            "category": self.spec.category,
            "surface": self.spec.surface,
            "description": self.spec.description,
            "expected_detectable": self.spec.expected_detectable,
            "responsible_oracle_declared": self.spec.responsible_oracle,
            "minimum_severity": self.spec.minimum_severity,
            "limitations": self.spec.limitations,
            "n_trials": len(self.trials),
            "seeds": sorted({t.seed for t in self.trials}),
            "detection_rate": self.detection_rate,
            "layers_observed": self.layers,
            "oracles_observed": self.oracles,
            "classifications_observed": self.classifications,
            "first_violation_steps": self.localized_steps,
            "consistent_with_declaration": self.consistent,
            "verdicts": sorted({t.verdict for t in self.trials}),
            "reason_codes": sorted({c for t in self.trials for c in t.reason_codes}),
        }


@dataclass
class DetectabilityMatrix:
    """The full campaign result."""

    taxonomy_version: str
    manifest_name: str
    seeds: list[int]
    rows: list[FaultRow] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        """Whether every row matched its declaration."""
        return all(row.consistent for row in self.rows)

    @property
    def false_positives(self) -> int:
        """Detections on the negative control."""
        row = next((r for r in self.rows if r.spec.name == "none"), None)
        return 0 if row is None else sum(t.detected for t in row.trials)

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "taxonomy_version": self.taxonomy_version,
            "manifest": self.manifest_name,
            "seeds": self.seeds,
            "n_faults": len(self.rows),
            "n_trials": sum(len(r.trials) for r in self.rows),
            "all_consistent": self.consistent,
            "false_positives_on_negative_control": self.false_positives,
            "rows": [r.to_jsonable() for r in self.rows],
            "scope_limitation": (
                "The taxonomy is author-generated. A fully consistent matrix establishes that "
                "IVF detects the fault classes its authors enumerated, not that it detects fault "
                "classes nobody thought of."
            ),
        }


def _inject_into_manifest(base_text: str, fault: str, seed: int) -> str:
    """Return manifest YAML with the named fault injected into the candidate subject."""
    doc = yaml.safe_load(base_text)
    candidate = doc["subjects"]["candidate"]
    if fault != "none":
        candidate["fault"] = fault
    doc.setdefault("workload", {})["seeds"] = [seed]
    doc["name"] = f"{doc['name']}.{fault.replace('_', '-')}.s{seed}"
    return yaml.safe_dump(doc, sort_keys=False)


def _analyze(result: Any) -> TrialResult:
    """Turn a :class:`ivf.runner.RunResult` into a trial record."""
    validity_failed = result.validity is not None and not result.validity.valid
    failing = [o for o in result.outcomes if o.status in ("fail", "unsupported")]
    detected = bool(
        validity_failed
        or failing
        or result.verdict in (Verdict.INVALID_EXPERIMENT, Verdict.FAIL, Verdict.UNSUPPORTED,
                              Verdict.ERROR)
    )
    if validity_failed or result.verdict is Verdict.INVALID_EXPERIMENT:
        layer = "validity"
    elif failing:
        layer = "oracle"
    elif result.verdict is Verdict.ERROR:
        layer = "runtime"
    else:
        layer = "none"

    divergences = [o.divergence for o in failing if o.divergence is not None]
    first = min((d.first_tolerance_violation_step for d in divergences
                 if d.first_tolerance_violation_step is not None), default=None)
    classification = divergences[0].classification if divergences else None
    quantity = divergences[0].signal if divergences else None
    return TrialResult(
        fault="", seed=0, verdict=result.verdict.value, reason_codes=list(result.reason_codes),
        detected=detected, layer=layer,
        responsible_oracles=[o.name for o in failing], first_step=first,
        classification=classification, quantity=quantity,
    )


def run_calibration(
    *,
    manifest_path: str | Path,
    results_root: str | Path = "ivf-results",
    seeds: list[int] | None = None,
    faults: list[str] | None = None,
    keep_bundles: bool = False,
) -> DetectabilityMatrix:
    """Run the campaign and return the detectability matrix."""
    seeds = list(seeds or [11, 23, 47])
    base_text = Path(manifest_path).read_text(encoding="utf-8")
    base = parse_manifest(base_text, source_path=str(manifest_path))
    names = list(faults or TAXONOMY)

    workdir = Path(results_root) if keep_bundles else Path(tempfile.mkdtemp(prefix="ivf-calibration-"))
    matrix = DetectabilityMatrix(
        taxonomy_version=TAXONOMY_VERSION, manifest_name=base.name, seeds=seeds
    )
    try:
        for name in names:
            spec = TAXONOMY[name]
            row = FaultRow(spec=spec)
            for seed in seeds:
                text = _inject_into_manifest(base_text, name, seed)
                manifest = parse_manifest(text, source_path=str(manifest_path))
                result = validate(manifest, results_root=workdir, seed=seed,
                                  command=["ivf", "calibrate"])
                trial = _analyze(result)
                trial.fault, trial.seed = name, seed
                row.trials.append(trial)
            matrix.rows.append(row)
    finally:
        if not keep_bundles:
            unseal(workdir)
            shutil.rmtree(workdir, ignore_errors=True)
    return matrix


def render_matrix_markdown(matrix: DetectabilityMatrix) -> str:
    """Render the detectability matrix as markdown."""
    lines = [
        "# IVF fault detectability matrix",
        "",
        f"- taxonomy: `{matrix.taxonomy_version}`",
        f"- base manifest: `{matrix.manifest_name}`",
        f"- seeds: {matrix.seeds}",
        f"- trials: {sum(len(r.trials) for r in matrix.rows)} across {len(matrix.rows)} fault classes",
        f"- false positives on the negative control: **{matrix.false_positives}**",
        f"- every row matches its declaration: **{matrix.consistent}**",
        "",
        "> The taxonomy is author-generated. A fully consistent matrix establishes that IVF",
        "> detects the fault classes its authors enumerated, not that it detects fault classes",
        "> nobody thought of.",
        "",
        "| fault | category | expected | layer | detection rate | oracles | classification | "
        "first step | consistent |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in matrix.rows:
        rate = row.detection_rate
        lines.append(
            f"| `{row.spec.name}` | {row.spec.category} | "
            f"{'yes' if row.spec.expected_detectable else 'no'} | "
            f"{', '.join(row.layers) or '-'} | {rate:.0%} | "
            f"{', '.join(row.oracles) or '-'} | {', '.join(row.classifications) or '-'} | "
            f"{', '.join(str(s) for s in row.localized_steps) or '-'} | "
            f"{'yes' if row.consistent else '**NO**'} |"
        )
    lines.append("")
    lines.append("## Declared limitations")
    lines.append("")
    for row in matrix.rows:
        if row.spec.limitations:
            lines.append(f"- **{row.spec.name}** (min severity: {row.spec.minimum_severity}): "
                         f"{row.spec.limitations}")
    return "\n".join(lines) + "\n"
