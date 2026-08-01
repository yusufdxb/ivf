# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Orchestration for ``ivf validate``: manifest in, sealed evidence bundle out.

The runner is the only place that decides a verdict. Oracles report; validity vetoes;
the runner rolls up under the manifest's declared policy. Precedence is fixed and
documented because a verdict model nobody can predict is a verdict model nobody trusts:

``INVALID_EXPERIMENT`` > ``FAIL`` > ``UNSUPPORTED`` > ``INCONCLUSIVE`` > ``PASS``

with ``ERROR`` sitting outside the ladder entirely, reserved for infrastructure that
broke before evidence existed. ``FAIL`` outranks ``UNSUPPORTED`` deliberately: if one
oracle could not run because a feature is missing but another genuinely failed on real
evidence, the real failure is the actionable result.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .bundle import BundleContractError
from .evidence import EVIDENCE_SCHEMA_VERSION, EvidenceBundle
from .manifest import Manifest
from .oracles import OracleContext, OracleOutcome, get_oracle
from .report import render_html
from .signals import RuntimeUnavailable, SignalSet, SignalSourceError, load_subject
from .validity import ValidityReport, check_experiment
from .verdicts import Verdict

_STATISTICAL_TYPES = frozenset({"statistical_equivalence"})


@dataclass
class RunResult:
    """The outcome of one ``ivf validate`` invocation."""

    verdict: Verdict
    reason_codes: list[str]
    run_id: str
    bundle_path: Path
    outcomes: list[OracleOutcome] = field(default_factory=list)
    validity: ValidityReport | None = None
    error: str | None = None

    @property
    def exit_code(self) -> int:
        """CI exit status for this run."""
        return self.verdict.exit_code


def make_run_id(manifest: Manifest, *, now: datetime | None = None) -> str:
    """Return a run identifier that is unique, sortable and traceable to the manifest.

    Shape: ``<name>-<UTC timestamp>-<first 8 of the manifest digest>``. The digest makes
    it obvious at a glance when two runs came from the same declared experiment.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{manifest.name}-{stamp}-{manifest.digest()[:8]}"


def _load_subject_multiseed(manifest: Manifest, role: str) -> SignalSet:
    """Load one subject, concatenating multiple seeds along the environment axis.

    Multiple seeds are the only defensible way to make a paired statistical oracle mean
    anything: a single seed gives one sample of the initial-condition distribution, and
    an interval computed from it describes that one draw. Concatenating along the
    environment axis keeps the pairing intact, because both subjects walk the same seed
    list in the same order.

    Sources that carry their own seed (a recorded trajectory bundle) are loaded once and
    their recorded seed is authoritative; the manifest's seed list does not apply.
    """
    subject = manifest.subjects[role]
    workload = manifest.workload
    if subject.kind != "synthetic":
        return load_subject(subject, steps=workload.steps, num_envs=workload.num_envs,
                            seed=workload.seeds[0])

    per_seed = [
        load_subject(subject, steps=workload.steps, num_envs=workload.num_envs, seed=seed)
        for seed in workload.seeds
    ]
    if len(per_seed) == 1:
        return per_seed[0]

    names = sorted(per_seed[0].signals)
    for other in per_seed[1:]:
        if sorted(other.signals) != names:
            raise SignalSourceError(
                f"subjects.{role}: seeds produced different signal sets; cannot pool"
            )
    merged = SignalSet(
        role=role,
        signals={n: np.concatenate([s.signals[n] for s in per_seed], axis=1) for n in names},
        metadata=dict(per_seed[0].metadata),
        actions=per_seed[0].actions,
        complete=all(s.complete for s in per_seed),
    )
    merged.metadata["seed"] = list(workload.seeds)
    merged.metadata["seed_pooling"] = (
        f"{len(per_seed)} seeds concatenated along the environment axis "
        f"({workload.num_envs} envs each)"
    )
    return merged


def _run_oracles(manifest: Manifest, baseline: SignalSet, candidate: SignalSet,
                 *, alpha: float, seed: int) -> list[OracleOutcome]:
    """Execute every declared oracle, converting oracle bugs into honest outcomes."""
    n_statistical = sum(1 for o in manifest.oracles if o.type in _STATISTICAL_TYPES)
    # Sidak split of the family-wise false-alarm budget across statistical oracles: with
    # k independent tests at alpha_adj, the family-wise rate is 1-(1-alpha_adj)^k = alpha.
    alpha_adj = 1.0 - (1.0 - alpha) ** (1.0 / n_statistical) if n_statistical > 1 else alpha

    outcomes: list[OracleOutcome] = []
    for spec in manifest.oracles:
        ctx = OracleContext(
            manifest=manifest, spec=spec, baseline=baseline, candidate=candidate,
            warmup_steps=manifest.workload.warmup_steps or 0,
            alpha=alpha_adj if spec.type in _STATISTICAL_TYPES else alpha,
            seed=seed,
        )
        try:
            fn = get_oracle(spec.type)
        except KeyError as exc:
            outcomes.append(OracleOutcome(
                name=spec.name, type=spec.type, status="unsupported",
                summary=str(exc), reason_codes=["IVF-ORACLE-UNKNOWN"],
            ))
            continue
        try:
            outcomes.append(fn(ctx))
        except Exception as exc:  # an oracle bug must not masquerade as a science result
            outcomes.append(OracleOutcome(
                name=spec.name, type=spec.type, status="inconclusive",
                summary=f"oracle raised {type(exc).__name__}: {exc}",
                reason_codes=["IVF-RUNTIME-EXECUTION-ERROR"],
                metrics={"traceback": traceback.format_exc(limit=6)},
                limitations="This is an IVF defect, not evidence about the subjects.",
            ))
    return outcomes


def decide(manifest: Manifest, validity: ValidityReport,
           outcomes: list[OracleOutcome]) -> tuple[Verdict, list[str]]:
    """Roll validity and oracle outcomes up into one verdict plus its reason codes."""
    policy = manifest.verdict_policy
    codes: list[str] = []

    if not validity.valid and policy.invalid_experiment_on_control_violation:
        return Verdict.INVALID_EXPERIMENT, validity.reason_codes()
    if not validity.valid:
        codes.extend(validity.reason_codes())

    statuses = [o.status for o in outcomes]
    for outcome in outcomes:
        if outcome.status in ("fail", "unsupported", "inconclusive"):
            codes.extend(outcome.reason_codes)
    codes = list(dict.fromkeys(codes))  # de-duplicate, preserve order

    if "fail" in statuses:
        return Verdict.FAIL, codes
    if "unsupported" in statuses:
        if policy.unsupported_is_failure:
            return Verdict.FAIL, codes
        return Verdict.UNSUPPORTED, codes
    if "inconclusive" in statuses:
        return Verdict.INCONCLUSIVE, codes
    if statuses and all(s == "skipped" for s in statuses):
        return Verdict.INCONCLUSIVE, [*codes, "IVF-SAMPLE-INSUFFICIENT"]
    return Verdict.PASS, codes


def _provenance(manifest: Manifest, command: list[str] | None) -> dict[str, Any]:
    """Collect audit provenance without publishing machine-local identity or paths."""
    from .env import run_doctor

    doctor = run_doctor()
    tracked_names = (
        "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
        "OMNI_KIT_ACCEPT_EULA", "IVF_RESULTS_ROOT", "PYTHONPATH",
    )
    tracked_env = {}
    for name in tracked_names:
        if name not in os.environ:
            continue
        if name == "PYTHONPATH":
            tracked_env[name] = "set (value redacted)"
        else:
            tracked_env[name] = _portable_text(os.environ[name])
    return {
        "ivf_version": __version__,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "manifest_schema_version": manifest.schema_version,
        "manifest_digest_sha256": manifest.digest(),
        "manifest_source_path": _portable_text(manifest.source_path or "<in-memory>"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [_portable_text(str(item)) for item in (command or sys.argv)],
        "cwd": "$REPOSITORY_ROOT",
        "python": {"version": platform.python_version(), "executable": "python"},
        "host": {
            "hostname": "redacted",
            "user": "redacted",
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        },
        "environment_variables": tracked_env,
        "doctor": _portable_value(doctor.to_jsonable()),
    }


def _portable_text(value: str) -> str:
    """Replace the repository and home prefixes in a provenance string."""
    cwd = os.getcwd().rstrip("/")
    home = str(Path.home()).rstrip("/")
    return value.replace(cwd, "$REPOSITORY_ROOT").replace(home, "$HOME")


def _portable_value(value: Any) -> Any:
    """Recursively remove machine-local absolute prefixes from doctor output."""
    if isinstance(value, dict):
        return {str(key): _portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, str):
        return _portable_text(value)
    return value


def _reproduce_script(manifest: Manifest, run_id: str, results_root: Path) -> str:
    """Return the contents of ``reproduce.sh``."""
    manifest_path = manifest.source_path or "<manifest not loaded from a file>"
    try:
        relative_manifest = Path(manifest_path).resolve().relative_to(Path.cwd().resolve()).as_posix()
        script_manifest = f'"${{repository_root}}/{relative_manifest}"'
    except (OSError, ValueError):
        script_manifest = _shell_quote(_portable_text(manifest_path))
    return f"""#!/usr/bin/env bash
# Reproduce IVF run {run_id}
#
# This script re-executes the manifest that produced this evidence bundle. It does not
# guarantee identical output on a different machine: identical output is a property of
# the workload, not of this script. `ivf reproduce` compares the two and reports what
# changed.
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
ivf --results-root ivf-results validate {script_manifest}
"""


def _shell_quote(value: str) -> str:
    """Quote a value for inclusion in a shell command."""
    if all(c.isalnum() or c in "._-/=" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def validate(
    manifest: Manifest,
    *,
    results_root: str | Path = "ivf-results",
    alpha: float = 0.05,
    seed: int = 0,
    command: list[str] | None = None,
    run_id: str | None = None,
) -> RunResult:
    """Execute a manifest end to end and write a sealed evidence bundle.

    Always writes a bundle, including on ``ERROR``: an infrastructure failure with no
    record of what was attempted is the hardest kind to debug.
    """
    run_id = run_id or make_run_id(manifest)
    root = Path(results_root)
    # Two runs of the same manifest inside one second would otherwise collide. Suffixing
    # rather than overwriting is deliberate: an evidence bundle is never silently replaced.
    base_id, attempt = run_id, 1
    while (root / run_id).exists():
        attempt += 1
        run_id = f"{base_id}-{attempt:02d}"
    bundle = EvidenceBundle.create(root, run_id)
    started = time.time()

    verdict = Verdict.ERROR
    codes: list[str] = []
    outcomes: list[OracleOutcome] = []
    validity: ValidityReport | None = None
    error: str | None = None
    baseline: SignalSet | None = None
    candidate: SignalSet | None = None

    try:
        baseline = _load_subject_multiseed(manifest, "baseline")
        candidate = _load_subject_multiseed(manifest, "candidate")
        validity = check_experiment(manifest, baseline, candidate)
        outcomes = _run_oracles(manifest, baseline, candidate, alpha=alpha, seed=seed)
        verdict, codes = decide(manifest, validity, outcomes)
    except BundleContractError as exc:
        # A capture that violates the boundary is uninterpretable, not broken
        # infrastructure, and it must never reach an oracle. INVALID_EXPERIMENT is the
        # honest verdict: we cannot judge the subject, and we know exactly why.
        verdict, codes, error = Verdict.INVALID_EXPERIMENT, [exc.reason_code], str(exc)
    except RuntimeUnavailable as exc:
        verdict, codes, error = Verdict.UNSUPPORTED, ["IVF-RUNTIME-UNAVAILABLE"], str(exc)
    except SignalSourceError as exc:
        verdict, codes, error = Verdict.ERROR, ["IVF-RUNTIME-EXECUTION-ERROR"], str(exc)
    except Exception as exc:
        verdict, codes = Verdict.ERROR, ["IVF-RUNTIME-EXECUTION-ERROR"]
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=10)}"

    elapsed = time.time() - started
    verdict_payload = {
        "verdict": verdict.value,
        "exit_code": verdict.exit_code,
        "reason_codes": codes,
        "run_id": run_id,
        "experiment": manifest.name,
        "description": manifest.description,
        "manifest_digest_sha256": manifest.digest(),
        "ivf_version": __version__,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": round(elapsed, 4),
        "alpha": alpha,
        "seed": seed,
        "error": error,
        "counts": {
            status: sum(1 for o in outcomes if o.status == status)
            for status in ("pass", "fail", "inconclusive", "unsupported", "skipped")
        },
    }

    try:
        bundle.write_text("manifest.original.yaml", manifest.source_text or "")
        bundle.write_json("manifest.resolved.json", manifest.to_jsonable())
        bundle.write_json("verdict.json", verdict_payload)
        bundle.write_json("validity.json", validity.to_jsonable() if validity else {"valid": None})
        bundle.write_json("oracles.json", [o.to_jsonable() for o in outcomes])
        bundle.write_jsonl(
            "divergence.jsonl",
            [o.divergence.to_jsonable() for o in outcomes if o.divergence is not None],
        )
        bundle.write_json("provenance.json", _provenance(manifest, command))
        for role, sset in (("baseline", baseline), ("candidate", candidate)):
            if sset is not None:
                bundle.write_signals(role, sset.signals)
                bundle.write_json(f"signals/{role}.metadata.json", _jsonable(sset.metadata))
        bundle.write_text("reproduce.sh", _reproduce_script(manifest, run_id, root))
        os.chmod(bundle.root / "reproduce.sh", 0o755)
        bundle.write_text(
            "report.html",
            render_html(
                verdict=verdict_payload,
                manifest=manifest.to_jsonable(),
                validity=validity.to_jsonable() if validity else {"valid": None, "checks": []},
                outcomes=[o.to_jsonable() for o in outcomes],
                provenance=json.loads((bundle.root / "provenance.json").read_text(encoding="utf-8")),
                run_id=run_id,
            ),
        )
        bundle.finalize()
    except Exception as exc:  # writing evidence is the last thing that may fail
        return RunResult(
            verdict=Verdict.ERROR, reason_codes=["IVF-EVIDENCE-WRITE-ERROR"], run_id=run_id,
            bundle_path=bundle.root, outcomes=outcomes, validity=validity,
            error=f"{type(exc).__name__}: {exc}",
        )

    return RunResult(
        verdict=verdict, reason_codes=codes, run_id=run_id, bundle_path=bundle.root,
        outcomes=outcomes, validity=validity, error=error,
    )


def _jsonable(value: Any) -> Any:
    """Coerce metadata into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
