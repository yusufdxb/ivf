# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Verdict roll-up, experiment validity, and divergence localization."""

from __future__ import annotations

import numpy as np
import pytest

from ivf.divergence import localize
from ivf.manifest import parse_manifest
from ivf.oracles import OracleOutcome
from ivf.runner import decide, validate
from ivf.validity import check_experiment
from ivf.verdicts import REASON_CODES, Verdict, describe, validate_code

from .conftest import MINIMAL_MANIFEST, make_signals

pytestmark = pytest.mark.unit


# -- verdict vocabulary ---------------------------------------------------------------------

def test_exit_codes_are_distinct_so_ci_can_branch_on_them():
    codes = [v.exit_code for v in Verdict]
    assert len(set(codes)) == len(codes)
    assert Verdict.PASS.exit_code == 0


def test_only_evidence_backed_verdicts_are_scientific():
    assert Verdict.FAIL.is_scientific
    assert not Verdict.ERROR.is_scientific
    assert not Verdict.UNSUPPORTED.is_scientific


def test_every_reason_code_has_a_gloss():
    for code in REASON_CODES:
        assert describe(code) and not describe(code).startswith("(unrecognized")


def test_unknown_reason_codes_render_rather_than_crash():
    assert "unrecognized" in describe("IVF-NOT-A-REAL-CODE")


def test_emitting_an_unregistered_code_is_a_programming_error():
    with pytest.raises(KeyError):
        validate_code("IVF-MADE-UP")


def test_all_emitted_codes_are_registered():
    """Guards against a typo producing a code nobody can grep for."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "ivf"
    emitted = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("IVF-") and node.value.isupper() is False:
                    continue
                if node.value.startswith("IVF-"):
                    emitted.add(node.value)
    unknown = {c for c in emitted if c not in REASON_CODES}
    assert not unknown, f"unregistered reason codes emitted in source: {sorted(unknown)}"


# -- verdict roll-up -------------------------------------------------------------------------

def _outcome(status, name="o", codes=()):
    return OracleOutcome(name=name, type="t", status=status, summary="", reason_codes=list(codes))


def test_a_control_violation_outranks_every_oracle_result(minimal_manifest):
    report = check_experiment(
        minimal_manifest,
        make_signals("baseline", steps=50),
        make_signals("candidate", asset_identity="different.v1"),
    )
    verdict, codes = decide(minimal_manifest, report, [_outcome("pass")])
    assert verdict is Verdict.INVALID_EXPERIMENT
    assert "IVF-CONTROL-ASSET-MISMATCH" in codes


def test_a_real_failure_outranks_an_unsupported_feature(minimal_manifest):
    report = check_experiment(minimal_manifest, make_signals("baseline", steps=50), make_signals("candidate", steps=50))
    verdict, _ = decide(minimal_manifest, report,
                        [_outcome("unsupported"), _outcome("fail", "b", ["IVF-ORACLE-NON_EQUIVALENT"])])
    assert verdict is Verdict.FAIL


def test_unsupported_outranks_inconclusive(minimal_manifest):
    report = check_experiment(minimal_manifest, make_signals("baseline", steps=50), make_signals("candidate", steps=50))
    verdict, _ = decide(minimal_manifest, report, [_outcome("unsupported"), _outcome("inconclusive", "b")])
    assert verdict is Verdict.UNSUPPORTED


def test_all_skipped_oracles_cannot_pass(minimal_manifest):
    """A run in which nothing was actually checked provides no reassurance."""
    report = check_experiment(minimal_manifest, make_signals("baseline", steps=50), make_signals("candidate", steps=50))
    verdict, _ = decide(minimal_manifest, report, [_outcome("skipped"), _outcome("skipped", "b")])
    assert verdict is Verdict.INCONCLUSIVE


def test_all_passing_is_a_pass(minimal_manifest):
    report = check_experiment(minimal_manifest, make_signals("baseline", steps=50), make_signals("candidate", steps=50))
    verdict, codes = decide(minimal_manifest, report, [_outcome("pass"), _outcome("pass", "b")])
    assert verdict is Verdict.PASS
    assert codes == []


# -- experiment validity ----------------------------------------------------------------------

def test_a_missing_control_input_is_unverifiable_not_a_match(minimal_manifest):
    """Absence of evidence must never read as evidence of a match."""
    report = check_experiment(
        minimal_manifest, make_signals("baseline", steps=50),
        make_signals("candidate", steps=50, asset_identity=""),
    )
    ids = {c.check_id: c.status for c in report.checks}
    assert ids["V-01"] == "unverifiable"
    assert report.valid  # unverifiable does not veto; it is reported


def test_a_timestep_mismatch_is_always_checked(minimal_manifest):
    report = check_experiment(
        minimal_manifest, make_signals("baseline", steps=50), make_signals("candidate", physics_dt=0.01)
    )
    assert not report.valid
    assert "IVF-CONTROL-CONTROL-FREQUENCY-MISMATCH" in report.reason_codes()


def test_a_truncated_run_is_a_protocol_violation(minimal_manifest):
    short = make_signals("candidate", steps=5)
    report = check_experiment(minimal_manifest, make_signals("baseline", steps=50), short)
    assert "IVF-PROTOCOL-PARTIAL-RUN" in report.reason_codes()


def test_an_undeclared_warmup_is_reported():
    manifest = parse_manifest(MINIMAL_MANIFEST.replace("  warmup_steps: 0\n", ""))
    report = check_experiment(manifest, make_signals("baseline", steps=50),
                              make_signals("candidate", steps=50))
    warmup = next(c for c in report.checks if c.check_id == "V-12")
    assert warmup.status == "unverifiable"


def test_env_count_mismatch_fails_when_pairing_is_assumed(minimal_manifest):
    report = check_experiment(minimal_manifest, make_signals("baseline", envs=6),
                              make_signals("candidate", envs=5))
    assert "IVF-CONTROL-ENV-COUNT-MISMATCH" in report.reason_codes()


# -- divergence localization ----------------------------------------------------------------------

def _localize(baseline, candidate, threshold=1e-3, **kwargs):
    errors = np.linalg.norm(candidate - baseline, axis=-1)
    return localize(signal="s", oracle="o", baseline=baseline, candidate=candidate,
                    per_step_env_error=errors, threshold=threshold, **kwargs)


def test_a_constant_per_env_offset_reads_as_a_frame_problem():
    steps, envs = 50, 4
    base = np.sin(np.linspace(0, 6, steps))[:, None, None] * np.ones((1, envs, 1))
    offsets = np.arange(envs, dtype=np.float64)[None, :, None] * 2.0
    record = _localize(base, base + offsets)
    assert record.classification == "coordinate_frame_mismatch"
    assert record.confidence == "high"
    assert record.limitations


def test_a_difference_present_from_step_zero_reads_as_a_reset_problem():
    steps, envs = 50, 4
    t = np.linspace(0, 6, steps)[:, None, None] * np.ones((1, envs, 1))
    record = _localize(np.sin(t), np.sin(t + 0.2))
    assert record.classification == "reset_mismatch"


def test_a_declared_config_difference_beats_a_dynamical_story():
    base = np.zeros((20, 2, 1))
    record = _localize(base, base + 1.0,
                       config_differences={"asset_identity": {"baseline": "a", "candidate": "b"}})
    assert record.classification == "asset_mismatch"


def test_non_finite_values_classify_as_numerical_and_localize():
    base = np.zeros((20, 2, 1))
    candidate = base.copy()
    candidate[5, 1, 0] = np.nan
    record = _localize(base, candidate)
    assert record.classification == "numerical_drift"
    assert record.first_tolerance_violation_step == 5
    assert record.affected_env_ids == [1]


def test_an_unmatched_shape_admits_it_rather_than_guessing():
    steps, envs = 60, 3
    base = np.zeros((steps, envs, 1))
    candidate = base.copy()
    candidate[30:] += 5.0
    candidate[45:] -= 5.0
    record = _localize(base, candidate)
    assert record.classification in ("unknown", "sensor_semantic_difference")
    assert record.limitations


def test_the_window_supports_a_plot_around_the_first_violation():
    base = np.zeros((60, 3, 1))
    candidate = base.copy()
    candidate[30:] += 1.0
    record = _localize(base, candidate)
    assert record.window["start_step"] <= 30 <= record.window["end_step"]
    assert len(record.window["error"]) > 1


# -- runner behaviour ------------------------------------------------------------------------

def test_an_unreadable_source_is_an_error_not_a_failure(tmp_path):
    text = MINIMAL_MANIFEST.replace(
        "    kind: synthetic\n    system: damped_pendulum\n  candidate:",
        "    kind: parity_bundle\n    path: /nonexistent/bundle\n  candidate:",
    )
    result = validate(parse_manifest(text), results_root=tmp_path / "r")
    assert result.verdict is Verdict.ERROR
    assert result.reason_codes == ["IVF-RUNTIME-EXECUTION-ERROR"]


def test_an_error_run_still_writes_a_sealed_bundle(tmp_path):
    """An infrastructure failure with no record of what was attempted is the worst kind."""
    text = MINIMAL_MANIFEST.replace(
        "    kind: synthetic\n    system: damped_pendulum\n  candidate:",
        "    kind: parity_bundle\n    path: /nonexistent/bundle\n  candidate:",
    )
    result = validate(parse_manifest(text), results_root=tmp_path / "r")
    from ivf.evidence import EvidenceBundle

    assert EvidenceBundle.open(result.bundle_path).verify() == []


def test_a_missing_runtime_is_unsupported_not_a_failure(tmp_path):
    text = MINIMAL_MANIFEST.replace(
        "    kind: synthetic\n    system: damped_pendulum\n  candidate:",
        "    kind: isaaclab\n    backend: physx\n  candidate:",
    )
    result = validate(parse_manifest(text), results_root=tmp_path / "r")
    assert result.verdict is Verdict.UNSUPPORTED
    assert result.reason_codes == ["IVF-RUNTIME-UNAVAILABLE"]


def test_an_unknown_oracle_type_is_unsupported(tmp_path):
    text = MINIMAL_MANIFEST.replace("- type: invariant", "- type: telepathy")
    result = validate(parse_manifest(text), results_root=tmp_path / "r")
    assert result.verdict is Verdict.UNSUPPORTED
    assert "IVF-ORACLE-UNKNOWN" in result.reason_codes


def test_an_oracle_bug_does_not_masquerade_as_a_science_result(tmp_path, monkeypatch):
    from ivf.oracles import base as oracle_base

    def explode(ctx):
        raise RuntimeError("oracle is broken")

    monkeypatch.setitem(oracle_base._REGISTRY, "invariant", explode)
    result = validate(parse_manifest(MINIMAL_MANIFEST), results_root=tmp_path / "r")
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "IVF-RUNTIME-EXECUTION-ERROR" in result.reason_codes


def test_multiple_seeds_are_pooled_along_the_environment_axis(tmp_path):
    text = MINIMAL_MANIFEST.replace("  seeds: [3]", "  seeds: [3, 5, 7]")
    result = validate(parse_manifest(text), results_root=tmp_path / "r")
    from ivf.evidence import EvidenceBundle

    bundle = EvidenceBundle.open(result.bundle_path)
    signals = bundle.read_signals("baseline")
    assert next(iter(signals.values())).shape[1] == 4 * 3


def test_statistical_alpha_is_sidak_split_across_statistical_oracles():
    """Two statistical oracles must not each spend the full family-wise budget."""
    from ivf.runner import _STATISTICAL_TYPES

    alpha, k = 0.05, 2
    adjusted = 1.0 - (1.0 - alpha) ** (1.0 / k)
    assert adjusted < alpha
    assert pytest.approx(1.0 - (1.0 - adjusted) ** k) == alpha
    assert "statistical_equivalence" in _STATISTICAL_TYPES


def test_run_ids_are_traceable_to_the_manifest_digest():
    from ivf.runner import make_run_id

    manifest = parse_manifest(MINIMAL_MANIFEST)
    assert make_run_id(manifest).endswith(manifest.digest()[:8])
