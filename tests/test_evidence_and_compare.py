# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Evidence bundles: sealing, verification, tamper detection, and offline comparison."""

from __future__ import annotations

import json
import stat

import numpy as np
import pytest

from ivf.compare import compare_bundles, render_comparison_text
from ivf.evidence import CHECKSUM_FILE, SEAL_FILE, EvidenceBundle, EvidenceError, unseal
from ivf.manifest import parse_manifest
from ivf.runner import validate

from .conftest import MINIMAL_MANIFEST

pytestmark = pytest.mark.unit


@pytest.fixture
def sealed(tmp_path):
    """A finalized evidence bundle produced by the real runner."""
    manifest = parse_manifest(MINIMAL_MANIFEST)
    result = validate(manifest, results_root=tmp_path / "r", command=["ivf", "validate", "x"])
    return EvidenceBundle.open(result.bundle_path)


@pytest.mark.reproduction
def test_a_sealed_bundle_verifies(sealed):
    assert sealed.is_sealed
    assert sealed.verify() == []


@pytest.mark.reproduction
def test_every_expected_artifact_is_present(sealed):
    for name in ("manifest.original.yaml", "manifest.resolved.json", "verdict.json",
                 "validity.json", "oracles.json", "divergence.jsonl", "provenance.json",
                 "report.html", "reproduce.sh", CHECKSUM_FILE, SEAL_FILE,
                 "signals/baseline.npz", "signals/candidate.npz"):
        assert (sealed.root / name).is_file(), f"missing {name}"


@pytest.mark.reproduction
def test_sealed_files_are_read_only(sealed):
    mode = (sealed.root / "verdict.json").stat().st_mode
    assert not mode & stat.S_IWUSR


@pytest.mark.reproduction
def test_editing_a_sealed_file_is_detected(sealed, tmp_path):
    copy_root = tmp_path / "copy"
    copy_root.mkdir()
    target = copy_root / sealed.root.name
    import shutil

    shutil.copytree(sealed.root, target)
    unseal(target)
    payload = json.loads((target / "verdict.json").read_text())
    payload["verdict"] = "PASS"
    (target / "verdict.json").write_text(json.dumps(payload))
    problems = EvidenceBundle.open(target).verify()
    assert any("checksum mismatch" in p for p in problems)


@pytest.mark.reproduction
def test_an_added_file_is_detected(sealed, tmp_path):
    import shutil

    target = tmp_path / "added" / sealed.root.name
    target.parent.mkdir()
    shutil.copytree(sealed.root, target)
    unseal(target)
    (target / "extra.json").write_text("{}")
    problems = EvidenceBundle.open(target).verify()
    assert any("not recorded" in p for p in problems)


@pytest.mark.reproduction
def test_a_removed_file_is_detected(sealed, tmp_path):
    import shutil

    target = tmp_path / "removed" / sealed.root.name
    target.parent.mkdir()
    shutil.copytree(sealed.root, target)
    unseal(target)
    (target / "oracles.json").unlink()
    problems = EvidenceBundle.open(target).verify()
    assert any("missing from the bundle" in p for p in problems)


def test_finalizing_twice_is_refused(sealed):
    with pytest.raises(EvidenceError, match="already finalized"):
        sealed.finalize()


def test_creating_over_an_existing_bundle_is_refused(sealed):
    with pytest.raises(EvidenceError, match="refusing to overwrite"):
        EvidenceBundle.create(sealed.root.parent, sealed.run_id)


def test_an_incompatible_evidence_schema_is_refused_not_half_parsed(sealed, tmp_path):
    import shutil

    target = tmp_path / "future" / sealed.root.name
    target.parent.mkdir()
    shutil.copytree(sealed.root, target)
    unseal(target)
    seal = json.loads((target / SEAL_FILE).read_text())
    seal["evidence_schema_version"] = "ivf.evidence/v99"
    (target / SEAL_FILE).write_text(json.dumps(seal))
    with pytest.raises(EvidenceError, match="not compatible"):
        EvidenceBundle.open(target)


def test_a_bundle_is_readable_without_ivf(sealed):
    """A reviewer with json and numpy must be able to check the numbers by hand."""
    verdict = json.loads((sealed.root / "verdict.json").read_text())
    assert verdict["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE", "UNSUPPORTED",
                                 "INVALID_EXPERIMENT", "ERROR")
    with np.load(sealed.root / "signals" / "baseline.npz") as payload:
        assert payload.files


def test_public_provenance_redacts_identity_paths_and_pythonpath(sealed):
    text = (sealed.root / "provenance.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "/home/" not in text
    assert payload["host"]["hostname"] == "redacted"
    assert payload["host"]["user"] == "redacted"
    assert payload["python"]["executable"] == "python"
    if "PYTHONPATH" in payload["environment_variables"]:
        assert payload["environment_variables"]["PYTHONPATH"] == "set (value redacted)"


def test_generated_reproduction_command_uses_global_options_before_the_subcommand(sealed):
    script = (sealed.root / "reproduce.sh").read_text(encoding="utf-8")
    assert "ivf --results-root ivf-results validate" in script
    assert "repository_root=" in script
    assert "ivf validate" not in script
    assert "/home/" not in script


# -- comparison ----------------------------------------------------------------------------

def _run(tmp_path, text, name):
    manifest = parse_manifest(text)
    return validate(manifest, results_root=tmp_path / name).bundle_path


def test_identical_reruns_compare_as_unchanged(tmp_path):
    a = _run(tmp_path, MINIMAL_MANIFEST, "a")
    b = _run(tmp_path, MINIMAL_MANIFEST, "b")
    report = compare_bundles(a, b)
    assert report.comparable
    assert not report.verdict_change["changed"]
    assert not report.changed
    assert "No material differences." in render_comparison_text(report)


def test_comparison_refuses_two_different_experiments(tmp_path):
    a = _run(tmp_path, MINIMAL_MANIFEST, "a")
    b = _run(tmp_path, MINIMAL_MANIFEST.replace("name: minimal", "name: other"), "b")
    report = compare_bundles(a, b)
    assert not report.comparable
    assert any("different experiments" in r for r in report.incomparability_reasons)


def test_comparison_reports_a_changed_acceptance_criterion_as_such(tmp_path):
    """A verdict difference across two different criteria is not evidence about the subjects."""
    with_oracle = MINIMAL_MANIFEST + """  - type: trajectory_equivalence
    name: pole_angle_agreement
    signal: pole_angle
    tolerance:
      value: {value}
      unit: rad
      scope: per-step absolute error
      rationale: a deliberately different budget between the two runs
      aggregation: max
      min_samples: 10
      kind: engineering
"""
    a = _run(tmp_path, with_oracle.format(value="1.0e-3"), "a")
    b = _run(tmp_path, with_oracle.format(value="2.0e-3"), "b")
    report = compare_bundles(a, b)
    changes = [c for c in report.oracle_changes if c["change"] == "tolerance"]
    assert changes
    assert "not evidence about the subjects" in changes[0]["note"]


def test_comparison_flags_a_tampered_input(tmp_path):
    a = _run(tmp_path, MINIMAL_MANIFEST, "a")
    b = _run(tmp_path, MINIMAL_MANIFEST, "b")
    unseal(b)
    (b / "verdict.json").write_text("{}")
    report = compare_bundles(a, b)
    assert not report.comparable
    assert any("integrity" in r for r in report.incomparability_reasons)
