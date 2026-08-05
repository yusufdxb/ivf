# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Offline guards for the narrow simulator capture specification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ADAPTER_SRC = Path(__file__).resolve().parents[1] / "adapters" / "parity_capture" / "src"
sys.path.insert(0, str(ADAPTER_SRC))

from parity_capture.bundle_writer import finalize_v1  # noqa: E402
from parity_capture.capture import _module_git_revision  # noqa: E402
from parity_capture.cli import _update_ivf_manifest  # noqa: E402
from parity_capture.spec import SpecError, parse_spec  # noqa: E402

pytestmark = pytest.mark.unit

BASE = """schema_version: parity.capture/v1
name: guard
task: cartpole_passive
backend: physx
num_envs: 2
steps: 4
seed: 0
"""


def test_reset_perturbation_is_inactive_when_omitted_or_explicitly_false():
    assert parse_spec(BASE).defect.drop_reset_velocity is False
    assert parse_spec(BASE + "defect:\n  drop_reset_velocity: false\n").defect.drop_reset_velocity is False


def test_reset_perturbation_requires_explicit_boolean_true():
    assert parse_spec(BASE + "defect:\n  drop_reset_velocity: true\n").defect.drop_reset_velocity is True
    with pytest.raises(SpecError, match="expected a YAML boolean"):
        parse_spec(BASE + 'defect:\n  drop_reset_velocity: "false"\n')


def test_backend_and_test_switch_do_not_change_task_identity():
    physx = parse_spec(BASE)
    newton = parse_spec(BASE.replace("backend: physx", "backend: newton"))
    perturbed = parse_spec(BASE + "defect:\n  drop_reset_velocity: true\n")
    assert physx.config_identity() == newton.config_identity() == perturbed.config_identity()


def test_ivf_manifest_wiring_does_not_change_task_identity():
    wired = parse_spec(
        BASE
        + "ivf_manifest:\n"
        + "  role: baseline\n"
        + "  template: template.yaml\n"
        + "  filename: generated.yaml\n"
    )
    assert wired.config_identity() == parse_spec(BASE).config_identity()


@pytest.mark.parametrize("block", [
    "ivf_manifest:\n  role: reference\n  template: template.yaml\n",
    "ivf_manifest:\n  role: baseline\n",
    "ivf_manifest:\n  role: baseline\n  template: template.yaml\n  filename: ../escape.yaml\n",
])
def test_ivf_manifest_wiring_is_strict(block):
    with pytest.raises(SpecError, match="ivf_manifest"):
        parse_spec(BASE + block)


def test_two_captures_generate_one_hash_locked_ivf_manifest(tmp_path):
    template = tmp_path / "template.yaml"
    template.write_text(
        "schema_version: ivf.validation/v1\n"
        "name: generated\n"
        "description: fixture\n"
        "subjects:\n"
        "  baseline: {kind: parity_bundle, label: baseline}\n"
        "  candidate: {kind: parity_bundle, label: candidate}\n"
        "workload: {task: fixture, steps: 4}\n"
        "controls: {require_same: [horizon]}\n"
        "oracles:\n"
        "  - {type: invariant, name: finite, check: finite_state}\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "COMPLETE").write_text(
        '{"checksums_sha256": "' + "a" * 64 + '"}', encoding="utf-8"
    )
    (candidate / "COMPLETE").write_text(
        '{"checksums_sha256": "' + "b" * 64 + '"}', encoding="utf-8"
    )
    common = (
        "ivf_manifest:\n"
        f"  template: {template}\n"
        "  filename: generated.yaml\n"
    )
    baseline_spec = parse_spec(BASE + common + "  role: baseline\n")
    candidate_spec = parse_spec(BASE + common + "  role: candidate\n")

    output, pending = _update_ivf_manifest(baseline_spec, baseline)
    assert pending == ["candidate"]
    assert output == tmp_path / "generated.yaml"
    output, pending = _update_ivf_manifest(candidate_spec, candidate)
    assert pending == []

    import yaml

    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert generated["subjects"]["baseline"]["bundle_sha256"] == "a" * 64
    assert generated["subjects"]["candidate"]["bundle_sha256"] == "b" * 64


def test_source_revision_probe_resolves_the_actual_ivf_checkout():
    import subprocess

    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    revision = _module_git_revision("ivf", "ivf")
    assert revision["status"] == "resolved"
    assert revision["commit"] == expected


def test_adapter_has_no_ivf_runtime_dependency(repo_root):
    # tomllib is stdlib from 3.11, and this project supports 3.10. The assertion is a
    # static repository invariant rather than a property of the run, so checking it on the
    # interpreters that can read TOML without a third-party parser is sufficient; adding a
    # dependency to the test environment to re-check it on 3.10 would buy nothing.
    tomllib = pytest.importorskip(
        "tomllib", reason="stdlib TOML parser is 3.11+; covered by the 3.11 and 3.12 jobs"
    )

    project = tomllib.loads(
        (repo_root / "adapters" / "parity_capture" / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert all(not requirement.lower().startswith("ivf") for requirement in project["dependencies"])


def test_adapter_finalizer_writes_the_versioned_file_boundary(tmp_path):
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "trajectories.npz").write_bytes(b"payload")
    checksums = finalize_v1(tmp_path, run_status="completed")

    import json

    marker = json.loads((tmp_path / "COMPLETE").read_text(encoding="utf-8"))
    assert marker["schema"] == "trajectory_bundle/v1"
    assert marker["run_status"] == "completed"
    assert set(checksums) == {"metadata.json", "trajectories.npz"}
