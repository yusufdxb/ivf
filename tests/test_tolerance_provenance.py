# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Machine checks for every tolerance shipped in a public exemplar."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ivf.tolerance_provenance import derive_value, load_registry, render_markdown

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "validation" / "provenance" / "tolerances.yaml"
DOC_PATH = ROOT / "docs" / "engineering" / "tolerance-provenance.md"
REAL_MANIFESTS = {
    "cartpole_benign_difference.yaml",
    "cartpole_corrected.yaml",
    "cartpole_reset_defect.yaml",
}


def _records():
    return load_registry(REGISTRY_PATH)


def test_every_record_derives_its_declared_value_from_named_inputs():
    for record in _records():
        assert derive_value(record) == pytest.approx(float(record["value"]), rel=1e-12, abs=1e-15)


def test_every_manifest_tolerance_has_exactly_one_provenance_record():
    expected = {
        item
        for record in _records()
        for item in record.get("applies_to", [])
    }
    assert len(expected) == sum(len(record.get("applies_to", [])) for record in _records())

    actual = set()
    for path in sorted((ROOT / "validation" / "examples").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for oracle in payload.get("oracles", []):
            if "tolerance" in oracle:
                actual.add(f"validation/examples/{path.name}:{oracle['name']}")
    assert actual == expected


def test_manifest_values_and_units_match_the_registry():
    manifests = {
        path.relative_to(ROOT).as_posix(): yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (ROOT / "validation" / "examples").glob("*.yaml")
    }
    for record in _records():
        for application in record["applies_to"]:
            manifest_path, oracle_name = application.split(":", 1)
            oracle = next(
                item for item in manifests[manifest_path]["oracles"] if item["name"] == oracle_name
            )
            tolerance = oracle["tolerance"]
            assert float(tolerance["value"]) == pytest.approx(float(record["value"]), rel=1e-12)
            assert tolerance["unit"] == record["unit"]
            assert tolerance["aggregation"] == record["aggregation"]
            assert int(tolerance["min_samples"]) == int(record["minimum_samples"])


def test_cartpole_measurements_reproduce_the_angle_and_velocity_budgets():
    root = ROOT / "artifacts" / "cartpole-physx-baseline"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    contract = metadata["capture_contract"]
    with np.load(root / "trajectories.npz") as payload:
        angle = np.asarray(payload["abs_pole_angle"])[:, :, 0]
        pole_velocity = np.asarray(payload["pole_velocity"])[:, :, 0]

    crossings = np.asarray([np.flatnonzero(angle[:, env] > 1.0)[0] for env in range(angle.shape[1])])
    rates = np.asarray([abs(pole_velocity[step, env]) for env, step in enumerate(crossings)])
    frequency = 1.0 / float(contract["timing"]["control_dt"])

    assert frequency == pytest.approx(120.0)
    assert float(np.median(rates)) == pytest.approx(3.455533981323242)
    assert round(float(np.median(rates)), 3) == 3.456
    assert float(np.median(crossings)) == 37.0
    median_elapsed = (float(np.median(crossings)) + 1.0) / frequency
    assert median_elapsed == pytest.approx(38.0 / 120.0)

    angle_budget = round(float(np.median(rates)), 3) / frequency
    velocity_budget = angle_budget / median_elapsed
    assert angle_budget == pytest.approx(0.0288)
    assert velocity_budget == pytest.approx(0.09094736842105263)


def test_affected_exemplars_use_a_homogeneous_pole_rate_signal_and_no_corrupt_formula():
    for name in REAL_MANIFESTS:
        text = (ROOT / "validation" / "examples" / name).read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        velocity = next(item for item in payload["oracles"] if item["name"] == "pole_velocity_agreement")
        assert velocity["signal"] == "pole_velocity"
        assert velocity["tolerance"]["unit"] == "rad/s"
        assert "0.333" not in text
        assert "0.0864" not in text


def test_no_shipped_tolerance_norms_a_mixed_joint_vector():
    for path in (ROOT / "validation" / "examples").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for oracle in payload.get("oracles", []):
            if "tolerance" in oracle:
                assert oracle.get("signal") not in {"joint_pos", "joint_vel"}, (
                    f"{path.name}:{oracle['name']} uses a mixed-dimension joint vector"
                )


def test_documented_tolerance_index_is_generated_from_the_registry():
    text = DOC_PATH.read_text(encoding="utf-8")
    generated = text.split("<!-- BEGIN GENERATED -->\n", 1)[1].split("<!-- END GENERATED -->", 1)[0]
    assert generated == render_markdown(_records())
