# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The ``trajectory_bundle/v1`` capture boundary.

Every test here asserts a *refusal*. The property under test is not "IVF reads good
bundles" but "IVF refuses bad ones before an oracle ever sees them", because an oracle
handed an uninterpretable capture produces a confident number rather than an error, and
that number is worse than no answer at all.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ivf.bundle import (
    CHECKSUM_FILE,
    COMPLETION_MARKER,
    CONTRACT_KEY,
    SCHEMA,
    BundleContractError,
    finalize_v1,
    is_v1,
    load_v1,
)
from ivf.manifest import parse_manifest
from ivf.runner import validate
from ivf.verdicts import REASON_CODES, Verdict

pytestmark = pytest.mark.unit

STEPS, ENVS = 24, 4


def write_bundle(root, *, run_status="completed", captured_steps=STEPS, declared_steps=STEPS,
                 contract_edits=None, array_edits=None, finalize=True):
    """Write a minimal but fully valid v1 bundle, then apply the requested corruption."""
    root.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, captured_steps, dtype=np.float32)[:, None, None]
    pos = np.repeat(t, ENVS, axis=1) * np.ones((1, 1, 3), dtype=np.float32)
    quat = np.zeros((captured_steps, ENVS, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    arrays = {"root_pos_w": pos, "root_quat_w": quat}
    if array_edits:
        arrays = array_edits(arrays)
    np.savez_compressed(root / "trajectories.npz", **arrays,
                        __actions__=np.zeros((captured_steps, ENVS), dtype=np.float32))

    contract = {
        "schema": SCHEMA,
        "run_status": run_status,
        "declared_steps": declared_steps,
        "captured_steps": captured_steps,
        "task": {"id": "fixture.v1", "variant": "default",
                 "config_digest_sha256": "0" * 64},
        "backend": {"id": "fixture-backend", "solver_settings": {"iters": 4}, "features": []},
        "software": {"python": "3.12", "numpy": np.__version__},
        "hardware": {"gpu": "fixture", "driver": "fixture"},
        "seed": {"value": 0, "env_ids": list(range(ENVS)), "env_order": "clone_index"},
        "timing": {"physics_dt": 0.005, "control_dt": 0.005, "decimation": 1,
                   "action_applied": "before_physics_step",
                   "capture_hook": "post_step_post_update"},
        "frames": {"convention": "world_z_up_right_handed", "length_unit": "m",
                   "angle_unit": "rad"},
        "quaternion": {"layout": "wxyz", "scalar_first": True, "normalized": True},
        "reset": {"semantics": "writes_pose_and_velocity", "initial_state_digest": "a" * 64},
        "termination": {"declared": False, "reason": "fixture cannot terminate"},
        "arrays": {
            "root_pos_w": {"shape": [captured_steps, ENVS, 3], "dtype": "float32",
                           "unit": "m", "frame": "world"},
            "root_quat_w": {"shape": [captured_steps, ENVS, 4], "dtype": "float32",
                            "unit": "dimensionless", "frame": "world"},
        },
    }
    if contract_edits:
        contract = contract_edits(contract)
    (root / "metadata.json").write_text(
        json.dumps({"scenario_name": "fixture", CONTRACT_KEY: contract}, indent=2), encoding="utf-8"
    )
    if finalize:
        finalize_v1(root, run_status=run_status)
    return root


@pytest.fixture
def good(tmp_path):
    """A valid v1 bundle."""
    return write_bundle(tmp_path / "good")


def test_a_valid_bundle_loads(good):
    bundle = load_v1(good)
    assert bundle.contract.run_status == "completed"
    assert bundle.contract.scalar_first is True
    assert set(bundle.arrays) == {"root_pos_w", "root_quat_w"}
    assert bundle.arrays["root_pos_w"].shape == (STEPS, ENVS, 3)


def test_finalize_writes_the_completion_marker_last(good):
    """The ordering is the contract: payload, checksums, then the marker."""
    marker = json.loads((good / COMPLETION_MARKER).read_text())
    assert marker["schema"] == SCHEMA
    recorded = (good / CHECKSUM_FILE).read_text()
    assert COMPLETION_MARKER not in recorded  # the marker cannot checksum itself
    assert "trajectories.npz" in recorded and "metadata.json" in recorded


def test_every_refusal_code_is_registered():
    """A refusal nobody can grep for is not a contract."""
    import ast

    from ivf import bundle as bundle_module

    tree = ast.parse(Path(bundle_module.__file__).read_text(encoding="utf-8"))
    emitted = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value.startswith("IVF-")}
    assert emitted, "no reason codes found in ivf.bundle"
    assert not (emitted - set(REASON_CODES))


# -- the five named failure modes ---------------------------------------------------------

def test_missing_completion_marker_is_refused(tmp_path):
    """The case the marker exists for: a killed capture leaves a plausible payload."""
    root = write_bundle(tmp_path / "b")
    (root / COMPLETION_MARKER).unlink()
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-INCOMPLETE"


def test_invalid_checksum_is_refused(tmp_path):
    root = write_bundle(tmp_path / "b")
    meta = json.loads((root / "metadata.json").read_text())
    meta["scenario_name"] = "tampered"
    (root / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-CHECKSUM-MISMATCH"


def test_a_file_added_after_finalization_is_refused(tmp_path):
    root = write_bundle(tmp_path / "b")
    (root / "extra.json").write_text("{}")
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-CHECKSUM-MISMATCH"


@pytest.mark.parametrize(
    "edit,label",
    [
        (lambda c: {**c, "quaternion": {"scalar_first": True}}, "layout absent"),
        (lambda c: {**c, "quaternion": {"layout": "jpl", "scalar_first": True}}, "unknown layout"),
        (lambda c: {**c, "quaternion": {"layout": "wxyz", "scalar_first": False}}, "contradictory"),
        (lambda c: {**c, "quaternion": {"layout": "xyzw", "scalar_first": True}}, "contradictory"),
    ],
)
def test_ambiguous_quaternion_convention_is_refused(tmp_path, edit, label):
    """wxyz read as xyzw yields a plausible, wrong rotation, so there is no safe default."""
    root = write_bundle(tmp_path / f"b-{label.replace(' ', '-')}", contract_edits=edit)
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-QUATERNION-AMBIGUOUS"


def test_non_unit_quaternions_discredit_the_declaration(tmp_path):
    def scale(arrays):
        arrays["root_quat_w"] = arrays["root_quat_w"] * 0.5
        return arrays

    root = write_bundle(tmp_path / "b", array_edits=scale)
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-QUATERNION-AMBIGUOUS"


def test_declared_shape_that_contradicts_the_payload_is_refused(tmp_path):
    def edit(c):
        c["arrays"]["root_pos_w"]["shape"] = [STEPS, ENVS, 7]
        return c

    root = write_bundle(tmp_path / "b", contract_edits=edit)
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT"


def test_declared_env_count_that_contradicts_the_env_ids_is_refused(tmp_path):
    def edit(c):
        c["seed"]["env_ids"] = [0, 1]
        return c

    root = write_bundle(tmp_path / "b", contract_edits=edit)
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT"


def test_an_undeclared_partial_run_is_refused(tmp_path):
    """A short rollout claiming to be complete is the defect the field exists for."""
    root = write_bundle(tmp_path / "b", captured_steps=12, declared_steps=STEPS,
                        run_status="completed")
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-PARTIAL-UNDECLARED"


def test_a_declared_partial_run_loads_and_is_marked_incomplete(tmp_path):
    """Declaring the truncation is what makes it usable: honest, not hidden."""
    root = write_bundle(tmp_path / "b", captured_steps=12, declared_steps=STEPS,
                        run_status="partial")
    bundle = load_v1(root)
    assert bundle.contract.is_partial
    from ivf.signals import load_trajectory_bundle_v1

    assert load_trajectory_bundle_v1(root).complete is False


# -- other contract violations --------------------------------------------------------------

def test_a_payload_array_nobody_declared_is_refused(tmp_path):
    def add(arrays):
        arrays["undeclared"] = np.zeros((STEPS, ENVS, 1), dtype=np.float32)
        return arrays

    root = write_bundle(tmp_path / "b", array_edits=add)
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-ARRAY-SHAPE-INCONSISTENT"


def test_a_dtype_that_contradicts_the_declaration_is_refused(tmp_path):
    def edit(c):
        c["arrays"]["root_pos_w"]["dtype"] = "float64"
        return c

    root = write_bundle(tmp_path / "b", contract_edits=edit)
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-ARRAY-DTYPE-INCONSISTENT"


@pytest.mark.parametrize("section", ["task", "software", "timing", "frames", "reset",
                                     "termination", "seed", "arrays"])
def test_every_required_declaration_is_enforced(tmp_path, section):
    root = write_bundle(tmp_path / f"b-{section}",
                        contract_edits=lambda c: {k: v for k, v in c.items() if k != section})
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-CONTRACT-INCOMPLETE"


def test_an_unknown_frame_convention_is_refused(tmp_path):
    root = write_bundle(tmp_path / "b",
                        contract_edits=lambda c: {**c, "frames": {"convention": "whatever"}})
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-FRAME-UNDECLARED"


def test_an_unknown_reset_semantics_is_refused(tmp_path):
    root = write_bundle(tmp_path / "b",
                        contract_edits=lambda c: {**c, "reset": {"semantics": "magic"}})
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-RESET-UNDECLARED"


def test_a_future_schema_is_refused_not_half_parsed(tmp_path):
    root = write_bundle(tmp_path / "b")
    meta = json.loads((root / "metadata.json").read_text())
    meta[CONTRACT_KEY]["schema"] = "trajectory_bundle/v99"
    (root / "metadata.json").write_text(json.dumps(meta))
    finalize_v1_again(root)
    with pytest.raises(BundleContractError) as exc:
        load_v1(root)
    assert exc.value.reason_code == "IVF-BUNDLE-SCHEMA-UNSUPPORTED"


def finalize_v1_again(root):
    """Re-seal a bundle after an intentional metadata edit, so the test isolates one failure."""
    (root / CHECKSUM_FILE).unlink(missing_ok=True)
    (root / COMPLETION_MARKER).unlink(missing_ok=True)
    finalize_v1(root, run_status="completed")


# -- routing and end-to-end refusal -----------------------------------------------------------

def test_is_v1_keys_on_the_marker_so_unfinished_captures_take_the_strict_path(tmp_path):
    root = write_bundle(tmp_path / "b")
    assert is_v1(root)
    (root / COMPLETION_MARKER).unlink()
    assert is_v1(root), "a declared-but-unfinished capture must still route to the strict reader"


def test_a_legacy_parity_bundle_still_loads(repo_root):
    """Reconciliation, not replacement: pre-v1 captures keep working."""
    from ivf.signals import load_parity_bundle

    legacy = next((repo_root / "validation" / "bundles").iterdir())
    assert not is_v1(legacy)
    assert load_parity_bundle(legacy).signals


MANIFEST = """
schema_version: ivf.validation/v1
name: bundle-contract-gate
subjects:
  baseline:
    kind: parity_bundle
    path: {a}
  candidate:
    kind: parity_bundle
    path: {b}
workload:
  task: fixture.v1
  num_envs: 4
  steps: 24
  warmup_steps: 0
  seeds: [0]
controls:
  require_same: [asset_identity, observation_definition, control_frequency, num_envs, horizon]
oracles:
  - type: invariant
    name: finite_state
    check: finite_state
"""


@pytest.mark.integration
def test_a_broken_capture_never_reaches_the_oracle_layer(tmp_path, results_root):
    """End to end: the run is INVALID_EXPERIMENT and no oracle outcome was produced."""
    good_a = write_bundle(tmp_path / "a")
    bad_b = write_bundle(tmp_path / "b")
    (bad_b / COMPLETION_MARKER).unlink()

    manifest = parse_manifest(MANIFEST.format(a=good_a, b=bad_b))
    result = validate(manifest, results_root=results_root)

    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert result.reason_codes == ["IVF-BUNDLE-INCOMPLETE"]
    assert result.outcomes == [], "an oracle ran on an uninterpretable capture"


@pytest.mark.integration
@pytest.mark.parametrize(
    "corrupt,expected",
    [
        (lambda p: (p / COMPLETION_MARKER).unlink(), "IVF-BUNDLE-INCOMPLETE"),
        (lambda p: (p / "metadata.json").write_text("{}"), "IVF-BUNDLE-CHECKSUM-MISMATCH"),
        (lambda p: (p / "sneaky.txt").write_text("x"), "IVF-BUNDLE-CHECKSUM-MISMATCH"),
    ],
)
def test_each_corruption_yields_its_own_stable_reason_code(tmp_path, results_root, corrupt, expected):
    good_a = write_bundle(tmp_path / "a")
    bad_b = write_bundle(tmp_path / "b")
    corrupt(bad_b)
    result = validate(parse_manifest(MANIFEST.format(a=good_a, b=bad_b)), results_root=results_root)
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert result.reason_codes == [expected]


@pytest.mark.integration
def test_a_refused_capture_still_produces_verifiable_evidence(tmp_path, results_root):
    """A refusal is a finding, so it gets the same auditable record as any other verdict."""
    from ivf.evidence import EvidenceBundle

    good_a = write_bundle(tmp_path / "a")
    bad_b = write_bundle(tmp_path / "b")
    shutil.rmtree(bad_b / "trajectories.npz", ignore_errors=True)
    (bad_b / "trajectories.npz").unlink(missing_ok=True)

    result = validate(parse_manifest(MANIFEST.format(a=good_a, b=bad_b)), results_root=results_root)
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert EvidenceBundle.open(result.bundle_path).verify() == []


V1_CONTROL_MANIFEST = MANIFEST.replace(
    "require_same: [asset_identity, observation_definition, control_frequency, num_envs, horizon]",
    "require_same: [asset_identity, observation_definition, control_frequency, num_envs, horizon, "
    "frame_convention, quaternion_convention, reset_semantics, environment_ordering, action_timing]\n"
    "  unsupported_or_unverifiable: [asset_binary_identity, initial_state_realization, "
    "backend_internal_state]",
)


@pytest.mark.integration
def test_v1_declared_conventions_are_verified_and_unsupported_controls_stay_explicit(
    tmp_path, results_root
):
    a = write_bundle(tmp_path / "a")
    b = write_bundle(tmp_path / "b")
    result = validate(
        parse_manifest(V1_CONTROL_MANIFEST.format(a=a, b=b)),
        results_root=results_root,
    )
    assert result.verdict is Verdict.PASS
    checks = {check.name: check.status for check in result.validity.checks}
    assert checks["frame convention matches"] == "pass"
    assert checks["quaternion convention matches"] == "pass"
    assert checks["reset semantics matches"] == "pass"
    assert checks["environment ordering matches"] == "pass"
    assert checks["action timing matches"] == "pass"
    assert sum(status == "unverifiable" for status in checks.values()) >= 3
