# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Fault taxonomy, detectability calibration, and signal sources."""

from __future__ import annotations

import numpy as np
import pytest

from ivf.calibration import render_matrix_markdown, run_calibration
from ivf.faults import TAXONOMY, UnknownFault, apply_generation_fault, apply_trace_fault, get_fault
from ivf.signals import SignalSourceError, load_parity_bundle

from .conftest import BUNDLES, EXAMPLES, make_signals

pytestmark = pytest.mark.unit


# -- taxonomy ------------------------------------------------------------------------------

def test_every_fault_declares_its_expectation_and_limits():
    for name, spec in TAXONOMY.items():
        assert spec.name == name
        assert spec.surface in ("generation", "trace", "none")
        assert spec.minimum_severity, f"{name} declares no minimum severity"
        assert spec.description
        if not spec.expected_detectable and name != "none":
            assert spec.limitations, f"{name} claims to be undetectable without saying why"


def test_the_taxonomy_contains_a_negative_control():
    assert TAXONOMY["none"].expected_detectable is False


def test_unknown_faults_are_rejected():
    with pytest.raises(UnknownFault, match="unknown fault"):
        get_fault("cosmic_rays")


@pytest.mark.fault_injection
def test_generation_faults_change_only_generation_parameters():
    kwargs = {"steps": 10, "num_envs": 2, "seed": 0, "drive_amplitude": 1.0}
    out = apply_generation_fault("ignored_reset_velocity", kwargs, {})
    assert out["apply_reset_velocity"] is False
    assert out["steps"] == 10


@pytest.mark.fault_injection
def test_trace_faults_leave_the_original_untouched():
    original = make_signals("candidate")
    before = original.signals["x"].copy()
    mutated = apply_trace_fault("silent_nan", original, {"step": 3, "env": 1})
    assert np.array_equal(original.signals["x"], before)
    assert not np.isfinite(mutated.signals["x"]).all() or not np.isfinite(mutated.signals["y"]).all()


@pytest.mark.fault_injection
def test_a_trace_fault_with_no_applicable_signal_says_so():
    signals = make_signals("candidate")
    with pytest.raises(UnknownFault, match="last dimension 4"):
        apply_trace_fault("quaternion_ordering", signals, {})


@pytest.mark.fault_injection
def test_truncation_marks_the_run_incomplete():
    mutated = apply_trace_fault("truncated_rollout", make_signals("candidate"), {"keep_fraction": 0.5})
    assert mutated.complete is False
    assert mutated.steps < 40


# -- calibration campaign --------------------------------------------------------------------

@pytest.mark.fault_injection
@pytest.mark.slow
def test_detectability_matrix_matches_every_declaration(tmp_path):
    """The load-bearing claim: IVF behaves exactly as the taxonomy declares it will.

    Includes the negative control, so a validator aggressive enough to flag everything
    fails this test just as hard as one that flags nothing.
    """
    matrix = run_calibration(
        manifest_path=EXAMPLES / "calibration_base.yaml",
        results_root=tmp_path / "cal",
        seeds=[11, 23],
    )
    inconsistent = [r.spec.name for r in matrix.rows if not r.consistent]
    assert not inconsistent, (
        f"faults whose measured behaviour contradicts the taxonomy: {inconsistent}"
    )
    assert matrix.false_positives == 0
    assert len(matrix.rows) == len(TAXONOMY)


@pytest.mark.fault_injection
@pytest.mark.slow
def test_the_matrix_renders_with_its_scope_limitation(tmp_path):
    matrix = run_calibration(
        manifest_path=EXAMPLES / "calibration_base.yaml",
        results_root=tmp_path / "cal", seeds=[11], faults=["none", "ignored_reset_velocity"],
    )
    markdown = render_matrix_markdown(matrix)
    assert "author-generated" in markdown
    assert "ignored_reset_velocity" in markdown


@pytest.mark.fault_injection
def test_calibration_leaves_no_bundles_behind_by_default(tmp_path):
    root = tmp_path / "cal"
    run_calibration(manifest_path=EXAMPLES / "calibration_base.yaml", results_root=root,
                    seeds=[11], faults=["none"])
    assert not root.exists()


# -- parity bundle source ----------------------------------------------------------------------

BUNDLE_DIRS = sorted(p for p in BUNDLES.iterdir() if p.is_dir())


@pytest.mark.parametrize("path", BUNDLE_DIRS, ids=lambda p: p.name)
def test_real_bundles_load_without_isaac_lab(path):
    """The vendored fixtures are genuine PhysX/Newton captures; reading them uses numpy only."""
    import sys

    signals = load_parity_bundle(path)
    assert signals.signals
    assert signals.metadata["backend"] in ("physx", "newton:mjwarp")
    assert all(arr.ndim == 3 for arr in signals.signals.values())
    assert "isaaclab" not in sys.modules


def test_a_bundle_whose_payload_contradicts_its_metadata_is_rejected(tmp_path):
    import json
    import shutil

    src = BUNDLE_DIRS[0]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    meta = json.loads((dst / "metadata.json").read_text())
    name = next(iter(meta["quantities"]))
    meta["quantities"][name] = [1, 1, 1]
    (dst / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(SignalSourceError, match="does not match the declared"):
        load_parity_bundle(dst)


def test_a_bundle_missing_a_declared_quantity_is_rejected(tmp_path):
    import json
    import shutil

    src = BUNDLE_DIRS[0]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    meta = json.loads((dst / "metadata.json").read_text())
    meta["quantities"]["ghost_quantity"] = [1, 1, 1]
    (dst / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(SignalSourceError, match="payload lacks it"):
        load_parity_bundle(dst)


def test_a_directory_that_is_not_a_bundle_says_so(tmp_path):
    with pytest.raises(SignalSourceError, match="not a parity trajectory bundle"):
        load_parity_bundle(tmp_path)


def test_bundles_without_solver_settings_report_an_empty_mapping():
    """The released bundles genuinely lack solver settings; the reader must not invent them."""
    signals = load_parity_bundle(BUNDLE_DIRS[0])
    assert signals.metadata["solver_settings"] == {}


def test_an_asset_identity_derived_from_a_weak_source_is_labelled_weak():
    signals = load_parity_bundle(BUNDLE_DIRS[0])
    assert signals.metadata["asset_identity"].startswith("weak:")


# -- synthetic source --------------------------------------------------------------------------

def test_the_synthetic_system_is_deterministic():
    from ivf.signals import _damped_pendulum

    a, _, _ = _damped_pendulum(steps=20, num_envs=3, seed=5)
    b, _, _ = _damped_pendulum(steps=20, num_envs=3, seed=5)
    assert np.array_equal(a["pole_angle"], b["pole_angle"])


def test_injected_defects_are_silent_in_the_recorded_provenance():
    """The property that makes the fixture a fair test of the oracles."""
    from ivf.signals import _damped_pendulum

    clean_sig, clean_act, clean_meta = _damped_pendulum(
        steps=50, num_envs=4, seed=5, drive_amplitude=2.5, declared_drive_amplitude=2.5)
    faulted_sig, faulted_act, faulted_meta = _damped_pendulum(
        steps=50, num_envs=4, seed=5, drive_amplitude=2.5, declared_drive_amplitude=2.5,
        apply_reset_velocity=False)
    assert clean_meta["asset_identity"] == faulted_meta["asset_identity"]
    assert clean_meta["initial_state_digest"] == faulted_meta["initial_state_digest"]
    assert np.array_equal(clean_act, faulted_act)
    assert not np.allclose(clean_sig["pole_angle"], faulted_sig["pole_angle"])


def test_an_unknown_synthetic_system_lists_the_known_ones():
    from ivf.manifest import Subject
    from ivf.signals import load_subject

    subject = Subject(role="baseline", kind="synthetic", params={"system": "double_pendulum"})
    with pytest.raises(SignalSourceError, match="known:"):
        load_subject(subject, steps=10, num_envs=2, seed=0)
