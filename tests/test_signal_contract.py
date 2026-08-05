# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial tests for automatic unit/frame enforcement and identity detection.

Each mutation below produced ``PASS`` in v0.1.0-rc1. The point of the suite is not that
the checks exist but that they fire without being asked for: none of these manifests
requests a unit control, because a reviewer cannot be relied upon to request the check
that catches the mistake they did not know they made.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ivf.bundle import finalize_v1
from ivf.manifest import ManifestError, parse_manifest
from ivf.runner import validate
from ivf.verdicts import Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "artifacts" / "cartpole-physx-baseline"
CORRECTED = REPO_ROOT / "artifacts" / "cartpole-physx-corrected"
BENIGN = REPO_ROOT / "artifacts" / "cartpole-physx-benign"

pytestmark = pytest.mark.skipif(
    not (BASELINE / "trajectories.npz").is_file(),
    reason="shipped real captures are absent from this checkout",
)


def _remutate(src: Path, dst: Path, mutate) -> Path:
    """Copy a strict bundle, mutate its declarations, and finalize it properly.

    Re-finalizing matters. A hand-edited bundle is caught by the checksum seal, which
    would let a unit mistake hide behind an integrity failure and prove nothing about
    whether the unit check works. These fixtures are legitimately sealed bundles that
    happen to declare the wrong thing, which is the case that must be caught.
    """
    shutil.copytree(src, dst)
    (dst / "COMPLETE").unlink()
    (dst / "CHECKSUMS.sha256").unlink()
    meta = json.loads((dst / "metadata.json").read_text())
    mutate(meta)
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    finalize_v1(dst, run_status="completed")
    return dst


def _manifest(baseline: Path, candidate: Path, *, name: str = "mutation",
              mode: str | None = None, tolerance_unit: str = "rad") -> str:
    """Build a minimal two-subject manifest gating pole_angle."""
    mode_line = f"experiment_mode: {mode}\n" if mode else ""
    return f"""
schema_version: ivf.validation/v1
name: {name}
{mode_line}subjects:
  baseline:
    kind: parity_bundle
    path: {baseline}
  candidate:
    kind: parity_bundle
    path: {candidate}
workload:
  task: cartpole_passive
  num_envs: 16
  steps: 400
  warmup_steps: 0
  seeds: [0]
controls:
  require_same: [num_envs, horizon]
oracles:
  - type: trajectory_equivalence
    name: pole_angle_agreement
    signal: pole_angle
    metric: absolute
    tolerance:
      value: 2.88e-2
      unit: {tolerance_unit}
      scope: per-step absolute pole-angle error
      rationale: one control period of angular displacement near the crossing
      aggregation: second_largest
      min_samples: 1600
      kind: engineering
"""


def _run(tmp_path, text):
    return validate(parse_manifest(text), results_root=tmp_path / "r")


# -- 1. unit mutation ------------------------------------------------------------------

@pytest.mark.integration
def test_a_candidate_in_degrees_is_refused_not_compared(tmp_path):
    cand = _remutate(
        CORRECTED, tmp_path / "deg",
        lambda m: m["capture_contract"]["arrays"]["pole_angle"].update(unit="deg"),
    )
    result = _run(tmp_path, _manifest(BASELINE, cand, name="unit-mutation"))
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert "IVF-CONTROL-SIGNAL-UNIT-MISMATCH" in result.reason_codes
    assert result.outcomes == [], "no oracle may run on dimensionally incompatible signals"


# -- 2. frame mutation -----------------------------------------------------------------

@pytest.mark.integration
def test_a_candidate_in_a_different_frame_is_refused(tmp_path):
    cand = _remutate(
        CORRECTED, tmp_path / "frame",
        lambda m: m["capture_contract"]["arrays"]["pole_angle"].update(frame="world"),
    )
    result = _run(tmp_path, _manifest(BASELINE, cand, name="frame-mutation"))
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert "IVF-CONTROL-SIGNAL-FRAME-MISMATCH" in result.reason_codes
    assert result.outcomes == []


# -- 3. tolerance declared in the wrong unit -------------------------------------------

@pytest.mark.integration
def test_a_tolerance_in_the_wrong_unit_is_refused(tmp_path):
    """Both subjects agree on 'deg'; the tolerance still says 'rad'."""
    def to_deg(m):
        m["capture_contract"]["arrays"]["pole_angle"].update(unit="deg")

    base = _remutate(BASELINE, tmp_path / "b_deg", to_deg)
    cand = _remutate(CORRECTED, tmp_path / "c_deg", to_deg)
    result = _run(tmp_path, _manifest(base, cand, name="tolerance-unit",
                                      tolerance_unit="rad", mode="identity_check"))
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert "IVF-CONTROL-TOLERANCE-UNIT-MISMATCH" in result.reason_codes
    assert result.outcomes == []


# -- 4. mixed prismatic/revolute vector against one scalar threshold --------------------

@pytest.mark.integration
def test_a_mixed_unit_vector_cannot_enter_a_scalar_oracle(tmp_path):
    """joint_pos really does carry 'rad and m': one revolute and one prismatic joint.

    No single threshold can be correct for it, so the refusal does not depend on the two
    subjects disagreeing. They agree perfectly here.
    """
    text = _manifest(BASELINE, CORRECTED, name="mixed-unit", mode="identity_check")
    text = text.replace("signal: pole_angle", "signal: joint_pos")
    result = _run(tmp_path, text)
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert "IVF-CONTROL-MIXED-UNIT-REDUCTION-INVALID" in result.reason_codes
    assert result.outcomes == []


# -- 5. matching contracts still reach the oracles -------------------------------------

@pytest.mark.integration
def test_matching_contracts_still_reach_the_oracles(tmp_path):
    result = _run(tmp_path, _manifest(BASELINE, BENIGN, name="matching"))
    assert result.verdict is Verdict.PASS
    assert [o.name for o in result.outcomes] == ["pole_angle_agreement"]


# -- 6. legacy bundles are unverifiable, never matched ---------------------------------

@pytest.mark.integration
def test_a_legacy_bundle_without_declarations_is_unverifiable_not_matched(tmp_path):
    """Pre-v1 bundles declare no per-array units. Absence of evidence is not agreement."""
    legacy = REPO_ROOT / "validation" / "bundles" / "cartpole_passive_physx_cuda0_s0_29075be3"
    if not legacy.is_dir():  # pragma: no cover - depends on checkout
        pytest.skip("legacy bundle fixture absent")
    # Gate a signal the pre-v1 bundles actually carry; they predate pole_angle.
    text = _manifest(legacy, legacy, name="legacy", mode="identity_check",
                     tolerance_unit="m").replace("signal: pole_angle",
                                                 "signal: root_link_pos_w")
    result = _run(tmp_path, text)
    statuses = {c.check_id: c.status for c in result.validity.checks}
    assert statuses.get("V-SIGNAL-UNIT") == "unverifiable"
    assert statuses.get("V-SIGNAL-FRAME") == "unverifiable"
    assert not any(
        c.status == "pass" and c.check_id.startswith("V-SIGNAL")
        for c in result.validity.checks
    ), "a missing declaration must never be reported as a match"


# -- 7. self-comparison ----------------------------------------------------------------

@pytest.mark.integration
def test_identical_content_is_refused_by_default(tmp_path):
    result = _run(tmp_path, _manifest(BASELINE, CORRECTED, name="undeclared-identity"))
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert "IVF-EXPERIMENT-SELF-COMPARISON" in result.reason_codes
    assert result.outcomes == []


@pytest.mark.integration
def test_declared_identity_mode_runs_and_says_what_it_measures(tmp_path):
    result = _run(tmp_path, _manifest(BASELINE, CORRECTED, name="declared-identity",
                                      mode="identity_check"))
    assert result.verdict is Verdict.PASS
    check = next(c for c in result.validity.checks if c.check_id == "V-19")
    assert "validator identity behaviour" in check.detail
    assert "does not establish independent repeatability" in check.detail


@pytest.mark.integration
def test_identity_mode_with_differing_subjects_is_refused(tmp_path):
    """An identity check whose inputs differ is not an identity check."""
    result = _run(tmp_path, _manifest(BASELINE, BENIGN, name="bad-identity",
                                      mode="identity_check"))
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert "IVF-EXPERIMENT-IDENTITY-MODE-SUBJECTS-DIFFER" in result.reason_codes


def test_an_unknown_experiment_mode_is_rejected_at_load():
    with pytest.raises(ManifestError, match="experiment_mode"):
        parse_manifest(_manifest(BASELINE, CORRECTED, mode="whatever_i_like"))


# -- 8. distinct capture identity with identical arrays --------------------------------

@pytest.mark.integration
def test_distinct_capture_ids_with_identical_arrays_are_representable(tmp_path):
    """Independent deterministic re-execution is a real thing the schema must express.

    Two runs of a deterministic simulator produce identical arrays. That must remain
    declarable as two runs, otherwise the only way to look independent is to be
    non-deterministic.
    """
    def stamp(capture_id, when):
        def apply(m):
            m["capture_contract"]["capture"] = {
                "capture_id": capture_id,
                "created_utc": when,
                "independent_run": True,
                "producer": {"name": "parity-capture", "version": "0.1.0"},
                "execution": {"host_redacted": True, "user_redacted": True,
                              "paths_redacted": True},
            }
        return apply

    a = _remutate(BASELINE, tmp_path / "run_a", stamp("run-a", "2026-08-05T16:20:00Z"))
    b = _remutate(CORRECTED, tmp_path / "run_b", stamp("run-b", "2026-08-05T16:24:00Z"))

    from ivf.signals import load_trajectory_bundle_v1

    sa = load_trajectory_bundle_v1(a)
    sb = load_trajectory_bundle_v1(b)
    assert sa.metadata["capture_id"] == "run-a"
    assert sb.metadata["capture_id"] == "run-b"
    assert sa.metadata["capture_created_utc"] != sb.metadata["capture_created_utc"]
    # Distinct identities, identical science: still an A/A run in content terms, and
    # still refused unless declared. A different capture_id is not evidence of a
    # different capture, because a copy can carry any id its author types.
    assert sa.content_digest() == sb.content_digest()
    result = _run(tmp_path, _manifest(a, b, name="two-runs"))
    assert result.verdict is Verdict.INVALID_EXPERIMENT
    assert "IVF-EXPERIMENT-SELF-COMPARISON" in result.reason_codes


# -- 9. input identities reach the sealed evidence -------------------------------------

@pytest.mark.integration
def test_input_bundle_digests_appear_in_sealed_evidence(tmp_path):
    result = _run(tmp_path, _manifest(BASELINE, BENIGN, name="digest-binding"))
    provenance = json.loads((result.bundle_path / "provenance.json").read_text())
    blob = json.dumps(provenance)
    for role in ("baseline", "candidate"):
        assert role in provenance.get("subject_identities", {}), blob[:400]
    ids = provenance["subject_identities"]
    assert ids["baseline"]["payload_sha256"] != ids["candidate"]["payload_sha256"]
    assert ids["baseline"]["capture_schema"] == "trajectory_bundle/v1"
