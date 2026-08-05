# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The one test that actually launches a simulator.

Chain under test, with nothing mocked at any link:

    real Isaac Lab startup -> real rollout -> completed trajectory_bundle/v1
    -> IVF ingestion -> semantic oracle -> final verdict -> checksum verification

Everything else in this suite runs on synthetic or recorded data, which is what makes it
fast and portable. This module is the counterweight: it is the only place where a claim
about the simulator is backed by having started one.

When the runtime is absent the test **skips with the specific missing prerequisite**, and
that skip is not evidence of anything. A green suite on a laptop says the framework works
on a laptop; only a green run of this module says the framework works against Isaac Lab.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ivf.bundle import COMPLETION_MARKER, load_v1
from ivf.evidence import EvidenceBundle
from ivf.manifest import load_manifest, parse_manifest
from ivf.runner import validate
from ivf.verdicts import Verdict
from tests.simulator_gate import simulator_unavailable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.isaaclab,
    pytest.mark.physx,
    pytest.mark.slow,
]

CAPTURE_TIMEOUT_S = 900
SPEC_DIR = Path(__file__).resolve().parents[1] / "validation" / "capture"


def _capture_command() -> list[str] | None:
    """Return the ``parity-capture`` invocation for this machine, or ``None``.

    Resolution order, most explicit first: an interpreter named by
    ``IVF_CAPTURE_PYTHON``, then a ``parity-capture`` already on PATH. Nothing is
    guessed, because a capture command that silently resolves to the wrong environment
    would produce a bundle from the wrong simulator build.
    """
    explicit = os.environ.get("IVF_CAPTURE_PYTHON")
    if explicit and Path(explicit).exists():
        return [explicit, "-m", "parity_capture.cli"]
    found = shutil.which("parity-capture")
    if found:
        return [found]
    return None


def _missing_prerequisite() -> str | None:
    """Return a precise reason this machine cannot run the chain, or ``None``."""
    command = _capture_command()
    if command is None:
        return (
            "no capture command: set IVF_CAPTURE_PYTHON to a Python interpreter that has "
            "Isaac Lab and ivf-parity-capture installed, or put `parity-capture` on PATH. "
            "IVF core deliberately does not depend on Isaac Lab, so this cannot be "
            "installed by `uv sync`."
        )
    probe = subprocess.run(
        [*command, "doctor"], capture_output=True, text=True, timeout=300,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stdout + probe.stderr).strip().splitlines()
        tail = "; ".join(line.strip() for line in detail[-4:]) or "no output"
        return f"`parity-capture doctor` exited {probe.returncode}: {tail}"
    return None


@pytest.fixture(scope="module")
def capture_command() -> list[str]:
    """The capture command, skipping the module with a precise reason when unavailable."""
    reason = _missing_prerequisite()
    if reason is not None:
        simulator_unavailable(f"Isaac Lab capture unavailable. {reason}")
    return _capture_command()


def run_capture(command: list[str], spec: Path, output: Path) -> Path:
    """Run one real capture and return the bundle directory."""
    result = subprocess.run(
        [*command, str(spec), "--output", str(output)],
        capture_output=True, text=True, timeout=CAPTURE_TIMEOUT_S, check=False,
        env={
            **os.environ,
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "IVF_HARDWARE_LABEL": os.environ.get(
                "IVF_HARDWARE_LABEL", "NVIDIA (Blackwell) consumer GPU"
            ),
        },
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
        pytest.fail(f"parity-capture exited {result.returncode} for {spec.name}:\n{tail}")
    assert output.is_dir(), f"{spec.name}: capture reported success but wrote no directory"
    return output


@pytest.fixture(scope="module")
def live_captures(capture_command, tmp_path_factory) -> dict[str, Path]:
    """Produce three real captures: baseline, corrected, and the reset defect.

    Module-scoped because each capture boots Kit, which costs tens of seconds. Three is
    the minimum that proves both verdicts: without the corrected capture a FAIL could
    just mean the tolerances are too tight.
    """
    root = tmp_path_factory.mktemp("live-captures")
    specs = {
        "baseline": SPEC_DIR / "cartpole_physx_baseline.yaml",
        "corrected": SPEC_DIR / "cartpole_physx_candidate.yaml",
        "defect": SPEC_DIR / "cartpole_physx_reset_defect.yaml",
    }
    return {name: run_capture(capture_command, spec, root / name) for name, spec in specs.items()}


def test_the_capture_is_a_complete_trajectory_bundle_v1(live_captures):
    """The simulator produced a bundle that satisfies the capture boundary."""
    for name, path in live_captures.items():
        assert (path / COMPLETION_MARKER).is_file(), f"{name}: no completion marker"
        bundle = load_v1(path)  # full validation, including every checksum
        contract = bundle.contract
        assert contract.run_status == "completed", f"{name}: {contract.run_status}"
        assert contract.captured_steps == contract.declared_steps
        assert not contract.is_partial
        assert contract.backend["id"] == "physx"
        assert contract.quaternion["layout"] == "wxyz"
        assert contract.software.get("isaaclab"), f"{name}: no Isaac Lab version recorded"
        assert set(contract.software.get("source_commits", {})) == {
            "parity_capture", "isaaclab",
        }
        assert contract.backend["solver_settings"], f"{name}: no solver settings recorded"
        assert contract.task["asset"]["id"] == "isaaclab_assets.CARTPOLE_CFG"
        assert contract.timing["warmup_steps"] == 0
        assert contract.timing["timestamp_convention"]
        assert set(contract.arrays) >= {"pole_angle", "pole_velocity", "root_link_quat_w"}
        assert bundle.actions is not None and bundle.actions.shape == (400, 16, 2)


def test_the_completion_marker_is_written_after_the_checksums(live_captures):
    """The ordering that makes an interrupted capture structurally detectable."""
    path = live_captures["baseline"]
    marker = json.loads((path / COMPLETION_MARKER).read_text())
    assert marker["schema"] == "trajectory_bundle/v1"
    from ivf.bundle import CHECKSUM_FILE, sha256_file

    assert marker["checksums_sha256"] == sha256_file(path / CHECKSUM_FILE)


def _manifest_for(name: str, baseline: Path, candidate: Path):
    """Load a shipped manifest with fresh paths and their fresh finalized root locks."""
    import yaml

    source = Path(__file__).resolve().parents[1] / "validation" / "examples" / name
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    for role, path in (("baseline", baseline), ("candidate", candidate)):
        raw["subjects"][role]["path"] = str(path)
        raw["subjects"][role]["bundle_sha256"] = load_v1(path).bundle_sha256
    text = yaml.safe_dump(raw, sort_keys=False)
    return parse_manifest(text, source_path=str(source))


def test_a_corrected_capture_passes_end_to_end(live_captures, results_root):
    """Real simulator to PASS through the generated, hash-locked manifest.

    This is the negative control for the failing case below: if a clean rerun of the
    same workload does not pass, a failure elsewhere proves nothing about the defect.
    """
    generated = live_captures["baseline"].parent / "cartpole_real.yaml"
    assert generated.is_file(), "the two capture commands did not generate the IVF manifest"
    manifest = load_manifest(generated)
    for role in ("baseline", "candidate"):
        assert len(str(manifest.subjects[role].params.get("bundle_sha256", ""))) == 64
    result = validate(manifest, results_root=results_root)

    assert result.verdict is Verdict.PASS, (
        f"clean rerun did not pass: {result.reason_codes} / "
        + "; ".join(o.summary for o in result.outcomes if o.status != "pass")
    )
    assert result.validity is not None and result.validity.valid
    assert [o.status for o in result.outcomes].count("pass") == len(result.outcomes)
    assert EvidenceBundle.open(result.bundle_path).verify() == []


def test_an_explicit_simulator_reset_perturbation_is_caught_localized_and_sealed(
    live_captures, results_root
):
    """The full chain, ending in a semantic oracle firing on simulator output.

    The perturbation is applied before simulation, not to a recorded array: reset writes
    joint position with a zero velocity vector. Metadata records the switch, while the
    oracle independently establishes the numerical and event-level consequences.
    """
    manifest = _manifest_for("cartpole_reset_defect.yaml",
                             live_captures["baseline"], live_captures["defect"])
    result = validate(manifest, results_root=results_root)

    assert result.verdict is Verdict.FAIL
    assert result.validity is not None and result.validity.valid, (
        "the experiment must be valid: a defect that invalidates the comparison would "
        "prove nothing about the oracles"
    )

    # A semantic oracle, not only a numerical one, must have fired.
    failing = {o.name: o for o in result.outcomes if o.status == "fail"}
    assert "termination_timing" in failing, (
        f"no semantic oracle fired; failures were {sorted(failing)}"
    )
    assert "IVF-ORACLE-EVENT-TIMING-DELTA" in result.reason_codes
    assert "IVF-ORACLE-NON_EQUIVALENT" in result.reason_codes

    # Localization: the velocity signal carries the step-0 signature of a dropped reset.
    velocity = failing.get("pole_velocity_agreement")
    assert velocity is not None and velocity.divergence is not None
    record = velocity.divergence
    assert record.first_tolerance_violation_step == 0, (
        "a dropped reset velocity is wrong from the first captured step"
    )
    assert record.classification == "reset_mismatch", record.classification_basis
    assert len(record.affected_env_ids) == manifest.workload.num_envs

    # The event oracle localizes to a step and an environment, not just a magnitude.
    event = failing["termination_timing"].divergence
    assert event is not None and event.first_event_disagreement_step is not None
    assert event.affected_env_ids

    # And the whole finding is sealed and independently verifiable.
    evidence = EvidenceBundle.open(result.bundle_path)
    assert evidence.verify() == []
    assert evidence.verdict["verdict"] == "FAIL"


def test_the_shipped_demonstration_manifests_match_the_committed_artifacts(repo_root):
    """The committed evidence must describe the committed captures.

    Runs without a simulator; it is here rather than in the offline suite because it
    guards the artifacts this module produces.
    """
    for name in ("cartpole_reset_defect.yaml", "cartpole_corrected.yaml",
                 "cartpole_benign_difference.yaml"):
        manifest = load_manifest(repo_root / "validation" / "examples" / name)
        for subject in manifest.subjects.values():
            path = repo_root / subject.params["path"]
            assert path.is_dir(), f"{name}: {subject.role} points at a missing capture {path}"
            assert load_v1(path).contract.run_status == "completed"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
