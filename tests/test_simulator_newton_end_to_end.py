# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""One genuine Newton runtime path through capture, IVF, sealing, and verification."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ivf.bundle import load_v1
from ivf.evidence import EvidenceBundle
from ivf.manifest import parse_manifest
from ivf.runner import validate
from ivf.verdicts import Verdict
from tests.simulator_gate import simulator_unavailable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.isaaclab,
    pytest.mark.newton,
    pytest.mark.slow,
]

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_TIMEOUT_S = 900


def _capture_command() -> list[str] | None:
    explicit = os.environ.get("IVF_CAPTURE_PYTHON")
    if explicit and Path(explicit).is_file():
        return [explicit, "-m", "parity_capture.cli"]
    return None


def test_real_newton_capture_reaches_a_sealed_typed_ivf_verdict(tmp_path):
    command = _capture_command()
    if command is None:
        simulator_unavailable(
            "Newton capture unavailable: set IVF_CAPTURE_PYTHON to the Isaac Lab interpreter "
            "with ivf-parity-capture installed"
        )

    doctor = subprocess.run(
        [*command, "doctor"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
    )
    if doctor.returncode != 0:
        simulator_unavailable(
            f"Newton capture prerequisites unavailable: {(doctor.stdout + doctor.stderr).strip()}"
        )

    capture = tmp_path / "cartpole-newton"
    run = subprocess.run(
        [
            *command,
            str(ROOT / "validation" / "capture" / "cartpole_newton.yaml"),
            "--output",
            str(capture),
        ],
        capture_output=True,
        text=True,
        timeout=CAPTURE_TIMEOUT_S,
        check=False,
        env={
            **os.environ,
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "IVF_HARDWARE_LABEL": os.environ.get(
                "IVF_HARDWARE_LABEL", "NVIDIA Blackwell consumer GPU"
            ),
        },
    )
    assert run.returncode == 0, (run.stdout + run.stderr)[-6000:]

    bundle = load_v1(capture)
    assert bundle.contract.backend["id"] == "newton"
    assert bundle.contract.run_status == "completed"
    assert bundle.contract.captured_steps == bundle.contract.declared_steps == 400
    assert bundle.contract.backend["solver_settings"]["manager"] == "NewtonCfg"
    assert bundle.contract.software["newton"]
    assert bundle.contract.software["kit"]
    assert bundle.contract.hardware["gpu"]
    assert bundle.contract.hardware["driver"]
    assert bundle.contract.reset["initial_state"]
    assert bundle.actions is not None and bundle.actions.shape == (400, 16, 2)
    assert set(bundle.arrays) >= {"pole_angle", "pole_velocity", "abs_pole_angle"}

    manifest_path = ROOT / "validation" / "examples" / "cartpole_physx_vs_newton_v1.yaml"
    import yaml

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["subjects"]["candidate"]["path"] = str(capture)
    raw["subjects"]["candidate"]["bundle_sha256"] = bundle.bundle_sha256
    text = yaml.safe_dump(raw, sort_keys=False)
    manifest = parse_manifest(text, source_path=str(manifest_path))
    result = validate(manifest, results_root=tmp_path / "evidence")

    assert result.validity is not None and result.validity.valid
    assert result.verdict is Verdict.FAIL
    assert "IVF-ORACLE-NON_EQUIVALENT" in result.reason_codes
    assert "IVF-ORACLE-EVENT-TIMING-DELTA" in result.reason_codes
    assert EvidenceBundle.open(result.bundle_path).verify() == []
