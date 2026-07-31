# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Environment probing for ``ivf doctor``.

The job of the doctor is to answer, before anything runs, "can this machine produce a
valid experiment, and of which kind". It reports three capability levels rather than a
single yes/no, because the honest answer is usually "yes for the offline half":

``offline``
    read, compare, verify and render evidence bundles. Needs only Python and numpy.

``synthetic``
    additionally execute synthetic reference workloads. Needs nothing more.

``isaaclab``
    additionally execute live Isaac Lab workloads. Needs Isaac Lab, Isaac Sim, a
    supported backend and a GPU.

Every probe is best-effort and never raises: a doctor that crashes on an exotic machine
is useless precisely where it is most needed.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CheckStatus = Literal["ok", "warn", "fail", "absent"]

CAPABILITY_LEVELS = ("offline", "synthetic", "isaaclab")

# Version combinations this build claims to work with. Absence from this table is not a
# claim that something is broken; it is a claim that nobody has verified it.
COMPATIBILITY_MATRIX: list[dict[str, str]] = [
    {
        "isaac_lab": "any (not imported)",
        "path": "offline / synthetic",
        "status": "supported",
        "evidence": "CPU test suite in this repository",
    },
    {
        "isaac_lab": "2.x / 4.5.x-era contrib tree",
        "path": "parity_bundle ingest",
        "status": "supported",
        "evidence": "reads bundle schema 1.1 and 1.2; verified against 12 live PhysX/Newton "
                    "bundles generated 2026-07-12/13",
    },
    {
        "isaac_lab": "3.0 (develop)",
        "path": "parity_bundle ingest",
        "status": "supported",
        "evidence": "same bundle schema; the reader does not import Isaac Lab",
    },
    {
        "isaac_lab": "any",
        "path": "live execution (subjects.kind = isaaclab)",
        "status": "not implemented",
        "evidence": "generate bundles with isaaclab_contrib.parity and ingest them instead",
    },
]


@dataclass
class Check:
    """One environment probe."""

    name: str
    status: CheckStatus
    value: str = ""
    detail: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {"name": self.name, "status": self.status, "value": self.value, "detail": self.detail}


@dataclass
class DoctorReport:
    """The result of a full environment probe."""

    checks: list[Check] = field(default_factory=list)
    capabilities: dict[str, bool] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def add(self, name: str, status: CheckStatus, value: str = "", detail: str = "") -> None:
        """Append a check."""
        self.checks.append(Check(name, status, value, detail))

    def get(self, name: str) -> Check | None:
        """Return the check with this name, if present."""
        return next((c for c in self.checks if c.name == name), None)

    @property
    def failures(self) -> list[Check]:
        """Checks that failed."""
        return [c for c in self.checks if c.status == "fail"]

    def supports(self, level: str) -> bool:
        """Whether the environment supports a capability level."""
        return bool(self.capabilities.get(level, False))

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "checks": [c.to_jsonable() for c in self.checks],
            "capabilities": self.capabilities,
            "examples": self.examples,
            "compatibility_matrix": COMPATIBILITY_MATRIX,
        }


def _module_version(name: str) -> tuple[bool, str, str]:
    """Return ``(importable, version, detail)`` for a module, without raising."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "")
    if not version:
        try:
            from importlib.metadata import version as _pkg_version

            version = _pkg_version(name)
        except Exception:
            version = "unknown"
    return True, str(version), str(getattr(module, "__file__", "") or "")


def _git_commit(path: Path) -> str:
    """Return the short commit of the repository containing ``path``, or ``""``."""
    if shutil.which("git") is None:
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _nvidia_smi() -> tuple[str, str]:
    """Return ``(gpu_name, driver_version)`` from ``nvidia-smi``, or empty strings.

    ``IVF_HARDWARE_LABEL`` overrides the reported device name. Evidence bundles are meant
    to be shared, and an exact device model is sometimes more identifying than a team
    wants to publish; a coarse label such as "NVIDIA (Blackwell) consumer GPU" keeps the
    provenance meaningful without the detail.
    """
    override = os.environ.get("IVF_HARDWARE_LABEL")
    if override:
        return override, os.environ.get("IVF_DRIVER_LABEL", "redacted")
    if shutil.which("nvidia-smi") is None:
        return "", ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return "", ""
        first = out.stdout.strip().splitlines()[0]
        name, _, driver = first.partition(",")
        return name.strip(), driver.strip()
    except Exception:
        return "", ""


def run_doctor(*, results_root: str | Path = "ivf-results",
               examples_dir: str | Path = "validation/examples") -> DoctorReport:
    """Probe the environment and return a :class:`DoctorReport`."""
    from . import __version__
    from .oracles import registered_oracles

    report = DoctorReport()

    # -- interpreter and IVF itself ----------------------------------------------------
    py_ok = sys.version_info >= (3, 10)
    report.add(
        "python", "ok" if py_ok else "fail",
        f"{platform.python_version()} ({sys.executable})",
        "" if py_ok else "IVF requires Python 3.10 or newer",
    )
    ivf_root = Path(__file__).resolve().parents[2]
    commit = _git_commit(ivf_root)
    report.add("ivf", "ok", f"{__version__}" + (f" @ {commit}" if commit else ""), str(ivf_root))
    report.add("platform", "ok", f"{platform.system()} {platform.release()} {platform.machine()}")

    # -- required packages -------------------------------------------------------------
    for pkg in ("numpy", "yaml"):
        ok, version, detail = _module_version(pkg)
        report.add(f"package:{pkg}", "ok" if ok else "fail", version, "" if ok else detail)

    report.add("oracles", "ok", ", ".join(registered_oracles()))

    # -- optional simulator stack ------------------------------------------------------
    lab_ok, lab_version, lab_detail = _module_version("isaaclab")
    if lab_ok:
        lab_commit = _git_commit(Path(lab_detail).parent) if lab_detail else ""
        report.add("isaaclab", "ok", lab_version + (f" @ {lab_commit}" if lab_commit else ""), lab_detail)
    else:
        report.add("isaaclab", "absent", "", "not importable; offline and synthetic paths are unaffected")

    sim_ok, sim_version, sim_detail = _module_version("isaacsim")
    report.add("isaacsim", "ok" if sim_ok else "absent", sim_version, "" if sim_ok else sim_detail)

    contrib_ok, contrib_version, _ = _module_version("isaaclab_contrib")
    report.add(
        "isaaclab_contrib", "ok" if contrib_ok else "absent", contrib_version,
        "" if contrib_ok else "parity bundles are still readable without it: IVF parses the "
                             "bundle format directly",
    )

    backends = []
    for name, module in (("physx", "isaaclab_physx"), ("newton", "isaaclab_newton")):
        ok, version, _ = _module_version(module)
        if ok:
            backends.append(f"{name}={version or 'present'}")
    report.add(
        "backends", "ok" if backends else "absent", ", ".join(backends),
        "" if backends else "no physics backend importable; live execution is unavailable",
    )

    # -- torch, GPU, determinism -------------------------------------------------------
    torch_ok, torch_version, _ = _module_version("torch")
    cuda_available = False
    if torch_ok:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    report.add(
        "torch", "ok" if torch_ok else "absent", torch_version,
        f"cuda_available={cuda_available}" if torch_ok else "",
    )

    gpu_name, driver = _nvidia_smi()
    report.add(
        "gpu", "ok" if gpu_name else "absent", gpu_name,
        f"driver {driver}" if driver else "nvidia-smi not available or reported no device",
    )
    cuda_env = {k: v for k, v in os.environ.items()
                if k in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUBLAS_WORKSPACE_CONFIG",
                         "OMNI_KIT_ACCEPT_EULA", "PYTORCH_CUDA_ALLOC_CONF")}
    report.add("cuda_env", "ok" if cuda_env else "absent", json.dumps(cuda_env, sort_keys=True))

    determinism = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
    }
    if torch_ok:
        try:
            import torch

            determinism["torch.deterministic_algorithms"] = str(
                torch.are_deterministic_algorithms_enabled()
            )
            determinism["torch.backends.cudnn.deterministic"] = str(torch.backends.cudnn.deterministic)
        except Exception:
            pass
    report.add(
        "determinism", "warn" if determinism.get("CUBLAS_WORKSPACE_CONFIG") == "<unset>" else "ok",
        json.dumps(determinism, sort_keys=True),
        "CUBLAS_WORKSPACE_CONFIG is unset; GPU reductions may not be bitwise reproducible. "
        "This does not affect the offline or synthetic paths.",
    )

    # -- artifacts and examples --------------------------------------------------------
    root = Path(results_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".ivf-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report.add("artifact_dir", "ok", str(root.resolve()), "writable")
    except Exception as exc:
        report.add("artifact_dir", "fail", str(root), f"not writable: {exc}")

    examples_path = Path(examples_dir)
    examples = sorted(str(p) for p in examples_path.glob("*.yaml")) if examples_path.is_dir() else []
    report.examples = examples
    report.add(
        "example_suites", "ok" if examples else "warn", f"{len(examples)} manifest(s)",
        "\n".join(examples) if examples else f"no manifests found under {examples_dir}",
    )

    # -- capability roll-up ------------------------------------------------------------
    numpy_ok = report.get("package:numpy").status == "ok"
    yaml_ok = report.get("package:yaml").status == "ok"
    artifact_ok = report.get("artifact_dir").status == "ok"
    report.capabilities = {
        "offline": bool(py_ok and numpy_ok and yaml_ok),
        "synthetic": bool(py_ok and numpy_ok and yaml_ok and artifact_ok),
        "isaaclab": bool(lab_ok and sim_ok and backends and cuda_available),
    }
    return report


def render_doctor_text(report: DoctorReport, *, require: str | None = None) -> tuple[str, int]:
    """Render the doctor report as text and return ``(text, exit_code)``.

    ``require`` names the capability level the caller needs. When it is not available,
    the exit code is nonzero and the text says which probe is responsible, so a CI job
    fails on the cause rather than on a downstream symptom.
    """
    symbols = {"ok": "  ok  ", " warn": " warn ", "warn": " warn ", "fail": " FAIL ", "absent": "absent"}
    lines = ["IVF environment report", "=" * 60]
    for check in report.checks:
        mark = symbols.get(check.status, check.status)
        line = f"[{mark}] {check.name:<20} {check.value}"
        lines.append(line.rstrip())
        if check.detail and check.status in ("fail", "warn"):
            lines.append(f"{'':>9}{check.detail}")
    lines.append("")
    lines.append("Capabilities")
    lines.append("-" * 60)
    for level in CAPABILITY_LEVELS:
        available = report.capabilities.get(level, False)
        lines.append(f"[{'  ok  ' if available else ' none '}] {level}")
    lines.append("")
    lines.append("Compatibility")
    lines.append("-" * 60)
    for row in COMPATIBILITY_MATRIX:
        lines.append(f"  {row['isaac_lab']:<34} {row['path']:<32} {row['status']}")

    exit_code = 0
    if report.failures:
        lines.append("")
        lines.append("Failing probes:")
        for check in report.failures:
            lines.append(f"  - {check.name}: {check.detail or check.value}")
        exit_code = 1
    if require:
        if require not in CAPABILITY_LEVELS:
            lines.append(f"\nunknown capability level {require!r}; expected one of {CAPABILITY_LEVELS}")
            exit_code = 2
        elif not report.supports(require):
            lines.append("")
            lines.append(f"REQUIRED capability {require!r} is NOT available on this machine.")
            for name in _blockers_for(require):
                check = report.get(name)
                if check and check.status in ("absent", "fail"):
                    lines.append(f"  - {check.name}: {check.detail or 'not available'}")
            exit_code = 1
    return "\n".join(lines) + "\n", exit_code


def _blockers_for(level: str) -> tuple[str, ...]:
    """Return the probe names that gate a capability level."""
    if level == "offline":
        return ("python", "package:numpy", "package:yaml")
    if level == "synthetic":
        return ("python", "package:numpy", "package:yaml", "artifact_dir")
    return ("isaaclab", "isaacsim", "backends", "gpu", "torch")
