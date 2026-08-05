# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The shipped version must be one number, declared once per package and agreeing.

v0.1.0-rc2 was very nearly tagged on a tree whose packages still built and installed as
``0.1.0rc1``: the release work was merged, public CI was green, and the artifact the GPU
job installed was ``ivf==0.1.0rc1``. Nothing in the suite noticed, because nothing asserted
it.

That is the drift this file exists to stop. Every source of truth for a version is compared
against every other one, for both the root package and the capture adapter, so a release
that forgets one of them fails here instead of shipping under the wrong name.

The expected version is written once, below, and the release checklist is to change that
constant and the four declarations it guards.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import ivf

#: The version this working tree is supposed to be. Update it in the same commit that
#: updates the four declarations, which is exactly the commit these tests are here to gate.
EXPECTED_VERSION = "0.1.0rc2"

#: PEP 440 uses ``0.1.0rc2``; the git tag and the release notes use ``v0.1.0-rc2``. Both
#: forms are correct in their own place, and neither is correct in the other's.
EXPECTED_TAG = "v0.1.0-rc2"

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
ADAPTER_PYPROJECT = REPO_ROOT / "adapters" / "parity_capture" / "pyproject.toml"

_PROJECT_VERSION_RE = re.compile(
    r"^\[project\]$.*?^version\s*=\s*[\"']([^\"']+)[\"']",
    re.MULTILINE | re.DOTALL,
)


def project_version(pyproject: Path) -> str:
    """Return the ``[project].version`` declared in ``pyproject``.

    Deliberately does not use ``tomllib``: that is stdlib only from 3.11 and this project
    supports 3.10, and a version-drift guard that silently skips on one of the supported
    interpreters is most of the way back to having no guard. Where ``tomllib`` is
    available it is used as a cross-check that this regex read the same value.
    """
    text = pyproject.read_text(encoding="utf-8")
    match = _PROJECT_VERSION_RE.search(text)
    assert match is not None, f"{pyproject}: no [project] version found"
    version = match.group(1)

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - only on 3.10
        return version
    parsed = tomllib.loads(text)["project"]["version"]
    assert parsed == version, (
        f"{pyproject}: the TOML parser read {parsed!r} but the release guard read "
        f"{version!r}; the guard's parsing is wrong"
    )
    return version


# -- root package ----------------------------------------------------------------------

def test_the_root_package_declares_the_expected_version():
    assert ivf.__version__ == EXPECTED_VERSION


def test_the_root_pyproject_agrees_with_the_imported_module():
    assert project_version(ROOT_PYPROJECT) == ivf.__version__


def test_the_installed_root_distribution_agrees_with_the_imported_module():
    """Guards the case that actually bit: the artifact CI installs carrying a stale version."""
    import importlib.metadata

    try:
        installed = importlib.metadata.version("ivf")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source-only run
        pytest.skip("ivf is not installed as a distribution in this environment")
    assert installed == ivf.__version__


def test_the_cli_reports_the_same_version():
    result = subprocess.run(
        [sys.executable, "-m", "ivf.cli", "--version"],
        capture_output=True, text=True, check=False,
    )
    output = (result.stdout + result.stderr).strip()
    assert EXPECTED_VERSION in output, output


# -- capture adapter -------------------------------------------------------------------

def test_the_adapter_pyproject_declares_the_expected_version():
    assert project_version(ADAPTER_PYPROJECT) == EXPECTED_VERSION


def test_the_adapter_module_agrees_with_its_pyproject():
    """Read as text rather than imported: the adapter is a separate distribution and is
    not necessarily installed in the environment running the core suite."""
    source = (
        REPO_ROOT / "adapters" / "parity_capture" / "src" / "parity_capture" / "__init__.py"
    ).read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", source, re.MULTILINE)
    assert match is not None, "the adapter declares no __version__"
    assert match.group(1) == project_version(ADAPTER_PYPROJECT)


# -- release notes ---------------------------------------------------------------------

def test_release_notes_exist_for_this_version_and_name_it():
    notes = REPO_ROOT / "docs" / "releases" / f"{EXPECTED_TAG}.md"
    assert notes.is_file(), f"{notes} is missing"
    first_heading = next(
        line for line in notes.read_text(encoding="utf-8").splitlines() if line.startswith("# ")
    )
    assert EXPECTED_TAG in first_heading, first_heading


def test_the_package_form_is_not_used_as_the_tag_form():
    """``0.1.0rc2`` is the package version; ``v0.1.0-rc2`` is the tag. Mixing them produces
    a tag nobody can resolve or a package version PEP 440 rejects."""
    assert "v" + EXPECTED_VERSION.replace("rc", "-rc") == EXPECTED_TAG
    assert "-" not in EXPECTED_VERSION
