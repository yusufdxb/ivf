# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end CLI behaviour, including exit codes CI depends on."""

from __future__ import annotations

import json

import pytest

from ivf.cli import main

from .conftest import EXAMPLES, REPO_ROOT

pytestmark = pytest.mark.integration


def cli(*args, results_root=None):
    """Invoke the CLI in-process and return its exit code."""
    argv = list(args)
    if results_root is not None:
        argv = ["--results-root", str(results_root), *argv]
    return main(argv)


def latest(results_root, needle=""):
    """Return the newest bundle directory whose name contains ``needle``."""
    matches = sorted((p for p in results_root.iterdir() if needle in p.name),
                     key=lambda p: p.name)
    assert matches, f"no bundle matching {needle!r} in {results_root}"
    return matches[-1]


def test_doctor_reports_the_offline_capability(capsys, results_root):
    assert cli("doctor", results_root=results_root) == 0
    out = capsys.readouterr().out
    assert "Capabilities" in out
    assert "offline" in out


def test_doctor_json_is_machine_readable(capsys, results_root):
    cli("doctor", "--json", results_root=results_root)
    payload = json.loads(capsys.readouterr().out)
    assert payload["capabilities"]["offline"] is True
    assert payload["compatibility_matrix"]


def test_doctor_fails_clearly_when_a_required_capability_is_absent(capsys, results_root):
    """The environment cannot produce a live Isaac Lab experiment here, and must say so."""
    code = cli("doctor", "--require", "isaaclab", results_root=results_root)
    out = capsys.readouterr().out
    assert code == 1
    assert "REQUIRED capability 'isaaclab' is NOT available" in out


def test_validate_fails_on_the_injected_defect(capsys, results_root):
    code = cli("validate", str(EXAMPLES / "synthetic_fault.yaml"), results_root=results_root)
    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("FAIL")
    assert "IVF-ORACLE-NON_EQUIVALENT" in out
    assert "first violation at step" in out


def test_validate_passes_once_the_defect_is_removed(capsys, results_root):
    """The negative control for the whole validator."""
    code = cli("validate", str(EXAMPLES / "synthetic_fixed.yaml"), results_root=results_root)
    assert code == 0
    assert capsys.readouterr().out.startswith("PASS")


def test_validate_on_real_cross_backend_bundles_without_isaac_lab(capsys, results_root):
    code = cli("validate", str(EXAMPLES / "cartpole_cross_backend.yaml"), results_root=results_root)
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("INCONCLUSIVE")
    assert "unverifiable controls" in out
    assert "solver" in out


def test_validate_rejects_a_malformed_manifest(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: ivf.validation/v1\nname: x\n")
    code = cli("validate", str(bad), results_root=tmp_path)
    assert code == 64
    assert "subjects" in capsys.readouterr().err


def test_validate_reports_a_missing_manifest_as_a_usage_error(tmp_path, capsys):
    assert cli("validate", str(tmp_path / "nope.yaml"), results_root=tmp_path) == 64


def test_report_prints_the_bundled_report_path(capsys, results_root):
    cli("validate", str(EXAMPLES / "synthetic_fixed.yaml"), results_root=results_root)
    capsys.readouterr()
    run_id = latest(results_root, "fixed").name
    assert cli("report", run_id, "--print-verdict", results_root=results_root) == 0
    out = capsys.readouterr().out
    assert "report.html" in out
    assert "PASS" in out


def test_report_can_rerender_from_a_sealed_bundle(tmp_path, results_root, capsys):
    cli("validate", str(EXAMPLES / "synthetic_fault.yaml"), results_root=results_root)
    capsys.readouterr()
    out_path = tmp_path / "rendered.html"
    assert cli("report", latest(results_root).name, "--output", str(out_path),
               results_root=results_root) == 0
    html = out_path.read_text()
    assert "<title>" in html and "FAIL" in html
    assert "http://" not in html and "https://" not in html  # self-contained, no external assets


def test_compare_diffs_two_runs_offline(capsys, results_root):
    cli("validate", str(EXAMPLES / "synthetic_fixed.yaml"), results_root=results_root)
    cli("validate", str(EXAMPLES / "synthetic_fixed.yaml"), results_root=results_root)
    capsys.readouterr()
    runs = sorted(p.name for p in results_root.iterdir())
    code = cli("compare", runs[0], runs[1], results_root=results_root)
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS == PASS" in out


def test_compare_refuses_two_unrelated_experiments(capsys, results_root):
    cli("validate", str(EXAMPLES / "synthetic_fixed.yaml"), results_root=results_root)
    cli("validate", str(EXAMPLES / "synthetic_fault.yaml"), results_root=results_root)
    capsys.readouterr()
    code = cli("compare", latest(results_root, "fixed").name, latest(results_root, "defect").name,
               results_root=results_root)
    out = capsys.readouterr().out
    assert code == 4
    assert "NOT COMPARABLE" in out


@pytest.mark.reproduction
def test_reproduce_verifies_and_reruns(capsys, results_root):
    cli("validate", str(EXAMPLES / "synthetic_fault.yaml"), results_root=results_root)
    capsys.readouterr()
    code = cli("reproduce", latest(results_root, "defect").name, results_root=results_root)
    out = capsys.readouterr().out
    assert code == 0
    assert "integrity verified against included seal" in out
    assert "No material differences." in out


@pytest.mark.reproduction
def test_reproduce_verify_only_needs_no_runtime(capsys, results_root):
    cli("validate", str(EXAMPLES / "synthetic_fixed.yaml"), results_root=results_root)
    capsys.readouterr()
    assert cli("reproduce", latest(results_root).name, "--verify-only",
               results_root=results_root) == 0


@pytest.mark.reproduction
def test_reproduce_refuses_when_the_manifest_has_moved_on(tmp_path, capsys, results_root):
    manifest = tmp_path / "m.yaml"
    manifest.write_text((EXAMPLES / "synthetic_fixed.yaml").read_text())
    cli("validate", str(manifest), results_root=results_root)
    capsys.readouterr()
    manifest.write_text(manifest.read_text().replace("value: 2.0e-3", "value: 9.0e-3"))
    code = cli("reproduce", latest(results_root).name, results_root=results_root)
    out = capsys.readouterr().out
    assert code == 4
    assert "REFUSING to reproduce" in out


def test_unknown_bundle_is_a_usage_error(capsys, results_root):
    results_root.mkdir(parents=True, exist_ok=True)
    assert cli("report", "no-such-run", results_root=results_root) == 64


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "ivf" in capsys.readouterr().out


# -- seal threat model -----------------------------------------------------------------

@pytest.mark.integration
def test_the_seal_detects_uncoordinated_change_but_not_a_coordinated_reseal(tmp_path):
    """The documented threat-model boundary, asserted rather than only written down.

    An unkeyed digest over content the editor controls cannot establish authorship. The
    first four cases are what the seal is for; the fifth is what it is not for, and it is
    tested so the limitation cannot quietly stop being true.
    """
    import json as _json
    import shutil as _shutil

    from ivf.evidence import EvidenceBundle

    source = next(iter((REPO_ROOT / "artifacts" / "evidence").iterdir()))

    def fresh(name):
        dst = tmp_path / name
        _shutil.copytree(source, dst)
        for path in dst.rglob("*"):
            if path.is_file():
                path.chmod(0o644)
        return dst

    # 1. a naive content edit
    edited = fresh("edited")
    (edited / "verdict.json").write_text(
        (edited / "verdict.json").read_text() + "\n", encoding="utf-8"
    )
    assert EvidenceBundle.open(edited).verify()

    # 2. a deleted file
    deleted = fresh("deleted")
    (deleted / "divergence.jsonl").unlink()
    assert EvidenceBundle.open(deleted).verify()

    # 3. an unrecorded added file
    added = fresh("added")
    (added / "extra.json").write_text("{}", encoding="utf-8")
    assert EvidenceBundle.open(added).verify()

    # 4. an incompatible schema major is refused at open, before verification
    from ivf.evidence import EvidenceError

    schema = fresh("schema")
    seal = _json.loads((schema / "SEAL.json").read_text())
    seal["evidence_schema_version"] = "ivf.evidence/v99"
    (schema / "SEAL.json").write_text(_json.dumps(seal, indent=2, sort_keys=True) + "\n")
    with pytest.raises(EvidenceError, match="not compatible"):
        EvidenceBundle.open(schema)

    # 5. a coordinated edit plus reseal verifies as internally consistent. This is the
    #    boundary: the seal establishes integrity, never authenticity.
    resealed = fresh("resealed")
    (resealed / "verdict.json").write_text(
        (resealed / "verdict.json").read_text() + "\n", encoding="utf-8"
    )
    bundle = EvidenceBundle.open(resealed)
    lines = [
        f"{digest}  {name}"
        for name, digest in sorted(bundle._compute_checksums().items())
    ]
    (resealed / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from ivf.evidence import sha256_file

    seal = _json.loads((resealed / "SEAL.json").read_text())
    seal["checksums_sha256"] = sha256_file(resealed / "CHECKSUMS.sha256")
    seal["n_files"] = len(lines)
    (resealed / "SEAL.json").write_text(_json.dumps(seal, indent=2, sort_keys=True) + "\n")
    assert EvidenceBundle.open(resealed).verify() == [], (
        "a coordinated edit plus reseal is internally consistent by construction; "
        "this is the documented limit of an unkeyed seal"
    )
