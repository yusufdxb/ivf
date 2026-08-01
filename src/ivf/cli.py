# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The ``ivf`` command line.

Six verbs, each of which does one thing and exits with a status a CI job can branch on:

``doctor``     what can this machine actually do
``validate``   run a manifest, seal an evidence bundle, return a verdict
``report``     render or re-render the static report for a bundle
``compare``    diff two evidence bundles, offline
``reproduce``  verify a bundle's checksums and re-run it when the runtime allows
``calibrate``  measure the declared synthetic fault-detectability matrix

Exit codes follow :class:`ivf.verdicts.Verdict`: 0 pass, 1 fail, 2 inconclusive,
3 unsupported, 4 invalid experiment, 5 error. Usage errors exit 64, following the
``sysexits`` convention, so a broken command line is never mistaken for a failed
experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

EX_USAGE = 64


def _resolve_bundle(path_or_id: str, results_root: str) -> Path:
    """Resolve a bundle argument that may be a path or a bare run id."""
    candidate = Path(path_or_id)
    if candidate.is_dir():
        return candidate
    nested = Path(results_root) / path_or_id
    if nested.is_dir():
        return nested
    raise FileNotFoundError(
        f"no evidence bundle at {path_or_id!r} or {nested}; pass a directory or a run id"
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what the environment can do."""
    from .env import render_doctor_text, run_doctor

    report = run_doctor(results_root=args.results_root, examples_dir=args.examples_dir)
    if args.json:
        print(json.dumps(report.to_jsonable(), indent=2, sort_keys=True))
        if args.require and not report.supports(args.require):
            return 1
        return 1 if report.failures else 0
    text, code = render_doctor_text(report, require=args.require)
    print(text, end="")
    return code


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute a manifest and seal an evidence bundle."""
    from .manifest import ManifestError, load_manifest
    from .runner import validate

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"ivf validate: {args.manifest}: {exc}", file=sys.stderr)
        return EX_USAGE

    result = validate(
        manifest, results_root=args.results_root, alpha=args.alpha, seed=args.seed,
        command=["ivf", *sys.argv[1:]],
    )

    print(f"{result.verdict.value}  {manifest.name}  ({result.run_id})")
    for code in result.reason_codes:
        print(f"  {code}")
    if result.error:
        print(f"\n{result.error}", file=sys.stderr)
    for outcome in result.outcomes:
        print(f"  [{outcome.status:>12}] {outcome.name}: {outcome.summary}")
    if result.validity and result.validity.unverifiable:
        print("\n  unverifiable controls (absence of evidence, not evidence of a match):")
        for check in result.validity.unverifiable:
            print(f"    {check.check_id} {check.name}: {check.detail}")
    print(f"\nevidence: {result.bundle_path}")
    print(f"report:   {result.bundle_path / 'report.html'}")
    print(f"rerun:    ivf --results-root {args.results_root} validate {args.manifest}")
    return result.exit_code


def cmd_report(args: argparse.Namespace) -> int:
    """Render (or re-render) the static report for an evidence bundle."""
    from .evidence import EvidenceBundle
    from .report import render_html

    try:
        root = _resolve_bundle(args.bundle, args.results_root)
    except FileNotFoundError as exc:
        print(f"ivf report: {exc}", file=sys.stderr)
        return EX_USAGE
    bundle = EvidenceBundle.open(root)

    if args.output is None:
        path = bundle.root / "report.html"
        if path.is_file():
            print(str(path))
            if args.print_verdict:
                verdict = bundle.verdict
                print(f"{verdict['verdict']}  {verdict.get('experiment')}  ({verdict['run_id']})")
            return 0
        print(f"ivf report: {path} is absent; pass --output to render elsewhere", file=sys.stderr)
        return 1

    html = render_html(
        verdict=bundle.verdict,
        manifest=bundle.read_json("manifest.resolved.json"),
        validity=bundle.read_json("validity.json"),
        outcomes=bundle.read_json("oracles.json"),
        provenance=bundle.read_json("provenance.json"),
        run_id=bundle.run_id,
    )
    Path(args.output).write_text(html, encoding="utf-8")
    print(args.output)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Diff two evidence bundles offline."""
    from .compare import compare_bundles, render_comparison_text
    from .evidence import EvidenceError

    try:
        a = _resolve_bundle(args.baseline, args.results_root)
        b = _resolve_bundle(args.candidate, args.results_root)
    except FileNotFoundError as exc:
        print(f"ivf compare: {exc}", file=sys.stderr)
        return EX_USAGE
    try:
        report = compare_bundles(a, b)
    except EvidenceError as exc:
        print(f"ivf compare: {exc}", file=sys.stderr)
        return 5

    if args.json:
        print(json.dumps(report.to_jsonable(), indent=2, sort_keys=True))
    else:
        print(render_comparison_text(report), end="")
    if not report.comparable:
        return 4
    if args.fail_on_change and report.changed:
        return 1
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Verify a bundle's checksums, then re-run it when the runtime allows."""
    from .evidence import EvidenceBundle

    try:
        root = _resolve_bundle(args.bundle, args.results_root)
    except FileNotFoundError as exc:
        print(f"ivf reproduce: {exc}", file=sys.stderr)
        return EX_USAGE
    bundle = EvidenceBundle.open(root)

    problems = bundle.verify()
    if problems:
        print(f"INTEGRITY FAILED  {bundle.run_id}")
        for problem in problems:
            print(f"  {problem}")
        return 5
    print(f"integrity ok      {bundle.run_id} ({len(bundle._compute_checksums())} files verified)")

    verdict = bundle.verdict
    print(f"recorded verdict  {verdict['verdict']}  {verdict.get('experiment')}")

    if args.verify_only:
        return 0

    manifest_path = bundle.read_json("provenance.json").get("manifest_source_path")
    if manifest_path:
        manifest_path = str(manifest_path).replace("$REPOSITORY_ROOT", str(Path.cwd()))
        manifest_path = manifest_path.replace("$HOME", str(Path.home()))
    if not manifest_path or not Path(manifest_path).is_file():
        print("\ncannot re-execute: the original manifest is not available at "
              f"{manifest_path!r}. The bundle remains fully verifiable and readable offline; "
              "copy manifest.original.yaml out of it to re-run.")
        return 3

    from .manifest import load_manifest
    from .runner import validate

    manifest = load_manifest(manifest_path)
    if manifest.digest() != verdict.get("manifest_digest_sha256"):
        print(f"\nREFUSING to reproduce: {manifest_path} no longer matches the manifest recorded "
              f"in this bundle (digest {manifest.digest()[:16]}… vs "
              f"{str(verdict.get('manifest_digest_sha256'))[:16]}…). Re-running it would compare "
              "a different experiment.")
        return 4

    result = validate(manifest, results_root=args.results_root, command=["ivf", *sys.argv[1:]])
    print(f"\nreproduced run    {result.run_id}: {result.verdict.value}")

    from .compare import compare_bundles, render_comparison_text

    print()
    print(render_comparison_text(compare_bundles(bundle.root, result.bundle_path)), end="")
    if result.verdict.value != verdict["verdict"]:
        print(f"VERDICT CHANGED: recorded {verdict['verdict']}, reproduced {result.verdict.value}")
        return 1
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run the fault-injection calibration campaign and print the detectability matrix."""
    from .calibration import render_matrix_markdown, run_calibration

    matrix = run_calibration(
        manifest_path=args.manifest, results_root=args.results_root, seeds=args.seeds,
        keep_bundles=args.keep_bundles,
    )
    if args.json:
        print(json.dumps(matrix.to_jsonable(), indent=2, sort_keys=True))
    else:
        print(render_matrix_markdown(matrix))
    if args.output:
        Path(args.output).write_text(render_matrix_markdown(matrix), encoding="utf-8")
    return 0 if matrix.consistent else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ivf",
        description="Auditable validation and acceptance for Isaac Lab changes.",
    )
    parser.add_argument("--version", action="version", version=f"ivf {__version__}")
    parser.add_argument(
        "--results-root", default="ivf-results",
        help="directory holding evidence bundles (default: ivf-results)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="report what this environment can validate")
    p_doctor.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_doctor.add_argument(
        "--require", choices=("offline", "synthetic", "isaaclab"),
        help="exit nonzero unless this capability level is available",
    )
    p_doctor.add_argument("--examples-dir", default="validation/examples")
    p_doctor.set_defaults(func=cmd_doctor)

    p_validate = sub.add_parser("validate", help="run an experiment manifest")
    p_validate.add_argument("manifest", help="path to a versioned manifest YAML file")
    p_validate.add_argument("--alpha", type=float, default=0.05,
                            help="family-wise false-alarm budget for statistical oracles")
    p_validate.add_argument("--seed", type=int, default=0, help="seed for bootstrap resampling")
    p_validate.set_defaults(func=cmd_validate)

    p_report = sub.add_parser("report", help="show or re-render a bundle's static report")
    p_report.add_argument("bundle", help="evidence bundle path or run id")
    p_report.add_argument("--output", help="write a freshly rendered report here")
    p_report.add_argument("--print-verdict", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_compare = sub.add_parser("compare", help="diff two evidence bundles (no GPU required)")
    p_compare.add_argument("baseline", help="baseline evidence bundle path or run id")
    p_compare.add_argument("candidate", help="candidate evidence bundle path or run id")
    p_compare.add_argument("--json", action="store_true")
    p_compare.add_argument("--fail-on-change", action="store_true",
                           help="exit 1 when anything material differs")
    p_compare.set_defaults(func=cmd_compare)

    p_repro = sub.add_parser("reproduce", help="verify checksums and re-run an evidence bundle")
    p_repro.add_argument("bundle", help="evidence bundle path or run id")
    p_repro.add_argument("--verify-only", action="store_true",
                         help="check integrity and stop, without re-executing")
    p_repro.set_defaults(func=cmd_reproduce)

    p_cal = sub.add_parser(
        "calibrate", help="run fault injection and print the detectability matrix",
    )
    p_cal.add_argument("--manifest", default="validation/examples/calibration_base.yaml")
    p_cal.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    p_cal.add_argument("--json", action="store_true")
    p_cal.add_argument("--output", help="also write the matrix as markdown here")
    p_cal.add_argument("--keep-bundles", action="store_true",
                       help="keep the evidence bundle for every injected trial")
    p_cal.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # pragma: no cover - depends on the consumer
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
