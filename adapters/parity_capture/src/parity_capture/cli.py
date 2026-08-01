# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""``parity-capture``: run one Isaac Lab workload and emit a ``trajectory_bundle/v1``.

The command boots the real simulator. There is no mock mode and no dry-run that
pretends to capture, because a capture tool whose happy path can be exercised without a
simulator will eventually be trusted without one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EX_USAGE = 64
EX_UNAVAILABLE = 3
EX_ERROR = 5


def check_prerequisites() -> list[str]:
    """Return the list of missing prerequisites, empty when the capture can run.

    Checked before Kit is launched so a missing simulator costs a second rather than a
    40-second boot followed by an obscure import error.
    """
    missing = []
    for module, label in (("isaacsim", "Isaac Sim"), ("isaaclab", "Isaac Lab"),
                          ("isaaclab_assets", "Isaac Lab assets")):
        try:
            __import__(module)
        except Exception as exc:
            missing.append(f"{label} ({module}) is not importable: {type(exc).__name__}: {exc}")
    try:
        import torch

        if not torch.cuda.is_available():
            missing.append("torch reports no CUDA device")
    except Exception as exc:
        missing.append(f"torch is not importable: {exc}")
    return missing


def launch_app(headless: bool = True):
    """Start the Isaac Lab application. Must happen before any isaaclab.* import."""
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=headless, enable_cameras=False)
    return launcher.app


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report whether this environment can capture."""
    missing = check_prerequisites()
    if missing:
        print("parity-capture: CANNOT capture on this machine")
        for item in missing:
            print(f"  - {item}")
        return EX_UNAVAILABLE
    import importlib.metadata as md

    print("parity-capture: ready")
    for pkg in ("isaacsim", "isaaclab", "isaaclab_physx", "isaaclab_newton", "torch"):
        try:
            print(f"  {pkg:18s} {md.version(pkg)}")
        except Exception:
            print(f"  {pkg:18s} absent")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    """Run one capture end to end."""
    from .spec import SpecError, load_spec

    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        print(f"parity-capture: {args.spec}: {exc}", file=sys.stderr)
        return EX_USAGE

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        print(f"parity-capture: {output} exists and is not empty; refusing to overwrite a capture",
              file=sys.stderr)
        return EX_USAGE

    missing = check_prerequisites()
    if missing:
        print("parity-capture: the Isaac Lab runtime is not available here.", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return EX_UNAVAILABLE

    if spec.defect.any_enabled():
        banner = "!" * 74
        print(banner, file=sys.stderr)
        print("!! THIS CAPTURE HAS A DELIBERATE TEST-ONLY DEFECT INJECTED", file=sys.stderr)
        for line in spec.defect.describe():
            print(f"!!   {line}", file=sys.stderr)
        print("!! It is not a valid baseline for anything.", file=sys.stderr)
        print(banner, file=sys.stderr)

    app = launch_app(headless=not args.gui)
    try:
        from .capture import run_capture

        result = run_capture(spec, output)
    except Exception as exc:
        print(f"parity-capture: capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return EX_ERROR
    finally:
        app.close()

    print()
    print(f"bundle:     {result.bundle_path}")
    print(f"status:     {result.run_status} ({result.captured_steps}/{result.declared_steps} steps)")
    print("schema:     trajectory_bundle/v1")
    print(f"files:      {len(result.checksums)} checksummed, completion marker written last")
    print()
    print("Validate it with IVF (no GPU required for this step):")
    print()
    print("  ivf validate <manifest.yaml>   # with a parity_bundle subject pointing at")
    print(f"                                 # {result.bundle_path}")
    print()
    print("Verify the capture in isolation:")
    print()
    print(f"  python -c \"from ivf.bundle import load_v1; "
          f"print(load_v1('{result.bundle_path}').contract.run_status)\"")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the capture form.

    ``doctor`` is dispatched before argparse rather than as a subparser: a subcommand
    and a bare positional path cannot coexist cleanly in argparse, and the documented
    form is the bare one (``parity-capture <spec> --output <dir>``).
    """
    parser = argparse.ArgumentParser(
        prog="parity-capture",
        description="Run one Isaac Lab workload and emit a trajectory_bundle/v1 capture.",
        epilog="Use `parity-capture doctor` to check prerequisites without booting Kit.",
    )
    parser.add_argument("spec", nargs="?", help="path to a capture spec YAML file")
    parser.add_argument("--output", help="directory the bundle is written to")
    parser.add_argument("--gui", action="store_true", help="run with the Kit UI instead of headless")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        return int(cmd_doctor(argparse.Namespace()))
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.spec or not args.output:
        parser.print_usage(sys.stderr)
        print("parity-capture: a spec path and --output are required", file=sys.stderr)
        return EX_USAGE
    return int(cmd_capture(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
