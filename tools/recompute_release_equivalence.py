#!/usr/bin/env python3
# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Recompute the publicly verifiable half of the release-equivalence record.

``docs/releases/*-equivalence.json`` mixes two very different kinds of statement. Some
are aggregate digests over files that ship in this repository: anyone can recompute them
and either match or not. Others compare the public tree against private pre-sanitization
history that no reader can fetch, so no reader can check them.

Publishing both as bare hashes invites a reader to assume the second kind was verified
the same way as the first. This script exists so the first kind actually is verifiable,
which is what makes labelling the second kind an author attestation honest rather than
convenient.

Usage::

    python tools/recompute_release_equivalence.py                 # print JSON
    python tools/recompute_release_equivalence.py --check         # compare to the record

A record names a *release*, so it is checked against that release's tree, not against
whatever the working branch currently holds::

    git worktree add /tmp/rc1 v0.1.0-rc1
    python tools/recompute_release_equivalence.py --check --root /tmp/rc1

Exit codes: ``0`` recomputed (and matched, under ``--check``), ``1`` a mismatch,
``2`` the recorded file is missing or unreadable.

Algorithm, stated exactly because an aggregate whose definition is implicit is not
reproducible:

* a group is a set of files selected by explicit glob patterns, rooted at the repository
  root;
* paths are made relative to the repository root, encoded as POSIX strings with ``/``
  separators, and sorted bytewise as UTF-8;
* for each file, ``sha256`` is taken over the raw bytes;
* the group digest is ``sha256`` over the concatenation, for each file in sorted order,
  of ``<relative path>`` + ``\\n`` + ``<hex file digest>`` + ``\\n``, UTF-8 encoded;
* a group that selects no files digests the empty string, and is reported with
  ``"n_files": 0`` so an empty group is distinguishable from a missing one rather than
  silently agreeing with another empty group.

Directories, symlinks and anything under ``.git`` are never included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Group name -> glob patterns, relative to the repository root. Each group is one line
#: in the published record. Patterns are explicit rather than "everything under x" so
#: that adding a file to the tree cannot silently change a published aggregate.
GROUPS: dict[str, tuple[str, ...]] = {
    "source_tree": ("src/ivf/**/*.py",),
    "tolerance_registry_tree": ("docs/engineering/tolerance-provenance.md",),
    "example_manifest_tree": ("validation/examples/*.yaml",),
    "raw_capture_trajectory_aggregate": (
        "artifacts/*/trajectories.npz",
        "validation/bundles/*/trajectories.npz",
    ),
    "sealed_signal_array_aggregate": (
        "artifacts/evidence/*/signals/*.npz",
        "validation/evidence/*/signals/*.npz",
    ),
    "sealed_oracle_output_aggregate": (
        "artifacts/evidence/*/oracles.json",
        "validation/evidence/*/oracles.json",
    ),
    "sealed_verdict_output_aggregate": (
        "artifacts/evidence/*/verdict.json",
        "validation/evidence/*/verdict.json",
    ),
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file's raw bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Return the sorted, de-duplicated regular files matched by ``patterns``."""
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and not path.is_symlink() and ".git" not in path.parts:
                seen.add(path)
    return sorted(seen, key=lambda p: p.relative_to(root).as_posix())


def group_digest(root: Path, patterns: tuple[str, ...]) -> tuple[str, list[str]]:
    """Return ``(aggregate digest, relative paths)`` for one group."""
    files = select(root, patterns)
    aggregate = hashlib.sha256()
    names: list[str] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        names.append(rel)
        aggregate.update(f"{rel}\n{sha256_file(path)}\n".encode())
    return aggregate.hexdigest(), names


def recompute(root: Path = REPO_ROOT) -> dict[str, object]:
    """Return the full recomputed record as a JSON-serializable mapping."""
    groups: dict[str, object] = {}
    for name, patterns in sorted(GROUPS.items()):
        digest, names = group_digest(root, patterns)
        groups[name] = {
            "sha256": digest,
            "n_files": len(names),
            "patterns": list(patterns),
            "files": names,
        }
    return {
        "schema": "ivf.release_equivalence.recomputed/v1",
        "algorithm": {
            "path_encoding": "POSIX relative path from the repository root, UTF-8",
            "ordering": "bytewise ascending by relative path",
            "per_file": "sha256 over raw bytes",
            "aggregate": "sha256 over concat(rel_path + '\\n' + file_sha256 + '\\n')",
            "empty_group": "sha256 of the empty string, reported with n_files 0",
            "excluded": ["directories", "symlinks", "anything under .git"],
        },
        "groups": groups,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", metavar="PATH", nargs="?",
                        const="docs/releases/v0.1.0-rc1-equivalence.json",
                        help="compare against the publicly_recomputable block of a record")
    parser.add_argument("--full", action="store_true",
                        help="include the per-group file lists in the printed JSON")
    parser.add_argument("--root", metavar="DIR", default=None,
                        help="recompute against another checkout, e.g. a worktree at the "
                             "tag the record names (default: this checkout)")
    args = parser.parse_args(argv)

    record = recompute(Path(args.root).resolve() if args.root else REPO_ROOT)
    if not args.full:
        for group in record["groups"].values():  # type: ignore[union-attr]
            group.pop("files", None)  # type: ignore[union-attr]

    if not args.check:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    path = REPO_ROOT / args.check
    try:
        published = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.check}: {exc}", file=sys.stderr)
        return 2

    expected = published.get("publicly_recomputable", {}).get("groups", {})
    if not expected:
        print(f"{args.check} records no publicly_recomputable.groups block", file=sys.stderr)
        return 2

    mismatches = []
    for name, group in sorted(record["groups"].items()):  # type: ignore[union-attr]
        want = expected.get(name, {}).get("sha256")
        got = group["sha256"]  # type: ignore[index]
        status = "ok" if want == got else "MISMATCH"
        if want is None:
            status = "MISSING from the record"
        print(f"{status:22s} {name}  {got}")
        if want != got:
            mismatches.append(name)
    if mismatches:
        print(f"\n{len(mismatches)} group(s) do not match the published record: "
              f"{', '.join(mismatches)}", file=sys.stderr)
        return 1
    print(f"\nall {len(expected)} publicly recomputable group(s) match {args.check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
