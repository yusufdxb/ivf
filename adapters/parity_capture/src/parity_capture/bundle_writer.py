# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Write the finalized files of the versioned ``trajectory_bundle/v1`` format.

This module deliberately does not import IVF. The producer and consumer share the file
contract, not a Python API. Keeping this writer small also makes byte-level review of the
handoff practical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "trajectory_bundle/v1"
RUN_STATUSES = frozenset({"completed", "partial", "failed"})


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_v1(root: str | Path, *, run_status: str) -> dict[str, str]:
    """Write checksums and then the completion marker, in normative v1 order."""
    if run_status not in RUN_STATUSES:
        raise ValueError(f"run_status {run_status!r} is not one of {sorted(RUN_STATUSES)}")
    path = Path(root)
    checksums = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name not in ("CHECKSUMS.sha256", "COMPLETE")
    }
    if not checksums:
        raise ValueError(f"{path}: nothing to checksum")
    checksum_path = path / "CHECKSUMS.sha256"
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    marker = {
        "schema": SCHEMA,
        "run_status": run_status,
        "n_files": len(checksums),
        "checksums_sha256": sha256_file(checksum_path),
    }
    (path / "COMPLETE").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checksums
