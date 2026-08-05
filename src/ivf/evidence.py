# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The evidence bundle: a finalized, checksummed, offline-readable record of one run.

An evidence bundle is what makes a verdict auditable six months later on a different
machine. Its properties are deliberate:

* **Readable without Isaac Lab, without a GPU, and without IVF itself.** Everything is
  JSON, YAML, plain text, or ``.npz``. A reviewer with Python and numpy can check the
  numbers by hand.
* **Immutable after finalization.** Finalizing writes a checksum manifest, seals it with
  a digest of the checksum file, and drops write permission on every file. Editing a
  bundle after the fact is detectable, which is the whole point.
* **Explicitly versioned.** A bundle written by an incompatible schema is refused with
  a clear message rather than half-parsed.

Layout::

    ivf-results/<run_id>/
      manifest.original.yaml     the file the user wrote, byte for byte
      manifest.resolved.json     defaults materialized; this is what was hashed
      verdict.json               verdict, reason codes, identity, timings
      validity.json              every experiment-validity check
      oracles.json               every oracle outcome, with metrics and tolerances
      divergence.jsonl           one localization record per failing oracle
      provenance.json            versions, hardware, env vars, exact command
      signals/{baseline,candidate}.npz
      report.html                the static report
      reproduce.sh               the exact rerun command
      CHECKSUMS.sha256           sha256 of every file above
      SEAL.json                  schema version + digest of CHECKSUMS.sha256
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EVIDENCE_SCHEMA_VERSION = "ivf.evidence/v1"
"""Evidence-bundle schema identifier."""

CHECKSUM_FILE = "CHECKSUMS.sha256"
SEAL_FILE = "SEAL.json"
_EXCLUDED_FROM_CHECKSUMS = frozenset({CHECKSUM_FILE, SEAL_FILE})


class EvidenceError(RuntimeError):
    """Raised when an evidence bundle cannot be written, read, or verified."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, streamed so large payloads do not load fully."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class EvidenceBundle:
    """A handle to an evidence bundle directory."""

    root: Path
    run_id: str

    @classmethod
    def create(cls, results_root: str | Path, run_id: str) -> EvidenceBundle:
        """Create a new, unsealed bundle directory."""
        root = Path(results_root) / run_id
        if root.exists():
            raise EvidenceError(f"{root} already exists; refusing to overwrite an evidence bundle")
        (root / "signals").mkdir(parents=True)
        return cls(root=root, run_id=run_id)

    @classmethod
    def open(cls, path: str | Path) -> EvidenceBundle:
        """Open an existing bundle, refusing incompatible schema versions."""
        root = Path(path)
        if not root.is_dir():
            raise EvidenceError(f"{root}: not a directory")
        seal_path = root / SEAL_FILE
        if seal_path.is_file():
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            version = seal.get("evidence_schema_version", "")
            if _major(version) != _major(EVIDENCE_SCHEMA_VERSION):
                raise EvidenceError(
                    f"{root}: evidence schema {version!r} is not compatible with this IVF build "
                    f"({EVIDENCE_SCHEMA_VERSION!r}). No migration path exists; read it with the "
                    "IVF version that wrote it."
                )
        elif not (root / "verdict.json").is_file():
            raise EvidenceError(f"{root}: not an IVF evidence bundle (no verdict.json)")
        return cls(root=root, run_id=root.name)

    # -- writing -----------------------------------------------------------------------

    def write_text(self, name: str, text: str) -> Path:
        """Write a text file into the bundle."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        """Write a JSON file with stable key ordering."""
        return self.write_text(name, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    def write_jsonl(self, name: str, rows: list[Any]) -> Path:
        """Write a JSON Lines file."""
        body = "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
        return self.write_text(name, body)

    def write_signals(self, role: str, signals: dict[str, np.ndarray]) -> Path:
        """Write one subject's signal arrays, compressed."""
        path = self.root / "signals" / f"{role}.npz"
        np.savez_compressed(path, **signals)
        return path

    # -- finalization ------------------------------------------------------------------

    def finalize(self) -> dict[str, str]:
        """Write the checksum manifest and seal, then make every file read-only.

        Returns the mapping of relative path to digest. Idempotence is not offered on
        purpose: finalizing twice means something wrote to a sealed bundle.
        """
        if (self.root / SEAL_FILE).exists():
            raise EvidenceError(f"{self.root}: already finalized")
        checksums = self._compute_checksums()
        lines = [f"{digest}  {name}" for name, digest in sorted(checksums.items())]
        (self.root / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
        seal = {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "n_files": len(checksums),
            "checksums_sha256": sha256_file(self.root / CHECKSUM_FILE),
        }
        (self.root / SEAL_FILE).write_text(
            json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._make_read_only()
        return checksums

    def _compute_checksums(self) -> dict[str, str]:
        """Return relative-path to digest for every file except the seal files."""
        out: dict[str, str] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            if rel in _EXCLUDED_FROM_CHECKSUMS:
                continue
            out[rel] = sha256_file(path)
        return out

    def _make_read_only(self) -> None:
        """Drop write permission on every file in the bundle."""
        for path in self.root.rglob("*"):
            if path.is_file():
                mode = path.stat().st_mode
                path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    # -- verification ------------------------------------------------------------------

    def verify(self) -> list[str]:
        """Return a list of integrity problems; empty means the bundle is intact.

        Checks the seal, then every recorded checksum, then looks for files that appeared
        after finalization. All three matter: an uncoordinated edit usually gains a file or
        loses one rather than editing in place.
        """
        problems: list[str] = []
        seal_path = self.root / SEAL_FILE
        checksum_path = self.root / CHECKSUM_FILE
        if not seal_path.is_file():
            return [f"{SEAL_FILE} is missing: the bundle was never finalized"]
        if not checksum_path.is_file():
            return [f"{CHECKSUM_FILE} is missing but {SEAL_FILE} is present: the bundle is broken"]

        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if _major(seal.get("evidence_schema_version", "")) != _major(EVIDENCE_SCHEMA_VERSION):
            problems.append(
                f"evidence schema {seal.get('evidence_schema_version')!r} is incompatible with "
                f"{EVIDENCE_SCHEMA_VERSION!r}"
            )
        actual_seal = sha256_file(checksum_path)
        if actual_seal != seal.get("checksums_sha256"):
            problems.append(f"{CHECKSUM_FILE} does not match the digest recorded in {SEAL_FILE}")

        recorded: dict[str, str] = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            recorded[name] = digest

        for name, digest in sorted(recorded.items()):
            path = self.root / name
            if not path.is_file():
                problems.append(f"{name}: recorded in {CHECKSUM_FILE} but missing from the bundle")
                continue
            actual = sha256_file(path)
            if actual != digest:
                problems.append(f"{name}: checksum mismatch (recorded {digest[:16]}…, found {actual[:16]}…)")

        present = {
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file() and p.relative_to(self.root).as_posix() not in _EXCLUDED_FROM_CHECKSUMS
        }
        for extra in sorted(present - set(recorded)):
            problems.append(f"{extra}: present in the bundle but not recorded in {CHECKSUM_FILE}")
        return problems

    # -- reading -----------------------------------------------------------------------

    def read_json(self, name: str) -> Any:
        """Read a JSON file from the bundle."""
        path = self.root / name
        if not path.is_file():
            raise EvidenceError(f"{self.root}: missing {name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_jsonl(self, name: str) -> list[Any]:
        """Read a JSON Lines file from the bundle, returning ``[]`` when absent."""
        path = self.root / name
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def read_signals(self, role: str) -> dict[str, np.ndarray]:
        """Read one subject's signal arrays."""
        path = self.root / "signals" / f"{role}.npz"
        if not path.is_file():
            raise EvidenceError(f"{self.root}: missing signals/{role}.npz")
        with np.load(path) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}

    @property
    def verdict(self) -> dict[str, Any]:
        """The parsed ``verdict.json``."""
        return self.read_json("verdict.json")

    @property
    def is_sealed(self) -> bool:
        """Whether the bundle has been finalized."""
        return (self.root / SEAL_FILE).is_file()


def _major(schema: str) -> str:
    """Return the major part of a ``family/vN`` schema identifier."""
    return schema.split("/")[-1].split(".")[0] if schema else ""


def unseal(root: str | Path) -> None:
    """Restore write permission on a bundle so it can be deleted.

    Not part of the normal workflow. Provided because a read-only tree is otherwise
    awkward to remove, and because a hidden ``chmod`` inside a delete path would be
    worse than an explicit function.
    """
    for path in Path(root).rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    os.chmod(root, os.stat(root).st_mode | stat.S_IWUSR)
