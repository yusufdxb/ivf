# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause
"""Shared paths, the frozen-IVF guard, and case pairing for the corpus evaluation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO / "research" / "failure_corpus"
DATA = Path(os.environ.get("IVF_CORPUS_DATA", Path.home() / "Projects" / "ivf-corpus-data"))
CAPTURES = DATA / "captures"
RESULTS = RESEARCH / "results"

FROZEN_SRC_TREE = "4932658076919e65c87ecb0e358b3fd3f27336c3"
FROZEN_COMMIT = "d64f74049a7e2b64bba5289f5627f136d71bbe4f"


def assert_ivf_frozen() -> None:
    """Refuse to evaluate if IVF source differs from the frozen tree (committed or not)."""
    head_tree = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD:src/ivf"],
                               capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src/ivf", "uv.lock"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if head_tree != FROZEN_SRC_TREE or dirty:
        raise SystemExit(f"IVF source is not the frozen tree ({head_tree} dirty={bool(dirty)}); refusing")
    import ivf
    if not str(Path(ivf.__file__).resolve()).startswith(str(REPO / "src" / "ivf")):
        raise SystemExit(f"ivf imported from {ivf.__file__}, not this repository")


def load_labels(split: str) -> list[dict[str, Any]]:
    return json.loads((RESEARCH / "corpus" / f"labels_{split}.json").read_text())


def load_result(split: str, capture_id: str) -> dict[str, Any] | None:
    p = CAPTURES / split / capture_id / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def bundle_dir(split: str, capture_id: str) -> Path:
    return CAPTURES / split / capture_id / "bundle"


def reference_for(labels: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    """The PhysX clean reference capture of the same split, platform and seed."""
    for lab in labels:
        if (lab["is_reference"] and lab["platform"] == case["platform"] and lab["seed"] == case["seed"]
                and lab["split"] == case["split"]):
            return lab
    raise KeyError(f"no reference for {case['capture_id']}")


def evaluation_cases(split: str) -> list[dict[str, Any]]:
    """All non-reference cases of a split, each with its reference attached."""
    labels = load_labels(split)
    out = []
    for lab in labels:
        if lab["is_reference"]:
            continue
        case = dict(lab)
        case["reference_id"] = reference_for(labels, lab)["capture_id"]
        out.append(case)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
