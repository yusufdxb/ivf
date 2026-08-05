# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""The release-equivalence record must be recomputable, not just published."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD = REPO_ROOT / "docs" / "releases" / "v0.1.0-rc1-equivalence.json"
sys.path.insert(0, str(REPO_ROOT / "tools"))


@pytest.fixture(scope="module")
def record():
    if not RECORD.is_file():  # pragma: no cover - depends on checkout
        pytest.skip("release record absent")
    return json.loads(RECORD.read_text())


@pytest.mark.integration
def test_every_published_aggregate_recomputes_from_this_clone(record):
    """The whole point of the record: a reader can check it without trusting the author."""
    from recompute_release_equivalence import main

    assert main(["--check", "docs/releases/v0.1.0-rc1-equivalence.json"]) == 0


def test_the_recomputation_is_deterministic():
    from recompute_release_equivalence import recompute

    assert recompute() == recompute()


def test_private_history_claims_are_labelled_as_attestations(record):
    """A claim a reader cannot check must say so, next to the claim."""
    assert record["author_attested"]["status"] == "author_attested"
    assert "not independently verifiable" in record["author_attested"]["meaning"]


def test_no_unreachable_object_ids_are_published(record):
    """Decorative hashes naming private history are worse than no hashes at all.

    A reader who sees a hex id assumes they could resolve it. These could not be
    resolved, compared or falsified from anything public, so they were removed rather
    than kept for the appearance of rigour.
    """
    import re

    # Aggregates under publicly_recomputable are 64-char digests and are recomputable.
    # Anything 7 to 40 hex characters long is a git object id from unpublished history.
    blob = json.dumps(record.get("author_attested", {}))
    offenders = re.findall(r"\b[0-9a-f]{7,40}\b", blob)
    assert not offenders, f"unreachable object ids published: {sorted(set(offenders))}"
