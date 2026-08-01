# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Manifest parsing, normalization and hashing."""

from __future__ import annotations

import pytest

from ivf.manifest import (
    SCHEMA_VERSION,
    ManifestError,
    Tolerance,
    load_manifest,
    parse_manifest,
)

from .conftest import EXAMPLES, MINIMAL_MANIFEST

pytestmark = pytest.mark.unit


def test_minimal_manifest_parses(minimal_manifest):
    assert minimal_manifest.name == "minimal"
    assert minimal_manifest.schema_version == SCHEMA_VERSION
    assert set(minimal_manifest.subjects) == {"baseline", "candidate"}
    assert minimal_manifest.workload.steps == 50


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_examples_are_valid(path):
    """Every manifest we ship must parse. A broken example is a broken quickstart."""
    manifest = load_manifest(path)
    assert manifest.oracles


def test_bare_float_tolerance_is_rejected():
    """A naked epsilon is the failure mode this schema exists to prevent."""
    text = MINIMAL_MANIFEST.replace(
        "    check: finite_state",
        "    check: finite_state\n    tolerance: 0.002",
    )
    with pytest.raises(ManifestError, match="bare number is not a tolerance"):
        parse_manifest(text)


@pytest.mark.parametrize("missing", ["unit", "scope", "rationale", "aggregation", "min_samples", "kind"])
def test_tolerance_requires_every_field(missing):
    raw = {
        "value": 0.002, "unit": "rad", "scope": "per step",
        "rationale": "a sufficiently long justification",
        "aggregation": "max", "min_samples": 10, "kind": "numerical",
    }
    raw.pop(missing)
    with pytest.raises(ManifestError, match="missing required field"):
        Tolerance.parse(raw, "oracles[0].tolerance")


def test_tolerance_rejects_a_stub_rationale():
    raw = {
        "value": 0.002, "unit": "rad", "scope": "per step", "rationale": "because",
        "aggregation": "max", "min_samples": 10, "kind": "numerical",
    }
    with pytest.raises(ManifestError, match="too short to be a rationale"):
        Tolerance.parse(raw, "t")


def test_statistical_tolerance_requires_a_minimum_meaningful_effect():
    """Without one, a large sample turns any difference into a failure."""
    raw = {
        "value": 0.002, "unit": "rad", "scope": "per env",
        "rationale": "a sufficiently long justification",
        "aggregation": "mean", "min_samples": 10, "kind": "statistical",
    }
    with pytest.raises(ManifestError, match="minimum_meaningful_effect"):
        Tolerance.parse(raw, "t")


def test_unknown_control_is_rejected():
    """Silently ignoring a control the user asked for would be the worst outcome."""
    text = MINIMAL_MANIFEST.replace("require_same: [asset_identity", "require_same: [gravity_vector")
    with pytest.raises(ManifestError, match="unknown control"):
        parse_manifest(text)


def test_contradictory_controls_are_rejected():
    text = MINIMAL_MANIFEST.replace(
        "  require_same: [asset_identity, observation_definition, control_frequency, num_envs, horizon]",
        "  require_same: [asset_identity]\n  allow_different: [asset_identity]",
    )
    with pytest.raises(ManifestError, match="more than one control partition"):
        parse_manifest(text)


def test_control_block_rejects_unknown_keys_instead_of_dropping_them():
    text = MINIMAL_MANIFEST.replace(
        "  require_same: [asset_identity, observation_definition, control_frequency, num_envs, horizon]",
        "  require_same: [asset_identity]\n  silently_ignored: [reset_semantics]",
    )
    with pytest.raises(ManifestError, match="controls: unknown key"):
        parse_manifest(text)


def test_unsupported_controls_are_preserved_and_cannot_overlap():
    text = MINIMAL_MANIFEST.replace(
        "  require_same: [asset_identity, observation_definition, control_frequency, num_envs, horizon]",
        "  require_same: [asset_identity]\n"
        "  unsupported_or_unverifiable: [frame_convention, asset_binary_identity]",
    )
    manifest = parse_manifest(text)
    assert manifest.controls.unsupported_or_unverifiable == (
        "frame_convention",
        "asset_binary_identity",
    )
    assert manifest.to_jsonable()["controls"]["unsupported_or_unverifiable"] == [
        "frame_convention",
        "asset_binary_identity",
    ]

    overlap = text.replace(
        "unsupported_or_unverifiable: [frame_convention, asset_binary_identity]",
        "unsupported_or_unverifiable: [asset_identity]",
    )
    with pytest.raises(ManifestError, match="more than one control partition"):
        parse_manifest(overlap)


def test_unverifiable_only_control_cannot_be_required():
    text = MINIMAL_MANIFEST.replace(
        "  require_same: [asset_identity, observation_definition, control_frequency, num_envs, horizon]",
        "  require_same: [asset_binary_identity]",
    )
    with pytest.raises(ManifestError, match="only appear in unsupported_or_unverifiable"):
        parse_manifest(text)


def test_unit_quaternion_has_no_unexplained_fallback_epsilon():
    text = MINIMAL_MANIFEST + """  - type: invariant
    name: unit_quaternion
    check: unit_quaternion
    signals: [pole_quat]
"""
    with pytest.raises(ManifestError, match="no implicit epsilon"):
        parse_manifest(text)


@pytest.mark.parametrize(
    "text,pattern",
    [
        ("not: a manifest", "schema_version"),
        ("schema_version: ivf.validation/v99\nname: x", "not supported"),
        ("[]", "expected a mapping"),
        ("key: [unclosed", "not valid YAML"),
    ],
)
def test_malformed_input_fails_with_a_useful_message(text, pattern):
    with pytest.raises(ManifestError, match=pattern):
        parse_manifest(text)


def test_missing_oracles_is_rejected():
    text = MINIMAL_MANIFEST.split("oracles:")[0]
    with pytest.raises(ManifestError, match="at least one oracle"):
        parse_manifest(text)


def test_unknown_top_level_key_is_rejected():
    """A typo in a top-level key must not be silently dropped."""
    with pytest.raises(ManifestError, match="unknown key"):
        parse_manifest(MINIMAL_MANIFEST + "\nverdict_polcy: {}\n")


def test_warmup_must_be_shorter_than_the_run():
    text = MINIMAL_MANIFEST.replace("warmup_steps: 0", "warmup_steps: 500")
    with pytest.raises(ManifestError, match="must be smaller"):
        parse_manifest(text)


def test_duplicate_oracle_names_are_rejected():
    text = MINIMAL_MANIFEST + """  - type: invariant
    name: finite_state
    check: finite_state
"""
    with pytest.raises(ManifestError, match="duplicate oracle name"):
        parse_manifest(text)


def test_canonicalization_is_insensitive_to_formatting(minimal_manifest):
    """Two manifests that mean the same thing must hash the same.

    Key order, quoting style and explicitly-spelled defaults are formatting, not
    meaning, and a digest that moves with formatting is useless for provenance.
    """
    reformatted = MINIMAL_MANIFEST.replace(
        "name: minimal", 'name: "minimal"'
    ).replace(
        "  seeds: [3]", "  seeds:\n    - 3"
    ) + "\nverdict_policy:\n  invalid_experiment_on_control_violation: true\n"
    assert parse_manifest(reformatted).digest() == minimal_manifest.digest()


def test_digest_changes_when_a_tolerance_changes():
    text = MINIMAL_MANIFEST + """  - type: trajectory_equivalence
    name: x_agreement
    signal: x
    tolerance:
      value: 0.002
      unit: rad
      scope: per step
      rationale: a sufficiently long justification
      aggregation: max
      min_samples: 10
      kind: numerical
"""
    a = parse_manifest(text)
    b = parse_manifest(text.replace("value: 0.002", "value: 0.003"))
    assert a.digest() != b.digest()


def test_defaults_are_materialized_in_the_resolved_form(minimal_manifest):
    """The evidence bundle must record what ran, not what the author left implicit."""
    resolved = minimal_manifest.to_jsonable()
    assert resolved["verdict_policy"]["invalid_experiment_on_control_violation"] is True
    assert resolved["workload"]["seeds"] == [3]
