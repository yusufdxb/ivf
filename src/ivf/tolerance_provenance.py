# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Derive and render the tolerance registry shipped with IVF.

The registry is data, not executable policy. This module deliberately supports only the
four small derivations used by the shipped exemplars so the arithmetic is reviewable and
tests do not duplicate formulas from prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = frozenset(
    {
        "name",
        "value",
        "unit",
        "signal",
        "scope",
        "source_measurement",
        "formula",
        "formula_inputs",
        "assumptions",
        "aggregation",
        "minimum_samples",
        "confidence_or_uncertainty",
        "engineering_interpretation",
        "limitations",
    }
)


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    """Load a tolerance registry and reject incomplete or duplicate records."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tolerances"), list):
        raise ValueError("tolerance registry must contain a 'tolerances' list")
    records = payload["tolerances"]
    names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"tolerances[{index}] must be a mapping")
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"{record.get('name', index)} missing fields: {sorted(missing)}")
        name = str(record["name"])
        if name in names:
            raise ValueError(f"duplicate tolerance provenance name: {name}")
        names.add(name)
    return records


def derive_value(record: dict[str, Any]) -> float:
    """Evaluate the record's declared arithmetic from its named inputs."""
    derivation = record.get("derivation")
    if not isinstance(derivation, dict):
        raise ValueError(f"{record.get('name')}: derivation must be a mapping")
    operation = derivation.get("operation")
    inputs = record["formula_inputs"]

    def value(name: str) -> float:
        item = inputs.get(name)
        if not isinstance(item, dict) or "value" not in item:
            raise ValueError(f"{record.get('name')}: formula input {name!r} has no value")
        return float(item["value"])

    if operation == "identity":
        return value(str(derivation["input"]))
    if operation == "multiply":
        result = 1.0
        for name in derivation["inputs"]:
            result *= value(str(name))
        return result
    if operation == "divide":
        return value(str(derivation["numerator"])) / value(str(derivation["denominator"]))
    if operation == "exact":
        return float(record["value"])
    raise ValueError(f"{record.get('name')}: unsupported derivation operation {operation!r}")


def render_markdown(records: list[dict[str, Any]]) -> str:
    """Render the compact human index that is checked into the engineering docs."""
    lines = [
        "| Provenance record | Value | Signal | Formula |",
        "|---|---:|---|---|",
    ]
    for record in records:
        value = format(float(record["value"]), ".12g")
        formula = str(record["formula"]).replace("|", "\\|")
        lines.append(
            f"| `{record['name']}` | {value} {record['unit']} | `{record['signal']}` | {formula} |"
        )
    return "\n".join(lines) + "\n"
