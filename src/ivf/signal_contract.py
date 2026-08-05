# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Per-signal unit and frame compatibility, enforced before any oracle runs.

The validity layer in :mod:`ivf.validity` judges controls the *manifest asked for*. That
is the right model for experiment design, and the wrong model for physical dimensions: a
comparison of a baseline in radians against a candidate in degrees is not a loose
comparison, it is a meaningless one, and no reviewer should have to remember to request
the check that catches it.

So the checks here are **not opt-in**. Every signal an oracle actually consumes is
verified for unit, frame, dtype and shape agreement between the two subjects, plus
agreement between the gated signal's unit and the unit its tolerance is declared in.

Three rules follow from the same principle:

* a declared mismatch is a **failure**, not a warning, and it vetoes the run;
* a *missing* declaration is **unverifiable**, never "matched". Legacy pre-v1 bundles
  record no per-array units, and absence of evidence must not read as agreement;
* a vector whose components carry different units cannot be reduced to one scalar
  against one single-unit threshold, so that is refused outright rather than averaged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .manifest import Manifest
    from .signals import SignalSet
    from .validity import ValidityCheck

UNIT_MISMATCH = "IVF-CONTROL-SIGNAL-UNIT-MISMATCH"
FRAME_MISMATCH = "IVF-CONTROL-SIGNAL-FRAME-MISMATCH"
DTYPE_MISMATCH = "IVF-CONTROL-SIGNAL-DTYPE-MISMATCH"
TOLERANCE_UNIT_MISMATCH = "IVF-CONTROL-TOLERANCE-UNIT-MISMATCH"
MIXED_UNIT_REDUCTION = "IVF-CONTROL-MIXED-UNIT-REDUCTION-INVALID"

#: Tolerance units that gate a property of the comparison rather than the signal's own
#: physical dimension, so they are never required to equal the signal unit.
DIMENSIONLESS_TOLERANCE_UNITS = frozenset({
    "steps", "fraction", "ulp", "dimensionless", "relative", "ratio", "percent", "count",
})

#: Separators that a capture uses to declare that one array carries more than one unit,
#: e.g. ``"rad and m (revolute and prismatic joints in one array)"``.
_MIXED_UNIT_MARKERS = (" and ", " / ", ", ")

#: Oracle types that reduce a signal to a scalar and compare it against one threshold.
_SCALAR_REDUCING_ORACLES = frozenset({
    "trajectory_equivalence", "statistical_equivalence",
})


def is_mixed_unit(unit: str) -> bool:
    """Return whether ``unit`` declares more than one physical dimension.

    A capture that packs a revolute and a prismatic joint into one array declares
    something like ``"rad and m"``. That is an honest declaration of a heterogeneous
    array, and it is exactly what must not be reduced against a single threshold.
    """
    text = (unit or "").strip().lower()
    if not text:
        return False
    head = text.split("(")[0]
    return any(marker in head for marker in _MIXED_UNIT_MARKERS)


def normalize_unit(unit: str) -> str:
    """Return the comparable core of a declared unit string.

    Captures append prose to units for human readers. The dimension is the part before
    any parenthetical, compared case-insensitively so ``"Rad"`` and ``"rad"`` agree.
    """
    return (unit or "").split("(")[0].strip().lower()


def _signals_consumed(manifest: Manifest) -> dict[str, set[str]]:
    """Map each signal an oracle consumes to the oracle names that consume it.

    Reading the manifest rather than instrumenting the oracles keeps this check ahead of
    execution, which is the whole point: the refusal must happen before a number is
    produced that somebody might quote.
    """
    consumed: dict[str, set[str]] = {}
    for spec in manifest.oracles:
        names: list[str] = []
        for key in ("signal", "signals"):
            value = spec.params.get(key)
            if isinstance(value, str):
                names.append(value)
            elif isinstance(value, (list, tuple)):
                names.extend(str(v) for v in value)
        for name in names:
            consumed.setdefault(name, set()).add(spec.name)
    return consumed


def _contract_of(sset: SignalSet) -> dict[str, dict[str, Any]]:
    """Return the per-array declarations a subject carries, empty when it declares none."""
    contract = sset.metadata.get("signal_contract")
    return contract if isinstance(contract, dict) else {}


def check_signal_contracts(
    manifest: Manifest, baseline: SignalSet, candidate: SignalSet
) -> list[ValidityCheck]:
    """Verify unit, frame, dtype and tolerance compatibility for every consumed signal.

    Returns one :class:`~ivf.validity.ValidityCheck` per signal per property. A ``fail``
    here is a veto: :mod:`ivf.runner` refuses to execute oracles at all, so an
    incompatible comparison never produces a number.
    """
    checks: list[ValidityCheck] = []
    consumed = _signals_consumed(manifest)
    base_contract, cand_contract = _contract_of(baseline), _contract_of(candidate)
    tolerance_units = _tolerance_units_by_signal(manifest)

    for signal in sorted(consumed):
        if signal not in baseline.signals or signal not in candidate.signals:
            # A missing signal is already the oracle layer's error to report.
            continue
        a_spec, b_spec = base_contract.get(signal), cand_contract.get(signal)
        checks.extend(_check_property(signal, a_spec, b_spec, "unit", UNIT_MISMATCH,
                                      normalize=normalize_unit))
        checks.extend(_check_property(signal, a_spec, b_spec, "frame", FRAME_MISMATCH))
        checks.extend(_check_property(signal, a_spec, b_spec, "dtype", DTYPE_MISMATCH))
        checks.extend(_check_tolerance_units(signal, a_spec, tolerance_units.get(signal, [])))
    return checks


def _check_property(
    signal: str, a_spec: dict[str, Any] | None, b_spec: dict[str, Any] | None,
    prop: str, code: str, *, normalize=lambda v: v,
) -> list[ValidityCheck]:
    """Compare one declared property of one signal across the two subjects."""
    from .validity import ValidityCheck

    check_id = f"V-SIGNAL-{prop.upper()}"
    name = f"{signal}: declared {prop} matches"
    a = None if a_spec is None else a_spec.get(prop)
    b = None if b_spec is None else b_spec.get(prop)
    if a in (None, "") or b in (None, ""):
        return [ValidityCheck(
            check_id, name, "unverifiable",
            f"at least one subject does not declare a {prop} for {signal!r}; the signals "
            "cannot be shown to be commensurable and are not treated as matched",
            baseline_value=a, candidate_value=b,
        )]
    if normalize(str(a)) != normalize(str(b)):
        return [ValidityCheck(
            check_id, name, "fail",
            f"baseline declares {a!r} and candidate declares {b!r} for {signal!r}; "
            "comparing them would produce a number with no physical meaning",
            code, baseline_value=a, candidate_value=b,
        )]
    return [ValidityCheck(check_id, name, "pass", f"both {a!r}",
                          baseline_value=a, candidate_value=b)]


def _tolerance_units_by_signal(manifest: Manifest) -> dict[str, list[tuple[str, str, str]]]:
    """Map signal name to ``(oracle name, oracle type, tolerance unit)`` triples."""
    out: dict[str, list[tuple[str, str, str]]] = {}
    for spec in manifest.oracles:
        if spec.tolerance is None:
            continue
        signal = spec.params.get("signal")
        if not isinstance(signal, str):
            continue
        out.setdefault(signal, []).append((spec.name, spec.type, spec.tolerance.unit))
    return out


def _check_tolerance_units(
    signal: str, a_spec: dict[str, Any] | None, tolerances: list[tuple[str, str, str]],
) -> list[ValidityCheck]:
    """Verify each tolerance gating ``signal`` is declared in a compatible unit.

    Two distinct refusals live here. A tolerance in the wrong dimension (``rad`` against a
    signal in ``deg``) is a mis-declared threshold. A *mixed-unit* signal reduced to one
    scalar is worse: no single threshold can be correct for it, whatever unit it names.
    """
    from .validity import ValidityCheck

    checks: list[ValidityCheck] = []
    signal_unit = None if a_spec is None else a_spec.get("unit")
    for oracle_name, oracle_type, tol_unit in tolerances:
        check_id, name = "V-TOLERANCE-UNIT", f"{oracle_name}: tolerance unit suits {signal}"
        if signal_unit in (None, ""):
            checks.append(ValidityCheck(
                check_id, name, "unverifiable",
                f"the baseline declares no unit for {signal!r}, so the tolerance declared in "
                f"{tol_unit!r} cannot be shown to gate the right dimension",
                baseline_value=signal_unit, candidate_value=tol_unit,
            ))
            continue
        if is_mixed_unit(str(signal_unit)) and oracle_type in _SCALAR_REDUCING_ORACLES:
            checks.append(ValidityCheck(
                check_id, f"{oracle_name}: {signal} is reducible to one threshold", "fail",
                f"{signal!r} declares mixed units {signal_unit!r}; reducing it to a single "
                f"scalar against one {tol_unit!r} threshold compares quantities of different "
                "physical dimensions, so no threshold value can be correct",
                MIXED_UNIT_REDUCTION, baseline_value=signal_unit, candidate_value=tol_unit,
            ))
            continue
        if normalize_unit(tol_unit) in DIMENSIONLESS_TOLERANCE_UNITS:
            checks.append(ValidityCheck(
                check_id, name, "pass",
                f"tolerance unit {tol_unit!r} gates a property of the comparison, not the "
                f"dimension of {signal!r}",
                baseline_value=signal_unit, candidate_value=tol_unit,
            ))
            continue
        if normalize_unit(tol_unit) != normalize_unit(str(signal_unit)):
            checks.append(ValidityCheck(
                check_id, name, "fail",
                f"{signal!r} is declared in {signal_unit!r} but oracle {oracle_name!r} gates it "
                f"with a tolerance declared in {tol_unit!r}",
                TOLERANCE_UNIT_MISMATCH, baseline_value=signal_unit, candidate_value=tol_unit,
            ))
        else:
            checks.append(ValidityCheck(
                check_id, name, "pass", f"both {tol_unit!r}",
                baseline_value=signal_unit, candidate_value=tol_unit,
            ))
    return checks
