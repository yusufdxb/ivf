# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Typed verdicts and the stable reason-code vocabulary.

A verdict answers "can I accept this change for my workload", and the reason codes
say why. Two rules are load-bearing and are enforced here rather than left to
convention:

1. **Infrastructure failure is never a scientific failure.** A crashed run, an
   unreadable bundle, or a missing runtime yields :data:`Verdict.ERROR` or
   :data:`Verdict.UNSUPPORTED`, never :data:`Verdict.FAIL`.
2. **An uninterpretable experiment outranks its own results.** If the experiment
   validity layer rejects the comparison, the verdict is
   :data:`Verdict.INVALID_EXPERIMENT` and oracle outcomes are reported but do not
   decide anything.

Reason codes are part of the public contract: they are what a CI job greps for and
what :mod:`ivf.compare` diffs across runs. Renaming one is a breaking change.
"""

from __future__ import annotations

import enum


class Verdict(str, enum.Enum):
    """Run-level outcome of a validation."""

    PASS = "PASS"
    """Every declared control held and every acceptance criterion was met."""

    FAIL = "FAIL"
    """At least one declared acceptance criterion failed on adequate evidence."""

    INCONCLUSIVE = "INCONCLUSIVE"
    """The run completed but the evidence cannot decide: too few samples, or the
    uncertainty interval straddles the decision boundary."""

    UNSUPPORTED = "UNSUPPORTED"
    """The requested combination relies on a backend or runtime feature that is not
    available. Not a defect in the subject under test."""

    INVALID_EXPERIMENT = "INVALID_EXPERIMENT"
    """A control was violated or the protocol was broken, so no comparison of the
    results would be interpretable."""

    ERROR = "ERROR"
    """Infrastructure failed before a scientific verdict was possible."""

    @property
    def exit_code(self) -> int:
        """Return the CI exit status for this verdict.

        ``0`` means "safe to accept". Everything else is distinguishable so a CI job
        can treat, for example, a broken experiment differently from a real regression.
        """
        return _EXIT_CODES[self]

    @property
    def is_scientific(self) -> bool:
        """Whether this verdict was decided by evidence rather than by plumbing."""
        return self in (Verdict.PASS, Verdict.FAIL, Verdict.INCONCLUSIVE)


_EXIT_CODES = {
    Verdict.PASS: 0,
    Verdict.FAIL: 1,
    Verdict.INCONCLUSIVE: 2,
    Verdict.UNSUPPORTED: 3,
    Verdict.INVALID_EXPERIMENT: 4,
    Verdict.ERROR: 5,
}


# --------------------------------------------------------------------------------------
# Reason codes
# --------------------------------------------------------------------------------------
# Grouped by the layer that can emit them. Every code that ships must appear in
# REASON_CODES with a one-line meaning; ``ivf.verdicts.describe`` is the only place a
# human-readable gloss lives, so reports and the CLI cannot drift apart.

REASON_CODES: dict[str, str] = {
    # -- experiment validity (emitted before any oracle runs) --------------------------
    "IVF-CONTROL-ASSET-MISMATCH": "Subjects do not share asset identity although the manifest requires it.",
    "IVF-CONTROL-ACTION-SEQUENCE-MISMATCH": "Subjects consumed different action streams.",
    "IVF-CONTROL-INITIAL-STATE-MISMATCH": "Subjects started from different initial-state semantics.",
    "IVF-CONTROL-OBSERVATION-DEFINITION-MISMATCH": "Subjects captured different signal sets or shapes.",
    "IVF-CONTROL-CONTROL-FREQUENCY-MISMATCH": "Subjects ran at different timesteps or control rates.",
    "IVF-CONTROL-SEED-MISMATCH": "Subjects used different seeds although pairing is assumed.",
    "IVF-CONTROL-ENV-COUNT-MISMATCH":
        "Subjects used different environment counts although pairing is assumed.",
    "IVF-CONTROL-SOLVER-PRESENTED-AS-EQUIVALENT":
        "Solver-specific parameters differ but were declared equivalent.",
    "IVF-CONTROL-TASK-VARIANT-MISMATCH": "Subjects ran different task variants.",
    "IVF-CONTROL-WARMUP-UNDECLARED":
        "The manifest declares no warm-up, so early-transient results are uninterpretable.",
    "IVF-CONTROL-HORIZON-MISMATCH": "Subjects captured different numbers of steps.",
    "IVF-PROTOCOL-PARTIAL-RUN": "At least one subject produced a truncated or failed run.",
    "IVF-PROTOCOL-STALE-ARTIFACT": "An input artifact predates the sources it claims to describe.",
    "IVF-PROTOCOL-TOLERANCE-RATIONALE-MISSING":
        "A tolerance was declared without a rationale.",
    # -- oracles -----------------------------------------------------------------------
    "IVF-ORACLE-INVARIANT-VIOLATION":
        "A per-run invariant was violated (non-finite state, norm drift, cap breach).",
    "IVF-ORACLE-NON_EQUIVALENT": "Signals differ by more than the declared tolerance.",
    "IVF-ORACLE-EVENT-TIMING-DELTA": "A semantic event occurred at incompatible times across subjects.",
    "IVF-ORACLE-EVENT-COUNT-MISMATCH":
        "A semantic event occurred a different number of times across subjects.",
    "IVF-ORACLE-DECISION-DISAGREEMENT":
        "Downstream decisions agreed less often than the declared minimum.",
    "IVF-ORACLE-METAMORPHIC-VIOLATION": "A declared metamorphic relation did not hold.",
    "IVF-ORACLE-ULP-EXCEEDED": "Bitwise distance exceeded the declared ULP budget.",
    # -- evidence sufficiency ----------------------------------------------------------
    "IVF-SAMPLE-INSUFFICIENT": "Fewer samples than the tolerance's declared minimum.",
    "IVF-EQUIVALENCE-UNDECIDED":
        "The confidence interval straddles the equivalence margin: neither equivalence nor "
        "difference is established.",
    "IVF-EFFECT-BELOW-MEANINGFUL":
        "A statistically detectable difference is smaller than the declared minimum "
        "meaningful effect.",
    # -- capability --------------------------------------------------------------------
    "IVF-BACKEND-FEATURE-UNSUPPORTED":
        "The requested backend feature is not available in this environment.",
    "IVF-RUNTIME-UNAVAILABLE": "The runtime required to execute this manifest is not installed.",
    "IVF-ORACLE-UNKNOWN": "The manifest requested an oracle type that is not registered.",
    # -- infrastructure ----------------------------------------------------------------
    "IVF-RUNTIME-EXECUTION-ERROR": "A subject failed to execute.",
    "IVF-EVIDENCE-WRITE-ERROR": "The evidence bundle could not be written.",
    "IVF-EVIDENCE-CHECKSUM-MISMATCH": "An evidence file does not match its recorded checksum.",
    "IVF-EVIDENCE-SCHEMA-INCOMPATIBLE":
        "The evidence bundle was written by an incompatible schema version.",
}


def describe(code: str) -> str:
    """Return the human-readable gloss for ``code``.

    Unknown codes are returned annotated rather than raising: a report must still
    render if an older bundle carries a code this version has retired.
    """
    return REASON_CODES.get(code, f"(unrecognized reason code: {code})")


def validate_code(code: str) -> str:
    """Return ``code`` if it is part of the shipped vocabulary, else raise.

    Used at emission sites so a typo fails a unit test rather than silently
    producing a code nobody can grep for.
    """
    if code not in REASON_CODES:
        raise KeyError(f"unknown reason code {code!r}; add it to ivf.verdicts.REASON_CODES")
    return code
