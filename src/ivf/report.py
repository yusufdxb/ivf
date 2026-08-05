# Copyright (c) 2026, Yusuf Guenena
# SPDX-License-Identifier: BSD-3-Clause

"""Static HTML reporting.

One self-contained file, no JavaScript, no external assets, no build step. Plots are
inline SVG generated here rather than by a plotting library, which keeps the dependency
list at numpy + PyYAML and keeps the report readable in any browser, in an email, or
as an artifact in a CI run.

The report is deliberately ordered the way a reviewer reads: verdict first, then whether
the experiment was even valid, then what failed, then where, then how to reproduce it.
Trace plots come last, after validity is established, because a polished plot of an
invalid experiment is exactly the thing this project exists to prevent.
"""

from __future__ import annotations

import html
from typing import Any

from .verdicts import describe

_VERDICT_COLORS = {
    "PASS": ("#1b5e20", "#e8f5e9"),
    "FAIL": ("#b71c1c", "#ffebee"),
    "INCONCLUSIVE": ("#e65100", "#fff3e0"),
    "UNSUPPORTED": ("#37474f", "#eceff1"),
    "INVALID_EXPERIMENT": ("#4a148c", "#f3e5f5"),
    "ERROR": ("#212121", "#eeeeee"),
}

_STATUS_COLORS = {
    "pass": "#1b5e20",
    "fail": "#b71c1c",
    "inconclusive": "#e65100",
    "unsupported": "#37474f",
    "skipped": "#616161",
    "unverifiable": "#e65100",
    "not_applicable": "#616161",
}

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 62rem; padding: 2rem 1.25rem 5rem; line-height: 1.55;
       color: #1a1a1a; background: #fff; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2.25rem 0 .5rem; padding-bottom: .3rem;
     border-bottom: 1px solid #ddd; }
h3 { font-size: .95rem; margin: 1.25rem 0 .35rem; }
.sub { color: #666; font-size: .85rem; margin: 0 0 1.5rem; }
.verdict { border-radius: 6px; padding: 1rem 1.25rem; margin: 1rem 0 1.5rem; }
.verdict .label { font-size: 1.6rem; font-weight: 700; letter-spacing: .02em; }
.verdict .why { margin-top: .5rem; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: .5rem 0 1rem; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #e5e5e5;
         vertical-align: top; }
th { background: #fafafa; font-weight: 600; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82em; }
pre { background: #f6f8fa; padding: .75rem 1rem; border-radius: 5px; overflow-x: auto;
      font-size: .8rem; }
.badge { display: inline-block; padding: .05rem .45rem; border-radius: 3px; color: #fff;
         font-size: .72rem; font-weight: 600; letter-spacing: .02em; }
.note { background: #f6f8fa; border-left: 3px solid #bbb; padding: .6rem .9rem;
        font-size: .82rem; margin: .75rem 0; }
.limit { color: #666; font-size: .8rem; font-style: italic; }
.scroll { overflow-x: auto; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #16181c; }
  th { background: #22252b; }
  th, td { border-bottom-color: #2c3037; }
  pre, .note { background: #1d2026; }
  h2 { border-bottom-color: #2c3037; }
  .sub, .limit { color: #9aa0a6; }
}
"""


def _e(value: Any) -> str:
    """HTML-escape a value."""
    return html.escape("" if value is None else str(value))


def _badge(status: str) -> str:
    """Render a coloured status badge."""
    color = _STATUS_COLORS.get(status, "#616161")
    return f'<span class="badge" style="background:{color}">{_e(status)}</span>'


def _svg_error_curve(window: dict[str, Any], width: int = 640, height: int = 160) -> str:
    """Render the divergence window as an inline SVG line plot with the threshold marked.

    Hand-rolled rather than pulled from a plotting library: one dependency saved, and a
    plot with no hidden defaults is easier to trust in a report whose whole purpose is
    auditability.
    """
    values = [v for v in window.get("error", []) if v is not None]
    if len(values) < 2:
        return ""
    start = int(window.get("start_step", 0))
    threshold = float(window.get("threshold", 0.0) or 0.0)
    pad_l, pad_r, pad_t, pad_b = 52, 12, 12, 26
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    y_max = max(max(values), threshold) * 1.15 or 1.0
    y_min = 0.0

    def x_of(i: int) -> float:
        return pad_l + plot_w * i / max(1, len(values) - 1)

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

    points = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
    thresh_y = y_of(threshold)
    labels = "".join(
        f'<text x="{x_of(i):.1f}" y="{height - 8}" font-size="9" text-anchor="middle" '
        f'fill="#888">{start + i}</text>'
        for i in range(0, len(values), max(1, len(values) // 6))
    )
    return f"""<svg width="100%" viewBox="0 0 {width} {height}" role="img"
     aria-label="error curve around the first tolerance violation">
  <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#ddd"/>
  <line x1="{pad_l}" y1="{thresh_y:.1f}" x2="{pad_l + plot_w}" y2="{thresh_y:.1f}"
        stroke="#b71c1c" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="{pad_l + plot_w - 4}" y="{thresh_y - 4:.1f}" font-size="9" text-anchor="end"
        fill="#b71c1c">tolerance {threshold:.3g}</text>
  <polyline points="{points}" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="6" y="{pad_t + 10}" font-size="9" fill="#888">{y_max:.3g}</text>
  <text x="6" y="{pad_t + plot_h}" font-size="9" fill="#888">0</text>
  {labels}
</svg>"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a table; returns an empty string when there are no rows."""
    if not rows:
        return ""
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_html(
    *,
    verdict: dict[str, Any],
    manifest: dict[str, Any],
    validity: dict[str, Any],
    outcomes: list[dict[str, Any]],
    provenance: dict[str, Any],
    run_id: str,
) -> str:
    """Render a complete, self-contained evidence report."""
    label = str(verdict.get("verdict", "ERROR"))
    fg, bg = _VERDICT_COLORS.get(label, ("#212121", "#eeeeee"))
    codes = verdict.get("reason_codes", []) or []
    counts = verdict.get("counts", {}) or {}

    parts: list[str] = [
        f"<title>IVF {label}: {_e(manifest.get('name', run_id))}</title>",
        f"<style>{_CSS}</style>",
        f"<h1>{_e(manifest.get('name', 'experiment'))}</h1>",
        f'<p class="sub mono">{_e(run_id)} &middot; IVF {_e(verdict.get("ivf_version"))} '
        f'&middot; {_e(verdict.get("created_utc"))}</p>',
    ]

    why = "".join(
        f"<div><code>{_e(c)}</code> - {_e(describe(c))}</div>" for c in codes
    ) or "<div>No reason codes: every declared control held and every criterion was met.</div>"
    parts.append(
        f'<div class="verdict" style="background:{bg};color:{fg}">'
        f'<div class="label">{_e(label)}</div>'
        f'<div class="why">{why}</div></div>'
    )
    if verdict.get("error"):
        parts.append(f"<pre>{_e(verdict['error'])}</pre>")

    if manifest.get("description"):
        parts.append(f"<p>{_e(manifest['description'])}</p>")

    # -- identity ----------------------------------------------------------------------
    parts.append("<h2>Subjects under comparison</h2>")
    subject_rows = []
    for role, subject in (manifest.get("subjects") or {}).items():
        detail = ", ".join(
            f"{k}={v}" for k, v in sorted(subject.items()) if k not in ("role", "kind", "label")
        )
        subject_rows.append([f"<strong>{_e(role)}</strong>", _e(subject.get("kind")),
                             f'<span class="mono">{_e(detail)}</span>'])
    parts.append(_table(["role", "source", "configuration"], subject_rows))

    workload = manifest.get("workload", {})
    parts.append(_table(
        ["task", "steps", "envs", "warm-up", "seeds"],
        [[_e(workload.get("task")), _e(workload.get("steps")), _e(workload.get("num_envs")),
          _e(workload.get("warmup_steps")), _e(workload.get("seeds"))]],
    ))

    # -- experiment validity -----------------------------------------------------------
    parts.append("<h2>Experiment validity</h2>")
    checks = validity.get("checks", []) or []
    if not checks:
        parts.append('<p class="limit">No validity checks ran; the experiment did not reach '
                     "that stage.</p>")
    else:
        summary = (f"{validity.get('n_checks', 0)} checks, {validity.get('n_failed', 0)} failed, "
                   f"{validity.get('n_unverifiable', 0)} unverifiable")
        parts.append(f'<p class="sub">{_e(summary)}</p>')
        parts.append(_table(
            ["id", "check", "status", "detail"],
            [[f'<span class="mono">{_e(c["check_id"])}</span>', _e(c["name"]),
              _badge(c["status"]), _e(c["detail"])] for c in checks],
        ))
        if validity.get("n_unverifiable"):
            parts.append('<div class="note">An <strong>unverifiable</strong> control is not a '
                         "match. It means a subject did not record the metadata the control needs, "
                         "so IVF cannot establish that the two runs were comparable in that respect."
                         "</div>")

    # -- oracles -----------------------------------------------------------------------
    parts.append("<h2>Acceptance criteria</h2>")
    parts.append(
        f'<p class="sub">{counts.get("pass", 0)} passed, {counts.get("fail", 0)} failed, '
        f'{counts.get("inconclusive", 0)} inconclusive, {counts.get("unsupported", 0)} unsupported, '
        f'{counts.get("skipped", 0)} skipped</p>'
    )
    oracle_rows = []
    for outcome in outcomes:
        tol = outcome.get("tolerance")
        tol_text = (
            f"{tol['value']:g} {tol['unit']} ({tol['kind']}, {tol['aggregation']}, "
            f"n&ge;{tol['min_samples']})" if tol else "-"
        )
        oracle_rows.append([
            f'<span class="mono">{_e(outcome["name"])}</span>',
            _e(outcome["type"]),
            _badge(outcome["status"]),
            _e(outcome["summary"]),
            tol_text,
        ])
    parts.append(_table(["oracle", "type", "status", "result", "tolerance"], oracle_rows))

    for outcome in outcomes:
        if outcome.get("tolerance") and outcome["status"] in ("fail", "inconclusive"):
            tol = outcome["tolerance"]
            parts.append(
                f'<div class="note"><strong>{_e(outcome["name"])}</strong> tolerance rationale: '
                f'{_e(tol["rationale"])}<br><span class="limit">scope: {_e(tol["scope"])}</span></div>'
            )

    # -- effect sizes and intervals ------------------------------------------------------
    stat_rows = []
    for outcome in outcomes:
        metrics = outcome.get("metrics") or {}
        if "ci_low" not in metrics:
            continue
        stat_rows.append([
            f'<span class="mono">{_e(outcome["name"])}</span>',
            (f"{metrics['mean_difference']:+.4g}"
             if metrics.get("mean_difference") is not None else "-"),
            f"[{metrics.get('ci_low'):+.4g}, {metrics.get('ci_high'):+.4g}]",
            f"{metrics.get('cohens_dz'):+.3g}" if metrics.get("cohens_dz") is not None else "-",
            f"&plusmn;{metrics.get('equivalence_margin'):g}",
            _e(metrics.get("n_pairs")),
        ])
    if stat_rows:
        parts.append("<h3>Effect sizes and uncertainty</h3>")
        parts.append(_table(
            ["oracle", "mean paired difference", "bootstrap CI", "Cohen's dz", "margin", "n pairs"],
            stat_rows,
        ))

    # -- divergence ----------------------------------------------------------------------
    divergences = [(o["name"], o["divergence"]) for o in outcomes if o.get("divergence")]
    if divergences:
        parts.append("<h2>First divergence</h2>")
        for name, record in divergences:
            parts.append(f"<h3>{_e(name)} &middot; {_e(record.get('signal'))}</h3>")
            parts.append(_table(
                ["first numerical difference", "first tolerance violation", "first event disagreement",
                 "environments", "components"],
                [[_e(record.get("first_numerical_difference_step")),
                  _e(record.get("first_tolerance_violation_step")),
                  _e(record.get("first_event_disagreement_step")),
                  _e(record.get("affected_env_ids")),
                  _e(record.get("affected_components"))]],
            ))
            parts.append(_table(
                ["classification", "confidence", "basis"],
                [[f'<strong>{_e(record.get("classification"))}</strong>',
                  _e(record.get("confidence")), _e(record.get("classification_basis"))]],
            ))
            if record.get("limitations"):
                parts.append(f'<p class="limit">Limitation: {_e(record["limitations"])}</p>')
            if record.get("baseline_values"):
                parts.append(_table(
                    ["", "values at the first violating step"],
                    [["baseline", f'<span class="mono">{_e(record["baseline_values"])}</span>'],
                     ["candidate", f'<span class="mono">{_e(record["candidate_values"])}</span>']],
                ))
            if record.get("action_at_violation"):
                parts.append(f'<p class="limit">Action at that step: '
                             f'<span class="mono">{_e(record["action_at_violation"])}</span></p>')
            if record.get("config_differences"):
                parts.append(_table(
                    ["configuration difference", "baseline", "candidate"],
                    [[_e(k), _e(v.get("baseline")), _e(v.get("candidate"))]
                     for k, v in sorted(record["config_differences"].items())],
                ))
            svg = _svg_error_curve(record.get("window") or {})
            if svg:
                parts.append(f'<p class="sub">Error around the first violation, environment '
                             f'{_e((record.get("window") or {}).get("env_id"))}</p>{svg}')

    # -- limitations ---------------------------------------------------------------------
    limitations = [(o["name"], o["limitations"]) for o in outcomes if o.get("limitations")]
    if limitations:
        parts.append("<h2>Limitations</h2>")
        parts.append(_table(["oracle", "what this result does not establish"],
                            [[f'<span class="mono">{_e(n)}</span>', _e(t)] for n, t in limitations]))
    parts.append(
        '<div class="note">IVF reports divergence between two subjects. It does not certify '
        "either subject as physically correct, and agreement within a tolerance is not evidence "
        "that either one is right.</div>"
    )

    # -- reproduction --------------------------------------------------------------------
    parts.append("<h2>Reproduction</h2>")
    command = provenance.get("command") or []
    parts.append("<pre>" + _e(" ".join(str(c) for c in command)) + "</pre>")
    parts.append(_table(
        ["", ""],
        [["manifest digest", f'<span class="mono">{_e(verdict.get("manifest_digest_sha256"))}</span>'],
         ["IVF version", _e(provenance.get("ivf_version"))],
         ["evidence schema", _e(provenance.get("evidence_schema_version"))],
         ["host", _e((provenance.get("host") or {}).get("platform"))],
         ["python", _e((provenance.get("python") or {}).get("version"))],
         ["wall time", f'{_e(verdict.get("wall_time_s"))} s']],
    ))
    parts.append(
        "<pre>ivf reproduce &lt;evidence-bundle-path&gt; --verify-only\n"
        "ivf compare &lt;baseline-evidence-path&gt; &lt;candidate-evidence-path&gt;</pre>"
    )
    parts.append(
        '<p class="limit">Per-file SHA-256 digests are in <code>CHECKSUMS.sha256</code>, sealed by '
        "<code>SEAL.json</code>. Run <code>ivf reproduce &lt;run&gt; --verify-only</code> to check them. "
        "These detect accidental corruption, incomplete transfer and uncoordinated "
        "modification. The seal is not a digital signature and does not establish "
        "authenticity against an actor who can modify and reseal the entire bundle."
        "</p>"
    )
    return "\n".join(parts)
