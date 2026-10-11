"""A single self-contained HTML scan report.

Meant to be forwarded to whoever has to act on it. It opens with a verdict
and a triage plan (which records to delete, quarantine, or review), then
the severity breakdown, a filterable list of findings with evidence and fix
steps, the rules that fired, and what the scan looked for.

Everything is inline: no fonts, scripts, or images are fetched, so the
file opens offline and leaks nothing when opened. The script only filters
and expands; without it every finding is still readable. Light and dark
themes follow the viewer's system setting, and printing expands every
finding.

All memory-derived text (ids, snippets, evidence) is HTML-escaped. Snippets
are secret-masked before they get here.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from html import escape

from ..models.enums import Action, Severity
from ..models.results import ScanFinding, ScanReport
from ..rules import REPO_URL, rule_for

_SEVERITY_ORDER = (
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
)

_ACTION_COPY: dict[Action, str] = {
    Action.REVIEW: "Review",
    Action.QUARANTINE: "Quarantine",
    Action.DELETE: "Delete",
}

_ACTION_GUIDE: dict[Action, str] = {
    Action.DELETE: "Remove these records. Rotate any credential they hold before you delete.",
    Action.QUARANTINE: "Keep these out of retrieval until someone confirms they are true.",
    Action.REVIEW: "A person should read these and decide. Most are fine to keep once checked.",
}

_MAX_CHIPS = 12

_CSS = """
:root {
  color-scheme: light;
  --bg: #edf1f5;
  --surface: #ffffff;
  --sunk: #f4f7fa;
  --ink: #142231;
  --muted: #586a7c;
  --rule: #d3dce5;
  --accent: #2747c9;
  --critical: #b3122e;
  --critical-tint: #fbe7ea;
  --high: #b8500b;
  --high-tint: #fcecdf;
  --medium: #8a6800;
  --medium-tint: #f7efd2;
  --low: #2f6489;
  --low-tint: #e3eef6;
  --info: #5f6c79;
  --info-tint: #eaeef2;
  --ok: #1d6b45;
  --ok-tint: #e2f2e9;
  --sans: "Avenir Next", Avenir, "Segoe UI Variable Text", "Segoe UI", "Helvetica Neue",
    "Noto Sans", Arial, sans-serif;
  --mono: "SF Mono", "Cascadia Code", "JetBrains Mono", Menlo, Consolas, "Liberation Mono",
    monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --bg: #0d141b;
    --surface: #152029;
    --sunk: #101921;
    --ink: #e3e9ef;
    --muted: #93a4b5;
    --rule: #263544;
    --accent: #93a8ff;
    --critical: #ff7186;
    --critical-tint: #3a1820;
    --high: #ff9d57;
    --high-tint: #3a2414;
    --medium: #e6c24e;
    --medium-tint: #342b12;
    --low: #7db8e3;
    --low-tint: #142c3d;
    --info: #a5b1bd;
    --info-tint: #1d2832;
    --ok: #6fd3a0;
    --ok-tint: #12301f;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.55 var(--sans);
  font-variant-numeric: tabular-nums;
}
a { color: var(--accent); text-underline-offset: 2px; }
code, .mono { font-family: var(--mono); font-size: 0.88em; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
.page { max-width: 1060px; margin: 0 auto; padding: 44px 24px 72px; }
.brand { font-weight: 700; letter-spacing: -0.01em; margin: 0 0 36px; color: var(--muted); }
.brand b { color: var(--ink); }
.scope { margin: 0 0 6px; color: var(--muted); }
.scope code { color: var(--ink); }
h1 {
  font-size: clamp(30px, 5vw, 46px);
  line-height: 1.08;
  letter-spacing: -0.025em;
  font-weight: 700;
  margin: 0 0 12px;
  max-width: 18em;
}
.lede { font-size: 18px; color: var(--muted); margin: 0 0 22px; max-width: 42em; }
.facts { display: flex; flex-wrap: wrap; gap: 6px 28px; margin: 0; font-size: 14px; }
.facts div { display: flex; gap: 8px; }
.facts dt { color: var(--muted); }
.facts dd { margin: 0; font-weight: 600; }
.notice {
  margin: 22px 0 0;
  padding: 12px 16px;
  border-radius: 8px;
  background: var(--medium-tint);
  color: var(--ink);
  font-size: 15px;
  max-width: 48em;
}
section { margin-top: 56px; }
h2 { font-size: 22px; letter-spacing: -0.01em; margin: 0 0 6px; }
.section-note { color: var(--muted); margin: 0 0 18px; max-width: 46em; }

/* Triage plan: the part of the page people act on. */
.plan { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.lane {
  background: var(--surface);
  border-radius: 12px;
  padding: 18px 20px 20px;
  border-top: 6px solid var(--lane);
  box-shadow: 0 1px 0 var(--rule);
}
.lane.delete { --lane: var(--critical); }
.lane.quarantine { --lane: var(--high); }
.lane.review { --lane: var(--low); }
.lane.empty { --lane: var(--rule); color: var(--muted); }
.lane h3 { margin: 0; font-size: 17px; }
.lane .n { font-size: 40px; font-weight: 700; line-height: 1.1; letter-spacing: -0.02em; }
.lane .n small { font-size: 15px; font-weight: 600; color: var(--muted); letter-spacing: 0; }
.lane p { margin: 8px 0 14px; color: var(--muted); font-size: 14.5px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0; padding: 0; list-style: none; }
.chips a {
  display: inline-block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font: 13px/1 var(--mono);
  padding: 6px 8px;
  border-radius: 6px;
  background: var(--sunk);
  color: var(--ink);
  text-decoration: none;
  border: 1px solid var(--rule);
}
.chips a:hover { border-color: var(--lane); }
.chips .more { font-size: 13px; color: var(--muted); align-self: center; }
.allclear {
  background: var(--ok-tint);
  color: var(--ink);
  border-radius: 12px;
  padding: 20px 22px;
  max-width: 48em;
}
.allclear strong { color: var(--ok); }
.notice.incomplete { background: var(--critical-tint); }
.notice.incomplete ul { margin: 8px 0 0; padding-left: 20px; }
.notice.incomplete code { font: 13px var(--mono); }

/* Severity breakdown. */
.spread { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 28px; }
.bar {
  display: flex;
  height: 18px;
  border-radius: 9px;
  overflow: hidden;
  background: var(--rule);
  margin: 4px 0 14px;
}
.bar span { display: block; min-width: 3px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; margin: 0; padding: 0; list-style: none; }
.legend li { display: flex; align-items: baseline; gap: 8px; font-size: 15px; }
.legend b { font-size: 20px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.byrule { margin: 0; padding: 0; list-style: none; font-size: 14.5px; }
.byrule li {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid var(--rule);
}
.byrule li:last-child { border-bottom: 0; }
.byrule code { color: var(--muted); font-size: 12.5px; margin-left: 6px; }

.s-critical { --sev: var(--critical); --sev-tint: var(--critical-tint); }
.s-high { --sev: var(--high); --sev-tint: var(--high-tint); }
.s-medium { --sev: var(--medium); --sev-tint: var(--medium-tint); }
.s-low { --sev: var(--low); --sev-tint: var(--low-tint); }
.s-info { --sev: var(--info); --sev-tint: var(--info-tint); }
.bar .s-critical, .bar .s-high, .bar .s-medium, .bar .s-low, .bar .s-info,
.swatch { background: var(--sev); }
.sev {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 650;
  color: var(--sev);
  background: var(--sev-tint);
  padding: 4px 9px 4px 8px;
  border-radius: 999px;
  white-space: nowrap;
}
.sev::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 1px;
  background: var(--sev);
  transform: rotate(45deg);
}

/* Findings. */
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  align-items: center;
  margin: 0 0 14px;
  padding: 12px;
  background: var(--surface);
  border-radius: 10px;
  box-shadow: 0 1px 0 var(--rule);
}
.toolbar input[type="search"], .toolbar select {
  font: inherit;
  font-size: 14.5px;
  color: var(--ink);
  background: var(--sunk);
  border: 1px solid var(--rule);
  border-radius: 7px;
  padding: 7px 10px;
}
.toolbar input[type="search"] { flex: 1 1 240px; min-width: 0; }
.toggles { display: flex; flex-wrap: wrap; gap: 6px; }
.toggles button {
  font: inherit;
  font-size: 13px;
  border: 1px solid var(--rule);
  background: var(--sunk);
  color: var(--muted);
  border-radius: 999px;
  padding: 5px 10px;
  cursor: pointer;
}
.toggles button[aria-pressed="true"] {
  color: var(--sev);
  background: var(--sev-tint);
  border-color: transparent;
  font-weight: 650;
}
.toolbar .linkish {
  font: inherit;
  font-size: 14px;
  background: none;
  border: 0;
  color: var(--accent);
  cursor: pointer;
  padding: 4px;
}
.count { color: var(--muted); font-size: 14px; margin-left: auto; }

.findings { display: flex; flex-direction: column; gap: 8px; }
.finding {
  background: var(--surface);
  border-radius: 10px;
  box-shadow: 0 1px 0 var(--rule);
  border-left: 4px solid var(--sev);
}
.finding:target { outline: 2px solid var(--accent); outline-offset: 2px; }
.finding > summary {
  list-style: none;
  cursor: pointer;
  display: grid;
  grid-template-columns: 104px minmax(0, 1fr) minmax(0, 220px) 104px 64px;
  gap: 14px;
  align-items: center;
  padding: 12px 16px;
}
.finding > summary::-webkit-details-marker { display: none; }
.f-title { font-weight: 650; }
.f-title code { display: block; font-weight: 400; color: var(--muted); font-size: 12.5px; }
.f-record {
  font: 13.5px/1.3 var(--mono);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.f-action { font-size: 14px; font-weight: 600; }
.f-action.delete { color: var(--critical); }
.f-action.quarantine { color: var(--high); }
.f-owasp { font-size: 13px; color: var(--muted); }
.f-body { padding: 4px 20px 20px 20px; border-top: 1px solid var(--rule); }
.f-body h4 { font-size: 13.5px; margin: 18px 0 6px; color: var(--muted); font-weight: 600; }
.f-body p { margin: 12px 0 0; max-width: 60em; }
.snippet {
  margin: 0;
  font: 13.5px/1.55 var(--mono);
  background: var(--sunk);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 10px 12px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0 28px; }
.fix { margin: 0; padding-left: 20px; }
.fix li { margin: 0 0 4px; }
.why { margin: 0; display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 14px;
  font-size: 14.5px; }
.why dt { color: var(--muted); }
.why dd { margin: 0; overflow-wrap: anywhere; }
.refs { margin: 16px 0 0; font-size: 14px; display: flex; flex-wrap: wrap; gap: 4px 16px; }
.fp { margin: 10px 0 0; font-size: 12.5px; color: var(--muted); }
.nomatch { color: var(--muted); padding: 16px 4px; }

/* Rules and scan setup. */
table { width: 100%; border-collapse: collapse; font-size: 14.5px; background: var(--surface);
  border-radius: 10px; overflow: hidden; box-shadow: 0 1px 0 var(--rule); }
th, td {
  text-align: left;
  vertical-align: top;
  padding: 10px 14px;
  border-bottom: 1px solid var(--rule);
}
th { font-weight: 600; color: var(--muted); background: var(--sunk); }
tr:last-child td { border-bottom: 0; }
td.num { text-align: right; }
td code, td a { white-space: nowrap; }
.desc { color: var(--muted); }

footer {
  margin-top: 64px;
  padding-top: 18px;
  border-top: 1px solid var(--rule);
  color: var(--muted);
  font-size: 14px;
  max-width: 52em;
}

@media (max-width: 820px) {
  .plan, .spread, .cols { grid-template-columns: minmax(0, 1fr); }
  .finding > summary { grid-template-columns: auto minmax(0, 1fr) auto; gap: 6px 12px; }
  .f-title { grid-column: 2 / 4; }
  .f-record { grid-column: 2; }
  .f-action { grid-column: 3; }
  .f-owasp { display: none; }
  td.desc { min-width: 240px; }
  .page { padding: 28px 16px 56px; }
  table { display: block; overflow-x: auto; }
}
@media (prefers-reduced-motion: no-preference) {
  .finding[open] > .f-body { animation: reveal 160ms ease-out; }
  @keyframes reveal { from { opacity: 0; transform: translateY(-3px); } to { opacity: 1; } }
}
@media print {
  :root { --bg: #fff; }
  .toolbar { display: none !important; }
  .finding, .lane, tr { break-inside: avoid; }
  .finding > .f-body { display: block; }
  a { color: inherit; }
}
""".strip()

_SCRIPT = """
(function () {
  var list = Array.prototype.slice.call(document.querySelectorAll(".finding"));
  var bar = document.getElementById("toolbar");
  if (!bar || !list.length) return;
  bar.hidden = false;
  var q = document.getElementById("q");
  var act = document.getElementById("act");
  var shown = document.getElementById("shown");
  var none = document.getElementById("nomatch");
  var toggles = Array.prototype.slice.call(bar.querySelectorAll("[data-sev]"));
  function apply() {
    var text = q.value.trim().toLowerCase();
    var on = {};
    toggles.forEach(function (b) {
      on[b.dataset.sev] = b.getAttribute("aria-pressed") === "true";
    });
    var n = 0;
    list.forEach(function (el) {
      var ok = on[el.dataset.severity] !== false &&
        (!act.value || el.dataset.action === act.value) &&
        (!text || el.dataset.text.indexOf(text) !== -1);
      el.hidden = !ok;
      if (ok) n++;
    });
    shown.textContent =
      n === list.length ? "Showing all " + n : "Showing " + n + " of " + list.length;
    none.hidden = n !== 0;
  }
  toggles.forEach(function (b) {
    b.addEventListener("click", function () {
      b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") === "true" ? "false" : "true");
      apply();
    });
  });
  q.addEventListener("input", apply);
  act.addEventListener("change", apply);
  var expand = document.getElementById("expand");
  expand.addEventListener("click", function () {
    var open = expand.dataset.open !== "true";
    list.forEach(function (el) { if (!el.hidden) el.open = open; });
    expand.dataset.open = String(open);
    expand.textContent = open ? "Collapse all" : "Expand all";
  });
  function reveal() {
    var id = decodeURIComponent(location.hash.slice(1));
    var el = id && document.getElementById(id);
    if (el && el.classList.contains("finding")) { el.hidden = false; el.open = true; }
  }
  window.addEventListener("hashchange", reveal);
  window.addEventListener("beforeprint", function () {
    list.forEach(function (el) { el.open = true; });
  });
  reveal();
  apply();
})();
""".strip()


def finding_label(code: str) -> str:
    """Turn a finding code into the short title shown in reports.

    Args:
        code: A finding code such as `"secret_detected"`.

    Returns:
        A title such as `"Leaked secret"`. Unknown codes become the code
        with underscores turned into spaces and the first letter capitalized.
    """
    return rule_for(code).title


def render_html(report: ScanReport) -> str:
    """Build a standalone HTML document for `report`.

    Args:
        report: The scan result. Snippets are escaped. They are expected
            to already be secret-masked.

    Returns:
        A complete HTML document, including the stylesheet and a small
        filtering script, as one string. Write it to a file and open it in
        a browser.
    """
    title = _headline(report)
    body = [
        _masthead(report, title),
        _plan(report),
    ]
    if report.findings:
        body += [_spread(report), _findings(report), _rules(report)]
    body += [_setup(report), _footer(report)]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="generator" content="Mimvo {escape(report.mimvo_version)}">
<title>{escape(title)} · Mimvo</title>
<style>{_CSS}</style>
</head>
<body>
<main class="page">
{"".join(body)}
</main>
<script>{_SCRIPT}</script>
</body>
</html>
"""


# -- header ---------------------------------------------------------------------


def _headline(report: ScanReport) -> str:
    if report.clean and not report.complete:
        return (
            f"Scan incomplete: {report.records_with_errors:,} of {_records(report.total)} "
            "not fully checked"
        )
    if report.clean:
        return f"No problems found in {_records(report.total)}"
    return f"{report.flagged:,} of {_records(report.total)} need action"


def _masthead(report: ScanReport, title: str) -> str:
    source = (
        f"Memory scan of <code>{escape(report.source)}</code>" if report.source else "Memory scan"
    )
    if report.clean and not report.complete:
        lede = (
            "Nothing was flagged, but a check or detector failed, so some records were not "
            "fully checked. A failure is not a finding: fix it (see below) and scan again "
            "before acting on this report."
        )
    elif report.clean:
        lede = (
            "None of the checks matched a poisoned fact, hidden instruction, or leaked "
            "secret. The default detectors are heuristics, so see what was checked below "
            "before treating this as a clean bill of health."
        )
    else:
        plan = report.action_plan()
        parts = [
            f"{len(ids):,} to {_ACTION_COPY[action].lower()}" for action, ids in plan.items() if ids
        ]
        lede = (
            f"{_join(parts).capitalize()}. That is {report.flagged_pct:g}% of the records "
            "scanned. Each finding below says why it was flagged and how to fix it."
        )
    facts = [("Scanned", _when(report.generated_at)), ("Records", f"{report.total:,}")]
    if report.duration_seconds is not None:
        facts.append(("Took", _duration(report.duration_seconds)))
    facts.append(("Mimvo", report.mimvo_version))
    facts_html = "".join(
        f"<div><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>" for k, v in facts if v
    )
    notice = ""
    if report.sample and report.total >= report.sample:
        notice = (
            f'<p class="notice">Only the first {report.sample:,} records were scanned '
            "(<code>--sample</code>). Records after that were not checked.</p>"
        )
    if report.errors:
        items = "".join(
            f"<li><code>{escape(e.check)}{'/' + escape(e.detector) if e.detector else ''}</code> "
            f"raised <code>{escape(e.error_type)}</code> on {_records(e.records)}"
            + (f": {escape(e.message)}" if e.message else "")
            + "</li>"
            for e in report.errors[:10]
        )
        notice += (
            f'<div class="notice incomplete" role="alert"><strong>Scan incomplete.</strong> '
            f"{_records(report.records_with_errors)} were not fully checked because a check or "
            f"detector failed. Records are not flagged for this; fix the failure and scan "
            f"again.<ul>{items}</ul></div>"
        )
    return f"""
<header>
  <p class="brand"><b>Mimvo</b> report</p>
  <p class="scope">{source}</p>
  <h1>{escape(title)}</h1>
  <p class="lede">{escape(lede)}</p>
  <dl class="facts">{facts_html}</dl>
  {notice}
</header>"""


# -- triage plan ----------------------------------------------------------------


def _plan(report: ScanReport) -> str:
    if report.clean and not report.complete:
        return f"""
<section aria-labelledby="plan-h">
  <h2 id="plan-h">What to do</h2>
  <div class="notice incomplete"><strong>Fix the failing check first.</strong>
  {_records(report.records_with_errors)} could not be fully checked, so this scan cannot
  say they are clean.</div>
</section>"""
    if report.clean:
        return f"""
<section aria-labelledby="plan-h">
  <h2 id="plan-h">What to do</h2>
  <div class="allclear"><strong>Nothing to act on.</strong> {_records(report.total)} scanned,
  none flagged. Scan again after the store changes, or add model detectors for
  paraphrased attacks the heuristics miss.</div>
</section>"""
    anchors = _first_anchor_by_record(report.findings)
    lanes = []
    for action, ids in report.action_plan().items():
        css = action.value
        if not ids:
            lanes.append(
                f'<div class="lane empty"><h3>{_ACTION_COPY[action]}</h3>'
                f'<div class="n">0 <small>records</small></div><p>Nothing here.</p></div>'
            )
            continue
        chips = "".join(
            f'<li><a href="#{escape(anchors[i])}" title="{escape(i)}">{escape(i)}</a></li>'
            for i in ids[:_MAX_CHIPS]
        )
        if len(ids) > _MAX_CHIPS:
            chips += f'<li class="more">and {len(ids) - _MAX_CHIPS:,} more</li>'
        noun = "record" if len(ids) == 1 else "records"
        lanes.append(
            f'<div class="lane {css}"><h3>{_ACTION_COPY[action]}</h3>'
            f'<div class="n">{len(ids):,} <small>{noun}</small></div>'
            f"<p>{escape(_ACTION_GUIDE[action])}</p>"
            f'<ul class="chips" aria-label="{_ACTION_COPY[action]} these records">'
            f"{chips}</ul></div>"
        )
    return f"""
<section aria-labelledby="plan-h">
  <h2 id="plan-h">What to do</h2>
  <p class="section-note">Each flagged record is listed once, under the strongest action any of
  its findings calls for. Select a record to jump to its findings.</p>
  <div class="plan">{"".join(lanes)}</div>
</section>"""


# -- severity breakdown ---------------------------------------------------------


def _spread(report: ScanReport) -> str:
    counts = report.by_severity
    total = sum(counts.values()) or 1
    segments = "".join(
        f'<span class="s-{sev.value}" style="width:{100 * counts[sev.value] / total:.3f}%"'
        f' title="{counts[sev.value]:,} {sev.value}"></span>'
        for sev in _SEVERITY_ORDER
        if counts.get(sev.value)
    )
    legend = "".join(
        f'<li class="s-{sev.value}"><span class="swatch" aria-hidden="true"></span>'
        f"<b>{counts[sev.value]:,}</b> {sev.value.capitalize()}</li>"
        for sev in _SEVERITY_ORDER
        if counts.get(sev.value)
    )
    by_rule = "".join(
        f"<li><span>{escape(rule_for(code).title)}<code>{escape(code)}</code></span>"
        f"<b>{n:,}</b></li>"
        for code, n in report.by_rule.items()
    )
    n_findings = len(report.findings)
    return f"""
<section aria-labelledby="spread-h">
  <h2 id="spread-h">{n_findings:,} {"finding" if n_findings == 1 else "findings"}</h2>
  <p class="section-note">A record with two problems has two findings. Severity and action come
  from the rule, not from the detector that raised it.</p>
  <div class="spread">
    <div>
      <div class="bar" role="img" aria-label="{escape(_severity_sentence(counts))}">{segments}</div>
      <ul class="legend">{legend}</ul>
    </div>
    <ul class="byrule" aria-label="Findings by rule">{by_rule}</ul>
  </div>
</section>"""


# -- findings -------------------------------------------------------------------


def _findings(report: ScanReport) -> str:
    counts = report.by_severity
    toggles = "".join(
        f'<button type="button" class="s-{sev.value}" data-sev="{sev.value}" aria-pressed="true">'
        f"{sev.value.capitalize()} {counts[sev.value]:,}</button>"
        for sev in _SEVERITY_ORDER
        if counts.get(sev.value)
    )
    actions = "".join(
        f'<option value="{a.value}">{_ACTION_COPY[a]}</option>'
        for a in (Action.DELETE, Action.QUARANTINE, Action.REVIEW)
        if report.by_action.get(a.value)
    )
    anchors = _anchors(report.findings)
    items = "".join(
        _finding(item, anchor) for item, anchor in zip(report.findings, anchors, strict=True)
    )
    return f"""
<section aria-labelledby="list-h">
  <h2 id="list-h">Findings</h2>
  <p class="section-note">Worst first. Open a finding for the masked excerpt, the evidence the
  detectors reported, and the steps to fix it.</p>
  <div class="toolbar" id="toolbar" hidden>
    <input type="search" id="q" placeholder="Filter by record id, rule, or text"
      aria-label="Filter findings">
    <div class="toggles" role="group" aria-label="Severity">{toggles}</div>
    <select id="act" aria-label="Recommended action">
      <option value="">Any action</option>{actions}
    </select>
    <button type="button" class="linkish" id="expand">Expand all</button>
    <span class="count" id="shown" aria-live="polite"></span>
  </div>
  <div class="findings">{items}</div>
  <p class="nomatch" id="nomatch" hidden>No findings match these filters.</p>
</section>"""


def _finding(item: ScanFinding, anchor: str) -> str:
    sev = item.severity.value
    rule = rule_for(item.code)
    owasp_id = item.owasp.split(":", 1)[0]
    search = " ".join(
        [item.record_id, item.code, item.title, item.snippet, item.check or "", *item.detectors]
    ).lower()
    snippet = (
        f'<h4>Masked excerpt</h4><pre class="snippet">{escape(item.snippet)}</pre>'
        if item.snippet
        else ""
    )
    steps = "".join(f"<li>{escape(step)}</li>" for step in item.remediation)
    why_rows = [("Detectors", ", ".join(item.detectors) or "—")]
    if item.confidence is not None:
        why_rows.append(("Confidence", f"{item.confidence:.2f}"))
    if item.check:
        why_rows.append(("Check", item.check))
    why_rows += [(_humanize(k), _evidence_text(v)) for k, v in item.evidence.items()]
    why = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in why_rows)
    refs = "".join(
        f'<a href="{escape(ref.url)}" rel="noopener noreferrer">{escape(ref.label)}</a>'
        for ref in rule.references
    )
    message = item.message or rule.summary
    action = item.recommended_action
    return f"""
<details class="finding s-{escape(sev)}" id="{escape(anchor)}" data-severity="{escape(sev)}"
  data-action="{escape(action.value)}" data-text="{escape(search)}">
  <summary>
    <span class="sev">{escape(sev.capitalize())}</span>
    <span class="f-title">{escape(item.title)}<code>{escape(item.code)}</code></span>
    <span class="f-record" title="{escape(item.record_id)}">{escape(item.record_id)}</span>
    <span class="f-action {escape(action.value)}">{_ACTION_COPY[action]}</span>
    <span class="f-owasp">{escape(owasp_id)}</span>
  </summary>
  <div class="f-body">
    <p>{escape(message)}</p>
    {snippet}
    <div class="cols">
      <div><h4>How to fix</h4><ol class="fix">{steps}</ol></div>
      <div><h4>Why it was flagged</h4><dl class="why">{why}</dl></div>
    </div>
    <p class="refs">{refs}</p>
    <p class="fp">Fingerprint <span class="mono">{escape(item.fingerprint)}</span></p>
  </div>
</details>"""


# -- rules and setup ------------------------------------------------------------


def _rules(report: ScanReport) -> str:
    worst: dict[str, Severity] = {}
    for item in report.findings:
        if item.code not in worst or item.severity.rank > worst[item.code].rank:
            worst[item.code] = item.severity
    rows = []
    for code, n in report.by_rule.items():
        rule = rule_for(code)
        sev = worst[code].value
        cwe = ", ".join(
            f'<a href="{escape(r.url)}">{escape(r.label)}</a>'
            for r in rule.references
            if r.label.startswith("CWE-")
        )
        rows.append(
            f"<tr><td><b>{escape(rule.title)}</b><br><code>{escape(code)}</code></td>"
            f'<td><span class="sev s-{sev}">{sev.capitalize()}</span></td>'
            f'<td class="num">{n:,}</td>'
            f'<td><a href="{escape(rule.owasp_url)}">{escape(rule.owasp_id)}</a></td>'
            f"<td>{cwe or '—'}</td>"
            f'<td class="desc">{escape(rule.summary)}</td></tr>'
        )
    return f"""
<section aria-labelledby="rules-h">
  <h2 id="rules-h">Rules that fired</h2>
  <p class="section-note">The rule id is the finding code in the JSON and SARIF output.</p>
  <table>
    <thead><tr><th>Rule</th><th>Severity</th><th>Findings</th><th>OWASP</th><th>CWE</th>
    <th>What it means</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>"""


def _setup(report: ScanReport) -> str:
    if not report.checks:
        return ""
    rows = "".join(
        f"<tr><td><b>{escape(name)}</b></td>"
        f"<td>{escape(', '.join(detectors)) if detectors else '—'}</td></tr>"
        for name, detectors in report.checks.items()
    )
    return f"""
<section aria-labelledby="setup-h">
  <h2 id="setup-h">What was checked</h2>
  <p class="section-note">Every record went through each check below. A check flags a record
  when its detectors match; a detector that cannot run (no stored vectors, no callback) adds
  nothing. Heuristic detectors catch the phrasings they were written for and miss paraphrases
  and encodings, so a finding is a signal, not proof.</p>
  <table>
    <thead><tr><th>Check</th><th>Detectors</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>"""


def _footer(report: ScanReport) -> str:
    return f"""
<footer>
  Generated by <a href="{REPO_URL}">Mimvo</a> {escape(report.mimvo_version)}
  on this machine. Snippets are masked and secret values are never written to the report.
  Recommended actions are suggestions; Mimvo never changes the store.
</footer>"""


# -- helpers --------------------------------------------------------------------


def _anchors(findings: Iterable[ScanFinding]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for item in findings:
        base = f"f-{item.fingerprint[:16]}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}-{n}")
    return out


def _first_anchor_by_record(findings: list[ScanFinding]) -> dict[str, str]:
    first: dict[str, str] = {}
    for item, anchor in zip(findings, _anchors(findings), strict=True):
        first.setdefault(item.record_id, anchor)
    return first


def _records(n: int) -> str:
    return f"{n:,} record" + ("" if n == 1 else "s")


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _when(when: datetime | None) -> str:
    stamp = when or datetime.now(UTC)
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    return f"{stamp.day} {stamp:%b %Y, %H:%M} UTC"


def _duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes} min {secs} s"


def _severity_sentence(counts: dict[str, int]) -> str:
    parts = [f"{counts[s.value]:,} {s.value}" for s in _SEVERITY_ORDER if counts.get(s.value)]
    return "Findings by severity: " + ", ".join(parts)


def _humanize(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _evidence_text(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


__all__ = ["finding_label", "render_html"]
