"""GitHub-flavoured Markdown output, for job summaries and PR comments.

Write it to `$GITHUB_STEP_SUMMARY` and the scan shows up on the workflow
run page, or post it as a pull-request comment. The layout follows what
CI security tools print there: a one-line verdict, counts by severity, the
records to act on, a findings table, then folded details per finding.

Memory text is attacker-controlled, so snippets go in fenced code blocks
and record ids in code spans; neither can inject links, images, or HTML.
"""

from __future__ import annotations

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
_MARK: dict[Severity, str] = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
    Severity.INFO: "⚪",
}
_ACTION_COPY: dict[Action, str] = {
    Action.DELETE: "Delete",
    Action.QUARANTINE: "Quarantine",
    Action.REVIEW: "Review",
}
_ACTION_HINT: dict[Action, str] = {
    Action.DELETE: "rotate any credential first",
    Action.QUARANTINE: "keep it out of retrieval until confirmed",
    Action.REVIEW: "a person should read it",
}
_MAX_IDS = 15


def render_markdown(report: ScanReport, *, max_findings: int = 50) -> str:
    """Build a Markdown summary of `report`.

    Args:
        report: The scan result.
        max_findings: Rows in the findings table and the details block.
            The rest are counted, not listed, to stay under comment limits.

    Returns:
        Markdown text ending in a newline.
    """
    parts: list[str] = [f"## Mimvo scan: {_verdict(report)}", "", _meta(report), ""]
    if report.sample and report.total >= report.sample:
        parts += [
            f"> [!NOTE]\n> Only the first {report.sample:,} records were scanned (`--sample`).",
            "",
        ]
    if report.errors:
        parts += _incomplete(report)
    if report.clean:
        parts += [
            "No poisoned facts, hidden instructions, or leaked secrets were found by the "
            f"{_checks_phrase(report)}.",
            "",
            _footer(report),
        ]
        return "\n".join(parts) + "\n"

    parts += _severity_table(report)
    parts += _plan(report)
    shown = report.findings[: max(max_findings, 0)]
    parts += _findings_table(shown)
    hidden = len(report.findings) - len(shown)
    if hidden > 0:
        parts += [
            f"_{hidden:,} more findings are in the HTML or JSON report._",
            "",
        ]
    parts += _details(shown)
    parts.append(_footer(report))
    return "\n".join(parts) + "\n"


def _verdict(report: ScanReport) -> str:
    if report.clean and not report.complete:
        unchecked = report.records_with_errors
        return f"⚠️ incomplete, {unchecked:,} of {_records(report.total)} not fully checked"
    if report.clean:
        return f"no problems in {_records(report.total)}"
    worst = report.worst_severity()
    mark = _MARK[worst] + " " if worst is not None else ""
    return f"{mark}{report.flagged:,} of {_records(report.total)} need action"


def _meta(report: ScanReport) -> str:
    bits = []
    if report.source:
        bits.append(f"Source {_code(report.source)}")
    if report.generated_at is not None:
        bits.append(report.generated_at.strftime("%Y-%m-%d %H:%M UTC"))
    if report.duration_seconds is not None:
        secs = report.duration_seconds
        bits.append(f"{secs * 1000:.0f} ms" if secs < 1 else f"{secs:.1f} s")
    bits.append(f"Mimvo {report.mimvo_version}")
    return " · ".join(bits)


def _incomplete(report: ScanReport) -> list[str]:
    lines = [
        "> [!WARNING]",
        f"> Scan incomplete: {_records(report.records_with_errors)} were not fully checked "
        "because a check or detector failed. A failure is not a finding; fix it and scan again.",
        ">",
    ]
    for error in report.errors[:10]:
        where = _code(error.check + (f"/{error.detector}" if error.detector else ""))
        lines.append(f"> - {where} raised {_code(error.error_type)} on {_records(error.records)}")
    return [*lines, ""]


def _severity_table(report: ScanReport) -> list[str]:
    counts = report.by_severity
    lines = ["| Severity | Findings |", "| :-- | --: |"]
    for sev in _SEVERITY_ORDER:
        n = counts.get(sev.value, 0)
        if n:
            lines.append(f"| {_MARK[sev]} {sev.value.capitalize()} | {n:,} |")
    lines.append(f"| **Records flagged** | **{report.flagged:,}** ({report.flagged_pct:g}%) |")
    return [*lines, ""]


def _plan(report: ScanReport) -> list[str]:
    lines = ["### What to do", ""]
    for action, ids in report.action_plan().items():
        if not ids:
            continue
        listed = ", ".join(_code(i) for i in ids[:_MAX_IDS])
        more = f" and {len(ids) - _MAX_IDS:,} more" if len(ids) > _MAX_IDS else ""
        lines.append(
            f"- **{_ACTION_COPY[action]} {_records(len(ids))}** "
            f"({_ACTION_HINT[action]}): {listed}{more}"
        )
    return [*lines, ""]


def _findings_table(findings: list[ScanFinding]) -> list[str]:
    lines = [
        "### Findings",
        "",
        "| Severity | Finding | Record | Action | Confidence | OWASP |",
        "| :-- | :-- | :-- | :-- | --: | :-- |",
    ]
    for item in findings:
        owasp_id = item.owasp.split(":", 1)[0]
        lines.append(
            f"| {_MARK[item.severity]} {item.severity.value} "
            f"| {_cell(item.title)} {_cell(_code(item.code))} "
            f"| {_cell(_code(item.record_id))} "
            f"| {_ACTION_COPY[item.recommended_action]} "
            f"| {_confidence(item.confidence)} "
            f"| [{_cell(owasp_id)}]({rule_for(item.code).owasp_url}) |"
        )
    return [*lines, ""]


def _details(findings: list[ScanFinding]) -> list[str]:
    if not findings:
        return []
    lines = ["<details>", "<summary>Evidence and fixes for each finding</summary>", ""]
    for item in findings:
        rule = rule_for(item.code)
        lines += [
            f"#### {_MARK[item.severity]} {_text(item.title)} in {_code(item.record_id)}",
            "",
            _text(item.message or rule.summary),
            "",
        ]
        if item.snippet:
            lines += [_fence(item.snippet), ""]
        why = f"Detectors: {', '.join(_code(d) for d in item.detectors) or 'n/a'}"
        for key, value in item.evidence.items():
            why += f"; {_text(key.replace('_', ' '))}: {_code(_evidence_text(value))}"
        lines += [why, ""]
        lines += [f"{i}. {_text(step)}" for i, step in enumerate(item.remediation, start=1)]
        refs = " · ".join(f"[{_text(ref.label)}]({ref.url})" for ref in rule.references)
        lines += ["", refs, ""]
    lines += ["</details>", ""]
    return lines


def _confidence(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _footer(report: ScanReport) -> str:
    return (
        f"<sub>Generated by [Mimvo]({REPO_URL}) {report.mimvo_version}. "
        "Snippets are masked; secret values never leave the scanner.</sub>"
    )


def _checks_phrase(report: ScanReport) -> str:
    names = list(report.checks)
    if not names:
        return "checks that ran"
    return ", ".join(names) + (" check" if len(names) == 1 else " checks")


def _evidence_text(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _records(n: int) -> str:
    return f"{n:,} record" + ("" if n == 1 else "s")


def _code(text: str) -> str:
    """Wrap `text` in a code span that no backtick inside can close."""
    text = " ".join(text.split())
    run = max((len(r) for r in _backtick_runs(text)), default=0)
    fence = "`" * (run + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _backtick_runs(text: str) -> list[str]:
    runs: list[str] = []
    current = ""
    for ch in text:
        if ch == "`":
            current += ch
        elif current:
            runs.append(current)
            current = ""
    if current:
        runs.append(current)
    return runs


def _fence(text: str) -> str:
    longest = max((len(r) for r in _backtick_runs(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _cell(text: str) -> str:
    """Escape a table cell so a `|` in the content cannot split the row."""
    return text.replace("|", "\\|")


def _text(text: str) -> str:
    """Escape prose so content cannot add links, images, or HTML."""
    out = text.replace("\\", "\\\\")
    for ch in "`*_[]<>#!|~":
        out = out.replace(ch, "\\" + ch)
    return " ".join(out.split())


__all__ = ["render_markdown"]
