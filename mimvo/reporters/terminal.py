"""The plain-text summary printed after a scan, and `str(report)`.

Like the other reporters, this module only reads a `ScanReport`; the data
model does not know how it is displayed.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models.results import ScanReport

_RESET = "\033[0m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_AMBER = "\033[38;5;214m"
_UNDERLINE = "\033[4m"
_SEVERITY_COLOR = {
    "critical": "\033[31m",
    "high": "\033[38;5;208m",
    "medium": "\033[33m",
    "low": "\033[38;5;67m",
    "info": "\033[90m",
}
_SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")


def format_scan_summary(
    report: ScanReport,
    *,
    report_path: str | None = None,
    json_path: str | None = None,
    color: bool = False,
    details: bool = False,
    max_rows: int = 20,
    outputs: Sequence[tuple[str, str]] = (),
) -> str:
    """Build the text printed after a scan.

    The lines are: how many records were scanned, how many were flagged,
    a count per severity, optionally a findings table, then the files that
    were written.

    Args:
        report: The scan result to summarize.
        report_path: HTML file that was written. Omitted from the text
            when `None`.
        json_path: JSON file that was written. Omitted when `None`.
        color: When `True`, wrap marks in ANSI color: green checks, an
            amber flag, and a colored dot per severity. When `False`,
            return plain text.
        details: When `True`, add a table of findings (severity, rule,
            record id, action, confidence, OWASP) under the counts. Memory
            text is never printed, so the table is safe for CI logs.
        max_rows: Most table rows to print before summarising the rest.
        outputs: More written files as `(label, path)` pairs, such as
            `("SARIF", "results.sarif")`, listed after the JSON file.

    Returns:
        A multi-line string. Print it, or use `str(report)` for the
        plain-text version.
    """
    check = _paint("✓", _GREEN, enabled=color)
    lines = [f"{check} Scanned {_records(report.total)}"]
    if report.flagged:
        bang = _paint("!", _AMBER, enabled=color)
        lines.append(f"{bang} {_records(report.flagged)} flagged ({report.flagged_pct:.2f}%)")
        severity = _severity_line(report, color=color)
        if severity:
            lines.append(severity)
    else:
        lines.append(f"{check} {_records(0)} flagged")
    if report.errors:
        cross = _paint("✗", _RED, enabled=color)
        lines.append(
            f"{cross} Scan incomplete: {_records(report.records_with_errors)} not fully checked"
        )
        for error in report.errors[:3]:
            where = error.check + (f"/{error.detector}" if error.detector else "")
            lines.append(f"  • {where} raised {error.error_type} on {_records(error.records)}")
        if len(report.errors) > 3:
            lines.append(f"  • … {len(report.errors) - 3} more in the report")

    if details and report.findings:
        lines.append("")
        lines.extend(_findings_table(report, color=color, max_rows=max_rows))

    written: list[str] = []
    if report_path:
        path = _paint(report_path, _UNDERLINE, enabled=color)
        written.append(f"{check} Report written to {path}")
    if json_path:
        written.append(
            f"{check} Findings written to {_paint(json_path, _UNDERLINE, enabled=color)}"
        )
    for label, out_path in outputs:
        written.append(f"{check} {label} written to {_paint(out_path, _UNDERLINE, enabled=color)}")
    if written:
        lines.append("")
        lines.extend(written)
    return "\n".join(lines)


_DIM = "\033[2m"
_ID_WIDTH = 28


def _findings_table(report: ScanReport, *, color: bool, max_rows: int) -> list[str]:
    """Render findings as aligned columns, worst first.

    Args:
        report: The scan result. Findings are already sorted by severity.
        color: Paint the severity column and dim the header.
        max_rows: Rows to show before a "more in the report" line.

    Returns:
        Lines of text, each indented by two spaces.
    """
    rows = [
        (
            item.severity.value,
            item.code,
            _clip(item.record_id, _ID_WIDTH),
            item.recommended_action.value,
            "—" if item.confidence is None else f"{item.confidence:.2f}",
            item.owasp.split(":", 1)[0],
        )
        for item in report.findings[: max(max_rows, 0)]
    ]
    header = ("SEVERITY", "RULE", "RECORD", "ACTION", "CONF", "OWASP")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def line(cells: tuple[str, ...], *, sev: str | None = None) -> str:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        if sev is not None:
            padded[0] = _paint(padded[0], _SEVERITY_COLOR.get(sev, ""), enabled=color)
        return "  " + "  ".join(padded).rstrip()

    out = [_paint(line(header), _DIM, enabled=color)]
    out.extend(line(row, sev=row[0]) for row in rows)
    hidden = len(report.findings) - len(rows)
    if hidden > 0:
        noun = "finding" if hidden == 1 else "findings"
        out.append(_paint(f"  … {hidden:,} more {noun}", _DIM, enabled=color))
    return out


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _records(n: int) -> str:
    noun = "record" if n == 1 else "records"
    return f"{n:,} {noun}"


def _severity_line(report: ScanReport, *, color: bool) -> str:
    counts = report.by_severity
    parts: list[str] = []
    for name in _SEVERITY_ORDER:
        count = counts.get(name, 0)
        if not count:
            continue
        label = f"• {count:,} {name}"
        parts.append(_paint(label, _SEVERITY_COLOR[name], enabled=color))
    if not parts:
        return ""
    return "  " + "  ".join(parts)


def _paint(text: str, code: str, *, enabled: bool) -> str:
    if not enabled or not code:
        return text
    return f"{code}{text}{_RESET}"


__all__ = ["format_scan_summary"]
