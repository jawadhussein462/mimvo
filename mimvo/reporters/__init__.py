"""Render a `ScanReport` for the people and tools that act on it.

* `render_html` — one self-contained file to forward to whoever acts on it.
* `render_markdown` — a summary for CI job pages and pull-request comments.
* `render_sarif` — SARIF 2.1.0 for GitHub code scanning and other viewers.
* `format_scan_summary` — the plain-text summary the CLI prints.
* `ScanReport.model_dump_json()` — the full machine-readable report.
"""

from __future__ import annotations

from .html import render_html
from .markdown import render_markdown
from .sarif import render_sarif
from .terminal import format_scan_summary

__all__ = ["format_scan_summary", "render_html", "render_markdown", "render_sarif"]
