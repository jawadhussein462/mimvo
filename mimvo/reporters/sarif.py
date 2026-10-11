"""SARIF 2.1.0 output, for GitHub code scanning and other SARIF viewers.

SARIF is the format Semgrep, CodeQL, Trivy, Gitleaks, and Bandit emit.
Uploading it with `github/codeql-action/upload-sarif` puts Mimvo
findings in a repository's Security tab, with severity, rule help, and
de-duplication across runs.

Memory records are not source files, so each result points at the scanned
store (the JSONL path, or a `store/collection` label) and names the record
as a logical location. Snippets are left out by default: SARIF usually
goes to a third-party dashboard, and the record id is enough to find it.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from .._version import __version__
from ..models.enums import Severity
from ..models.results import ScanFinding, ScanReport
from ..rules import REPO_URL, Rule, rule_for

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
SARIF_VERSION = "2.1.0"
FINGERPRINT_KEY = "mimvoFingerprint/v1"

# GitHub reads `security-severity` on a rule: >= 9.0 critical, 7.0-8.9 high,
# 4.0-6.9 medium, 0.1-3.9 low.
_SECURITY_SEVERITY: dict[Severity, str] = {
    Severity.CRITICAL: "9.5",
    Severity.HIGH: "8.0",
    Severity.MEDIUM: "5.5",
    Severity.LOW: "3.0",
}
_LEVEL: dict[Severity, str] = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}
_UNSAFE_URI = re.compile(r"[^A-Za-z0-9._~/-]+")


def render_sarif(report: ScanReport, *, include_snippets: bool = False) -> str:
    """Build a SARIF 2.1.0 log for `report`.

    Args:
        report: The scan result.
        include_snippets: Put the masked snippet on each result's region.
            Off by default so memory text does not reach the dashboard.

    Returns:
        The SARIF log as an indented JSON string.
    """
    return json.dumps(sarif_log(report, include_snippets=include_snippets), indent=2) + "\n"


def sarif_log(report: ScanReport, *, include_snippets: bool = False) -> dict[str, Any]:
    """Build the SARIF log as a Python dict. See `render_sarif`.

    Args:
        report: The scan result.
        include_snippets: Put the masked snippet on each result's region.

    Returns:
        A dict ready for `json.dumps`.
    """
    worst: dict[str, Severity] = {}
    for item in report.findings:
        current = worst.get(item.code)
        if current is None or item.severity.rank > current.rank:
            worst[item.code] = item.severity
    rule_ids = sorted(worst)
    index = {rule_id: i for i, rule_id in enumerate(rule_ids)}
    artifact = _artifact_uri(report.source)

    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": "Mimvo",
                "version": __version__,
                "semanticVersion": __version__,
                "informationUri": REPO_URL,
                "rules": [_rule(rule_for(rid), worst[rid]) for rid in rule_ids],
            }
        },
        "automationDetails": {"id": f"mimvo/{_slug(report.source) or 'scan'}/"},
        "results": [
            _result(item, index[item.code], artifact, include_snippets=include_snippets)
            for item in report.findings
        ],
        "columnKind": "unicodeCodePoints",
        "properties": {
            "source": report.source,
            "recordsScanned": report.total,
            "recordsFlagged": report.flagged,
            "recordsWithErrors": report.records_with_errors,
            "sample": report.sample,
            "checks": report.checks,
        },
    }
    invocation: dict[str, Any] = {"executionSuccessful": report.complete}
    if report.errors:
        invocation["toolExecutionNotifications"] = [
            {
                "level": "error",
                "descriptor": {"id": "check-failure"},
                "message": {
                    "text": (
                        f"{error.check}"
                        + (f"/{error.detector}" if error.detector else "")
                        + f" raised {error.error_type} on {error.records} record(s); "
                        "those records were not fully checked."
                    )
                },
                "properties": {
                    "check": error.check,
                    "detector": error.detector,
                    "errorType": error.error_type,
                    "records": error.records,
                    "recordIds": error.record_ids,
                },
            }
            for error in report.errors
        ]
    if report.generated_at is not None:
        invocation["startTimeUtc"] = _iso(report.generated_at)
        if report.duration_seconds is not None:
            end = report.generated_at + timedelta(seconds=report.duration_seconds)
            invocation["endTimeUtc"] = _iso(end)
    run["invocations"] = [invocation]
    return {"$schema": SARIF_SCHEMA, "version": SARIF_VERSION, "runs": [run]}


def _rule(rule: Rule, severity: Severity) -> dict[str, Any]:
    tags = ["security", "ai-memory", rule.category, rule.owasp_id, *rule.cwe]
    properties: dict[str, Any] = {"tags": tags, "precision": "medium"}
    if severity in _SECURITY_SEVERITY:
        properties["security-severity"] = _SECURITY_SEVERITY[severity]
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(rule.remediation, start=1))
    refs = "\n".join(f"- [{ref.label}]({ref.url})" for ref in rule.references)
    return {
        "id": rule.id,
        "name": "".join(part.capitalize() for part in re.split(r"[\s_-]+", rule.title)),
        "shortDescription": {"text": rule.title},
        "fullDescription": {"text": rule.summary},
        "help": {
            "text": rule.summary + "\n\n" + " ".join(rule.remediation),
            "markdown": f"{rule.summary}\n\n**How to fix**\n\n{steps}\n\n**References**\n\n{refs}",
        },
        "helpUri": rule.help_uri,
        "defaultConfiguration": {"level": _LEVEL[severity]},
        "properties": properties,
    }


def _result(
    item: ScanFinding, rule_index: int, artifact: str, *, include_snippets: bool
) -> dict[str, Any]:
    region: dict[str, Any] = {"startLine": 1}
    if include_snippets and item.snippet:
        region["snippet"] = {"text": item.snippet}
    message = f"{item.title} in record '{item.record_id}'. {item.message}".strip()
    return {
        "ruleId": item.code,
        "ruleIndex": rule_index,
        "level": _LEVEL[item.severity],
        "message": {"text": message},
        "locations": [
            {
                "physicalLocation": {"artifactLocation": {"uri": artifact}, "region": region},
                "logicalLocations": [
                    {
                        "name": item.record_id,
                        "fullyQualifiedName": f"{artifact}/{item.record_id}",
                        "kind": "object",
                    }
                ],
            }
        ],
        "partialFingerprints": {FINGERPRINT_KEY: item.fingerprint},
        "properties": {
            "recordId": item.record_id,
            "severity": item.severity.value,
            "action": item.recommended_action.value,
            "check": item.check,
            "detectors": item.detectors,
            "confidence": item.confidence,
            "owasp": item.owasp,
            "cwe": item.cwe,
        },
    }


def _artifact_uri(source: str) -> str:
    """Pick the URI results point at.

    A JSONL scan points at the file itself. A store scan has no file, so it
    points at `store/collection`, which code-scanning UIs show as the path.
    """
    kind, _, name = source.partition(":")
    if kind == "jsonl" and name and name != "stdin":
        return name.replace("\\", "/").lstrip("/")
    slug = _slug(source)
    return slug or "memory-store"


def _slug(source: str) -> str:
    return _UNSAFE_URI.sub("-", source.replace(":", "/")).strip("-/")


def _iso(when: datetime) -> str:
    if when.tzinfo is not None:
        when = when.astimezone(UTC)
    return when.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


__all__ = ["FINGERPRINT_KEY", "SARIF_SCHEMA", "SARIF_VERSION", "render_sarif", "sarif_log"]
