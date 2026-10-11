"""What `Mimvo.scan` gives back: `ScanFinding` rows and a `ScanReport`."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from .._version import __version__
from ..exceptions import CheckError
from ..rules import rule_for
from .enums import Action, Severity
from .finding import Finding

SCHEMA_VERSION = "3.0"
"""Version of the JSON report layout. Bumped when fields change meaning."""


def finding_fingerprint(record_id: str, code: str) -> str:
    """Stable id for one finding on one record, for baselines and dedupe.

    The same rule on the same record gets the same fingerprint on every
    scan, so a ticket, a SARIF alert, or a suppression list can track it.
    Only the record id and the rule are hashed, never the memory text.

    Args:
        record_id: Id of the stored record.
        code: Finding code, such as `"secret_detected"`.

    Returns:
        32 hex characters.
    """
    return hashlib.sha256(f"{code}\x1f{record_id}".encode()).hexdigest()[:32]


class ScanFinding(Finding):
    """A `Finding` located on one stored record, safe to forward.

    Everything a check put on the `Finding` is here under the same names
    (`code`, `severity`, `message`, `check`, `recommended_action`,
    `evidence`, `confidence`), so code that handles findings handles scan
    findings too. The scan adds where it was found and the rule text a
    reader needs. The snippet is secret-masked; the raw memory text is not
    included.

    Attributes:
        record_id: Id of the stored record.
        recommended_action: Always set on a scan finding: review,
            quarantine, or delete.
        message: The check's sentence. Filled from the rule when empty.
        snippet: A short, masked excerpt of the record.
        title: Short human title for the rule, such as `"Leaked secret"`.
            Filled from the rule catalogue when left empty.
        remediation: Steps to fix it, from the rule catalogue.
        cwe: CWE ids for the weakness, such as `["CWE-312"]`.
        fingerprint: Stable hash of rule and record id. See
            `finding_fingerprint`.
        detectors: Names of the detectors that agreed on it. Taken from
            the check's `evidence["detectors"]` before evidence is masked,
            or the check name when a plain check raised it.
    """

    record_id: str
    recommended_action: Action
    message: str = ""
    snippet: str = ""
    title: str = ""
    remediation: list[str] = Field(default_factory=list)
    cwe: list[str] = Field(default_factory=list)
    fingerprint: str = ""
    detectors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_from_rule(self) -> ScanFinding:
        rule = rule_for(self.code)
        if not self.message:
            self.message = rule.message
        if not self.detectors:
            raw = self.evidence.get("detectors")
            if isinstance(raw, list):
                self.detectors = [str(name) for name in raw]
            elif self.check:
                self.detectors = [self.check]
        if not self.title:
            self.title = rule.title
        if not self.remediation:
            self.remediation = list(rule.remediation)
        if not self.cwe:
            self.cwe = list(rule.cwe)
        if not self.fingerprint:
            self.fingerprint = finding_fingerprint(self.record_id, self.code)
        return self

    @classmethod
    def from_finding(
        cls, finding: Finding, *, record_id: str, snippet: str = "", **updates: Any
    ) -> ScanFinding:
        """Locate a check's `Finding` on a record without losing any field.

        Args:
            finding: What the check returned.
            record_id: Id of the record it was found on.
            snippet: Masked excerpt of the record.
            **updates: Fields to override, such as a resolved
                `recommended_action` or masked `evidence`.

        Returns:
            A scan finding carrying every field of `finding`.
        """
        data = finding.model_dump()
        data.update(updates)
        return cls(record_id=record_id, snippet=snippet, **data)


class ScanError(BaseModel):
    """A check or detector that failed during a scan, grouped across records.

    An error is not a finding. It says the affected records were not fully
    checked, not that anything is wrong with them, so it never adds a
    record to the action plan.

    Attributes:
        check: Name of the check, such as `"injection"`.
        detector: Name of the detector that raised, or `None` when the
            whole check raised.
        error_type: Exception class name, such as `"BackendError"`.
        message: The first error message, secret-masked and shortened.
        records: How many records hit this error.
        record_ids: The first few affected record ids.
    """

    model_config = ConfigDict(extra="forbid")

    check: str
    detector: str | None = None
    error_type: str
    message: str = ""
    records: int = 0
    record_ids: list[str] = Field(default_factory=list)


class ScanReport(BaseModel):
    """The result of `Mimvo.scan`.

    Counts every record read and lists each problem (poisoned fact, hidden
    instruction, leaked secret) with a recommended action.

    Attributes:
        schema_version: Layout version of the JSON form of this report.
        mimvo_version: Mimvo version that produced it.
        total: How many records were read.
        findings: One `ScanFinding` per problem, not per record. A record
            can appear more than once if it has several problems.
        source: Label of where the records came from, such as
            `"chroma:agent_memory"`. Empty when the caller did not set it.
        sample: Cap on how many records were requested. `None` means the
            scan was not capped.
        generated_at: When the scan started, in UTC. `None` if not set.
        duration_seconds: How long the checks took. `None` if not measured.
        checks: Each check that ran, mapped to its detector names, so a
            reader can tell what was looked for as well as what was found.
        errors: Checks and detectors that failed, grouped by check,
            detector, and exception type. Empty on a complete scan.
        records_with_errors: Distinct records at least one check or
            detector failed on.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = SCHEMA_VERSION
    mimvo_version: str = __version__
    total: int = 0
    findings: list[ScanFinding] = Field(default_factory=list)
    source: str = ""
    sample: int | None = None
    generated_at: datetime | None = None
    duration_seconds: float | None = None
    checks: dict[str, list[str]] = Field(default_factory=dict)
    errors: list[ScanError] = Field(default_factory=list)
    records_with_errors: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def complete(self) -> bool:
        """Whether every check and detector ran on every record.

        Returns:
            `False` when any check or detector raised. Findings from the
            parts that did run are still in `findings`, but a record with
            no finding may not have been fully checked. The CLI exits `2`
            on an incomplete scan.
        """
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise when the scan is incomplete, like `requests`' `raise_for_status`.

        The engine never raises because a check failed: it records the
        failure in `errors`. Call this when you would rather treat an
        incomplete scan as an exception.

        Raises:
            CheckError: At least one check or detector raised. The message
                names each failing check and detector and how many records
                it affected.
        """
        if self.complete:
            return
        parts = [
            f"{e.check}/{e.detector or '*'} {e.error_type} on {e.records} record(s)"
            for e in self.errors
        ]
        raise CheckError("Scan incomplete: " + "; ".join(parts))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def flagged(self) -> int:
        """How many distinct records have at least one finding.

        Returns:
            The number of unique `id` values in `findings`. A record with
            two problems counts once.
        """
        return len({item.record_id for item in self.findings})

    @computed_field  # type: ignore[prop-decorator]
    @property
    def flagged_pct(self) -> float:
        """Share of scanned records that were flagged, as a percent.

        Returns:
            `100 * flagged / total`, rounded to two decimals so a large
            store reads `0.28` rather than `0.3`. `0.0` when `total` is 0.
        """
        if not self.total:
            return 0.0
        return round(100.0 * self.flagged / self.total, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def by_severity(self) -> dict[str, int]:
        """Count findings by severity name.

        Returns:
            A dict such as `{"high": 2, "critical": 1}`. Severities with
            zero findings are omitted. Keys are the severity strings.
        """
        counts: dict[str, int] = {}
        for item in self.findings:
            key = item.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def by_rule(self) -> dict[str, int]:
        """Count findings by finding code, most frequent first.

        Returns:
            A dict such as `{"memory_poisoning": 3, "secret_detected": 1}`.
        """
        counts: dict[str, int] = {}
        for item in self.findings:
            counts[item.code] = counts.get(item.code, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def by_action(self) -> dict[str, int]:
        """Count flagged records by the strongest action they need.

        A record with a `delete` finding and a `review` finding counts once,
        under `delete`. The counts add up to `flagged`.

        Returns:
            A dict such as `{"delete": 1, "quarantine": 1, "review": 2}`.
            Actions with no records are omitted.
        """
        return {action.value: len(ids) for action, ids in self.action_plan().items() if ids}

    @property
    def clean(self) -> bool:
        """Whether the scan found nothing.

        Returns:
            `True` when `findings` is empty.
        """
        return not self.findings

    def worst_severity(self) -> Severity | None:
        """Return the most serious severity in this report.

        Returns:
            The highest `Severity` among `findings`, or `None` when the
            report has no findings.
        """
        if not self.findings:
            return None
        return max((item.severity for item in self.findings), key=lambda s: s.rank)

    def action_plan(self) -> dict[Action, list[str]]:
        """Group flagged records by the strongest action each one needs.

        This is the to-do list a report opens with: which records to
        delete, which to quarantine, which to review.

        Returns:
            Delete, quarantine, and review, in that order, each mapped to
            record ids in report order. Every flagged record appears once.
        """
        strongest: dict[str, Action] = {}
        for item in self.findings:
            current = strongest.get(item.record_id)
            if current is None or item.recommended_action.precedence > current.precedence:
                strongest[item.record_id] = item.recommended_action
        plan: dict[Action, list[str]] = {
            Action.DELETE: [],
            Action.QUARANTINE: [],
            Action.REVIEW: [],
        }
        for record_id, action in strongest.items():
            plan[action].append(record_id)
        return plan

    def at_or_above(
        self, severity: Severity, *, min_confidence: float | None = None
    ) -> list[ScanFinding]:
        """Findings as serious as `severity`, or worse.

        Args:
            severity: The threshold, such as `Severity.HIGH`.
            min_confidence: Also drop findings whose confidence is below
                this. Findings with no confidence (`None`) are kept, so an
                unscored detector is never silently ignored.

        Returns:
            The matching findings, in report order. This is what the CLI's
            `--fail-on` and `--min-confidence` gate on.
        """
        return [
            item
            for item in self.findings
            if item.severity.is_at_least(severity)
            and (
                min_confidence is None
                or item.confidence is None
                or item.confidence >= min_confidence
            )
        ]

    def __str__(self) -> str:
        # Imported here: the reporters depend on this module, not the reverse.
        from ..reporters.terminal import format_scan_summary

        return format_scan_summary(self)


__all__ = [
    "SCHEMA_VERSION",
    "ScanError",
    "ScanFinding",
    "ScanReport",
    "finding_fingerprint",
]
