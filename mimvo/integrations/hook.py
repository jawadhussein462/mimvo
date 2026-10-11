"""Check a memory before it is written, or filter records at retrieve time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..client import Mimvo
from ..exceptions import ConfigurationError
from ..models.enums import Action, Severity
from ..models.finding import Finding
from ..models.memory import MemoryRecord
from ..models.results import ScanError, ScanFinding, ScanReport


def _strongest_action(findings: Sequence[Finding]) -> Action | None:
    actions = [f.recommended_action for f in findings if f.recommended_action is not None]
    if not actions:
        return None
    return max(actions, key=lambda action: action.precedence)


def _worst_severity(findings: Sequence[Finding]) -> Severity | None:
    if not findings:
        return None
    return max((f.severity for f in findings), key=lambda item: item.rank)


@dataclass(frozen=True)
class WriteDecision:
    """Whether a memory may be written, and why.

    Attributes:
        allow: `True` when nothing reached `block_at` and, under
            `fail_closed`, every check ran.
        findings: Problems found in the candidate text, with every field the
            checks set (evidence included).
        action: Strongest recommended action among the findings, or `None`.
        report: The scan report for this one record. `report.errors`
            lists any check or detector that failed.
    """

    allow: bool
    findings: list[ScanFinding]
    action: Action | None
    report: ScanReport


class WriteGuard:
    """Reject a memory before it is stored.

    Call `inspect` (or `allow`) in the writer's path; the store scanner
    remains the after-the-fact audit.

    When a check or detector fails on the candidate and the client is
    `fail_closed` (the default), the write is refused: the text was not
    fully checked. The decision carries no finding for that; the failure is
    in `decision.report.errors`.

    Args:
        client: Scanner used for the check. `None` builds `Mimvo()`.
        block_at: Lowest severity that refuses the write. Default `HIGH`.
    """

    def __init__(
        self,
        *,
        client: Mimvo | None = None,
        block_at: Severity = Severity.HIGH,
    ) -> None:
        self.client = client or Mimvo()
        self.block_at = block_at

    def inspect(
        self,
        content: str,
        *,
        metadata: dict[str, object] | None = None,
        memory_id: str = "pending",
        neighbours: Sequence[MemoryRecord] = (),
    ) -> WriteDecision:
        """Scan one candidate and decide whether it may be written.

        Args:
            content: Text about to be stored.
            metadata: Optional store fields copied onto the record.
            memory_id: Id used in the report. Default `"pending"`. Must
                differ from every neighbour's id.
            neighbours: Stored memories near the candidate, such as the
                top few results of your store's similarity search for
                `content`. Corpus detectors (TrustRAG clusters, NLI
                contradictions, hubness) compare the candidate with them;
                without neighbours those detectors see the candidate alone.
                Neighbours are context only: their own findings and
                failures do not affect the decision.

        Returns:
            A `WriteDecision`. `allow` is `False` when any finding on the
            candidate is at least `block_at`, or when a check failed on the
            candidate and the client is `fail_closed`.

        Raises:
            ConfigurationError: A neighbour has the candidate's id.
        """
        if any(n.id == memory_id for n in neighbours):
            raise ConfigurationError(f"A neighbour has the candidate's id {memory_id!r}.")
        record = MemoryRecord(id=memory_id, content=content, metadata=metadata or {})
        report = self.client.scan([record, *neighbours])
        if neighbours:
            report = report.model_copy(
                update={
                    "findings": [f for f in report.findings if f.record_id == memory_id],
                    "errors": [e for e in report.errors if _touches(e, memory_id)],
                }
            )
        findings = list(report.findings)
        worst = _worst_severity(findings)
        blocked = worst is not None and worst.is_at_least(self.block_at)
        if not report.complete and self.client.config.fail_closed:
            blocked = True
        return WriteDecision(
            allow=not blocked,
            findings=findings,
            action=_strongest_action(findings),
            report=report,
        )

    def allow(
        self,
        content: str,
        *,
        metadata: dict[str, object] | None = None,
        memory_id: str = "pending",
        neighbours: Sequence[MemoryRecord] = (),
    ) -> bool:
        """Return whether `content` may be written.

        Args:
            content: Text about to be stored.
            metadata: See `inspect`.
            memory_id: See `inspect`.
            neighbours: See `inspect`.

        Returns:
            `True` when `inspect` would allow the write.
        """
        return self.inspect(
            content, metadata=metadata, memory_id=memory_id, neighbours=neighbours
        ).allow


def _touches(error: ScanError, record_id: str) -> bool:
    """Whether a grouped scan error may include `record_id`.

    True when the id is listed, or when the error affected more records
    than it lists ids for (the list is capped), so the candidate cannot be
    ruled out.
    """
    return record_id in error.record_ids or error.records > len(error.record_ids)


class RetrieveGuard:
    """Drop retrieved memories that the scanner flags, given a query.

    FilterRAG-style: this runs at inference time, when a query exists. A
    store scan has no query, so this class is not on the CLI path.
    Probe-query detection is the scan-time counterpart.

    A record a check or detector failed on is dropped when the client is
    `fail_closed` (the default), because it was not fully checked.

    Args:
        client: Scanner used on the retrieved batch.
        block_at: Lowest severity that drops a record. Default `HIGH`.
    """

    def __init__(
        self,
        *,
        client: Mimvo | None = None,
        block_at: Severity = Severity.HIGH,
    ) -> None:
        self.client = client or Mimvo()
        self.block_at = block_at

    def filter(
        self,
        records: Sequence[MemoryRecord],
        *,
        query: str,
    ) -> list[MemoryRecord]:
        """Return records that did not reach `block_at`.

        Args:
            records: Memories just retrieved for `query`.
            query: The retrieval question. Passed through to detectors
                that read `context.query`.

        Returns:
            The subset of `records` that the scan did not flag at
            `block_at` or worse (nor, under `fail_closed`, fail to check).
            Order is preserved.
        """
        report = self.client.scan(records, query=query)
        blocked = {
            item.record_id for item in report.findings if item.severity.is_at_least(self.block_at)
        }
        if self.client.config.fail_closed:
            for error in report.errors:
                if len(error.record_ids) < error.records:
                    return []  # more failures than listed ids: nothing is known safe
                blocked.update(error.record_ids)
        return [record for record in records if record.id not in blocked]


__all__ = ["RetrieveGuard", "WriteDecision", "WriteGuard"]
