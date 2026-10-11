"""Run every check over stored records and build a `ScanReport`.

A scan has two passes so a large store does not have to sit in memory as
`MemoryRecord` objects:

1. **Stream.** Records are read one at a time. Text-only detectors (the
   phrase lists, secret patterns, classifiers, hosted APIs) run right away,
   and the record is packed into a `Corpus`: its text and metadata, and its
   vector as 4-byte floats.
2. **Corpus.** Detectors that compare records with each other (TrustRAG,
   hubness, NLI, probe queries, and any custom detector that reads
   `context.existing`) run once every record has been read, against one
   shared nearest-neighbour table.

The engine only knows the `MemoryCheck` contract: each check says which
passes it runs in (`phases`), what it saw on each (`observe`), and its
findings after its last pass (`conclude`). A `SecurityCheck` uses this to
run text-only detectors on the stream and corpus detectors afterwards,
voting (`min_detectors`) over both. A check can add findings; it cannot
hide a finding another check raised.

A check or detector that raises does not stop the scan and does not become
a finding. It is reported in `ScanReport.errors` and the scan is marked
incomplete (`report.complete` is `False`); the other detectors' results
are kept.

Both `Mimvo` and `AsyncMimvo` use this engine.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from .checks.base import MemoryCheck, Phase
from .config import Config
from .context import CheckContext, CheckFailure
from .corpus import Corpus, NeighbourIndex
from .mask import mask_snippet, safe_evidence
from .models.enums import Action, Severity
from .models.finding import Finding
from .models.memory import MemoryCandidate, MemoryRecord
from .models.results import ScanError, ScanFinding, ScanReport
from .telemetry import (
    ATTR_FINDING_COUNT,
    ATTR_OPERATION,
    SPAN_SCAN,
    NullTracer,
    Tracer,
    get_logger,
)

_logger = get_logger(__name__)
_MAX_ERROR_IDS = 20
_MAX_ERROR_MESSAGE = 200


def _schedule(checks: list[MemoryCheck]) -> list[tuple[int, MemoryCheck, frozenset[Phase]]]:
    """Each check with its position and the passes it asked for."""
    schedule = []
    for position, chk in enumerate(checks):
        phases = chk.phases()
        _logger.debug(
            "check {!r} runs in: {}", chk.name, ", ".join(sorted(p.value for p in phases))
        )
        schedule.append((position, chk, phases))
    return schedule


class Evaluator:
    """Run the check list and collect a `ScanReport`.

    Attributes:
        config: Settings for this scan, including `fail_closed`.
        checks: Checks to run, in order, on every record.
        tracer: Receives one span per scan. A missing tracer is replaced
            with `NullTracer`, which records nothing.
        index: Nearest-neighbour implementation for corpus detectors.
            `None` uses the exact search in `mimvo.corpus`.
    """

    def __init__(
        self,
        *,
        config: Config,
        checks: list[MemoryCheck],
        tracer: Tracer | None = None,
        index: NeighbourIndex | None = None,
    ) -> None:
        """Store the settings, checks, and tracer used by `scan`.

        Args:
            config: Settings for this scan.
            checks: Checks to run on every record, in this order.
            tracer: Span destination. `None` means record nothing.
            index: Nearest-neighbour implementation. `None` uses exact search.
        """
        self.config = config
        self.checks = checks
        self.tracer: Tracer = tracer or NullTracer()
        self.index = index

    def scan(self, records: Iterable[MemoryRecord], *, query: str | None = None) -> ScanReport:
        """Audit stored records for poisoned facts, hidden instructions, and secrets.

        Args:
            records: Memories to scan. Read once, as a stream; only a packed
                copy of each record is kept for the corpus pass.
            query: The retrieval question for this batch. Passed through as
                `context.query`. A store scan has none; probe-query
                detectors generate one when they need it.

        Returns:
            A `ScanReport`. `total` is how many records were read. `findings`
            lists each problem, highest severity first, then in record and
            check order. Secret text in snippets is masked. `errors` lists
            checks and detectors that failed.
        """
        now = datetime.now(UTC)
        corpus = Corpus(index=self.index)
        ctx = CheckContext(
            config=self.config,
            now=now,
            operation="scan",
            existing=corpus.records(),
            query=query,
            corpus=corpus,
        )
        schedule = _schedule(self.checks)
        report = ScanReport(generated_at=now, checks=_check_inventory(self.checks))
        started = time.perf_counter()
        # (row, check position, finding)
        rows: list[tuple[int, int, Finding]] = []
        # Stream observations kept for checks that also run in the corpus pass.
        pending: dict[tuple[int, int], list[object]] = {}
        batch_size = self.config.batch_size

        def run_pass(batch: list[tuple[int, MemoryCandidate]], phase: Phase) -> None:
            """Run every check that wants `phase` on one batch of rows."""
            for position, chk, phases in schedule:
                if phase not in phases:
                    continue
                seen = _observe_batch(chk, [c for _, c in batch], ctx, phase)
                for (row, candidate), observation in zip(batch, seen, strict=True):
                    key = (row, position)
                    if phase is Phase.STREAM and Phase.CORPUS in phases:
                        if any(observation):  # empty observations carry nothing to keep
                            pending[key] = observation
                        continue
                    observations = pending.pop(key, []) + observation
                    found = _conclude(chk, observations, candidate, ctx)
                    rows.extend((row, position, f) for f in found)

        with self.tracer.span(SPAN_SCAN, {ATTR_OPERATION: "scan"}) as span:
            buffer: list[tuple[int, MemoryCandidate]] = []
            for record in records:
                buffer.append((corpus.add(record), _candidate(record)))
                if len(buffer) >= batch_size:
                    run_pass(buffer, Phase.STREAM)
                    buffer = []
            if buffer:
                run_pass(buffer, Phase.STREAM)
            report.total = len(corpus)

            if any(Phase.CORPUS in phases for _, _, phases in schedule):
                for start in range(0, len(corpus), batch_size):
                    rows_in_batch = range(start, min(start + batch_size, len(corpus)))
                    run_pass([(row, corpus.candidate(row)) for row in rows_in_batch], Phase.CORPUS)

            rows.sort(key=lambda item: (-item[2].severity.rank, item[0], item[1]))
            snippets: dict[int, str] = {}
            for row, _, finding in rows:
                if row not in snippets:
                    snippets[row] = mask_snippet(corpus.contents[row])
                report.findings.append(_scan_finding(corpus.ids[row], snippets[row], finding))
            report.errors, report.records_with_errors = _errors(ctx.failures)
            report.duration_seconds = round(time.perf_counter() - started, 4)
            span.set_attribute("mimvo.scanned", report.total)
            span.set_attribute("mimvo.flagged", report.flagged)
            span.set_attribute("mimvo.errors", report.records_with_errors)
            span.set_attribute(ATTR_FINDING_COUNT, len(report.findings))
        return report


def _observe_batch(
    chk: MemoryCheck, candidates: list[MemoryCandidate], ctx: CheckContext, phase: Phase
) -> list[list[object]]:
    """One pass of one check over a batch; a crash becomes a per-record failure.

    Returns:
        For each candidate, a list holding its observation, or an empty
        list when the check failed on it.
    """
    try:
        seen = chk.observe_batch(candidates, ctx, phase)
        if len(seen) == len(candidates):
            return [[observation] for observation in seen]
        _logger.warning(
            "check {!r}: observe_batch returned {} results for {} records",
            chk.name,
            len(seen),
            len(candidates),
        )
    except Exception as exc:
        if len(candidates) == 1:
            _fail(chk, candidates[0], ctx, exc)
            return [[]]
    # Retry one record at a time so a failure lands on the record that caused it.
    out: list[list[object]] = []
    for candidate in candidates:
        try:
            out.append([chk.observe(candidate, ctx, phase)])
        except Exception as exc:
            _fail(chk, candidate, ctx, exc)
            out.append([])
    return out


def _conclude(
    chk: MemoryCheck,
    observations: Sequence[object],
    candidate: MemoryCandidate,
    ctx: CheckContext,
) -> list[Finding]:
    """Turn a record's observations into findings; a crash becomes a failure."""
    if not observations:
        return []
    try:
        found = chk.conclude(observations)
    except Exception as exc:
        _fail(chk, candidate, ctx, exc)
        return []
    name = getattr(chk, "name", None)
    return [
        f.model_copy(update={"check": name}) if f.check is None and isinstance(name, str) else f
        for f in found
    ]


def _fail(chk: MemoryCheck, candidate: MemoryCandidate, ctx: CheckContext, exc: Exception) -> None:
    name = str(getattr(chk, "name", type(chk).__name__))
    _logger.warning("check {!r} raised {}", name, type(exc).__name__)
    ctx.record_failure(check=name, detector=None, error=exc, record_id=candidate.id)


def _candidate(record: MemoryRecord) -> MemoryCandidate:
    return MemoryCandidate.model_construct(
        content=record.content,
        metadata=dict(record.metadata),
        id=record.id,
        derived_from=list(record.derived_from),
        created_at=record.created_at,
        embedding=list(record.embedding) if record.embedding is not None else None,
    )


def _scan_finding(record_id: str, snippet: str, finding: Finding) -> ScanFinding:
    return ScanFinding.from_finding(
        finding,
        record_id=record_id,
        snippet=snippet,
        recommended_action=_report_action(finding.recommended_action, finding.severity),
        detectors=_detectors(finding),
        evidence=safe_evidence(finding.evidence),
    )


def _detectors(finding: Finding) -> list[str]:
    """The voters on a finding, read before `safe_evidence` drops the key."""
    raw = finding.evidence.get("detectors")
    if isinstance(raw, list):
        return [str(name) for name in raw]
    return [finding.check] if finding.check else []


def _errors(failures: list[CheckFailure]) -> tuple[list[ScanError], int]:
    """Group failures by check, detector, and exception type.

    Returns:
        The grouped errors, most records first, and how many distinct
        records had at least one failure.
    """
    groups: dict[tuple[str, str | None, str], ScanError] = {}
    affected: set[str] = set()
    anonymous = 0
    for failure in failures:
        key = (failure.check, failure.detector, failure.error_type)
        error = groups.get(key)
        if error is None:
            error = ScanError(
                check=failure.check,
                detector=failure.detector,
                error_type=failure.error_type,
                message=mask_snippet(failure.message, width=_MAX_ERROR_MESSAGE),
            )
            groups[key] = error
        error.records += 1
        if failure.record_id is None:
            anonymous += 1
        else:
            affected.add(failure.record_id)
            if len(error.record_ids) < _MAX_ERROR_IDS and failure.record_id not in error.record_ids:
                error.record_ids.append(failure.record_id)
    ordered = sorted(groups.values(), key=lambda e: (-e.records, e.check, e.detector or ""))
    return ordered, len(affected) + anonymous


def _check_inventory(checks: list[MemoryCheck]) -> dict[str, list[str]]:
    """Name each check and the detectors it runs, for the report header.

    Args:
        checks: The checks this engine will run.

    Returns:
        `{check name: [detector names]}`. A check without detectors (a
        plain `MemoryCheck`) maps to an empty list.
    """
    inventory: dict[str, list[str]] = {}
    for chk in checks:
        detectors = getattr(chk, "detectors", None) or []
        inventory[str(getattr(chk, "name", type(chk).__name__))] = [
            str(getattr(d, "name", type(d).__name__)) for d in detectors
        ]
    return inventory


def _report_action(action: Action | None, severity: Severity) -> Action:
    """Pick review, quarantine, or delete for one finding.

    Args:
        action: The action the check recommended. Used as-is when it is
            already review, quarantine, or delete.
        severity: Used when `action` is missing or is some other value.
            Critical becomes delete, high becomes quarantine, and anything
            lower becomes review.

    Returns:
        The action written on the scan finding.
    """
    if action in {Action.REVIEW, Action.QUARANTINE, Action.DELETE}:
        return action
    if severity.is_at_least(Severity.CRITICAL):
        return Action.DELETE
    if severity.is_at_least(Severity.HIGH):
        return Action.QUARANTINE
    return Action.REVIEW


__all__ = ["Evaluator"]
