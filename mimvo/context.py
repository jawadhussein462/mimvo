"""The `CheckContext` handed to every check.

A check receives the memory plus this object and returns findings. It should
not change global state or do I/O. The shape is:

    (candidate, context) -> list[Finding]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .config import Config
from .models.memory import MemoryRecord

if TYPE_CHECKING:
    from .corpus import Corpus


@dataclass(frozen=True)
class CheckFailure:
    """A check or detector that raised instead of answering.

    Recorded on the context and turned into `ScanReport.errors`. It is not a
    finding: a failure says nothing about the record, only that it was not
    fully checked.

    Attributes:
        check: Name of the check, such as `"injection"`.
        detector: Name of the detector that raised, or `None` when the whole
            check raised.
        error_type: Exception class name, such as `"BackendError"`.
        message: The exception message. Masked before it reaches a report.
        record_id: Id of the record being checked, when it had one.
    """

    check: str
    detector: str | None
    error_type: str
    message: str
    record_id: str | None


@dataclass
class CheckContext:
    """Extra facts a check may use while scanning one record.

    Attributes:
        config: Settings for this scan, including `fail_closed`.
        now: Clock time captured when the scan started.
        operation: What the engine is doing. Scans set this to `"scan"`.
        existing: Every record in this scan. Corpus detectors (TrustRAG,
            hubness, NLI) use these as neighbours. During a streamed scan
            this is a read-only view that rebuilds records on access;
            built-in detectors read `corpus` instead.
        query: The retrieval question, when the caller passed
            `Mimvo.scan(..., query=)`. Most checks ignore it. A store
            scan has no query; probe-query detectors generate one.
        cache: Scratch space shared by detectors on this scan.
        corpus: The packed records and neighbour index for this scan. The
            engine always sets it; use `mimvo.corpus.corpus_of(context)`
            to get one even when a check is called directly.
        failures: Checks and detectors that raised during this scan. A
            `SecurityCheck` records a failing detector here and keeps the
            results of the others.
    """

    config: Config
    now: datetime
    operation: str = "scan"
    existing: Sequence[MemoryRecord] = field(default_factory=list)
    query: str | None = None
    cache: dict[str, object] = field(default_factory=dict)
    corpus: Corpus | None = None
    failures: list[CheckFailure] = field(default_factory=list)

    def record_failure(
        self,
        *,
        check: str,
        detector: str | None,
        error: BaseException,
        record_id: str | None,
    ) -> None:
        """Note that a check or detector raised on one record.

        Args:
            check: Name of the check.
            detector: Name of the detector, or `None` for the whole check.
            error: The exception that was raised.
            record_id: Id of the record being checked.
        """
        self.failures.append(
            CheckFailure(
                check=check,
                detector=detector,
                error_type=type(error).__name__,
                message=str(error),
                record_id=record_id,
            )
        )


__all__ = ["CheckContext", "CheckFailure"]
