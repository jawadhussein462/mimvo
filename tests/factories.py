"""Functions that build memories, contexts, and findings for tests.

They live here, not in `conftest.py`, so a test module can import them
without loading pytest fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mimvo import MemoryCandidate, MemoryRecord
from mimvo.config import Config
from mimvo.context import CheckContext


def make_candidate(
    content: str = "hello",
    **fields: object,
) -> MemoryCandidate:
    """Build a `MemoryCandidate` for a test.

    Args:
        content: Memory text. Default `"hello"`.
        **fields: Extra `MemoryCandidate` fields, such as `id=`.

    Returns:
        A candidate the checks can scan.
    """
    return MemoryCandidate(content=content, **fields)


def make_record(
    content: str = "hello",
    *,
    id: str = "mem_x",
    status: str = "active",
    **fields: object,
) -> MemoryRecord:
    """Build a `MemoryRecord` for a test.

    Args:
        content: Memory text. Default `"hello"`.
        id: Record id. Default `"mem_x"`.
        status: `"active"`, `"revoked"`, or `"quarantined"`.
        **fields: Extra `MemoryRecord` fields.

    Returns:
        A stored record.
    """
    return MemoryRecord(id=id, content=content, status=status, **fields)


def make_context(
    *,
    existing: list[MemoryRecord] | None = None,
    config: Config | None = None,
    now_value: datetime | None = None,
    query: str | None = None,
) -> CheckContext:
    """Build the `CheckContext` a check receives.

    Args:
        existing: Neighbouring records. `None` means an empty list.
        config: Scan settings. `None` uses a default `Config`.
        now_value: Clock time stored on the context. `None` uses the
            current UTC time.
        query: Optional retrieval question.

    Returns:
        A context whose `operation` is `"scan"`.
    """
    return CheckContext(
        config=config or Config(),
        now=now_value or datetime.now(UTC),
        operation="scan",
        existing=existing or [],
        query=query,
    )
