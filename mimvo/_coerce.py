"""Turn loose caller input into `MemoryCandidate` and `MemoryRecord`.

Callers can pass a string, a dict, or an existing model. These helpers
accept all three so the common case stays short.
"""

from __future__ import annotations

from .exceptions import ConfigurationError
from .models.memory import MemoryCandidate, MemoryRecord


def coerce_candidate(content: str | MemoryCandidate | dict) -> MemoryCandidate:
    """Turn one input into a `MemoryCandidate`.

    Args:
        content: The memory to scan. A string becomes the `content` field.
            A dict is validated as a `MemoryCandidate`. A `MemoryCandidate`
            is returned unchanged.

    Returns:
        A `MemoryCandidate` the checks can read.

    Raises:
        ConfigurationError: `content` is not a string, a dict, or a
            `MemoryCandidate`.
    """
    if isinstance(content, MemoryCandidate):
        return content
    if isinstance(content, dict):
        return MemoryCandidate.model_validate(content)
    if isinstance(content, str):
        return MemoryCandidate(content=content)
    raise ConfigurationError(f"Cannot interpret content: {content!r}")


def coerce_record(item: object) -> MemoryRecord:
    """Turn one input into a `MemoryRecord`.

    Args:
        item: A `MemoryRecord`, returned unchanged, or a dict with at least
            an `id` and a `content` string.

    Returns:
        A `MemoryRecord` the scan engine can read.

    Raises:
        ConfigurationError: `item` is not a record or a dict.
    """
    if isinstance(item, MemoryRecord):
        return item
    if isinstance(item, dict):
        return MemoryRecord.model_validate(item)
    raise ConfigurationError(f"Cannot interpret record: {item!r}")


def coerce_records(records: list) -> list[MemoryRecord]:
    """Turn a list of records or dicts into `MemoryRecord` objects.

    Args:
        records: Each item is passed to `coerce_record`.

    Returns:
        One `MemoryRecord` per item, in the same order.

    Raises:
        ConfigurationError: Any item is not a record or a dict.
    """
    return [coerce_record(item) for item in records]


__all__ = ["coerce_candidate", "coerce_record", "coerce_records"]
