"""Read-only scan sources: stream records in batches, never write.

A scan source yields `MemoryRecord` objects. The connection is used to list
and fetch only. `sample` stops the scan after that many records, which is
how `--sample` caps a large store.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from ..exceptions import ConfigurationError, IntegrationError
from ..models.memory import MemoryRecord
from ..vectors import as_floats

DEFAULT_BATCH_SIZE = 500
_TEXT_KEYS = ("content", "text", "page_content", "document", "memory", "pageContent")
_CREATED_KEYS = ("created_at", "createdAt", "timestamp", "created", "inserted_at", "created_time")
_UPDATED_KEYS = ("updated_at", "updatedAt", "updated", "modified_at", "modified")
_SOURCE_KEYS = ("source", "provenance", "origin", "url", "uri")
_USER_KEYS = ("user", "user_id", "userId", "author", "owner")
_NS_KEYS = ("namespace", "collection", "tenant")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@runtime_checkable
class ScanSource(Protocol):
    """A read-only iterator of stored memories.

    Implementations must not insert, update, or delete.
    """

    def records(
        self, *, batch_size: int = DEFAULT_BATCH_SIZE, sample: int | None = None
    ) -> Iterator[MemoryRecord]:
        """Yield stored memories, one batch of fetches at a time.

        Args:
            batch_size: How many records to ask the store for per round trip.
                Default is `DEFAULT_BATCH_SIZE` (500).
            sample: Stop after this many records. `None` means read until
                the store is exhausted.

        Returns:
            An iterator of `MemoryRecord`. It may perform I/O as you pull
            from it.
        """
        ...


def take(
    items: Iterable[MemoryRecord],
    *,
    sample: int | None,
) -> Iterator[MemoryRecord]:
    """Yield records, stopping early when `sample` is set.

    Args:
        items: Records already being streamed from a store.
        sample: Maximum number to yield. `None` yields every item.

    Returns:
        An iterator. It stops after `sample` records, or when `items` ends.
    """
    for n, item in enumerate(items, start=1):
        yield item
        if sample is not None and n >= sample:
            return


def text_from_payload(
    content: object | None,
    metadata: object | None = None,
    *,
    field: str | None = None,
) -> str:
    """Pick the document string out of a store payload.

    Args:
        content: The store's document field, when it has one separate from
            metadata.
        metadata: A dict of extra fields, or anything else (ignored if it
            is not a dict).
        field: Metadata key to prefer, such as `"text"`. When that key is
            missing or blank, the function tries `content`, then the keys
            `content`, `text`, `page_content`, `document`, `memory`, and
            `pageContent`.

    Returns:
        The first non-blank string found. An empty string when every
        candidate is missing or blank and `content` itself is `None`.
    """
    meta = metadata if isinstance(metadata, dict) else {}
    if field:
        value = meta.get(field)
        if value is not None and str(value).strip():
            return str(value)
    if content is not None and str(content).strip():
        return str(content)
    for key in _TEXT_KEYS:
        value = meta.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(content) if content is not None else ""


def quote_ident(name: str) -> str:
    """Quote a SQL identifier so it can be interpolated into a query.

    Args:
        name: A column, table, or `schema.table`. Each part may contain
            letters, digits, and underscores, and must not start with a digit.

    Returns:
        The name with each part wrapped in double quotes, such as
        `"public"."memories"`.

    Raises:
        ConfigurationError: A part contains anything else. That rejects
            quote tricks and statement separators.
    """
    parts = name.split(".")
    if not parts or not all(_IDENT.fullmatch(part) for part in parts):
        raise ConfigurationError(
            f"Invalid SQL identifier {name!r}: use letters, digits, and underscores."
        )
    return ".".join(f'"{part}"' for part in parts)


def missing_extra(name: str, extra: str) -> IntegrationError:
    """Build the error raised when an optional store client is not installed.

    Args:
        name: Store name shown to the user, such as `"Chroma"`.
        extra: The pip extra, such as `"chroma"`.

    Returns:
        An `IntegrationError` whose message is
        `pip install "mimvo[<extra>]"`. The caller raises it.
    """
    return IntegrationError(f'{name} support requires `pip install "mimvo[{extra}]"`.')


def parse_datetime(value: object) -> datetime | None:
    """Parse a store timestamp into an aware UTC `datetime`.

    Args:
        value: A `datetime`, a Unix timestamp (seconds or milliseconds),
            or an ISO-8601 string. Anything else is ignored.

    Returns:
        An aware UTC datetime, or `None` when `value` cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float) and not isinstance(value, bool):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _first(meta: Mapping[str, Any], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in meta and meta[key] not in (None, ""):
            return meta[key]
    return None


def _as_metadata(metadata: object | None) -> dict[str, object]:
    if isinstance(metadata, Mapping):
        return {str(k): v for k, v in metadata.items()}
    return {}


def to_record(
    memory_id: Any,
    content: str,
    *,
    metadata: object | None = None,
    embedding: object | None = None,
    created_at: object | None = None,
    updated_at: object | None = None,
    namespace: str | None = None,
) -> MemoryRecord:
    """Build a `MemoryRecord` from a store row.

    Args:
        memory_id: Whatever the store uses as an id. It is converted with
            `str`.
        content: The memory text.
        metadata: Payload fields from the store. Source, user, and
            namespace are copied in when present. Timestamps in this
            mapping fill `created_at` / `updated_at` when those arguments
            are omitted.
        embedding: Stored vector. Lists, tuples, and arrays with
            `.tolist()` are accepted.
        created_at: Creation time, when the store has a dedicated field.
        updated_at: Last-update time, when the store has a dedicated field.
        namespace: Collection or namespace label. Stored under
            `metadata["namespace"]` when that key is not already set.

    Returns:
        A record with status `active`. Missing timestamps default to now.
    """
    meta = _as_metadata(metadata)
    if namespace and "namespace" not in meta:
        meta["namespace"] = namespace
    if "source" not in meta:
        source = _first(meta, _SOURCE_KEYS)
        if source is not None:
            meta["source"] = source
    if "user" not in meta:
        user = _first(meta, _USER_KEYS)
        if user is not None:
            meta["user"] = user
    if "namespace" not in meta:
        stored_ns = _first(meta, _NS_KEYS)
        if stored_ns is not None:
            meta["namespace"] = stored_ns
    created = parse_datetime(created_at) or parse_datetime(_first(meta, _CREATED_KEYS))
    updated = parse_datetime(updated_at) or parse_datetime(_first(meta, _UPDATED_KEYS))
    fields: dict[str, object] = {
        "id": str(memory_id),
        "content": content,
        "metadata": meta,
        "embedding": as_floats(embedding),
    }
    if created is not None:
        fields["created_at"] = created
    if updated is not None:
        fields["updated_at"] = updated
    return MemoryRecord.model_validate(fields)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "ScanSource",
    "missing_extra",
    "parse_datetime",
    "quote_ident",
    "take",
    "text_from_payload",
    "to_record",
]
