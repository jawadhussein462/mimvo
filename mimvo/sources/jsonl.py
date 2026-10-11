"""JSON Lines export: one JSON object with a `content` string per line."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from ..exceptions import ConfigurationError
from ..models.memory import MemoryRecord
from .base import DEFAULT_BATCH_SIZE, take, to_record


class JsonlScanSource:
    """Stream records from a JSON Lines file or an open text stream.

    Each non-blank line must be a JSON object with a string `content`.
    An `id` field is used when present. Otherwise the id is `line_N`.
    """

    def __init__(self, path: str | Path | IO[str]) -> None:
        """Remember where the lines will be read from. Nothing is read yet.

        Args:
            path: A filesystem path, or an open text stream such as `sys.stdin`.
        """
        self._path = path

    def records(
        self, *, batch_size: int = DEFAULT_BATCH_SIZE, sample: int | None = None
    ) -> Iterator[MemoryRecord]:
        """Yield one `MemoryRecord` per JSON line.

        Args:
            batch_size: Accepted so this matches `ScanSource`. Lines are
                already one at a time, so the value is ignored.
            sample: Stop after this many records. `None` reads every line.

        Returns:
            An iterator. A path is opened with UTF-8 and closed when the
            iterator is exhausted. A stream you passed in is left open.

        Raises:
            ConfigurationError: A line is not valid JSON, or the object has
                no string `content`. The message includes the line number.
        """
        del batch_size  # lines are already streamed
        stream = self._path
        if isinstance(stream, str | Path):
            with open(stream, encoding="utf-8") as fh:
                yield from take(_iter_jsonl(fh), sample=sample)
            return
        yield from take(_iter_jsonl(stream), sample=sample)


def _iter_jsonl(stream: IO[str]) -> Iterator[MemoryRecord]:
    for lineno, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            item: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"line {lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            raise ConfigurationError(f"line {lineno}: expected an object with a 'content' string")
        meta = item.get("metadata")
        extra = {
            key: item[key]
            for key in ("source", "user", "namespace", "created_at", "updated_at")
            if key in item and key != "metadata"
        }
        merged = {**extra, **meta} if isinstance(meta, dict) else extra
        yield to_record(
            item.get("id") or f"line_{lineno}",
            item["content"],
            metadata=merged,
            embedding=item.get("embedding"),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            namespace=item.get("namespace") if isinstance(item.get("namespace"), str) else None,
        )


__all__ = ["JsonlScanSource"]
