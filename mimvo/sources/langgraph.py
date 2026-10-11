"""Read every item in a LangGraph long-term memory store. Never writes.

LangGraph's `BaseStore` (`InMemoryStore`, `PostgresStore`, `RedisStore`, ...)
is where LangGraph agents and LangMem keep long-term memories: JSON values
under a namespace (such as `("memories", user_id)`) and a key. This source
pages through `store.search(namespace_prefix, limit=..., offset=...)`, which
every store implements, and turns each item into a record.

The memory text is read from the value's `content`, `text`, `memory`, or
`page_content` field (LangMem writes `{"kind": ..., "content": ...}`), or a
field you name. When none is found the whole value is scanned as JSON, so
nothing in the store goes unchecked. The store's vectors are not exposed by
`search`, so vector detectors do not run on this source.

    from langgraph.store.memory import InMemoryStore
    from mimvo import Mimvo
    from mimvo.sources import LangGraphStoreScanSource

    report = Mimvo().scan(LangGraphStoreScanSource(store, namespace=("memories",)))
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from ..exceptions import BackendError, ConfigurationError
from ..models.memory import MemoryRecord
from .base import DEFAULT_BATCH_SIZE, take, text_from_payload, to_record

_VALUE_TEXT_KEYS = ("content", "text", "memory", "page_content", "document", "data")


class LangGraphStoreScanSource:
    """List the items in a LangGraph `BaseStore`."""

    def __init__(
        self,
        store: Any,
        *,
        namespace: Sequence[str] = (),
        text_field: str | None = None,
    ) -> None:
        """Point at a store and a namespace prefix.

        Args:
            store: A LangGraph `BaseStore` instance.
            namespace: Namespace prefix to read, such as `("memories",)`.
                The empty prefix (the default) reads the whole store.
            text_field: Key in each item's value that holds the memory text.
                `None` tries `content`, `text`, `memory`, `page_content`,
                `document`, and `data`, then falls back to the JSON value.

        Raises:
            ConfigurationError: `store` has no `search` method.
        """
        if not callable(getattr(store, "search", None)):
            raise ConfigurationError(
                f"{type(store).__name__} is not a LangGraph store: it has no search()."
            )
        self._store = store
        self._namespace = tuple(namespace)
        self._text_field = text_field

    @property
    def label(self) -> str:
        """Source label for the report, such as `"langgraph:memories"`."""
        return "langgraph:" + ("/".join(self._namespace) or "*")

    def records(
        self, *, batch_size: int = DEFAULT_BATCH_SIZE, sample: int | None = None
    ) -> Iterator[MemoryRecord]:
        """Yield each stored item as a `MemoryRecord`.

        Args:
            batch_size: Items requested per `search` call.
            sample: Stop after this many items.

        Returns:
            An iterator of records. The record id is the item's namespace
            and key joined with `/`.

        Raises:
            BackendError: A `search` call failed.
        """
        yield from take(self._pages(batch_size), sample=sample)

    def _pages(self, batch_size: int) -> Iterator[MemoryRecord]:
        offset = 0
        while True:
            try:
                items = self._store.search(self._namespace, limit=batch_size, offset=offset)
            except Exception as exc:
                raise BackendError(f"LangGraph store search failed: {exc}") from exc
            items = list(items or [])
            for item in items:
                yield self._to_record(item)
            if len(items) < batch_size:
                return
            offset += len(items)

    def _to_record(self, item: Any) -> MemoryRecord:
        namespace = tuple(getattr(item, "namespace", ()) or ())
        key = str(getattr(item, "key", ""))
        value = getattr(item, "value", None)
        metadata: dict[str, Any] = {"namespace": "/".join(namespace), "key": key}
        if isinstance(value, dict):
            text = _value_text(value, self._text_field)
            metadata.update({k: v for k, v in value.items() if not isinstance(v, dict | list)})
        else:
            text = "" if value is None else str(value)
        return to_record(
            "/".join((*namespace, key)),
            text,
            metadata=metadata,
            created_at=getattr(item, "created_at", None),
            updated_at=getattr(item, "updated_at", None),
        )


def _value_text(value: dict[str, Any], field: str | None) -> str:
    candidate = value.get(field) if field else None
    if candidate is None:
        candidate = next(
            (value[k] for k in _VALUE_TEXT_KEYS if value.get(k) not in (None, "")), None
        )
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        nested = text_from_payload(None, candidate)
        if nested:
            return nested
    return json.dumps(value if candidate is None else candidate, default=str, ensure_ascii=False)


__all__ = ["LangGraphStoreScanSource"]
