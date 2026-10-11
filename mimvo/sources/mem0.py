"""Read memories from mem0, open-source or platform. Never writes.

Works with both mem0 clients:

* **`mem0.Memory`** (open source). With no filter the whole store is read
  through mem0's own vector store: a Qdrant or Chroma backend is paged
  with `QdrantScanSource` / `ChromaScanSource`, stored vectors included;
  other backends are read with `vector_store.list()`. With a `user_id`,
  `agent_id`, `run_id`, or `filters`, `Memory.get_all` is called instead.
* **`mem0.MemoryClient`** (hosted platform). `get_all` is paged with
  `page` / `page_size`. The platform needs a filter, so pass at least one
  of `user_id`, `agent_id`, `run_id`, or `filters`.

Both the current mem0 API (`get_all(filters=..., top_k=...)`) and the older
one (`get_all(user_id=..., limit=...)`) are supported; the call is chosen
from the installed client's signature. mem0 stores the memory text in a
payload field named `data`; records keep `user_id`, `agent_id`, `run_id`,
`hash`, and metadata.

    from mem0 import Memory
    from mimvo import Mimvo
    from mimvo.sources import Mem0ScanSource

    report = Mimvo().scan(Mem0ScanSource(Memory.from_config(config)))
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ..exceptions import BackendError, ConfigurationError
from ..models.memory import MemoryRecord
from .base import DEFAULT_BATCH_SIZE, take, text_from_payload, to_record
from .chroma import ChromaScanSource
from .qdrant import QdrantScanSource

MEM0_TEXT_FIELD = "data"
# Largest single read through APIs that do not page (Memory.get_all, vector_store.list).
DEFAULT_MAX_RECORDS = 100_000


class Mem0ScanSource:
    """List the memories in a mem0 `Memory` or `MemoryClient`."""

    def __init__(
        self,
        client: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        filters: Mapping[str, Any] | None = None,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> None:
        """Point at a mem0 client and, optionally, one user, agent, or run.

        Args:
            client: A `mem0.Memory` or `mem0.MemoryClient`. The async
                clients are not supported.
            user_id: Only this user's memories.
            agent_id: Only this agent's memories.
            run_id: Only this run's memories.
            filters: Extra mem0 filters, merged with the ids above.
            max_records: Cap for reads mem0 cannot page (`Memory.get_all`,
                `vector_store.list`). `--sample` lowers it further.

        Raises:
            ConfigurationError: `client` is neither kind of mem0 client, or a
                platform client was given no filter.
        """
        self._client = client
        merged: dict[str, Any] = dict(filters or {})
        for key, value in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
            if value:
                merged[key] = value
        self._filters = merged
        self._max_records = max_records
        if hasattr(client, "vector_store"):
            self.kind = "oss"
        elif callable(getattr(client, "get_all", None)):
            self.kind = "platform"
            if not merged:
                raise ConfigurationError(
                    "The mem0 platform API reads one user, agent, or run at a time: pass "
                    "user_id, agent_id, run_id, or filters."
                )
        else:
            raise ConfigurationError(
                f"{type(client).__name__} is not a mem0 Memory or MemoryClient."
            )

    @property
    def label(self) -> str:
        """Source label for the report, such as `"mem0:user_id=alice"`."""
        scope = ",".join(f"{k}={v}" for k, v in sorted(self._filters.items()) if isinstance(v, str))
        return f"mem0:{scope or 'all'}"

    def records(
        self, *, batch_size: int = DEFAULT_BATCH_SIZE, sample: int | None = None
    ) -> Iterator[MemoryRecord]:
        """Yield each memory as a `MemoryRecord`.

        Args:
            batch_size: Memories per page, where mem0 pages.
            sample: Stop after this many memories.

        Returns:
            An iterator of records.

        Raises:
            BackendError: A mem0 call failed.
        """
        limit = min(self._max_records, sample) if sample is not None else self._max_records
        if self.kind == "platform":
            pages = self._platform(batch_size)
        elif self._filters:
            pages = self._items(self._oss_get_all(limit))
        else:
            pages = self._oss_store(batch_size, limit)
        yield from take(pages, sample=sample)

    # -- open source ---------------------------------------------------------

    def _oss_store(self, batch_size: int, limit: int) -> Iterator[MemoryRecord]:
        store = self._client.vector_store
        qdrant = getattr(store, "client", None)
        if callable(getattr(qdrant, "scroll", None)) and getattr(store, "collection_name", None):
            yield from QdrantScanSource(
                client=qdrant, collection=store.collection_name, text_field=MEM0_TEXT_FIELD
            ).records(batch_size=min(batch_size, 256))
            return
        collection = getattr(store, "collection", None)
        if callable(getattr(collection, "get", None)):
            yield from ChromaScanSource(handle=collection, text_field=MEM0_TEXT_FIELD).records(
                batch_size=batch_size
            )
            return
        listing = getattr(store, "list", None)
        if not callable(listing):
            raise ConfigurationError(f"mem0 vector store {type(store).__name__} cannot list.")
        size = "top_k" if "top_k" in _parameters(listing) else "limit"
        try:
            result = listing(filters=None, **{size: limit})
        except Exception as exc:
            raise BackendError(f"mem0 vector_store.list failed: {exc}") from exc
        for point in _unwrap(result):
            payload = getattr(point, "payload", None) or {}
            yield to_record(
                getattr(point, "id", None) or payload.get("id") or "mem0_unknown",
                text_from_payload(payload.get(MEM0_TEXT_FIELD), payload, field=MEM0_TEXT_FIELD),
                metadata=payload,
                embedding=getattr(point, "vector", None),
            )

    def _oss_get_all(self, limit: int) -> Any:
        get_all = self._client.get_all
        params = _parameters(get_all)
        try:
            if "top_k" in params or ("filters" in params and "user_id" not in params):
                return get_all(filters=dict(self._filters), top_k=limit)
            ids: dict[str, Any] = {
                k: self._filters[k] for k in ("user_id", "agent_id", "run_id") if k in self._filters
            }
            extra = {k: v for k, v in self._filters.items() if k not in ids}
            kwargs: dict[str, Any] = {**ids, "limit": limit}
            if extra:
                kwargs["filters"] = extra
            return get_all(**kwargs)
        except Exception as exc:
            raise BackendError(f"mem0 get_all failed: {exc}") from exc

    # -- platform ------------------------------------------------------------

    def _platform(self, batch_size: int) -> Iterator[MemoryRecord]:
        get_all = self._client.get_all
        params = _parameters(get_all)
        extra: dict[str, Any] = {"version": "v2"} if "version" in params else {}
        page = 1
        while True:
            try:
                result = get_all(
                    filters=dict(self._filters), page=page, page_size=batch_size, **extra
                )
            except Exception as exc:
                raise BackendError(f"mem0 platform get_all failed: {exc}") from exc
            items = _results(result)
            yield from self._items(items)
            has_next = isinstance(result, Mapping) and result.get("next")
            if not items or not has_next:
                return
            page += 1

    # -- shared --------------------------------------------------------------

    def _items(self, result: Any) -> Iterator[MemoryRecord]:
        for item in _results(result):
            if not isinstance(item, Mapping):
                continue
            nested = item.get("metadata")
            metadata: dict[str, Any] = {
                k: v for k, v in item.items() if k not in {"memory", "metadata", "score"}
            }
            if isinstance(nested, Mapping):
                metadata.update(nested)
            yield to_record(
                item.get("id") or "mem0_unknown",
                str(item.get("memory") or item.get(MEM0_TEXT_FIELD) or ""),
                metadata=metadata,
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
            )


def _results(result: Any) -> list[Any]:
    if isinstance(result, Mapping):
        items = result.get("results") or result.get("memories") or []
    else:
        items = result or []
    return list(items)


def _unwrap(result: Any) -> list[Any]:
    """mem0 vector stores return either a flat list of points or `[points, ...]`."""
    if isinstance(result, list | tuple) and result and isinstance(result[0], list | tuple):
        return list(result[0])
    return list(result or [])


def _parameters(func: Callable[..., Any]) -> set[str]:
    try:
        return set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return set()


__all__ = ["DEFAULT_MAX_RECORDS", "MEM0_TEXT_FIELD", "Mem0ScanSource"]
