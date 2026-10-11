"""Scroll every point in a Qdrant collection. Never writes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..exceptions import BackendError, ConfigurationError
from ..models.memory import MemoryRecord
from .base import missing_extra, take, text_from_payload, to_record

_QDRANT_PAGE = 256


class QdrantScanSource:
    """Scroll every point in a Qdrant collection.

    Uses `scroll` with payloads and vectors. Nothing is upserted or
    deleted. Text is taken from the payload.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        collection: str,
        api_key: str | None = None,
        client: Any | None = None,
        text_field: str | None = None,
        metadata_field: str | None = None,
        vector_name: str | None = None,
        path: str | None = None,
    ) -> None:
        """Point at a Qdrant collection, or at an already-open client.

        Args:
            url: Qdrant HTTP URL. Required unless `client` is set.
            collection: Collection name to scroll.
            api_key: Sent to `QdrantClient` when opening a new client.
            client: An already-open client, used by tests. When set, `url`
                is not used.
            text_field: Payload key that holds the memory text. `None`
                tries common keys such as `content` and `text`.
            metadata_field: Payload key that holds a nested metadata dict
                (LangChain writes `"metadata"`). `None` uses the whole payload.
            vector_name: Named vector to read when the collection has
                several. `None` takes the unnamed or first vector.
            path: Local (embedded) Qdrant directory, instead of `url`.

        Raises:
            ConfigurationError: `client` is omitted and neither `url` nor
                `path` is given.
        """
        if client is None and not url and not path:
            raise ConfigurationError("qdrant scan needs --url.")
        self._url = url
        self._path = path
        self._metadata_field = metadata_field
        self._vector_name = vector_name
        self._api_key = api_key
        self._collection = collection
        self._client = client
        self._text_field = text_field

    def records(
        self, *, batch_size: int = _QDRANT_PAGE, sample: int | None = None
    ) -> Iterator[MemoryRecord]:
        """Yield each point's payload text as a `MemoryRecord`.

        Args:
            batch_size: Points per `scroll` call. Default 256, which is
                also Qdrant's typical page cap.
            sample: Stop after this many points. `None` reads the
                whole collection.

        Returns:
            An iterator. The first pull opens the client if one was not passed.

        Raises:
            IntegrationError: The Qdrant client is not installed.
            BackendError: A scroll call failed.
        """
        client = self._client if self._client is not None else self._open()
        yield from take(self._scroll(client, batch_size=batch_size), sample=sample)

    def _open(self) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise missing_extra("Qdrant", "qdrant") from exc
        if self._path and not self._url:
            return QdrantClient(path=self._path)
        return QdrantClient(url=self._url, api_key=self._api_key)

    def _scroll(self, client: Any, *, batch_size: int) -> Iterator[MemoryRecord]:
        offset: Any = None
        while True:
            try:
                points, offset = client.scroll(
                    collection_name=self._collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception as exc:
                raise BackendError(f"Qdrant scroll failed: {exc}") from exc
            for point in points or []:
                yield self._to_record(point)
            if offset is None:
                return

    def _to_record(self, point: Any) -> MemoryRecord:
        if isinstance(point, dict):
            payload = point.get("payload") or {}
            raw_id = point.get("id")
            vector = point.get("vector")
        else:
            payload = getattr(point, "payload", None) or {}
            raw_id = getattr(point, "id", None)
            vector = getattr(point, "vector", None)
        payload = payload if isinstance(payload, dict) else {}
        memory_id = payload.get("mimvo_id") or payload.get("id") or raw_id or "qdrant_unknown"
        content = text_from_payload(payload.get("content"), payload, field=self._text_field)
        metadata: dict[str, Any] = payload
        if self._metadata_field:
            nested = payload.get(self._metadata_field)
            metadata = dict(nested) if isinstance(nested, dict) else {}
        if self._vector_name and isinstance(vector, dict):
            vector = vector.get(self._vector_name)
        return to_record(
            memory_id,
            content,
            metadata=metadata,
            embedding=vector,
            namespace=self._collection,
        )


__all__ = ["QdrantScanSource"]
