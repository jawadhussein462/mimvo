"""Read every document in a Chroma collection. Never writes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..exceptions import BackendError, ConfigurationError
from ..models.memory import MemoryRecord
from .base import DEFAULT_BATCH_SIZE, missing_extra, take, text_from_payload, to_record


class ChromaScanSource:
    """List every document in a Chroma collection.

    The methods used are `get` only. Nothing is upserted or deleted.

    Pass `handle` in tests: it should look like a Chroma collection.
    Otherwise a persistent client is opened at `path` and `collection`
    is fetched.
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        collection: str | None = None,
        handle: Any | None = None,
        text_field: str | None = None,
    ) -> None:
        """Point at a collection on disk, or at a collection object.

        Args:
            path: Directory of the Chroma database. Required unless `handle`
                is set.
            collection: Collection name. Required unless `handle` is set.
            handle: An already-open collection. When set, `path` and
                `collection` are not used to connect.
            text_field: Metadata key that holds the text, for stores that
                keep it there instead of in `documents` (mem0 uses
                `"data"`). `None` uses the document, then common keys.

        Raises:
            ConfigurationError: `handle` is omitted and `path` or
                `collection` is missing.
        """
        if handle is None and (not path or not collection):
            raise ConfigurationError("chroma scan needs --path and --collection.")
        self._path = None if path is None else Path(path)
        self._name = collection
        self._handle = handle
        self._text_field = text_field

    def records(
        self, *, batch_size: int = DEFAULT_BATCH_SIZE, sample: int | None = None
    ) -> Iterator[MemoryRecord]:
        """Yield each document as a `MemoryRecord`.

        Args:
            batch_size: Documents requested per `get` call. Default 500.
            sample: Stop after this many documents. `None` reads the
                whole collection.

        Returns:
            An iterator. The first pull opens the database if `handle`
            was not passed.

        Raises:
            ConfigurationError: The path or collection does not exist.
            IntegrationError: Chroma is not installed.
            BackendError: A `get` call failed.
        """
        collection = self._handle if self._handle is not None else self._open()
        yield from take(self._pages(collection, batch_size=batch_size), sample=sample)

    def _open(self) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise missing_extra("Chroma", "chroma") from exc
        assert self._path is not None and self._name is not None
        if not self._path.exists():
            raise ConfigurationError(f"Chroma path not found: {self._path}")
        client = chromadb.PersistentClient(path=str(self._path))
        try:
            return client.get_collection(self._name)
        except Exception as exc:
            raise ConfigurationError(f"Chroma collection {self._name!r} not found: {exc}") from exc

    def _pages(self, collection: Any, *, batch_size: int) -> Iterator[MemoryRecord]:
        offset = 0
        while True:
            try:
                res = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas", "embeddings"],
                )
            except Exception:
                try:
                    res = collection.get(
                        limit=batch_size, offset=offset, include=["documents", "metadatas"]
                    )
                except Exception as exc:
                    raise BackendError(f"Chroma get failed: {exc}") from exc
            ids = _column(res, "ids")
            docs = _column(res, "documents")
            metas = _column(res, "metadatas")
            embs = _column(res, "embeddings")
            if not ids:
                return
            for i, memory_id in enumerate(ids):
                doc = docs[i] if i < len(docs) else None
                meta = metas[i] if i < len(metas) else None
                embedding = embs[i] if i < len(embs) else None
                yield to_record(
                    memory_id,
                    text_from_payload(doc, meta, field=self._text_field),
                    metadata=meta,
                    embedding=embedding,
                    namespace=self._name or getattr(collection, "name", None),
                )
            if len(ids) < batch_size:
                return
            offset += batch_size


def _column(result: Any, key: str) -> list[Any]:
    """One column of a Chroma `get()` result as a list.

    Chroma returns `embeddings` as a 2-D numpy array, so `value or []`
    would raise ("truth value of an array is ambiguous"); test for `None`.
    """
    if not isinstance(result, dict):
        return []
    value = result.get(key)
    return [] if value is None else list(value)


__all__ = ["ChromaScanSource"]
