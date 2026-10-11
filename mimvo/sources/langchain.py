"""Read every document in a LangChain vector store. Never writes.

LangChain's `VectorStore` interface has no "list everything" method, so
this source recognises the common stores and reads each one the way it
stores data. Nothing is added, updated, or deleted.

| Store | How it is read |
|---|---|
| `InMemoryVectorStore` (`langchain_core`) | its `store` dict, vectors included |
| `Chroma` (`langchain_chroma`) | `ChromaScanSource` on the underlying collection |
| `QdrantVectorStore` (`langchain_qdrant`) | `QdrantScanSource` on its client |
| `PineconeVectorStore` (`langchain_pinecone`) | `PineconeScanSource` on its index |
| `FAISS` (`langchain_community`) | its docstore, vectors from `index.reconstruct` |
| `PGVector` (`langchain_postgres`) | its own SQLAlchemy session, one collection |
| anything else | `get_by_ids(ids)`, when you pass `ids=` |

No LangChain package is imported here; stores are recognised by the
attributes they expose. Pass the store you already built:

    from mimvo import Mimvo
    from mimvo.sources import LangChainScanSource

    report = Mimvo().scan(LangChainScanSource(vector_store))
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from ..exceptions import BackendError, ConfigurationError
from ..models.memory import MemoryRecord
from .base import DEFAULT_BATCH_SIZE, take, to_record
from .chroma import ChromaScanSource
from .pinecone import PineconeScanSource
from .qdrant import QdrantScanSource


class LangChainScanSource:
    """List the documents in a LangChain `VectorStore`.

    Attributes:
        kind: Which reader was chosen, such as `"in_memory"` or `"chroma"`.
            Shown in error messages and the report source label.
    """

    def __init__(
        self,
        store: Any,
        *,
        ids: Sequence[str] | None = None,
        namespace: str | None = None,
    ) -> None:
        """Pick a reader for `store`.

        Args:
            store: A LangChain vector store instance.
            ids: Document ids to fetch with `get_by_ids`, for stores this
                source cannot list. Ignored for stores it can list.
            namespace: Label written to each record's metadata (and, for
                Pinecone, the namespace to read). `None` keeps the store's own.

        Raises:
            ConfigurationError: The store is not one of the supported kinds
                and `ids` was not given.
        """
        self._store = store
        self._ids = list(ids) if ids is not None else None
        self._namespace = namespace
        self.kind = _kind(store)
        if self.kind == "unknown" and self._ids is None:
            raise ConfigurationError(
                f"Cannot list the documents of {type(store).__name__}: LangChain has no generic "
                "'list all' call. Pass ids=[...] to read them with get_by_ids(), or scan the "
                "underlying database with its own scan source (chroma, qdrant, pgvector, pinecone)."
            )

    @property
    def label(self) -> str:
        """Source label for the report, such as `"langchain:chroma"`."""
        return f"langchain:{self.kind if self.kind != 'unknown' else type(self._store).__name__}"

    def records(
        self, *, batch_size: int = DEFAULT_BATCH_SIZE, sample: int | None = None
    ) -> Iterator[MemoryRecord]:
        """Yield each document as a `MemoryRecord`.

        Args:
            batch_size: Documents fetched per round trip, where the store pages.
            sample: Stop after this many documents.

        Returns:
            An iterator of records.

        Raises:
            BackendError: Reading the store failed.
        """
        reader = getattr(self, f"_read_{self.kind}", None)
        if reader is None:
            reader = self._read_by_ids
        yield from take(reader(batch_size), sample=sample)

    # -- readers ------------------------------------------------------------

    def _read_in_memory(self, batch_size: int) -> Iterator[MemoryRecord]:
        del batch_size
        for key, item in list(self._store.store.items()):
            if not isinstance(item, dict):
                continue
            yield to_record(
                item.get("id") or key,
                str(item.get("text") or ""),
                metadata=item.get("metadata"),
                embedding=item.get("vector"),
                namespace=self._namespace,
            )

    def _read_chroma(self, batch_size: int) -> Iterator[MemoryRecord]:
        source = ChromaScanSource(handle=self._store._collection)
        yield from source.records(batch_size=batch_size)

    def _read_qdrant(self, batch_size: int) -> Iterator[MemoryRecord]:
        source = QdrantScanSource(
            client=self._store.client,
            collection=self._store.collection_name,
            text_field=getattr(self._store, "content_payload_key", None) or "page_content",
            metadata_field=getattr(self._store, "metadata_payload_key", None) or "metadata",
            vector_name=getattr(self._store, "vector_name", None) or None,
        )
        yield from source.records(batch_size=min(batch_size, 256))

    def _read_pinecone(self, batch_size: int) -> Iterator[MemoryRecord]:
        namespace = self._namespace
        if namespace is None:
            namespace = getattr(self._store, "_namespace", None) or ""
        source = PineconeScanSource(
            handle=self._store.index,
            namespace=namespace,
            text_field=getattr(self._store, "_text_key", None) or "text",
        )
        yield from source.records(batch_size=min(batch_size, 100))

    def _read_faiss(self, batch_size: int) -> Iterator[MemoryRecord]:
        del batch_size
        index = getattr(self._store, "index", None)
        mapping: dict[int, str] = dict(self._store.index_to_docstore_id)
        docstore = self._store.docstore
        stored = getattr(docstore, "_dict", None)
        for position in sorted(mapping):
            doc_id = mapping[position]
            doc = stored.get(doc_id) if isinstance(stored, dict) else docstore.search(doc_id)
            if doc is None or isinstance(doc, str):
                continue
            vector = None
            reconstruct = getattr(index, "reconstruct", None)
            if callable(reconstruct):
                try:
                    vector = reconstruct(int(position))
                except Exception:
                    vector = None  # an index type that cannot reconstruct vectors
            yield _document_record(doc, doc_id, vector, self._namespace)

    def _read_pgvector(self, batch_size: int) -> Iterator[MemoryRecord]:
        store = self._store
        embedding_store = store.EmbeddingStore
        try:
            with store.session_maker() as session:
                collection = store.get_collection(session)
                if collection is None:
                    return
                query = (
                    session.query(embedding_store)
                    .filter(embedding_store.collection_id == collection.uuid)
                    .order_by(embedding_store.id)
                    .yield_per(batch_size)
                )
                for row in query:
                    yield to_record(
                        row.id,
                        "" if row.document is None else str(row.document),
                        metadata=row.cmetadata,
                        embedding=getattr(row, "embedding", None),
                        namespace=self._namespace or getattr(store, "collection_name", None),
                    )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise BackendError(f"LangChain PGVector read failed: {exc}") from exc

    def _read_by_ids(self, batch_size: int) -> Iterator[MemoryRecord]:
        ids = self._ids or []
        get_by_ids = getattr(self._store, "get_by_ids", None)
        if not callable(get_by_ids):
            raise ConfigurationError(f"{type(self._store).__name__} has no get_by_ids().")
        for start in range(0, len(ids), batch_size):
            try:
                docs = get_by_ids(ids[start : start + batch_size])
            except NotImplementedError as exc:
                raise ConfigurationError(
                    f"{type(self._store).__name__} does not implement get_by_ids()."
                ) from exc
            for doc in docs:
                yield _document_record(doc, getattr(doc, "id", None), None, self._namespace)


def _kind(store: Any) -> str:
    """Recognise a LangChain vector store by the attributes it exposes."""
    contents = getattr(store, "store", None)
    if isinstance(contents, dict) and all(
        isinstance(v, dict) and "text" in v for v in list(contents.values())[:5]
    ):
        return "in_memory"
    if hasattr(store, "index_to_docstore_id") and hasattr(store, "docstore"):
        return "faiss"
    if hasattr(store, "session_maker") and hasattr(store, "EmbeddingStore"):
        return "pgvector"
    if hasattr(store, "collection_name"):
        try:
            client = store.client
        except Exception:
            client = None
        if callable(getattr(client, "scroll", None)):
            return "qdrant"
    if hasattr(store, "_text_key") and hasattr(store, "index"):
        return "pinecone"
    try:
        collection = store._collection
    except Exception:
        collection = None
    if collection is not None and callable(getattr(collection, "get", None)):
        return "chroma"
    return "unknown"


def _document_record(doc: Any, doc_id: Any, vector: Any, namespace: str | None) -> MemoryRecord:
    metadata = getattr(doc, "metadata", None)
    record_id = (
        doc_id
        or getattr(doc, "id", None)
        or (metadata.get("id") if isinstance(metadata, dict) else None)
    )
    return to_record(
        record_id or "langchain_unknown",
        str(getattr(doc, "page_content", "") or ""),
        metadata=metadata,
        embedding=vector,
        namespace=namespace,
    )


__all__ = ["LangChainScanSource"]
