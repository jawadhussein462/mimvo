"""The scanned records, packed for corpus detectors.

TrustRAG, hubness, and temporal NLI compare each record with its nearest
neighbours across the whole scan. Done naively that is every record against
every other in Python, and every vector held as a list of Python floats
(about 32 bytes per number). A 50,000-record store with 1,536-dimension
embeddings needed gigabytes of memory and hours of compute.

`Corpus` keeps what those detectors need and nothing else:

* text, ids, status, timestamps, and metadata per record;
* stored vectors packed as 4-byte floats in one buffer;
* one k-nearest-neighbour table per scan, computed once and shared.

With numpy installed (`pip install "mimvo[fast]"`; the Chroma and
Qdrant clients already depend on it) the table is an exact matrix product,
computed in row blocks so its working memory stays near 64 MB. Without
numpy a pure-Python fallback gives the same answers and is practical up to
a few thousand records.

A different index (FAISS, HNSW, or the store's own search) can be plugged
in through `Corpus(index=...)`; see `NeighbourIndex`.

Detectors reach it through `corpus_of(context)`.
"""

from __future__ import annotations

import heapq
import importlib
import itertools
import math
from array import array
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

from .models.enums import MemoryStatus
from .models.memory import MemoryCandidate, MemoryRecord
from .telemetry import get_logger
from .text import tokenize

if TYPE_CHECKING:
    from .context import CheckContext

_logger = get_logger(__name__)

_CACHE_KEY = "mimvo.corpus"
# Bytes of similarity scores held at once while computing the k-NN table.
_BLOCK_BYTES = 64 * 1024 * 1024

Embed = Callable[[Sequence[str]], Sequence[Sequence[float]]]
Neighbours = list[tuple[int, float]]


def _numpy() -> Any | None:
    """numpy when installed, else `None`. Imported by name so it stays optional."""
    try:
        return importlib.import_module("numpy")
    except ImportError:  # pragma: no cover - exercised only without numpy
        return None


@runtime_checkable
class NeighbourIndex(Protocol):
    """Finds the k most similar rows for every row of a vector matrix.

    Implement this to use FAISS, HNSW, or a store's own similarity search
    instead of the exact search Mimvo ships.
    """

    def knn(self, vectors: Sequence[Sequence[float]] | Any, k: int) -> list[Neighbours]:
        """Return each row's `k` nearest other rows by cosine similarity.

        Args:
            vectors: One vector per row, all the same dimension. A numpy
                array when numpy is installed, otherwise a list of lists.
            k: Neighbours per row. Fewer are returned when there are fewer
                other rows.

        Returns:
            One list per row of `(row, cosine)` pairs, most similar first,
            never including the row itself.
        """
        ...


class ExactIndex:
    """Exact cosine k-NN. Uses numpy when it is installed."""

    def knn(self, vectors: Sequence[Sequence[float]] | Any, k: int) -> list[Neighbours]:
        """See `NeighbourIndex.knn`."""
        arrays = self.knn_arrays(vectors, k)
        if arrays is None:
            n = len(vectors)
            k = min(k, n - 1)
            if n == 0 or k < 1:
                return [[] for _ in range(n)]
            return _knn_python([list(map(float, v)) for v in vectors], k)
        top, values = arrays
        return [
            list(zip(row_idx.tolist(), row_val.tolist(), strict=True))
            for row_idx, row_val in zip(top, values, strict=True)
        ]

    def knn_arrays(
        self, vectors: Sequence[Sequence[float]] | Any, k: int
    ) -> tuple[Any, Any] | None:
        """Exact k-NN as two `(n, k)` numpy arrays: neighbour positions and cosines.

        Similarities are computed in blocks of rows so working memory stays
        near 64 MB whatever the store size.

        Returns:
            `(positions, cosines)`, most similar first per row, or `None`
            when numpy is not installed.
        """
        np = _numpy()
        if np is None:
            return None
        n = len(vectors)
        k = min(k, n - 1)
        if n == 0 or k < 1:
            return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32)
        # One float32 copy, normalised in place.
        unit = np.array(vectors, dtype=np.float32, copy=True)
        norms = np.linalg.norm(unit, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit /= norms
        block = max(1, _BLOCK_BYTES // (4 * n))
        top_all = np.empty((n, k), dtype=np.int64)
        val_all = np.empty((n, k), dtype=np.float32)
        for start in range(0, n, block):
            stop = min(n, start + block)
            sims = unit[start:stop] @ unit.T
            sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            values = np.take_along_axis(sims, top, axis=1)
            order = np.argsort(-values, axis=1, kind="stable")
            top_all[start:stop] = np.take_along_axis(top, order, axis=1)
            val_all[start:stop] = np.take_along_axis(values, order, axis=1)
        return top_all, val_all


def _knn_python(vectors: list[list[float]], k: int) -> list[Neighbours]:
    unit = []
    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        unit.append([x / norm for x in vec])
    out: list[Neighbours] = []
    for i, a in enumerate(unit):
        scored = (
            (j, sum(x * y for x, y in zip(a, b, strict=True))) for j, b in enumerate(unit) if j != i
        )
        out.append(heapq.nlargest(k, scored, key=lambda item: item[1]))
    return out


class KnnTable(Mapping[int, Neighbours]):
    """Each row's nearest neighbours, built into `(row, cosine)` lists on access.

    Backed by two numpy arrays when the exact index computed it, so a
    50,000-row table costs a few megabytes instead of millions of tuples.
    """

    def __init__(
        self,
        rows: list[int],
        k: int,
        *,
        arrays: tuple[Any, Any] | None = None,
        lists: list[Neighbours] | None = None,
    ) -> None:
        self._rows = rows
        self._pos = {row: i for i, row in enumerate(rows)}
        self.k = k
        self._arrays = arrays
        self._lists = lists

    def limit(self, k: int) -> KnnTable:
        """The same table, cut to `k` neighbours per row (shares storage)."""
        table = KnnTable.__new__(KnnTable)
        table._rows, table._pos, table._arrays, table._lists = (
            self._rows,
            self._pos,
            self._arrays,
            self._lists,
        )
        table.k = min(k, self.k)
        return table

    def __getitem__(self, row: int) -> Neighbours:
        pos = self._pos[row]
        if self._arrays is not None:
            positions, values = self._arrays
            return [
                (self._rows[p], v)
                for p, v in zip(
                    positions[pos, : self.k].tolist(), values[pos, : self.k].tolist(), strict=True
                )
            ]
        assert self._lists is not None
        return [(self._rows[p], v) for p, v in self._lists[pos][: self.k]]

    def __contains__(self, row: object) -> bool:
        return row in self._pos

    def __iter__(self) -> Iterator[int]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def occurrence(self) -> dict[int, int]:
        """How many rows list each row among their neighbours."""
        counts = dict.fromkeys(self._rows, 0)
        np = _numpy()
        if self._arrays is not None and np is not None:
            positions = self._arrays[0][:, : self.k].ravel()
            for pos, count in enumerate(np.bincount(positions, minlength=len(self._rows)).tolist()):
                counts[self._rows[pos]] = count
            return counts
        for row in self._rows:
            for neighbour, _ in self[row]:
                counts[neighbour] += 1
        return counts


class Corpus:
    """Every record of one scan, packed for neighbour searches.

    Rows are numbered in the order records were added. Vectors whose
    dimension differs from the first vector seen are ignored (and logged
    once), the same as the old pairwise code, which skipped them.

    Attributes:
        ids, contents, statuses, created, updated, metadata, derived_from:
            One entry per row.
    """

    def __init__(self, *, index: NeighbourIndex | None = None) -> None:
        """Start an empty corpus.

        Args:
            index: k-NN implementation. `None` uses `ExactIndex`.
        """
        self.index: NeighbourIndex = index or ExactIndex()
        self.ids: list[str] = []
        self.contents: list[str] = []
        self.statuses: list[MemoryStatus] = []
        self.created: list[datetime] = []
        self.updated: list[datetime] = []
        self.metadata: list[dict[str, object]] = []
        self.derived_from: list[list[str]] = []
        self._vectors = array("f")
        self._vector_slot: list[int] = []  # row -> slot in _vectors, or -1
        self.dim: int | None = None
        self._skipped_dims = 0
        self._knn: dict[tuple[str, bool], KnnTable] = {}
        self._embedded: dict[int, tuple[Embed, list[list[float]]]] = {}
        self._postings: dict[str, list[int]] | None = None
        self._postings_for: int | None = None
        self._frequency: Counter[str] = Counter()
        self._counts: dict[str, Any] = {}

    # -- building -------------------------------------------------------------

    def add(self, record: MemoryRecord) -> int:
        """Append one record and return its row number.

        Args:
            record: A stored memory. Its embedding is copied into the packed
                buffer; the `MemoryRecord` is not kept.

        Returns:
            The new row's index.
        """
        row = len(self.ids)
        self.ids.append(record.id)
        self.contents.append(record.content)
        self.statuses.append(record.status)
        self.created.append(record.created_at)
        self.updated.append(record.updated_at)
        self.metadata.append(record.metadata)
        self.derived_from.append(record.derived_from)
        slot = -1
        vector = record.embedding
        if vector:
            if self.dim is None:
                self.dim = len(vector)
            if len(vector) == self.dim:
                slot = len(self._vectors) // self.dim
                self._vectors.extend(vector)
            else:
                self._skipped_dims += 1
                if self._skipped_dims == 1:
                    _logger.warning(
                        "record {!r} has a {}-dimension vector; expected {}. Vectors with other "
                        "dimensions are ignored by vector detectors.",
                        record.id,
                        len(vector),
                        self.dim,
                    )
        self._vector_slot.append(slot)
        self._knn.clear()
        self._counts.clear()
        self._postings = None
        return row

    @classmethod
    def from_records(cls, records: Sequence[MemoryRecord]) -> Corpus:
        """Build a corpus from records already in memory."""
        corpus = cls()
        for record in records:
            corpus.add(record)
        return corpus

    # -- rows ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    def has_vector(self, row: int) -> bool:
        """Whether `row` has a usable stored vector."""
        return self._vector_slot[row] >= 0

    def vector(self, row: int) -> list[float] | None:
        """The stored vector of `row`, or `None`."""
        slot = self._vector_slot[row]
        if slot < 0 or self.dim is None:
            return None
        start = slot * self.dim
        return list(self._vectors[start : start + self.dim])

    def record(self, row: int) -> MemoryRecord:
        """Rebuild row `row` as a `MemoryRecord`, embedding included."""
        return MemoryRecord.model_construct(
            id=self.ids[row],
            content=self.contents[row],
            status=self.statuses[row],
            created_at=self.created[row],
            updated_at=self.updated[row],
            derived_from=list(self.derived_from[row]),
            metadata=dict(self.metadata[row]),
            embedding=self.vector(row),
        )

    def candidate(self, row: int) -> MemoryCandidate:
        """Row `row` as the `MemoryCandidate` checks receive."""
        return MemoryCandidate.model_construct(
            content=self.contents[row],
            metadata=dict(self.metadata[row]),
            id=self.ids[row],
            derived_from=list(self.derived_from[row]),
            created_at=self.created[row],
            embedding=self.vector(row),
        )

    def records(self) -> RecordsView:
        """A read-only sequence of the rows as `MemoryRecord` objects."""
        return RecordsView(self)

    def rows_for(self, candidate: MemoryCandidate) -> list[int]:
        """Rows whose id equals `candidate.id` (usually one)."""
        if candidate.id is None:
            return []
        index = self._counts.get("rows_by_id")
        if index is None:
            index = {}
            for row, rid in enumerate(self.ids):
                index.setdefault(rid, []).append(row)
            self._counts["rows_by_id"] = index
        return list(index.get(candidate.id, ()))

    def row_of(self, candidate: MemoryCandidate) -> int | None:
        """The row `candidate` came from, matched by id and content."""
        for row in self.rows_for(candidate):
            if self.contents[row] == candidate.content:
                return row
        return None

    def active_rows(self, *, exclude_id: str | None = None) -> list[int]:
        """Rows with status `active`, optionally without one id."""
        return [
            row
            for row, status in enumerate(self.statuses)
            if status == MemoryStatus.ACTIVE and (exclude_id is None or self.ids[row] != exclude_id)
        ]

    def active_id_counts(self) -> Counter[str]:
        """How many active rows carry each id (cached)."""
        cached = self._counts.get("active_ids")
        if cached is None:
            cached = Counter(
                rid
                for rid, status in zip(self.ids, self.statuses, strict=True)
                if status == MemoryStatus.ACTIVE
            )
            self._counts["active_ids"] = cached
        return cached

    def active_vector_id_counts(self) -> Counter[str]:
        """How many active rows with a usable vector carry each id (cached)."""
        cached = self._counts.get("active_vector_ids")
        if cached is None:
            cached = Counter(self.ids[row] for row in self.vector_rows())
            self._counts["active_vector_ids"] = cached
        return cached

    def neighbour_count(self, candidate_id: str | None, *, with_vector: bool = False) -> int:
        """Active rows other than those carrying `candidate_id`."""
        counts = self.active_vector_id_counts() if with_vector else self.active_id_counts()
        total = self.active_total(with_vector=with_vector)
        return total - (counts[candidate_id] if candidate_id is not None else 0)

    def active_total(self, *, with_vector: bool = False) -> int:
        """Number of active rows, or of active rows with a usable vector (cached)."""
        key = "active_total_vec" if with_vector else "active_total"
        cached = self._counts.get(key)
        if cached is None:
            counts = self.active_vector_id_counts() if with_vector else self.active_id_counts()
            cached = sum(counts.values())
            self._counts[key] = cached
        return int(cached)

    def k_occurrence(self, k: int) -> dict[str, int]:
        """How often each id appears in the other active rows' `k` nearest neighbours.

        A row that is a neighbour of many others is a hub (Radovanović et
        al., JMLR 2010). Computed from the shared k-NN table, once per `k`.

        Args:
            k: Neighbourhood size.

        Returns:
            `{record id: count}` for every active row with a vector.
        """
        key = f"k_occurrence:{k}"
        cached = self._counts.get(key)
        if cached is None:
            counts: dict[str, int] = {}
            for row, count in self.knn(k).occurrence().items():
                rid = self.ids[row]
                counts[rid] = counts.get(rid, 0) + count
            cached = counts
            self._counts[key] = cached
        return cached

    def vector_rows(self, *, active_only: bool = True) -> list[int]:
        """Rows that have a usable vector, active ones only by default."""
        return [
            row
            for row in range(len(self))
            if self._vector_slot[row] >= 0
            and (not active_only or self.statuses[row] == MemoryStatus.ACTIVE)
        ]

    # -- neighbours ----------------------------------------------------------

    def knn(self, k: int, *, active_only: bool = True) -> KnnTable:
        """Each vector row's `k` nearest vector rows, computed once per scan.

        Args:
            k: Neighbours per row.
            active_only: Search among, and return, active rows only.

        Returns:
            `{row: [(neighbour row, cosine), ...]}` for every row with a
            vector, most similar first, never including the row itself.
        """
        return self._table("stored", k, active_only, lambda rows: self._stored_matrix(rows))

    def knn_embedded(self, embed: Embed, k: int, *, active_only: bool = True) -> KnnTable:
        """Like `knn`, over fresh embeddings of every row's text.

        `embed` is called once per scan, in batches, not once per record.
        """
        vectors = self.embed_all(embed)
        return self._table(
            f"embed:{id(embed)}", k, active_only, lambda rows: [vectors[r] for r in rows]
        )

    def embed_all(self, embed: Embed, *, batch_size: int = 256) -> list[list[float]]:
        """Embed every row's text with `embed`, once, and cache the vectors."""
        cached = self._embedded.get(id(embed))
        if cached is not None and cached[0] is embed and len(cached[1]) == len(self):
            return cached[1]
        out: list[list[float]] = []
        for start in range(0, len(self), batch_size):
            texts = self.contents[start : start + batch_size]
            vectors = embed(texts)
            if len(vectors) != len(texts):
                from .exceptions import ConfigurationError

                raise ConfigurationError("embed() must return one vector per input text.")
            out.extend(list(map(float, v)) for v in vectors)
        self._embedded[id(embed)] = (embed, out)
        return out

    def nearest(
        self,
        query: Sequence[float],
        rows: Sequence[int],
        k: int,
        *,
        vectors: Sequence[Sequence[float]] | None = None,
    ) -> Neighbours:
        """The `k` rows among `rows` most similar to `query`, exactly.

        Args:
            query: The query vector.
            rows: Rows to search. Rows without a vector are skipped.
            k: How many to return.
            vectors: Per-row vectors to use instead of the stored ones.

        Returns:
            `(row, cosine)` pairs, most similar first.
        """
        if k < 1:
            return []
        if vectors is None:
            usable = [r for r in rows if self._vector_slot[r] >= 0]
            matrix: Any = self._stored_matrix(usable)
        else:
            usable = list(rows)
            matrix = [vectors[r] for r in usable]
        if not usable or len(query) != len(matrix[0]):
            return []
        np = _numpy()
        if np is not None:
            mat = np.asarray(matrix, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            q = np.asarray(query, dtype=np.float32)
            qn = float(np.linalg.norm(q)) or 1.0
            sims = (mat @ q) / (norms * qn)
            take = min(k, len(usable))
            top = np.argpartition(-sims, take - 1)[:take]
            top = top[np.argsort(-sims[top], kind="stable")]
            return [(usable[int(i)], float(sims[int(i)])) for i in top]
        qn = math.sqrt(sum(x * x for x in query)) or 1.0
        scored = []
        for row, vec in zip(usable, matrix, strict=True):
            vn = math.sqrt(sum(x * x for x in vec)) or 1.0
            scored.append((row, sum(x * y for x, y in zip(query, vec, strict=True)) / (qn * vn)))
        return heapq.nlargest(k, scored, key=lambda item: item[1])

    def _stored_matrix(self, rows: Sequence[int]) -> Any:
        assert self.dim is not None or not rows
        dim = self.dim or 0
        np = _numpy()
        if np is not None:
            if not dim:
                return np.zeros((0, 0), dtype=np.float32)
            buffer = np.frombuffer(self._vectors, dtype=np.float32).reshape(-1, dim)
            slots = [self._vector_slot[r] for r in rows]
            if slots == list(range(len(buffer))):
                return buffer  # every vector, in order: no copy
            return buffer[slots]
        return [self.vector(r) for r in rows]

    def _table(
        self,
        key: str,
        k: int,
        active_only: bool,
        matrix_for: Callable[[list[int]], Any],
    ) -> KnnTable:
        cached = self._knn.get((key, active_only))
        if cached is None or cached.k < k:
            if key == "stored":
                rows = self.vector_rows(active_only=active_only)
            else:
                rows = self.active_rows() if active_only else list(range(len(self)))
            matrix = matrix_for(rows) if rows else []
            arrays = None
            if rows and isinstance(self.index, ExactIndex):
                arrays = self.index.knn_arrays(matrix, k)
            if arrays is not None:
                cached = KnnTable(rows, k, arrays=arrays)
            else:
                lists = self.index.knn(matrix, k) if rows else []
                cached = KnnTable(rows, k, lists=lists)
            self._knn[(key, active_only)] = cached
        return cached.limit(k)

    # -- lexical candidates --------------------------------------------------

    def lexical_candidates(
        self,
        text: str,
        *,
        exclude: int | None = None,
        limit: int = 128,
        max_df: int = 200,
    ) -> list[int]:
        """Rows likely to be near-copies of `text`, for the lexical fallback.

        Each row is indexed under its word pairs (bigrams). A row becomes a
        candidate when it shares a pair with `text`, and candidates are
        ranked by how many they share. Pairs found in more than `max_df`
        rows are ignored: a phrase that common is a template ("User 41
        prefers plan 3"), not a sign of a planted cluster. This keeps the
        fallback near-linear even for small vocabularies.

        Args:
            text: The text to find candidates for.
            exclude: A row to leave out, usually the text's own row.
            limit: Most candidates to return.
            max_df: Ignore word pairs that occur in more rows than this.

        Returns:
            Candidate rows, most shared pairs first.
        """
        if self._postings is None or self._postings_for != max_df:
            self._build_postings(max_df)
        assert self._postings is not None
        shared: Counter[int] = Counter()
        for gram in _bigrams(text):
            shared.update(self._postings.get(gram, ()))
        if exclude is not None:
            shared.pop(exclude, None)
        return [row for row, _ in shared.most_common(limit)]

    def _build_postings(self, max_df: int) -> None:
        grams = [_bigrams(text) for text in self.contents]
        self._frequency = Counter(g for row in grams for g in row)
        postings: dict[str, list[int]] = {}
        for row, row_grams in enumerate(grams):
            for gram in row_grams:
                if self._frequency[gram] <= max_df:
                    postings.setdefault(gram, []).append(row)
        self._postings = postings
        self._postings_for = max_df


def _bigrams(text: str) -> frozenset[str]:
    words = tokenize(text)
    if len(words) < 2:
        return frozenset(words)
    return frozenset(f"{a} {b}" for a, b in itertools.pairwise(words))


class RecordsView(Sequence[MemoryRecord]):
    """`context.existing` for a streamed scan: records rebuilt on access.

    Built-in detectors use the `Corpus` directly. This view keeps custom
    detectors that iterate `context.existing` working unchanged.
    """

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus

    def __len__(self) -> int:
        return len(self._corpus)

    @overload
    def __getitem__(self, index: int) -> MemoryRecord: ...

    @overload
    def __getitem__(self, index: slice) -> list[MemoryRecord]: ...

    def __getitem__(self, index: int | slice) -> MemoryRecord | list[MemoryRecord]:
        if isinstance(index, slice):
            return [self._corpus.record(i) for i in range(*index.indices(len(self._corpus)))]
        if index < 0:
            index += len(self._corpus)
        if not 0 <= index < len(self._corpus):
            raise IndexError(index)
        return self._corpus.record(index)

    def __iter__(self) -> Iterator[MemoryRecord]:
        for row in range(len(self._corpus)):
            yield self._corpus.record(row)


def corpus_of(context: CheckContext) -> Corpus:
    """The corpus for this scan.

    The engine attaches one to every scan. When a check is run directly
    with a hand-built context (as in unit tests), one is built from
    `context.existing` on first use and cached on `context.cache`.

    Args:
        context: The scan context.

    Returns:
        A `Corpus` covering the scan's records.
    """
    if context.corpus is not None:
        return context.corpus
    cached = context.cache.get(_CACHE_KEY)
    if isinstance(cached, Corpus) and len(cached) == len(context.existing):
        return cached
    corpus = Corpus.from_records(list(context.existing))
    context.cache[_CACHE_KEY] = corpus
    return corpus


__all__ = [
    "Corpus",
    "ExactIndex",
    "KnnTable",
    "NeighbourIndex",
    "RecordsView",
    "corpus_of",
]
