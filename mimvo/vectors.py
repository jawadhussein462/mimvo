"""Nearest-neighbour helpers over stored embeddings.

These work on a list of records you already hold. During a scan, detectors
use `mimvo.corpus.Corpus` instead, which packs the vectors and computes
one shared nearest-neighbour table; `k_occurrence` here delegates to it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models.enums import MemoryStatus
from .models.memory import MemoryRecord
from .text import cosine


def as_floats(value: object) -> list[float] | None:
    """Turn a store vector into a Python list of floats.

    Args:
        value: A list or tuple of numbers, an array with `.tolist()`, or the
            text form pgvector returns when no vector adapter is registered
            (`"[0.1,0.2,0.3]"`). `None` and empty sequences become `None`.

    Returns:
        The values as `float`, or `None` when `value` is missing or not a
        sequence of numbers.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if len(text) < 2 or text[0] not in "[({" or text[-1] not in "])}":
            return None
        value = [part for part in text[1:-1].split(",") if part.strip()]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        # Named vectors: take the first dense one. A collection with a sparse
        # slot (mem0's "bm25", say) returns it next to the dense vector.
        for item in value.values():
            dense = as_floats(item)
            if dense is not None:
                return dense
        return None
    if not isinstance(value, list | tuple):
        return None
    try:
        out = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return out or None


def with_embeddings(
    records: Sequence[MemoryRecord],
    *,
    active_only: bool = True,
) -> list[MemoryRecord]:
    """Return records that carry a usable embedding.

    Args:
        records: Stored memories, possibly mixed.
        active_only: When `True` (the default), drop revoked and quarantined
            records.

    Returns:
        Records whose `embedding` is a non-empty list.
    """
    out: list[MemoryRecord] = []
    for record in records:
        if active_only and record.status != MemoryStatus.ACTIVE:
            continue
        if record.embedding:
            out.append(record)
    return out


def nearest(
    query: Sequence[float],
    records: Sequence[MemoryRecord],
    *,
    k: int,
    exclude_id: str | None = None,
    active_only: bool = True,
) -> list[tuple[MemoryRecord, float]]:
    """Return the `k` nearest records to `query` by cosine similarity.

    Args:
        query: Query vector. Usually the candidate's stored embedding.
        records: Corpus to search. Records without embeddings are skipped.
        k: Maximum number of neighbours to return. Values below 1 yield
            an empty list.
        exclude_id: Record id to skip, typically the candidate itself.
        active_only: When `True`, skip revoked and quarantined records.

    Returns:
        `(record, cosine)` pairs, highest similarity first. Shorter than
        `k` when fewer eligible records exist.
    """
    if k < 1:
        return []
    ranked: list[tuple[MemoryRecord, float]] = []
    query_list = list(query)
    for record in with_embeddings(records, active_only=active_only):
        if exclude_id is not None and record.id == exclude_id:
            continue
        assert record.embedding is not None
        try:
            score = cosine(query_list, record.embedding)
        except ValueError:
            continue
        ranked.append((record, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def k_occurrence(
    records: Sequence[MemoryRecord],
    *,
    k: int = 10,
    active_only: bool = True,
) -> dict[str, int]:
    """Count how often each record appears in the others' `k` nearest neighbours.

    A point that is a neighbour of many others is a hub. Poison written to
    be retrieved for many queries tends to become one (Radovanović et al.,
    JMLR 2010). The mean count is `k` when every record has an embedding.

    Args:
        records: Corpus. Records without embeddings are ignored.
        k: Neighbourhood size used for each record.
        active_only: When `True`, skip revoked and quarantined records.

    Returns:
        `{record_id: count}` for every record that had an embedding.
        Missing ids were skipped.
    """
    from .corpus import Corpus

    corpus = Corpus.from_records(list(records))
    if active_only:
        return corpus.k_occurrence(k)
    counts: dict[str, int] = {}
    for row, count in corpus.knn(k, active_only=False).occurrence().items():
        counts[corpus.ids[row]] = counts.get(corpus.ids[row], 0) + count
    return counts


__all__ = [
    "as_floats",
    "k_occurrence",
    "nearest",
    "with_embeddings",
]
