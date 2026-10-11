"""Probe-query traceback: a record that flips the answer on its own is suspect.

From RAGForensics (Zhang et al., WWW 2025, arXiv:2504.21668). Generate
probe questions from the record (doc2query or an LLM), retrieve, and
answer with and without the record. A record that changes the answer by
itself is treated as suspected poisoning.

This is also how a query-free store scan feeds query-time filters: the
generated probes are the queries FilterRAG would have needed.

All three steps are callbacks. Without them this detector returns nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ....context import CheckContext
from ....exceptions import ConfigurationError
from ....models.memory import MemoryCandidate, MemoryRecord
from ..base import BaseDetector, Detection
from ._corpus import active_neighbours

GenerateQueries = Callable[[str], Sequence[str]]
Retrieve = Callable[[str, Sequence[MemoryRecord]], Sequence[MemoryRecord]]
Answer = Callable[[str, Sequence[MemoryRecord]], str]


class ProbeQueryDetector(BaseDetector):
    """Flag a memory that changes the answer to its own probe questions.

    The finding code is `retrieval_flip`.
    """

    name = "probe_query"
    needs_corpus = True

    def __init__(
        self,
        *,
        generate_queries: GenerateQueries | None = None,
        retrieve: Retrieve | None = None,
        answer: Answer | None = None,
        min_flips: int = 1,
        max_queries: int = 3,
    ) -> None:
        """Store the three callbacks. Nothing is called until `detect`.

        Args:
            generate_queries: `(text) -> probe questions`. `None` disables
                the detector.
            retrieve: `(query, corpus) -> retrieved records`. `None`
                disables the detector.
            answer: `(query, retrieved) -> answer string`. `None`
                disables the detector.
            min_flips: How many probes must change the answer. Default `1`.
            max_queries: Probes to keep from `generate_queries`. Default `3`.

        Raises:
            ConfigurationError: `min_flips` or `max_queries` is below 1.
        """
        if min_flips < 1:
            raise ConfigurationError("min_flips must be >= 1.")
        if max_queries < 1:
            raise ConfigurationError("max_queries must be >= 1.")
        self.generate_queries = generate_queries
        self.retrieve = retrieve
        self.answer = answer
        self.min_flips = min_flips
        self.max_queries = max_queries

    def detect(self, candidate: MemoryCandidate, context: CheckContext) -> list[Detection]:
        """Ask whether removing this record changes answers to its probes.

        Args:
            candidate: The memory under test. Needs `candidate.id`.
            context: The materialised corpus in `context.existing`.

        Returns:
            One `Detection` with code `retrieval_flip` when at least
            `min_flips` probes change answer after the record is removed.
            Empty when any callback is missing.
        """
        if self.generate_queries is None or self.retrieve is None or self.answer is None:
            return []
        if candidate.id is None:
            return []
        queries = [q.strip() for q in self.generate_queries(candidate.content) if q.strip()]
        queries = queries[: self.max_queries]
        if not queries:
            return []
        self_record = next((r for r in context.existing if r.id == candidate.id), None)
        if self_record is None:
            return []
        corpus = [self_record, *active_neighbours(candidate, context)]
        without = [r for r in corpus if r.id != candidate.id]
        flipped: list[str] = []
        for query in queries:
            with_hit = self.retrieve(query, corpus)
            without_hit = self.retrieve(query, without)
            if self.answer(query, with_hit).strip() != self.answer(query, without_hit).strip():
                flipped.append(query)
        if len(flipped) < self.min_flips:
            return []
        return [
            self.hit(
                code="retrieval_flip",
                score=min(1.0, len(flipped) / max(len(queries), 1)),
                flipped=len(flipped),
                probes=len(queries),
            )
        ]


__all__ = ["ProbeQueryDetector"]
