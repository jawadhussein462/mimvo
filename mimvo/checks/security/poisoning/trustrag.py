"""TrustRAG: coordinated poison shows up as a tight cluster of near-copies.

From TrustRAG (Zhou et al., arXiv:2501.00879). An attacker who plants
several passages for one question pushes them all toward the same query
and answer, so the poisoned passages sit close together in embedding space
(pairwise cosine at least 0.85 in the paper) and share many word sequences
(ROUGE-L at least 0.25). That second test keeps ordinary documents on the
same topic from counting. One poisoned passage sitting among clean ones
is not caught here.

Applied to memory: a candidate is suspicious when at least `min_cluster`
neighbours are near-paraphrases of it. Neighbours come from the scanned
batch (`context.existing`).

Search order:

1. Stored vectors, via the scan's shared nearest-neighbour table (computed
   once per scan, with numpy when it is installed).
2. An `embed` callback: every record's text is embedded once per scan, in
   batches, then the same nearest-neighbour cut.
3. Lexical similarity. Candidates are the records that share rare word
   pairs with this one, so it no longer compares every record with every
   other; phrases shared by hundreds of records (templates) are ignored.
   Stores larger than `max_pairwise` records without vectors are skipped.

Pass `embed`, a function `(texts) -> vectors`, when the records have no
stored embeddings. Real copies of the same fact trigger this too, so the
finding is low severity and asks for review rather than quarantine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ....context import CheckContext
from ....corpus import Corpus, corpus_of
from ....exceptions import ConfigurationError
from ....models.enums import MemoryStatus
from ....models.memory import MemoryCandidate
from ....text import jaccard, ratio, rouge_l
from ..base import BaseDetector, Detection

Embed = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class TrustRAGDetector(BaseDetector):
    """Flag a memory that has several near-copy neighbours.

    The finding code is `poisoning_cluster`. A one-record scan has no
    neighbours, so this detector returns nothing for it.
    """

    name = "trustrag"
    needs_corpus = True

    def __init__(
        self,
        *,
        embed: Embed | None = None,
        cosine_threshold: float = 0.85,
        rouge_threshold: float = 0.25,
        min_cluster: int = 2,
        nn: int = 32,
        max_pairwise: int = 200_000,
    ) -> None:
        """Store the cluster thresholds. No model is loaded here.

        Args:
            embed: Function `(texts) -> one vector per text`. Used when the
                candidate and its neighbours have no stored embeddings.
            cosine_threshold: Minimum similarity with a neighbour, from just
                above 0 to 1. Default `0.85`, the paper's cutoff. Also used
                as the lexical cutoff when neither stored vectors nor
                `embed` are available.
            rouge_threshold: Minimum ROUGE-L overlap with a neighbour, from
                just above 0 to 1. Default `0.25`.
            min_cluster: How many neighbours must pass both tests. Default
                `2`. Must be at least 1.
            nn: How many nearest neighbours to inspect when vectors are
                available. Default `32`.
            max_pairwise: Largest batch that may fall back to lexical
                similarity. Larger batches without vectors are skipped.
                Default `200000`.

        Raises:
            ConfigurationError: A threshold is outside (0, 1], or
                `min_cluster` / `nn` / `max_pairwise` is below 1.
        """
        if not 0.0 < cosine_threshold <= 1.0 or not 0.0 < rouge_threshold <= 1.0:
            raise ConfigurationError("thresholds must be within (0, 1].")
        if min_cluster < 1:
            raise ConfigurationError("min_cluster must be >= 1.")
        if nn < 1:
            raise ConfigurationError("nn must be >= 1.")
        if max_pairwise < 1:
            raise ConfigurationError("max_pairwise must be >= 1.")
        self.embed = embed
        self.cosine_threshold = cosine_threshold
        self.rouge_threshold = rouge_threshold
        self.min_cluster = min_cluster
        self.nn = nn
        self.max_pairwise = max_pairwise

    def _ranked(
        self, candidate: MemoryCandidate, corpus: Corpus, neighbours: int
    ) -> tuple[list[tuple[int, float]], str]:
        row = corpus.row_of(candidate)
        if candidate.embedding and corpus.neighbour_count(candidate.id, with_vector=True) > 0:
            table = corpus.knn(self.nn)
            if row is not None and row in table:
                return table[row], "stored"
            rows = [r for r in corpus.vector_rows() if corpus.ids[r] != candidate.id]
            return corpus.nearest(candidate.embedding, rows, self.nn), "stored"
        if self.embed is not None:
            table = corpus.knn_embedded(self.embed, self.nn)
            if row is not None and row in table:
                return table[row], "cosine"
            vectors = corpus.embed_all(self.embed)
            (anchor,) = _embed_one(self.embed, candidate.content)
            rows = corpus.active_rows(exclude_id=candidate.id)
            return corpus.nearest(anchor, rows, self.nn, vectors=vectors), "cosine"
        if neighbours > self.max_pairwise:
            return [], "skipped"
        pool = corpus.lexical_candidates(candidate.content, exclude=row, limit=self.nn)
        scored: list[tuple[int, float]] = []
        for r in pool:
            if corpus.ids[r] == candidate.id or corpus.statuses[r] != MemoryStatus.ACTIVE:
                continue
            other = corpus.contents[r]
            overlap = jaccard(candidate.content, other)
            # difflib is slow, and near-identical character sequences share
            # most of their words, so the character ratio is only computed
            # when at least half the words already match.
            sim = max(overlap, ratio(candidate.content, other)) if overlap >= 0.5 else overlap
            scored.append((r, sim))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.nn], "lexical"

    def detect(self, candidate: MemoryCandidate, context: CheckContext) -> list[Detection]:
        """Look for a near-copy cluster around this memory.

        Args:
            candidate: The memory being scanned.
            context: The scan context. Neighbours are the scan's other
                active records; inactive records and the candidate's own
                id are skipped.

        Returns:
            One `Detection` with code `poisoning_cluster` when at least
            `min_cluster` neighbours are close enough. Otherwise an empty list.

        Raises:
            ConfigurationError: `embed` did not return one vector per text.
        """
        corpus = corpus_of(context)
        neighbours = corpus.neighbour_count(candidate.id)
        if neighbours < self.min_cluster:
            return []
        ranked, metric = self._ranked(candidate, corpus, neighbours)
        if not ranked:
            return []
        cluster: list[tuple[str, float, float]] = []
        for row, sim in ranked:
            if sim < self.cosine_threshold or corpus.ids[row] == candidate.id:
                continue
            overlap = rouge_l(candidate.content, corpus.contents[row])
            if overlap >= self.rouge_threshold:
                cluster.append((corpus.ids[row], sim, overlap))
        if len(cluster) < self.min_cluster:
            return []
        cluster.sort(key=lambda item: -item[1])
        return [
            self.hit(
                code="poisoning_cluster",
                score=min(1.0, sum(s for _, s, _ in cluster) / len(cluster)),
                cluster=[rid for rid, _, _ in cluster][:20],
                cluster_size=len(cluster),
                similarity_metric=metric,
                min_similarity=round(min(s for _, s, _ in cluster), 3),
                min_rouge_l=round(min(r for _, _, r in cluster), 3),
            )
        ]


def _embed_one(embed: Embed, text: str) -> list[list[float]]:
    vectors = embed([text])
    if len(vectors) != 1:
        raise ConfigurationError("embed() must return one vector per input text.")
    return [list(map(float, vectors[0]))]


__all__ = ["TrustRAGDetector"]
