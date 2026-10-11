"""Embedding-text consistency: the stored vector should match the text.

A store scanner is the only place this check can run. Re-embed the text
and compare it with the vector sitting in the store. A mismatch means
the vector was written directly or tampered with (AgentPoison-style
trigger optimisation works in embedding space).

Needs a stored embedding on the candidate and an `embed` callback that
uses the same model the store used. Without either, this detector
returns nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ....context import CheckContext
from ....exceptions import ConfigurationError
from ....models.memory import MemoryCandidate
from ....text import cosine
from ..base import BaseDetector, Detection

Embed = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class EmbeddingConsistencyDetector(BaseDetector):
    """Flag a memory whose stored vector does not match a fresh embedding.

    The finding code is `embedding_mismatch`.
    """

    name = "embedding_consistency"
    needs_corpus = False

    def __init__(
        self,
        *,
        embed: Embed | None = None,
        min_cosine: float = 0.85,
    ) -> None:
        """Store the embedder and the cosine floor.

        Args:
            embed: Function `(texts) -> one vector per text`. `None`
                disables the detector (it returns nothing).
            min_cosine: Minimum cosine between the stored vector and the
                re-embedded text. Default `0.85`. Must be within (-1, 1].

        Raises:
            ConfigurationError: `min_cosine` is outside (-1, 1].
        """
        if not -1.0 < min_cosine <= 1.0:
            raise ConfigurationError("min_cosine must be within (-1, 1].")
        self.embed = embed
        self.min_cosine = min_cosine

    def detect(self, candidate: MemoryCandidate, context: CheckContext) -> list[Detection]:
        """Compare the stored vector with a fresh embedding of the text.

        Args:
            candidate: Needs `candidate.embedding` and `candidate.content`.
            context: Unused. Accepted to match the detector protocol.

        Returns:
            One `Detection` with code `embedding_mismatch` when cosine
            similarity is below `min_cosine`. Empty when there is no
            stored vector or no `embed` function.

        Raises:
            ConfigurationError: `embed` did not return one vector.
        """
        del context
        if self.embed is None or not candidate.embedding:
            return []
        vectors = self.embed([candidate.content])
        if len(vectors) != 1:
            raise ConfigurationError("embed() must return one vector per input text.")
        try:
            score = cosine(list(candidate.embedding), list(vectors[0]))
        except ValueError:
            return [
                self.hit(
                    code="embedding_mismatch",
                    score=1.0,
                    cosine=None,
                    reason="dimension_mismatch",
                )
            ]
        if score >= self.min_cosine:
            return []
        return [
            self.hit(
                code="embedding_mismatch",
                score=min(1.0, (self.min_cosine - score) / max(self.min_cosine, 1e-6)),
                cosine=round(score, 4),
                min_cosine=self.min_cosine,
            )
        ]


__all__ = ["EmbeddingConsistencyDetector"]
