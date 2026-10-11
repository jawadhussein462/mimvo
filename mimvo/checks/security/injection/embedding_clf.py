"""Embedding-based prompt-injection classifier.

From Ayub & Majumdar, 2024, "Embedding-based classifiers can detect
prompt injection attacks". When the store's embedding model matches the
classifier, this runs on the vectors already in the store and costs
almost nothing.

Needs a stored embedding and a `score` callback `(vector) -> probability`.
Without either, this detector returns nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ....context import CheckContext
from ....exceptions import ConfigurationError
from ....models.memory import MemoryCandidate
from ..base import BaseDetector, Detection

Score = Callable[[Sequence[float]], float]


class EmbeddingInjectionDetector(BaseDetector):
    """Flag a memory whose stored vector looks like prompt injection.

    The finding code is `embedding_injection`.
    """

    name = "embedding_classifier"
    needs_corpus = False

    def __init__(self, *, score: Score | None = None, threshold: float = 0.5) -> None:
        """Store the classifier and the probability cutoff.

        Args:
            score: Function `(embedding) -> injection probability` in
                `[0, 1]`. `None` disables the detector.
            threshold: Minimum score to flag. Default `0.5`. Must be
                within (0, 1].

        Raises:
            ConfigurationError: `threshold` is outside (0, 1].
        """
        if not 0.0 < threshold <= 1.0:
            raise ConfigurationError("threshold must be within (0, 1].")
        self.score_fn = score
        self.threshold = threshold

    def detect(self, candidate: MemoryCandidate, context: CheckContext) -> list[Detection]:
        """Score the stored vector.

        Args:
            candidate: Needs `candidate.embedding`.
            context: Unused. Accepted to match the detector protocol.

        Returns:
            One `Detection` with code `embedding_injection` when the
            score is at or above `threshold`. Empty otherwise.
        """
        del context
        if self.score_fn is None or not candidate.embedding:
            return []
        probability = float(self.score_fn(candidate.embedding))
        if probability < self.threshold:
            return []
        return [
            self.hit(
                code="embedding_injection",
                score=probability,
                threshold=self.threshold,
            )
        ]


__all__ = ["EmbeddingInjectionDetector"]
