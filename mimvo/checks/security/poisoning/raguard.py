"""RAGuard: chunk-wise perplexity plus a text-similarity filter.

From Cheng et al., 2025, arXiv:2510.25025. Gradient-built attack suffixes
(HotFlip, GCG) look like gibberish to a causal language model, but scoring
the whole passage dilutes them. RAGuard scores overlapping chunks and
keeps a hit only when the surprising span is also unlike the rest of the
document. Fluent poison written by a language model is not caught here.

Pass `perplexity=` to supply the score yourself. The default loads a small
causal model through `transformers` (`gpt2` unless `model_id` says
otherwise), the same way `PerplexityDetector` does.
"""

from __future__ import annotations

from ....exceptions import ConfigurationError
from ....text import similarity
from ..base import BaseDetector, Detection
from .perplexity import DEFAULT_MODEL, Perplexity, _load_perplexity

_MIN_CHUNK_WORDS = 6


def _chunks(text: str, chunk_words: int) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text] if words else []
    step = max(1, chunk_words // 2)
    out: list[str] = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_words])
        if piece:
            out.append(piece)
        if start + chunk_words >= len(words):
            break
    return out


class RAGuardDetector(BaseDetector):
    """Flag a high-perplexity span that does not belong with the rest of the text.

    The finding code is `adversarial_text`. Short texts (fewer than
    `min_words`) are skipped.
    """

    name = "raguard"

    def __init__(
        self,
        *,
        threshold: float = 1000.0,
        perplexity: Perplexity | None = None,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 1024,
        min_words: int = 6,
        chunk_words: int = 32,
        max_context_similarity: float = 0.4,
    ) -> None:
        """Store the thresholds. The model is loaded on the first scan.

        Args:
            threshold: Flag a chunk at or above this perplexity. Must be
                greater than 1. Default `1000`. Calibrate it on clean memories.
            perplexity: Function `(text) -> float`. `None` loads `model_id`
                through `transformers` on first use.
            model_id: Hugging Face causal language model. Default `gpt2`.
            device: Device string passed to the model, such as `"cpu"`.
            max_length: Tokens kept before scoring one chunk.
            min_words: Skip text with fewer than this many words.
            chunk_words: Words per chunk. Default `32`. Must be at least 1.
            max_context_similarity: A high-perplexity chunk is kept only
                when its similarity to the rest of the document is below
                this value. A single-chunk document skips this filter.
                Default `0.4`.

        Raises:
            ConfigurationError: `threshold` is 1 or less, or `chunk_words`
                is below 1.
        """
        if threshold <= 1.0:
            raise ConfigurationError("threshold must be greater than 1 (perplexity is >= 1).")
        if chunk_words < 1:
            raise ConfigurationError("chunk_words must be >= 1.")
        if not 0.0 <= max_context_similarity <= 1.0:
            raise ConfigurationError("max_context_similarity must be within [0, 1].")
        self.threshold = threshold
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.min_words = max(1, min_words)
        self.chunk_words = chunk_words
        self.max_context_similarity = max_context_similarity
        self._perplexity: Perplexity | None = perplexity

    def perplexity(self, text: str) -> float:
        """Score how surprising `text` is to the language model.

        Args:
            text: One chunk of memory content.

        Returns:
            Perplexity, a float of at least 1.

        Raises:
            IntegrationError: The default model is used and `torch` or
                `transformers` is not installed.
        """
        if self._perplexity is None:
            self._perplexity = _load_perplexity(self.model_id, self.device, self.max_length)
        return self._perplexity(text)

    def detect_text(self, text: str) -> list[Detection]:
        if len(text.split()) < self.min_words:
            return []
        chunks = _chunks(text, self.chunk_words)
        if not chunks:
            return []
        hits: list[tuple[int, float, float]] = []
        for index, chunk in enumerate(chunks):
            if len(chunk.split()) < _MIN_CHUNK_WORDS:
                continue
            ppl = self.perplexity(chunk)
            if ppl < self.threshold:
                continue
            rest = " ".join(c for i, c in enumerate(chunks) if i != index)
            context_sim = similarity(chunk, rest) if rest else 0.0
            if rest and context_sim >= self.max_context_similarity:
                continue
            hits.append((index, ppl, context_sim))
        if not hits:
            return []
        _index, ppl, context_sim = max(hits, key=lambda item: item[1])
        score = min(1.0, 0.5 + 0.5 * (1 - self.threshold / ppl))
        return [
            self.hit(
                code="adversarial_text",
                score=score,
                perplexity=round(ppl, 1),
                threshold=self.threshold,
                chunks_flagged=len(hits),
                chunks=len(chunks),
                context_similarity=round(context_sim, 3),
                model=self.model_id,
            )
        ]


__all__ = ["RAGuardDetector"]
