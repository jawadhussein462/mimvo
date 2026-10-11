"""Flag text a language model finds implausible, using perplexity.

This is the baseline from Jain et al. (2023) and Alon and Kamfonas (2023).
Gradient-built attack suffixes (HotFlip, GCG, white-box PoisonedRAG) look
like gibberish to a causal language model and have much higher perplexity
than normal text. Fluent poison written by a language model does not: both
PoisonedRAG and TrustRAG show clean and poisoned perplexity overlapping.
Use this as one voter for the gibberish-suffix case, not as the only
poisoning detector.

Pass `perplexity=` to supply the score yourself. The default loads a small
causal model through `transformers` (`gpt2` unless `model_id` says
otherwise) and computes `exp(mean token negative log likelihood)`.

Calibrate `threshold` on your own clean memories. Ordinary English under
GPT-2 is often in the tens to low hundreds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ....exceptions import ConfigurationError, IntegrationError
from .._hf import require_transformers
from ..base import BaseDetector, Detection

Perplexity = Callable[[str], float]
DEFAULT_MODEL = "gpt2"


def _load_perplexity(model_id: str, device: str | None, max_length: int) -> Perplexity:
    transformers = require_transformers()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise IntegrationError("PerplexityDetector needs 'torch': pip install 'mimvo[hf]'") from exc
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    if device:
        model.to(device)
    model.eval()

    def perplexity(text: str) -> float:
        encoded: Any = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        if device:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**encoded, labels=encoded["input_ids"])
        return float(math.exp(float(out.loss)))

    return perplexity


class PerplexityDetector(BaseDetector):
    """Flag text whose language-model perplexity is above `threshold`.

    Short texts (fewer than `min_words`) are skipped, because a handful of
    words is not a stable perplexity. The finding code is `adversarial_text`.
    """

    name = "perplexity"

    def __init__(
        self,
        *,
        threshold: float = 1000.0,
        perplexity: Perplexity | None = None,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 1024,
        min_words: int = 6,
    ) -> None:
        """Store the threshold. The model is loaded on the first scan.

        Args:
            threshold: Flag text at or above this perplexity. Must be
                greater than 1, because perplexity is at least 1. Default
                `1000`. Calibrate it on clean memories.
            perplexity: Function `(text) -> float`. `None` loads `model_id`
                through `transformers` on first use.
            model_id: Hugging Face causal language model. Default `gpt2`.
            device: Device string passed to the model, such as `"cpu"` or
                `"cuda"`. `None` leaves the model where it loaded.
            max_length: Tokens kept before scoring. The rest are truncated.
            min_words: Skip text with fewer than this many whitespace-separated
                words. Values below 1 are raised to 1.

        Raises:
            ConfigurationError: `threshold` is 1 or less.
        """
        if threshold <= 1.0:
            raise ConfigurationError("threshold must be greater than 1 (perplexity is >= 1).")
        self.threshold = threshold
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.min_words = max(1, min_words)
        self._perplexity: Perplexity | None = perplexity

    def perplexity(self, text: str) -> float:
        """Score how surprising `text` is to the language model.

        Args:
            text: Memory content. Truncated to `max_length` tokens by the
                default model.

        Returns:
            Perplexity, a float of at least 1. Higher means the model found
            the text less like normal language.

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
        ppl = self.perplexity(text)
        if ppl < self.threshold:
            return []
        # Score saturates as perplexity runs past the threshold.
        score = min(1.0, 0.5 + 0.5 * (1 - self.threshold / ppl))
        return [
            self.hit(
                code="adversarial_text",
                score=score,
                perplexity=round(ppl, 1),
                threshold=self.threshold,
                model=self.model_id,
            )
        ]


__all__ = ["DEFAULT_MODEL", "PerplexityDetector"]
