"""RevPRAG: activation-based detection of poisoned generations.

From Tan et al., 2024. Optional; needs model weights. This detector is a
named hook for a `probe` callback `(text) -> score`. It does not load
the paper's weights. Without `probe` it returns nothing.
"""

from __future__ import annotations

from collections.abc import Callable

from ....exceptions import ConfigurationError
from ..base import BaseDetector, Detection

Probe = Callable[[str], float]


class RevPRAGDetector(BaseDetector):
    """Flag text whose internal activations look like a poisoned generation.

    The finding code is `adversarial_text`.
    """

    name = "revprag"

    def __init__(self, *, probe: Probe | None = None, threshold: float = 0.5) -> None:
        """Store the activation probe and the cutoff.

        Args:
            probe: Function `(text) -> score` in `[0, 1]`. `None`
                disables the detector.
            threshold: Minimum score to flag. Must be within (0, 1].

        Raises:
            ConfigurationError: `threshold` is outside (0, 1].
        """
        if not 0.0 < threshold <= 1.0:
            raise ConfigurationError("threshold must be within (0, 1].")
        self.probe = probe
        self.threshold = threshold

    def detect_text(self, text: str) -> list[Detection]:
        if self.probe is None:
            return []
        score = float(self.probe(text))
        if score < self.threshold:
            return []
        return [self.hit(code="adversarial_text", score=score, threshold=self.threshold)]


__all__ = ["RevPRAGDetector"]
