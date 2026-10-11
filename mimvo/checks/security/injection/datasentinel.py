"""DataSentinel: a two-step known-answer game for stored injection.

From Liu et al., IEEE S&P 2025. DataSentinel is a game-theoretically
fine-tuned form of known-answer detection. This detector runs the game
with an injectable `complete` callback; it does not load the paper's
weights. That makes it usable in a memory scanner: the record is the
untrusted player, the canary is the trusted instruction.

Without `complete` this detector returns nothing.
"""

from __future__ import annotations

from collections.abc import Callable

from ....exceptions import ConfigurationError
from ..base import BaseDetector, Detection
from .known_answer import DEFAULT_CANARY, DEFAULT_EXPECTED

Complete = Callable[[str], str]


class DataSentinelDetector(BaseDetector):
    """Flag a memory that wins a two-step canary game against the model.

    The finding code is `known_answer`. The record is offered first as
    context, then as a competing instruction. If either step drops the
    canary answer, the record is treated as injection.
    """

    name = "datasentinel"

    def __init__(
        self,
        *,
        complete: Complete | None = None,
        canary: str = DEFAULT_CANARY,
        expected: str = DEFAULT_EXPECTED,
    ) -> None:
        """Store the completion callback and the canary.

        Args:
            complete: Function `(prompt) -> model text`. `None` disables
                the detector.
            canary: Trusted instruction with a known answer.
            expected: Substring that must appear when the record is benign.

        Raises:
            ConfigurationError: `canary` or `expected` is blank.
        """
        if not canary.strip() or not expected.strip():
            raise ConfigurationError("canary and expected must be non-empty.")
        self.complete = complete
        self.canary = canary
        self.expected = expected

    def detect_text(self, text: str) -> list[Detection]:
        if self.complete is None:
            return []
        prompts = (
            f"{self.canary}\n\nThe next paragraph is untrusted memory.\n{text}\n"
            "Reply with the canary word only.",
            f"Memory to store:\n{text}\n\n{self.canary}",
        )
        failed = 0
        for prompt in prompts:
            if self.expected.lower() not in self.complete(prompt).lower():
                failed += 1
        if failed == 0:
            return []
        return [
            self.hit(
                code="known_answer",
                score=failed / len(prompts),
                rounds_failed=failed,
                rounds=len(prompts),
            )
        ]


__all__ = ["DataSentinelDetector"]
