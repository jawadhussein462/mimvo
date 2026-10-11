"""Training-free LLM-internal probes for stored injection.

These detectors wrap a `probe` callback `(text) -> score`. They do not
load paper weights. Each one is a named place to plug in:

* Attention Tracker (Hung et al., NAACL Findings 2025) — distraction
  effect in attention heads.
* TaskTracker (Abdelnabi et al., SaTML 2025) — activation-delta probes.
* PIShield (arXiv:2510.14005) — a linear probe on an LLM layer.

Without `probe` they return nothing. Treat the chat-prompt classifiers
as baselines; these are the hooks for memory-native internals.
"""

from __future__ import annotations

from collections.abc import Callable

from ....exceptions import ConfigurationError
from ..base import BaseDetector, Detection

Probe = Callable[[str], float]


class _ProbeDetector(BaseDetector):
    """Shared cutoff logic for an injectable activation or attention probe."""

    code: str = "persistent_instruction"

    def __init__(self, *, probe: Probe | None = None, threshold: float = 0.5) -> None:
        """Store the probe and the cutoff.

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
        return [self.hit(code=self.code, score=score, threshold=self.threshold)]


class AttentionTrackerDetector(_ProbeDetector):
    """Attention-head distraction effect. Needs a `probe` callback."""

    name = "attention_tracker"
    code = "persistent_instruction"


class TaskTrackerDetector(_ProbeDetector):
    """Activation-delta probe. Needs a `probe` callback."""

    name = "task_tracker"
    code = "persistent_instruction"


class PIShieldDetector(_ProbeDetector):
    """Linear probe on an LLM layer. Needs a `probe` callback."""

    name = "pishield"
    code = "persistent_instruction"


__all__ = [
    "AttentionTrackerDetector",
    "PIShieldDetector",
    "TaskTrackerDetector",
]
