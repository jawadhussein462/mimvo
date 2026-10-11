"""The injection check: instructions aimed at the agent, saved as memory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ....rules import Rule, rules_in
from ..base import Detector, SecurityCheck
from .heuristic import HeuristicInjectionDetector


class InjectionCheck(SecurityCheck):
    """Flag instructions that tell the agent to ignore its rules.

    Phrases like "ignore previous instructions" should not be stored as
    memory. The default detector is the offline phrase list. Chat-prompt
    classifiers (Prompt Guard, DeBERTa, Sentinel, Lakera, Azure) are
    optional baselines. Known-answer / DataSentinel and the embedding
    classifier are the memory-native methods.

    Example:
        Require both the phrase list and Prompt Guard to agree::

            InjectionCheck(
                detectors=[HeuristicInjectionDetector(), PromptGuardDetector()],
                min_detectors=2,
            )
    """

    name = "injection"
    default_code: ClassVar[str] = "persistent_instruction"
    rules: ClassVar[Mapping[str, Rule]] = rules_in("injection")

    @classmethod
    def default_detectors(cls) -> list[Detector]:
        """Return the offline phrase detector.

        Returns:
            A one-item list: `HeuristicInjectionDetector`.
        """
        return [HeuristicInjectionDetector()]


__all__ = ["InjectionCheck"]
