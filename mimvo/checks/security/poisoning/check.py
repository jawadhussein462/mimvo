"""The poisoning check: false or attacker-controlled facts planted in memory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ....rules import Rule, rules_in
from ..base import Detector, SecurityCheck
from .heuristic import HeuristicPoisoningDetector
from .hubness import HubnessDetector
from .trustrag import TrustRAGDetector


class PoisoningCheck(SecurityCheck):
    """Flag content that looks like an attempt to plant a false fact.

    Defaults are offline: the phrase list, TrustRAG (needs the other
    records in the batch), and hubness (needs stored embeddings). Model
    methods — RAGuard, NLI, probe-query, embedding consistency — are
    optional detectors you add yourself.

    Example:
        Run the phrase list and the cluster detector together::

            PoisoningCheck(detectors=[
                HeuristicPoisoningDetector(),
                TrustRAGDetector(),
            ])
    """

    name = "poisoning"
    default_code: ClassVar[str] = "memory_poisoning"
    rules: ClassVar[Mapping[str, Rule]] = rules_in("poisoning")

    @classmethod
    def default_detectors(cls) -> list[Detector]:
        """Return the offline corpus detectors.

        Returns:
            Phrase list, TrustRAG, and hubness. Hubness no-ops when the
            batch has no embeddings. TrustRAG uses stored vectors when
            present and lexical similarity on small batches otherwise.
        """
        return [HeuristicPoisoningDetector(), TrustRAGDetector(), HubnessDetector()]


__all__ = ["PoisoningCheck"]
