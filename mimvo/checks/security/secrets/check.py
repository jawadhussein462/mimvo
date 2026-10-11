"""The secrets check: credentials, keys, tokens, and, if you opt in, personal data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ....rules import Rule, rules_in
from ..base import Detector, SecurityCheck
from .gitleaks import GitleaksDetector
from .heuristic import HeuristicSecretsDetector


class SecretsCheck(SecurityCheck):
    """Flag secret-bearing content, and optionally personal data for review.

    Findings never contain the raw value. They list the kinds detected.
    Defaults are the offline pattern list plus a Gitleaks-rule port.
    Live verification (`SecretVerificationDetector`) is off unless you
    pass a `verify` callback.

    Personal-data detectors map only credential-like labels to
    `secret_detected` by default. Pass them an `*_ALL_LABELS` mapping to also
    emit `pii_detected` (high, review). The actions differ: delete and
    rotate a secret; redact or apply retention to PII.

    Example:
        Stack the pattern list, entropy, and Piiranha::

            SecretsCheck(detectors=[
                HeuristicSecretsDetector(),
                EntropyDetector(),
                PiiranhaDetector(),
            ])
    """

    name = "secrets"
    default_code: ClassVar[str] = "secret_detected"
    rules: ClassVar[Mapping[str, Rule]] = rules_in("secrets")

    @classmethod
    def default_detectors(cls) -> list[Detector]:
        """Return the offline format detectors.

        Returns:
            Heuristic patterns and the Gitleaks-rule port.
        """
        return [HeuristicSecretsDetector(), GitleaksDetector()]


__all__ = ["SecretsCheck"]
