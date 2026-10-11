"""Microsoft Presidio analyzer (`presidio-analyzer`).

Presidio mixes regex recognisers, checksums (Luhn for cards, IBAN mod-97),
context words, and a spaCy or transformers NER model. It returns
`RecognizerResult` spans with a type and a score. `CREDIT_CARD`,
`IBAN_CODE`, `US_SSN`, `US_BANK_NUMBER`, and `CRYPTO` map to
`secret_detected` by default. `PRESIDIO_ALL_LABELS` also maps `PERSON`,
`EMAIL_ADDRESS`, `PHONE_NUMBER`, and similar types to `pii_detected`.

Pass your own `AnalyzerEngine` as `analyzer` for custom recognisers or
another language. Otherwise a default English engine is built on first use,
which needs the `en_core_web_lg` spaCy model.

    pip install "mimvo[presidio]"
    python -m spacy download en_core_web_lg
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ....exceptions import ConfigurationError, IntegrationError
from ..base import BaseDetector, Detection

PRESIDIO_SECRET_LABELS: dict[str, str] = {
    "CREDIT_CARD": "secret_detected",
    "IBAN_CODE": "secret_detected",
    "US_SSN": "secret_detected",
    "US_BANK_NUMBER": "secret_detected",
    "US_PASSPORT": "secret_detected",
    "US_DRIVER_LICENSE": "secret_detected",
    "US_ITIN": "secret_detected",
    "UK_NHS": "secret_detected",
    "CRYPTO": "secret_detected",
}
PRESIDIO_PII_LABELS: dict[str, str] = {
    "PERSON": "pii_detected",
    "EMAIL_ADDRESS": "pii_detected",
    "PHONE_NUMBER": "pii_detected",
    "LOCATION": "pii_detected",
    "DATE_TIME": "pii_detected",
    "IP_ADDRESS": "pii_detected",
    "NRP": "pii_detected",
    "MEDICAL_LICENSE": "pii_detected",
    "URL": "pii_detected",
}
PRESIDIO_ALL_LABELS: dict[str, str] = {**PRESIDIO_SECRET_LABELS, **PRESIDIO_PII_LABELS}

# ``(text, entities) -> [obj with .entity_type and .score]``
Analyze = Callable[[str, list[str]], list[Any]]


class PresidioDetector(BaseDetector):
    """Read Presidio `AnalyzerEngine` results.

    By default only credential entity types are kept, as `secret_detected`.
    Pass `labels=PRESIDIO_ALL_LABELS` to also emit `pii_detected`.
    """

    name = "presidio"

    def __init__(
        self,
        *,
        analyzer: Any | None = None,
        labels: Mapping[str, str] | None = None,
        threshold: float = 0.5,
        language: str = "en",
        analyze: Analyze | None = None,
    ) -> None:
        """Store the engine settings. Presidio is imported on first use.

        Args:
            analyzer: A Presidio `AnalyzerEngine`. `None` builds a default
                English engine on the first scan.
            labels: Map of Presidio entity type to finding code. `None` uses
                `PRESIDIO_SECRET_LABELS`. Keys are compared in uppercase.
            threshold: Minimum Presidio score, from 0 to 1. Default `0.5`.
            language: Language code passed to `analyze`. Default `"en"`.
            analyze: Function `(text, entities) -> results with .entity_type
                and .score`. Pass one in tests. `None` uses `analyzer`.

        Raises:
            ConfigurationError: `threshold` is outside 0 to 1, or `labels`
                is empty.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("threshold must be within [0, 1].")
        self.labels: dict[str, str] = {
            k.upper(): v
            for k, v in (labels if labels is not None else PRESIDIO_SECRET_LABELS).items()
        }
        if not self.labels:
            raise ConfigurationError("PresidioDetector needs a non-empty labels mapping.")
        self.threshold = threshold
        self.language = language
        self._analyzer = analyzer
        self._analyze: Analyze | None = analyze

    def _load(self) -> Analyze:
        analyzer = self._analyzer
        if analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise IntegrationError(
                    "PresidioDetector needs 'presidio-analyzer': pip install 'mimvo[presidio]'"
                ) from exc
            analyzer = AnalyzerEngine()
            self._analyzer = analyzer
        return lambda text, entities: list(
            analyzer.analyze(
                text=text,
                language=self.language,
                entities=entities,
                score_threshold=self.threshold,
            )
        )

    def analyze(self, text: str) -> list[Any]:
        """Run Presidio on `text` for the labels this detector cares about.

        Args:
            text: Memory content.

        Returns:
            Presidio results. Each one has `entity_type` and `score`.
            The matched substring is not copied into Mimvo evidence.

        Raises:
            IntegrationError: Presidio is not installed and neither
                `analyzer` nor `analyze` was passed.
        """
        if self._analyze is None:
            self._analyze = self._load()
        return self._analyze(text, sorted(self.labels))

    def detect_text(self, text: str) -> list[Detection]:
        found: dict[str, tuple[set[str], float]] = {}
        for result in self.analyze(text):
            label = str(getattr(result, "entity_type", "")).upper()
            score = float(getattr(result, "score", 1.0))
            code = self.labels.get(label)
            if code is None or score < self.threshold:
                continue
            kinds, best = found.get(code, (set(), 0.0))
            kinds.add(label.lower())
            found[code] = (kinds, max(best, score))
        return [
            self.hit(code=code, score=best, kinds=sorted(kinds), provider="presidio")
            for code, (kinds, best) in sorted(found.items())
        ]


__all__ = [
    "PRESIDIO_ALL_LABELS",
    "PRESIDIO_PII_LABELS",
    "PRESIDIO_SECRET_LABELS",
    "PresidioDetector",
]
