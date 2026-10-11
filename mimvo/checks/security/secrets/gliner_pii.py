"""GLiNER PII (`urchade/gliner_multi_pii-v1`, Apache-2.0).

The original GLiNER zero-shot NER model, fine-tuned for personal data. You
choose the labels at inference time, and it recognises 50 or more types
out of the box, including `credit card number`, `social security number`,
`iban`, `password`, and `passport number`. Microsoft Presidio ships a
`GLiNERRecognizer` built on this model.

Credential-like labels map to `secret_detected` by default.
`GLINER_ALL_LABELS` adds contact and name labels as `pii_detected`.

Install with `pip install "mimvo[gliner]"` (the `gliner` package).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ....exceptions import ConfigurationError, IntegrationError
from ..base import BaseDetector, Detection

GLINER_PII = "urchade/gliner_multi_pii-v1"

GLINER_SECRET_LABELS: dict[str, str] = {
    "password": "secret_detected",
    "api key": "secret_detected",
    "credit card number": "secret_detected",
    "cvv": "secret_detected",
    "social security number": "secret_detected",
    "bank account number": "secret_detected",
    "iban": "secret_detected",
    "passport number": "secret_detected",
    "driver's license number": "secret_detected",
    "national id number": "secret_detected",
    "tax identification number": "secret_detected",
}
GLINER_PII_LABELS: dict[str, str] = {
    "person": "pii_detected",
    "email": "pii_detected",
    "phone number": "pii_detected",
    "address": "pii_detected",
    "date of birth": "pii_detected",
    "username": "pii_detected",
    "ip address": "pii_detected",
    "medical condition": "pii_detected",
}
GLINER_ALL_LABELS: dict[str, str] = {**GLINER_SECRET_LABELS, **GLINER_PII_LABELS}

# ``(text, labels) -> [{"label": ..., "score": ..., "start": ..., "end": ...}, ...]``
Predict = Callable[[str, list[str]], list[Mapping[str, Any]]]


class GLiNERPIIDetector(BaseDetector):
    """GLiNER zero-shot extraction, limited to credential labels by default.

    Pass `labels=GLINER_ALL_LABELS` to also emit `pii_detected`. Evidence
    lists label names, not the matched text.
    """

    name = "gliner_pii"

    def __init__(
        self,
        *,
        model_id: str = GLINER_PII,
        labels: Mapping[str, str] | None = None,
        threshold: float = 0.5,
        predict: Predict | None = None,
    ) -> None:
        """Store the model id and label map. Weights load on first use.

        Args:
            model_id: Hugging Face model id. Default
                `urchade/gliner_multi_pii-v1`.
            labels: Map of GLiNER label to finding code. `None` uses
                `GLINER_SECRET_LABELS`. Keys must match the label strings
                you want the model to look for, including spaces.
            threshold: Minimum entity score, from 0 to 1. Default `0.5`.
            predict: Function `(text, labels) -> [{"label", "score", ...}]`.
                Pass one in tests. `None` loads the `gliner` package on
                first use.

        Raises:
            ConfigurationError: `threshold` is outside 0 to 1, or `labels`
                is empty.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("threshold must be within [0, 1].")
        self.model_id = model_id
        self.labels: dict[str, str] = dict(labels if labels is not None else GLINER_SECRET_LABELS)
        if not self.labels:
            raise ConfigurationError("GLiNERPIIDetector needs a non-empty labels mapping.")
        self.threshold = threshold
        self._predict: Predict | None = predict

    def _load(self) -> Predict:
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise IntegrationError(
                "GLiNERPIIDetector needs the 'gliner' package: pip install 'mimvo[gliner]'"
            ) from exc
        model = GLiNER.from_pretrained(self.model_id)
        return lambda text, labels: list(
            model.predict_entities(text, labels, threshold=self.threshold)
        )

    def predict(self, text: str) -> list[Mapping[str, Any]]:
        """Ask GLiNER for entities in `text`.

        Args:
            text: Memory content. The model is asked only for the keys in
                `labels`.

        Returns:
            Entity dicts with at least `label` and `score`.

        Raises:
            IntegrationError: The `gliner` package is not installed and
                no `predict` callable was passed.
        """
        if self._predict is None:
            self._predict = self._load()
        return self._predict(text, sorted(self.labels))

    def detect_text(self, text: str) -> list[Detection]:
        found: dict[str, tuple[set[str], float]] = {}
        for entity in self.predict(text):
            label = str(entity.get("label", ""))
            code = self.labels.get(label)
            score = float(entity.get("score", 1.0))
            if code is None or score < self.threshold:
                continue
            kinds, best = found.get(code, (set(), 0.0))
            kinds.add(label)
            found[code] = (kinds, max(best, score))
        return [
            self.hit(code=code, score=best, kinds=sorted(kinds), model=self.model_id)
            for code, (kinds, best) in sorted(found.items())
        ]


__all__ = [
    "GLINER_ALL_LABELS",
    "GLINER_PII",
    "GLINER_PII_LABELS",
    "GLINER_SECRET_LABELS",
    "GLiNERPIIDetector",
]
