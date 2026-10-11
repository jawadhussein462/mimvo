"""Piiranha-v1 (`iiiorg/piiranha-v1-detect-personal-information`).

An mDeBERTa-v3 token classifier trained to tag 17 personal-data types in
six languages. On its test set it caught 98% of those tokens, and it is
especially strong on passwords, emails, phone numbers, and usernames.
Labels look like `I-PASSWORD` and `I-CREDITCARDNUMBER`. The window is 256
tokens, so long text is scanned in chunks. Licence is CC-BY-NC-ND-4.0.
Check that it fits your deployment before you ship it.

By default only credential-like labels map to `secret_detected` (delete).
Pass `labels=PIIRANHA_ALL_LABELS`, or your own map, to also report personal
data as `pii_detected` (review).

Install with `pip install "mimvo[hf]"`.
"""

from __future__ import annotations

from .._hf import HFTokenClassifierDetector

PIIRANHA_V1 = "iiiorg/piiranha-v1-detect-personal-information"

PIIRANHA_SECRET_LABELS: dict[str, str] = {
    "PASSWORD": "secret_detected",
    "CREDITCARDNUMBER": "secret_detected",
    "SOCIALNUM": "secret_detected",
    "ACCOUNTNUM": "secret_detected",
    "IDCARDNUM": "secret_detected",
    "DRIVERLICENSENUM": "secret_detected",
    "TAXNUM": "secret_detected",
}
PIIRANHA_PII_LABELS: dict[str, str] = {
    "EMAIL": "pii_detected",
    "TELEPHONENUM": "pii_detected",
    "USERNAME": "pii_detected",
    "GIVENNAME": "pii_detected",
    "SURNAME": "pii_detected",
    "DATEOFBIRTH": "pii_detected",
    "STREET": "pii_detected",
    "BUILDINGNUM": "pii_detected",
    "CITY": "pii_detected",
    "ZIPCODE": "pii_detected",
}
PIIRANHA_ALL_LABELS: dict[str, str] = {**PIIRANHA_SECRET_LABELS, **PIIRANHA_PII_LABELS}


class PiiranhaDetector(HFTokenClassifierDetector):
    """Piiranha-v1 token classifier.

    Credential labels become `secret_detected`. Personal-data labels become
    `pii_detected` only when you pass `labels=PIIRANHA_ALL_LABELS`.
    `chunk_chars` defaults to 600, about 256 tokens. Other constructor
    arguments come from `HFTokenClassifierDetector`.
    """

    name = "piiranha"
    model_id = PIIRANHA_V1
    default_labels = PIIRANHA_SECRET_LABELS

    def __init__(self, **kwargs: object) -> None:
        """Build a Piiranha detector with a 256-token-sized default chunk.

        Args:
            **kwargs: Forwarded to `HFTokenClassifierDetector`. `chunk_chars`
                defaults to 600 when you omit it.
        """
        kwargs.setdefault("chunk_chars", 600)  # ~256 mDeBERTa tokens
        super().__init__(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "PIIRANHA_ALL_LABELS",
    "PIIRANHA_PII_LABELS",
    "PIIRANHA_SECRET_LABELS",
    "PIIRANHA_V1",
    "PiiranhaDetector",
]
