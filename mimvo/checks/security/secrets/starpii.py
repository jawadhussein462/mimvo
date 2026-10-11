"""StarPII (`bigcode/starpii`): personal data and secrets inside code.

The BigCode model used to scrub The Stack before StarCoder training. It
tags `NAME`, `EMAIL`, `KEY`, `PASSWORD`, `IP_ADDRESS`, and `USERNAME` in
source code across 31 or more languages. Use it when memories contain
snippets, configs, or logs rather than prose.

The authors recommend ignoring keys shorter than nine characters and
usernames, which are noisy. The default label map follows that: only `KEY`
and `PASSWORD` are on, as `secret_detected`. Pass `labels=STARPII_ALL_LABELS`
to add the personal-data labels as `pii_detected`. The model is gated on
the Hugging Face Hub.

Install with `pip install "mimvo[hf]"`.
"""

from __future__ import annotations

from .._hf import HFTokenClassifierDetector

STARPII = "bigcode/starpii"

STARPII_SECRET_LABELS: dict[str, str] = {
    "KEY": "secret_detected",
    "PASSWORD": "secret_detected",
}
STARPII_PII_LABELS: dict[str, str] = {
    "EMAIL": "pii_detected",
    "NAME": "pii_detected",
    "IP_ADDRESS": "pii_detected",
    "USERNAME": "pii_detected",
}
STARPII_ALL_LABELS: dict[str, str] = {**STARPII_SECRET_LABELS, **STARPII_PII_LABELS}


class StarPIIDetector(HFTokenClassifierDetector):
    """StarPII token classifier for keys and passwords in code-like content.

    Constructor arguments come from `HFTokenClassifierDetector`. The default
    `labels` are `STARPII_SECRET_LABELS`.
    """

    name = "starpii"
    model_id = STARPII
    default_labels = STARPII_SECRET_LABELS


__all__ = [
    "STARPII",
    "STARPII_ALL_LABELS",
    "STARPII_PII_LABELS",
    "STARPII_SECRET_LABELS",
    "StarPIIDetector",
]
