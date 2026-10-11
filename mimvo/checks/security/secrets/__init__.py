"""Secrets: credentials, keys, tokens, and, if you opt in, personal data.

`SecretsCheck` runs one or more detectors:

* `HeuristicSecretsDetector` — known key formats and stated credentials.
  Default. Needs nothing extra.
* `EntropyDetector` — high-entropy strings, the detect-secrets approach.
  Needs nothing extra.
* `DetectSecretsDetector` — Yelp detect-secrets plugins. Needs
  `[detect-secrets]`.
* `PiiranhaDetector` — Piiranha-v1 personal-information tags. Needs `[hf]`.
* `StarPIIDetector` — StarPII, for secrets inside code. Needs `[hf]`.
* `GLiNER2PIIDetector` — GLiNER2 privacy filter. Needs `[gliner2]`.
* `GLiNERPIIDetector` — GLiNER multi-PII. Needs `[gliner]`.
* `PresidioDetector` — Microsoft Presidio. Needs `[presidio]`.

Personal-data models default to credential-like labels only, which emit
`secret_detected` (delete). Each module also exports an `*_ALL_LABELS` map.
Pass that as `labels=` to also report personal data as `pii_detected` (review).
"""

from __future__ import annotations

from .check import SecretsCheck
from .detect_secrets import DetectSecretsDetector
from .entropy import EntropyDetector, shannon_entropy
from .gitleaks import GitleaksDetector, gitleaks_kinds
from .gliner2_pii import (
    GLINER2_ALL_LABELS,
    GLINER2_PII_LABELS,
    GLINER2_SECRET_LABELS,
    GLiNER2PIIDetector,
)
from .gliner_pii import (
    GLINER_ALL_LABELS,
    GLINER_PII_LABELS,
    GLINER_SECRET_LABELS,
    GLiNERPIIDetector,
)
from .heuristic import HeuristicSecretsDetector, secret_kinds
from .piiranha import (
    PIIRANHA_ALL_LABELS,
    PIIRANHA_PII_LABELS,
    PIIRANHA_SECRET_LABELS,
    PiiranhaDetector,
)
from .presidio import (
    PRESIDIO_ALL_LABELS,
    PRESIDIO_PII_LABELS,
    PRESIDIO_SECRET_LABELS,
    PresidioDetector,
)
from .starpii import STARPII_ALL_LABELS, STARPII_PII_LABELS, STARPII_SECRET_LABELS, StarPIIDetector
from .verify import SecretVerificationDetector

__all__ = [
    "GLINER2_ALL_LABELS",
    "GLINER2_PII_LABELS",
    "GLINER2_SECRET_LABELS",
    "GLINER_ALL_LABELS",
    "GLINER_PII_LABELS",
    "GLINER_SECRET_LABELS",
    "PIIRANHA_ALL_LABELS",
    "PIIRANHA_PII_LABELS",
    "PIIRANHA_SECRET_LABELS",
    "PRESIDIO_ALL_LABELS",
    "PRESIDIO_PII_LABELS",
    "PRESIDIO_SECRET_LABELS",
    "STARPII_ALL_LABELS",
    "STARPII_PII_LABELS",
    "STARPII_SECRET_LABELS",
    "DetectSecretsDetector",
    "EntropyDetector",
    "GLiNER2PIIDetector",
    "GLiNERPIIDetector",
    "GitleaksDetector",
    "HeuristicSecretsDetector",
    "PiiranhaDetector",
    "PresidioDetector",
    "SecretVerificationDetector",
    "SecretsCheck",
    "StarPIIDetector",
    "gitleaks_kinds",
    "secret_kinds",
    "shannon_entropy",
]
