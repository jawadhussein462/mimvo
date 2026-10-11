"""A port of high-value Gitleaks rules for stored memory.

These are format regexes, not the full Gitleaks YAML. Evidence names the
rule id. The secret value is never stored on the detection. Pair with
`SecretVerificationDetector` if you want a live check; that one is off
by default.
"""

from __future__ import annotations

import re

from ..base import BaseDetector, Detection

# Rule ids follow Gitleaks naming where a rule exists there.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-token", re.compile(r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b")),
    ("github-pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b")),
    ("github-fine-grained-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("gitlab-pat", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    (
        "slack-webhook",
        re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+"),
    ),
    ("stripe-access-token", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    ("twilio-api-key", re.compile(r"\bSK[0-9a-fA-F]{32}\b")),
    ("sendgrid-api-key", re.compile(r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b")),
    (
        "discord-bot-token",
        re.compile(r"\b[MNO][A-Za-z0-9_-]{23,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}\b"),
    ),
    ("telegram-bot-token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    (
        "heroku-api-key",
        re.compile(r"(?i)heroku.{0,20}\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b"),
    ),
    ("npm-access-token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("pypi-upload-token", re.compile(r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}")),
    ("shopify-token", re.compile(r"\bshpca_[A-Fa-f0-9]{32}\b|\bshpat_[A-Fa-f0-9]{32}\b")),
    ("mailgun-api-key", re.compile(r"\bkey-[0-9a-zA-Z]{32}\b")),
    ("digitalocean-pat", re.compile(r"\bdop_v1_[a-f0-9]{64}\b")),
    ("square-access-token", re.compile(r"\bEAAA[A-Za-z0-9]{60,}\b")),
    ("private-key", re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}")),
]


def gitleaks_kinds(text: str) -> list[str]:
    """List Gitleaks rule ids that matched `text`.

    Args:
        text: Memory content.

    Returns:
        Sorted rule ids. The secret values are not included.
    """
    return sorted({name for name, pattern in _RULES if pattern.search(text)})


class GitleaksDetector(BaseDetector):
    """Match a port of common Gitleaks rules. Offline.

    The finding code is `secret_detected`. Evidence lists `kinds` only.
    Every rule matches a provider's fixed token format, so hits score 0.95
    (JWTs, which any service can issue, 0.85).
    """

    name = "gitleaks"

    def detect_text(self, text: str) -> list[Detection]:
        kinds = gitleaks_kinds(text)
        if not kinds:
            return []
        score = 0.85 if kinds == ["jwt"] else 0.95
        return [self.hit(code="secret_detected", score=score, kinds=kinds)]


__all__ = ["GitleaksDetector", "gitleaks_kinds"]
