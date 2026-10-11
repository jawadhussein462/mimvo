"""Known key formats and stated credentials, matched with regexes.

Evidence names the kinds found, such as `"aws_access_key_id"` or
`"credential"`. It never includes the secret value. The same text always
produces the same result, and nothing is downloaded.

Formats with no pattern here are missed. Add `EntropyDetector` or a model
detector for those.
"""

from __future__ import annotations

import re

from ..base import BaseDetector, Detection

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"),
    ),
    ("private_key", re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_api_key", re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    (
        "connection_string",
        re.compile(r"\b[a-z][a-z0-9+.-]{1,20}://[^\s/:@]+:[^\s/@]+@[^\s/]+", re.IGNORECASE),
    ),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{10,}=*")),
]

_CREDENTIAL = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|token)\b"
    r"(\s*[:=]\s*)"
    # A link is where to reset a password, not the password itself.
    r"(?!https?://|www\.)"
    r"([^\s\"']{6,})"
)

# Natural language: "my password is hunter2", "the PIN is 4821".
_STATED_CREDENTIAL = re.compile(
    r"(?i)\b(?:password|passwd|passcode|pwd|pin|secret|api[ _-]?key|access[ _-]?token"
    r"|auth[ _-]?token|token)\s+(?:is|was)\s+[\"']?([^\s\"',;]{4,})"
)
_HAS_DIGIT_OR_SYMBOL = re.compile(r"[\d!#$%&*+/=?@^_~|-]")


def _stated_credential(text: str) -> bool:
    """Whether `text` states a secret value, not just the word "password".

    A match of "my password is hunter2" counts when the value contains a
    digit or a symbol. "my password is correct" does not.

    Args:
        text: Memory content, scanned as written (not deobfuscated).

    Returns:
        `True` when at least one stated value looks like a secret.
    """
    for match in _STATED_CREDENTIAL.finditer(text):
        value = match.group(1).rstrip(".!?)")
        if len(value) >= 4 and _HAS_DIGIT_OR_SYMBOL.search(value):
            return True
    return False


def secret_kinds(text: str) -> list[str]:
    """List the kinds of secrets found in `text`.

    Args:
        text: Memory content.

    Returns:
        Sorted kind names such as `"openai_api_key"` and `"credential"`.
        The secret values are not included. Empty when nothing matched.
    """
    kinds: set[str] = set()
    for kind, pattern in _PATTERNS:
        if pattern.search(text):
            kinds.add(kind)
    if _CREDENTIAL.search(text) or _stated_credential(text):
        kinds.add("credential")
    return sorted(kinds)


def mask_secrets(text: str) -> str:
    """Replace secret-like substrings with bullets so a report can be shared.

    Args:
        text: Memory content.

    Returns:
        A copy of `text`. Matched keys, tokens, and stated secret values
        are replaced with `••••••••`. Surrounding words are kept.
    """
    masked = text
    for _, pattern in _PATTERNS:
        masked = pattern.sub("••••••••", masked)
    masked = _CREDENTIAL.sub(lambda m: f"{m.group(1)}{m.group(2)}••••••••", masked)

    def _stated(match: re.Match[str]) -> str:
        return match.group(0)[: match.start(1) - match.start(0)] + "••••••••"

    return _STATED_CREDENTIAL.sub(_stated, masked)


# How sure a match of each kind is. Provider formats with fixed prefixes and
# private-key blocks are near-certain; a `password = ...` line is likely; a
# sentence that states a value ("the PIN is 4821") is the weakest signal.
KIND_SCORES: dict[str, float] = {
    "private_key": 0.99,
    "anthropic_api_key": 0.97,
    "openai_api_key": 0.95,
    "stripe_key": 0.97,
    "aws_access_key_id": 0.97,
    "google_api_key": 0.95,
    "slack_token": 0.95,
    "github_token": 0.97,
    "github_pat": 0.97,
    "connection_string": 0.9,
    "jwt": 0.85,
    "bearer_token": 0.8,
    "credential": 0.75,
}


class HeuristicSecretsDetector(BaseDetector):
    """Match known key formats, `key = value`, and "my password is ..." lines.

    The finding code is `secret_detected`. Evidence lists `kinds` only. The
    score is that of the strongest kind found (`KIND_SCORES`).
    """

    name = "heuristic"

    def detect_text(self, text: str) -> list[Detection]:
        kinds = secret_kinds(text)
        if not kinds:
            return []
        score = max(KIND_SCORES.get(kind, 0.8) for kind in kinds)
        return [self.hit(code="secret_detected", score=score, kinds=kinds)]


__all__ = ["KIND_SCORES", "HeuristicSecretsDetector", "mask_secrets", "secret_kinds"]
