"""Flag high-entropy strings. Same idea as Yelp detect-secrets, standard library only.

Random tokens (API keys, session ids, signing secrets) have much higher
Shannon entropy than words. A run of base64-alphabet characters at least
`min_length` long is reported when its entropy is above `base64_limit`
(default 4.5 bits per character). A hex run is reported above `hex_limit`
(default 3.0). This catches key formats that have no regex, and it can
also flag content hashes. Raise the limits or `min_length` if that happens
on your data.

Evidence reports the token kind and its entropy, never the token.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ....exceptions import ConfigurationError
from ..base import BaseDetector, Detection

_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=_-]{16,}")
_HEX_RUN = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_ALPHA_ONLY = re.compile(r"^[A-Za-z]+$")


def shannon_entropy(value: str) -> float:
    """Measure how random `value` looks, in bits per character.

    Args:
        value: The string to score. An empty string is allowed.

    Returns:
        Shannon entropy. `0.0` for an empty string or a string of one
        repeated character. A fully random byte string approaches 8.
    """
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


class EntropyDetector(BaseDetector):
    """Flag long hex or base64 runs that look machine-generated.

    The finding code is `secret_detected`. Kinds look like
    `"high_entropy_hex"` and `"high_entropy_base64"`.
    """

    name = "entropy"

    def __init__(
        self,
        *,
        base64_limit: float = 4.5,
        hex_limit: float = 3.0,
        min_length: int = 20,
    ) -> None:
        """Store the entropy cutoffs.

        Args:
            base64_limit: Bits per character a base64-alphabet run must
                exceed. Default `4.5`. Must be from 0 to 8.
            hex_limit: Bits per character a hex run must exceed. Default
                `3.0`. Must be from 0 to 8.
            min_length: Shortest run that is considered. Default `20`.
                Must be at least 8.

        Raises:
            ConfigurationError: A limit is outside 0 to 8, or `min_length`
                is below 8.
        """
        if not 0.0 <= base64_limit <= 8.0 or not 0.0 <= hex_limit <= 8.0:
            raise ConfigurationError("entropy limits must be within [0, 8].")
        if min_length < 8:
            raise ConfigurationError("min_length must be >= 8.")
        self.base64_limit = base64_limit
        self.hex_limit = hex_limit
        self.min_length = min_length

    def high_entropy_kinds(self, text: str) -> dict[str, float]:
        """Find the strongest hex run and the strongest base64 run.

        Args:
            text: Memory content.

        Returns:
            A dict with whichever of `"hex"` and `"base64"` exceeded its
            limit. The value is that run's entropy in bits per character.
            Empty when nothing qualified. A run that is valid hex is judged
            only by the hex rule, so it is not also reported as base64.
        """
        found: dict[str, float] = {}
        for match in _HEX_RUN.finditer(text):
            token = match.group(0)
            if len(token) >= self.min_length:
                entropy = shannon_entropy(token)
                if entropy > self.hex_limit:
                    found["hex"] = max(found.get("hex", 0.0), entropy)
        for match in _BASE64_RUN.finditer(text):
            token = match.group(0).rstrip("=")
            if len(token) < self.min_length or _ALPHA_ONLY.match(token):
                continue
            if _HEX_RUN.fullmatch(token):
                continue  # already judged by the hex rule
            entropy = shannon_entropy(token)
            if entropy > self.base64_limit:
                found["base64"] = max(found.get("base64", 0.0), entropy)
        return found

    def mask(self, text: str) -> str:
        """Replace high-entropy runs with bullets.

        Args:
            text: Memory content.

        Returns:
            A copy of `text`. Runs that would be flagged are replaced with
            `••••••••`. Ordinary words are left as they are.
        """

        def _hex(match: re.Match[str]) -> str:
            token = match.group(0)
            if len(token) >= self.min_length and shannon_entropy(token) > self.hex_limit:
                return "••••••••"
            return token

        def _b64(match: re.Match[str]) -> str:
            token = match.group(0)
            body = token.rstrip("=")
            if len(body) < self.min_length or _ALPHA_ONLY.match(body) or _HEX_RUN.fullmatch(body):
                return token
            if shannon_entropy(body) > self.base64_limit:
                return "••••••••"
            return token

        return _BASE64_RUN.sub(_b64, _HEX_RUN.sub(_hex, text))

    def detect_text(self, text: str) -> list[Detection]:
        found = self.high_entropy_kinds(text)
        if not found:
            return []
        best = max(found.values())
        return [
            self.hit(
                code="secret_detected",
                score=min(1.0, best / 6.0),
                kinds=[f"high_entropy_{kind}" for kind in sorted(found)],
                entropy=round(best, 2),
            )
        ]


__all__ = ["EntropyDetector", "shannon_entropy"]
