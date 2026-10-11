"""Optional live verification of a candidate secret, off by default.

TruffleHog-style: after a format match, ask an injectable `verify`
callback whether the credential is live. This detector never ships a
network client. Without `verify` it returns nothing.

The secret value is passed only to `verify`. It is not written on the
detection.
"""

from __future__ import annotations

from collections.abc import Callable

from ....exceptions import BackendError
from ..base import BaseDetector, Detection
from .gitleaks import gitleaks_kinds
from .heuristic import secret_kinds

Verify = Callable[[str, str], bool]


class SecretVerificationDetector(BaseDetector):
    """Confirm a format match with a live check. Off unless `verify` is set.

    The finding code is `secret_detected`. Evidence lists `kinds` and
    `verified=True`. Unverified matches are left to the format detectors.
    """

    name = "secret_verify"

    def __init__(self, *, verify: Verify | None = None) -> None:
        """Store the verification callback.

        Args:
            verify: Function `(kind, text) -> True` when a live secret is
                confirmed. The callback may extract the value itself.
                `None` disables the detector.
        """
        self.verify = verify

    def detect_text(self, text: str) -> list[Detection]:
        if self.verify is None:
            return []
        kinds = sorted(set(secret_kinds(text)) | set(gitleaks_kinds(text)))
        if not kinds:
            return []
        confirmed: list[str] = []
        for kind in kinds:
            try:
                if self.verify(kind, text):
                    confirmed.append(kind)
            except Exception as exc:
                raise BackendError(
                    f"secret verify callback failed for {kind!r}: {type(exc).__name__}"
                ) from exc
        if not confirmed:
            return []
        return [self.hit(code="secret_detected", kinds=confirmed, verified=True)]


__all__ = ["SecretVerificationDetector"]
