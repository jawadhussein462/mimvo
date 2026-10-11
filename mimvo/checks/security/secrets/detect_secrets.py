"""Yelp `detect-secrets` plugins as a detector.

`detect-secrets` ships about 25 provider detectors (AWS, Azure Storage,
GitHub, GitLab, Slack, Stripe, Twilio, SendGrid, npm, PyPI, JWT, private
keys, basic-auth URLs, and others) plus keyword and high-entropy heuristics.
This detector runs `scan_line` on each line and reports the plugin types
that fired, such as `"AWS Access Key"`. Secret values stay inside the
plugin objects and are not copied into evidence.

Install with `pip install "mimvo[detect-secrets]"`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ....exceptions import IntegrationError
from ..base import BaseDetector, Detection

# ``(text) -> plugin type names that fired``
Scan = Callable[[str], Iterable[str]]


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")


class DetectSecretsDetector(BaseDetector):
    """Run Yelp detect-secrets plugins line by line.

    The finding code is `secret_detected`. `kinds` are slugified plugin
    names, such as `"aws_access_key"`. High-entropy plugins apply their own
    entropy limit here, so ordinary words are not reported.
    """

    name = "detect_secrets"

    def __init__(
        self,
        *,
        scan: Scan | None = None,
        exclude_types: Iterable[str] = (),
    ) -> None:
        """Choose which plugin results to keep.

        Args:
            scan: Function `(text) -> plugin type names`. `None` loads
                `detect-secrets` on first use.
            exclude_types: Plugin names to drop, such as `"Hex High Entropy String"`.
                Names are slugified before comparison, so spaces and case
                do not matter.

        The package is not imported until the first scan, unless you pass `scan`.
        """
        self.exclude_types = {_slug(t) for t in exclude_types}
        self._scan: Scan | None = scan

    def _load(self) -> Scan:
        try:
            from detect_secrets.core.plugins.initialize import from_secret_type
            from detect_secrets.core.scan import scan_line
            from detect_secrets.settings import default_settings
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise IntegrationError(
                "DetectSecretsDetector needs 'detect-secrets': pip install 'mimvo[detect-secrets]'"
            ) from exc

        def passes_entropy_limit(secret: object) -> bool:
            # ``scan_line`` is detect-secrets' adhoc path: its high-entropy plugins
            # return every candidate string and leave the entropy limit to the
            # caller (the CLI prints it). Apply the plugin's own limit here so
            # ordinary words are not reported.
            plugin = from_secret_type(str(getattr(secret, "type", "")))
            limit = getattr(plugin, "entropy_limit", None)
            if limit is None:
                return True
            value = getattr(secret, "secret_value", None)
            return bool(value) and plugin.calculate_shannon_entropy(value) > limit

        def scan(text: str) -> Iterable[str]:
            types: set[str] = set()
            with default_settings():
                for line in text.splitlines():
                    for secret in scan_line(line):
                        if passes_entropy_limit(secret):
                            types.add(str(secret.type))
            return types

        return scan

    def scan(self, text: str) -> set[str]:
        """Return the detect-secrets plugin names that fired.

        Args:
            text: Memory content, scanned line by line.

        Returns:
            Plugin type names, such as `{"AWS Access Key"}`. Empty when
            nothing fired. Values are not included.

        Raises:
            IntegrationError: `detect-secrets` is not installed and no
                `scan` callable was passed.
        """
        if self._scan is None:
            self._scan = self._load()
        return {str(t) for t in self._scan(text)}

    def detect_text(self, text: str) -> list[Detection]:
        kinds = sorted(
            slug for slug in (_slug(t) for t in self.scan(text)) if slug not in self.exclude_types
        )
        if not kinds:
            return []
        return [self.hit(code="secret_detected", kinds=kinds, provider="detect_secrets")]


__all__ = ["DetectSecretsDetector"]
