"""Lakera Guard, also called Check Point AI Guardrails. Hosted.

`POST https://api.lakera.ai/v2/guard` screens chat-style messages and
returns `flagged`. With `breakdown=true` it also returns one entry per
detector (`detector_type`, `detected`). Lakera Guard leads the PINT
prompt-injection benchmark at 95.2% and can screen tool output, so a
memory is sent as a `tool` message by default. Pass `role="user"` to
screen it as a direct prompt.

The API key comes from `api_key` or the `LAKERA_GUARD_API_KEY` environment
variable. Only the candidate text is sent.

If the Lakera project is in detect mode, `flagged` is forced to false.
This detector also reads per-detector `detected` flags in the breakdown
when `use_breakdown` is on (the default), and only for prompt-attack types.
"""

from __future__ import annotations

import os
from typing import Any

from ....exceptions import BackendError, ConfigurationError
from .._http import Transport, urllib_transport
from ..base import BaseDetector, Detection

LAKERA_GUARD_URL = "https://api.lakera.ai/v2/guard"
# Detector types that indicate an injection / jailbreak rather than e.g. PII.
_ATTACK_TYPES = ("prompt_attack", "prompt_injection", "jailbreak")


class LakeraGuardDetector(BaseDetector):
    """Ask Lakera Guard `/v2/guard` whether the text is a prompt attack.

    A hit means `flagged` is true, or a prompt-attack detector in the
    breakdown set `detected`. The score on that hit is `1.0` because the
    API returns a yes or no, not a probability.
    """

    name = "lakera_guard"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        role: str = "tool",
        url: str = LAKERA_GUARD_URL,
        use_breakdown: bool = True,
        transport: Transport | None = None,
        timeout: float = 10.0,
        concurrency: int = 8,
    ) -> None:
        """Store the API key and how the memory will be sent.

        Args:
            api_key: Bearer token. `None` reads `LAKERA_GUARD_API_KEY`.
            project_id: Optional Lakera project id sent in the body.
            role: Chat role for the memory text. `"tool"` (the default)
                treats it as indirect content. `"user"` treats it as a prompt.
            url: Endpoint. Defaults to `https://api.lakera.ai/v2/guard`.
            use_breakdown: When `True` (the default), request the per-detector
                breakdown and treat prompt-attack `detected` flags as hits.
            transport: Function `(url, headers, payload) -> parsed JSON`.
                `None` uses `urllib`.
            timeout: Seconds `urllib` waits. Ignored when you pass `transport`.
            concurrency: Requests in flight at once during a batched scan.
                `1` sends them one after another. A custom `transport` must
                be thread-safe when this is above 1.

        Raises:
            ConfigurationError: No API key was passed and the environment
                variable is unset, or `concurrency` is below 1.
        """
        api_key = api_key or os.environ.get("LAKERA_GUARD_API_KEY")
        if not api_key:
            raise ConfigurationError("LakeraGuardDetector needs api_key (or LAKERA_GUARD_API_KEY).")
        self.url = url
        self.project_id = project_id
        self.role = role
        self.use_breakdown = use_breakdown
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._transport = transport or urllib_transport(timeout)
        if concurrency < 1:
            raise ConfigurationError("concurrency must be >= 1.")
        self.concurrency = concurrency

    def guard(self, text: str) -> dict[str, Any]:
        """Call Lakera and return the parsed JSON.

        Args:
            text: Memory content. Sent as the only message.

        Returns:
            The response object. Typical keys are `flagged` and, when
            breakdown was requested, `breakdown`.

        Raises:
            ConfigurationError: The response is not a JSON object.
            BackendError: The HTTP call fails. Raised by the transport.
        """
        payload: dict[str, Any] = {"messages": [{"role": self.role, "content": text}]}
        if self.project_id:
            payload["project_id"] = self.project_id
        if self.use_breakdown:
            payload["breakdown"] = True
        response = self._transport(self.url, self._headers, payload)
        if not isinstance(response, dict):
            raise BackendError("Lakera Guard returned a non-object response.")
        return response

    def detect_text(self, text: str) -> list[Detection]:
        response = self.guard(text)
        detected_types: set[str] = set()
        for item in response.get("breakdown") or []:
            dtype = str(item.get("detector_type", ""))
            if item.get("detected") and dtype.startswith(_ATTACK_TYPES):
                detected_types.add(dtype)
        flagged = bool(response.get("flagged")) or bool(detected_types)
        if not flagged:
            return []
        if detected_types:
            return [
                self.hit(score=1.0, provider="lakera_guard", detector_types=sorted(detected_types))
            ]
        return [self.hit(score=1.0, provider="lakera_guard")]


__all__ = ["LAKERA_GUARD_URL", "LakeraGuardDetector"]
