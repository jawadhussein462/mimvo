"""Azure AI Content Safety Prompt Shields. Hosted.

`POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01`
analyses a `userPrompt` (a direct attack) and up to five `documents`
(instructions hidden in third-party content). A stored memory is
third-party content from the agent's point of view, so the candidate is
sent as a document by default. The detector reads
`documentsAnalysis[0].attackDetected`. The API returns yes or no, not a score.

Credentials come from `endpoint` and `key`, or from
`AZURE_CONTENT_SAFETY_ENDPOINT` and `AZURE_CONTENT_SAFETY_KEY`. Only the
candidate text is sent. Nothing is logged.
"""

from __future__ import annotations

import os
from typing import Any

from ....exceptions import BackendError, ConfigurationError
from .._http import Transport, urllib_transport
from ..base import BaseDetector, Detection

API_VERSION = "2024-09-01"
# The API accepts at most 10k characters per document.
MAX_DOCUMENT_CHARS = 10_000


class PromptShieldDetector(BaseDetector):
    """Ask Azure Prompt Shields whether a document hides an attack.

    Pass `as_user_prompt=True` to screen the text as a direct user prompt
    instead. A hit has score `1.0` because the API returns a boolean.
    """

    name = "prompt_shield"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        key: str | None = None,
        as_user_prompt: bool = False,
        api_version: str = API_VERSION,
        transport: Transport | None = None,
        timeout: float = 10.0,
        concurrency: int = 8,
    ) -> None:
        """Store the Azure endpoint and key.

        Args:
            endpoint: Content Safety endpoint URL. `None` reads
                `AZURE_CONTENT_SAFETY_ENDPOINT`.
            key: Subscription key. `None` reads `AZURE_CONTENT_SAFETY_KEY`.
            as_user_prompt: When `False` (the default), send the memory as a
                document and read `documentsAnalysis`. When `True`, send it
                as `userPrompt` and read `userPromptAnalysis`.
            api_version: Query-string version. Defaults to `2024-09-01`.
            transport: Function `(url, headers, payload) -> parsed JSON`.
                `None` uses `urllib`.
            timeout: Seconds `urllib` waits. Ignored when you pass `transport`.
            concurrency: Requests in flight at once during a batched scan.
                `1` sends them one after another. A custom `transport` must
                be thread-safe when this is above 1.

        Raises:
            ConfigurationError: `endpoint` or `key` is missing from both the
                arguments and the environment, or `concurrency` is below 1.
        """
        endpoint = endpoint or os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT")
        key = key or os.environ.get("AZURE_CONTENT_SAFETY_KEY")
        if not endpoint or not key:
            raise ConfigurationError(
                "PromptShieldDetector needs endpoint and key (or AZURE_CONTENT_SAFETY_ENDPOINT / "
                "AZURE_CONTENT_SAFETY_KEY)."
            )
        self.url = (
            f"{endpoint.rstrip('/')}/contentsafety/text:shieldPrompt?api-version={api_version}"
        )
        self._headers = {"Ocp-Apim-Subscription-Key": key}
        self.as_user_prompt = as_user_prompt
        self._transport = transport or urllib_transport(timeout)
        if concurrency < 1:
            raise ConfigurationError("concurrency must be >= 1.")
        self.concurrency = concurrency

    def shield(self, text: str) -> dict[str, Any]:
        """Call Prompt Shields and return the parsed JSON.

        Args:
            text: Memory content. Cut to 10,000 characters, which is the
                API's per-document limit.

        Returns:
            The response object. Look at `documentsAnalysis` or
            `userPromptAnalysis`, depending on `as_user_prompt`.

        Raises:
            ConfigurationError: The response is not a JSON object.
            BackendError: The HTTP call fails. Raised by the transport.
        """
        text = text[:MAX_DOCUMENT_CHARS]
        payload: dict[str, Any] = (
            {"userPrompt": text, "documents": []}
            if self.as_user_prompt
            else {"userPrompt": "", "documents": [text]}
        )
        response = self._transport(self.url, self._headers, payload)
        if not isinstance(response, dict):
            raise BackendError("Prompt Shields returned a non-object response.")
        return response

    def detect_text(self, text: str) -> list[Detection]:
        response = self.shield(text)
        if self.as_user_prompt:
            detected = bool((response.get("userPromptAnalysis") or {}).get("attackDetected"))
            mode = "user_prompt"
        else:
            docs = response.get("documentsAnalysis") or []
            detected = bool(docs and docs[0].get("attackDetected"))
            mode = "document"
        if not detected:
            return []
        return [self.hit(score=1.0, provider="azure_prompt_shields", mode=mode)]


__all__ = ["PromptShieldDetector"]
