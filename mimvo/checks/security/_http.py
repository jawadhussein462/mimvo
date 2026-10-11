"""POST JSON over HTTPS for hosted detectors. Uses only the standard library.

Azure Prompt Shields and Lakera Guard take a `transport` callable so tests
can fake the network and callers can plug in their own HTTP client.

The callable's shape is:

    (url, headers, payload) -> parsed JSON

`urllib_transport` is the default.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from ...exceptions import BackendError

Transport = Callable[[str, Mapping[str, str], Mapping[str, Any]], Any]


def urllib_transport(timeout: float = 10.0) -> Transport:
    """Build a transport that POSTs JSON with `urllib` and parses the reply.

    Args:
        timeout: Seconds to wait for the server before failing.

    Returns:
        A function `(url, headers, payload) -> parsed JSON`. `payload` is
        encoded as JSON. `headers` are added on top of `Content-Type`.

    Raises:
        BackendError: The server returns an HTTP error, the connection
            fails, the call times out, or the body is not JSON. The
            function that is returned raises this, not `urllib_transport`
            itself.
    """

    def post(url: str, headers: Mapping[str, str], payload: Mapping[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BackendError(f"{url}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise BackendError(f"{url}: {exc}") from exc

    return post


__all__ = ["Transport", "urllib_transport"]
