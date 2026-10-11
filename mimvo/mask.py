"""Masked snippets and evidence for scan reports.

Reports are meant to be forwarded. They show enough of the memory to act
on, and they do not show the secret values that triggered the finding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .checks.security.secrets.entropy import EntropyDetector
from .checks.security.secrets.heuristic import mask_secrets

_DEFAULT_WIDTH = 160
_EVIDENCE_WIDTH = 120
_MAX_LIST_ITEMS = 20

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list[JsonScalar] | dict[str, JsonScalar]


def mask_snippet(text: str, *, width: int = _DEFAULT_WIDTH) -> str:
    """Hide secrets in `text` and shorten it for a report line.

    Args:
        text: The full memory content.
        width: Maximum characters in the result. Default 160.

    Returns:
        One line: secrets replaced with bullets, whitespace collapsed, cut
        to `width` characters with a trailing ellipsis when it was longer.
    """
    masked = " ".join(EntropyDetector().mask(mask_secrets(text)).split())
    if len(masked) <= width:
        return masked
    return masked[: width - 1] + "…"


def safe_evidence(evidence: Mapping[str, object]) -> dict[str, JsonValue]:
    """Keep the parts of a finding's evidence that are safe to forward.

    Detectors already promise not to put raw secrets in evidence. This is
    a second guard for reports that leave the machine: every string is
    passed through `mask_snippet`, lists are capped, and anything that is
    not a JSON scalar, a list of scalars, or a flat mapping of scalars
    (such as per-detector `scores`) is dropped.

    Args:
        evidence: `Finding.evidence` as the check built it.

    Returns:
        A JSON-serialisable dict. The `detectors` key is left out because
        reports carry it as its own field.
    """
    out: dict[str, JsonValue] = {}
    for key, value in evidence.items():
        if key == "detectors":
            continue
        clean = _safe_value(value)
        if clean is not None and clean != [] and clean != {}:
            out[str(key)] = clean
    return out


def _safe_scalar(value: object) -> JsonScalar:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, str):
        return mask_snippet(value, width=_EVIDENCE_WIDTH)
    return None


def _safe_value(value: object) -> JsonValue:
    if isinstance(value, list | tuple | set | frozenset):
        items = sorted(value, key=str) if isinstance(value, set | frozenset) else list(value)
        scalars = [_safe_scalar(item) for item in items[:_MAX_LIST_ITEMS]]
        return [item for item in scalars if item is not None]
    if isinstance(value, Mapping):
        flat = {str(k): _safe_scalar(v) for k, v in value.items()}
        return {k: v for k, v in flat.items() if v is not None}
    return _safe_scalar(value)


__all__ = ["mask_snippet", "safe_evidence"]
