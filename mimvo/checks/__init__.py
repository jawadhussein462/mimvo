"""Built-in checks and `default_checks`.

`default_checks()` returns the standard pipeline: secrets, injection, then
poisoning. Each one uses an offline detector unless you pass your own.

Pass a configured `InjectionCheck`, `PoisoningCheck`, or `SecretsCheck` in
`Mimvo(checks=[...])` to replace the default check of the same name.
A check with a new name is added alongside the defaults.
"""

from __future__ import annotations

from .base import MemoryCheck, Phase
from .security import (
    BaseDetector,
    Detection,
    Detector,
    InjectionCheck,
    PoisoningCheck,
    SecretsCheck,
    SecurityCheck,
)


def default_checks() -> list[MemoryCheck]:
    """Return the checks `Mimvo()` runs, in order.

    The order is secrets, then injection, then poisoning. Each check uses
    its offline detector, so this list needs no network and no model download.
    The same input always produces the same findings.

    Returns:
        A new list of check instances. Call it again for a fresh list; do
        not share one list across clients.
    """
    return [
        SecretsCheck(),
        InjectionCheck(),
        PoisoningCheck(),
    ]


__all__ = [
    "BaseDetector",
    "Detection",
    "Detector",
    "InjectionCheck",
    "MemoryCheck",
    "Phase",
    "PoisoningCheck",
    "SecretsCheck",
    "SecurityCheck",
    "default_checks",
]
