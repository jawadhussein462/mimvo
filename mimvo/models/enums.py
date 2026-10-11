"""Labels shared across Mimvo: severity, risk, category, action, status.

Each enum is a `StrEnum`, so `Severity.HIGH == "high"` and JSON stores the
plain string. Severity and risk have a `rank` number instead of `<` and `>`
operators, because those operators would also compare the strings.
"""

from __future__ import annotations

from enum import StrEnum


class Severity(StrEnum):
    """How serious one finding is.

    From least to most serious: `INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Position on the severity scale.

        Returns:
            `0` for info through `4` for critical. A larger number is worse.
        """
        return _SEVERITY_ORDER[self]

    def is_at_least(self, other: Severity) -> bool:
        """Say whether this severity is as serious as `other`, or worse.

        Args:
            other: The severity to compare against.

        Returns:
            `True` when `self.rank` is greater than or equal to `other.rank`.
            `Severity.HIGH.is_at_least(Severity.MEDIUM)` is `True`.
            `Severity.LOW.is_at_least(Severity.HIGH)` is `False`.
        """
        return self.rank >= other.rank

    def lower(self) -> Severity:
        """Return the severity one step below this one.

        Returns:
            `HIGH` for `CRITICAL`, `MEDIUM` for `HIGH`, and so on. `INFO`
            stays `INFO`.
        """
        rank = max(0, self.rank - 1)
        return next(sev for sev, r in _SEVERITY_ORDER.items() if r == rank)


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Risk(StrEnum):
    """Overall risk for a record. Same names as `Severity`, plus `NONE`.

    `NONE` means nothing was found. The other values match a finding's
    severity string, so `"high"` is both `Severity.HIGH` and `Risk.HIGH`.
    """

    NONE = "none"
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Position on the risk scale.

        Returns:
            `0` for none through `5` for critical. A larger number is worse.
        """
        return _RISK_ORDER[self]

    @classmethod
    def from_severity(cls, severity: Severity) -> Risk:
        """Build the risk label that matches a finding severity.

        Args:
            severity: A finding severity. `NONE` is not a severity, so it
                cannot be passed here.

        Returns:
            The `Risk` member with the same string value, such as
            `Risk.HIGH` for `Severity.HIGH`.
        """
        return cls(severity.value)


_RISK_ORDER: dict[Risk, int] = {
    Risk.NONE: 0,
    Risk.INFO: 1,
    Risk.LOW: 2,
    Risk.MEDIUM: 3,
    Risk.HIGH: 4,
    Risk.CRITICAL: 5,
}


class Category(StrEnum):
    """Which family a finding belongs to.

    Built-in checks all use `SECURITY`.
    """

    SECURITY = "security"


class Action(StrEnum):
    """What the scan recommends doing with a stored record.

    From weakest to strongest: `REVIEW`, `QUARANTINE`, `DELETE`. When several
    findings disagree, the one with the higher `precedence` wins.
    """

    REVIEW = "review"
    QUARANTINE = "quarantine"
    DELETE = "delete"

    @property
    def precedence(self) -> int:
        """How strong this recommendation is compared with the others.

        Returns:
            `0` for review, `1` for quarantine, `2` for delete. The larger
            number wins when findings are combined.
        """
        return _ACTION_PRECEDENCE[self]


_ACTION_PRECEDENCE: dict[Action, int] = {
    Action.REVIEW: 0,
    Action.QUARANTINE: 1,
    Action.DELETE: 2,
}


class MemoryStatus(StrEnum):
    """Where a stored record is in its life.

    `ACTIVE` is normal. `QUARANTINED` is held back from retrieval.
    `REVOKED` is no longer in use.
    """

    ACTIVE = "active"
    REVOKED = "revoked"
    QUARANTINED = "quarantined"


__all__ = [
    "Action",
    "Category",
    "MemoryStatus",
    "Risk",
    "Severity",
]
