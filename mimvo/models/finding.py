"""A `Finding`: one problem a check noticed in a memory."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..owasp import ASI06_REF
from .enums import Action, Category, Severity


class Finding(BaseModel):
    """One problem a check noticed about a memory.

    A finding does not delete or quarantine by itself. The scan lists every
    finding in the report, and each one carries a recommended action.

    Attributes:
        code: Stable id such as `"memory_poisoning"` or `"secret_detected"`.
        category: Which family this belongs to. Built-in checks use security.
        severity: How serious it is. Defaults to medium.
        message: Sentence shown to the person reading the report.
        evidence: Extra facts such as detector names and scores. Never the
            raw secret or the full memory text.
        check: Name of the check that raised this, such as `"injection"`.
            `None` until the engine fills it in.
        recommended_action: What this finding argues for (review, quarantine,
            or delete). The report keeps this action.
        owasp: OWASP reference for this finding. Poisoning uses ASI06,
            injection uses LLM01, secrets and PII use LLM02.
        confidence: How sure the detectors are, from 0 to 1. Several
            agreeing detectors raise it. `None` when no detector scored
            its hit (a custom detector, for example).

    Example:
        A high-severity poisoning finding::

            Finding(
                code="memory_poisoning",
                category="security",
                severity="high",
                message="Content looks like an attempt to plant a false security fact.",
                confidence=0.8,
            )
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Stable machine-readable identifier, e.g. 'memory_poisoning'.")
    category: Category = Field(default=Category.SECURITY)
    severity: Severity = Field(default=Severity.MEDIUM)
    message: str = Field(description="Human-readable explanation.")
    evidence: dict[str, object] = Field(
        default_factory=dict,
        description="Structured, non-sensitive supporting data. Never contains raw secrets.",
    )
    check: str | None = Field(default=None, description="Name of the check that raised this.")
    recommended_action: Action | None = Field(
        default=None,
        description="Action this finding argues for; the engine aggregates across findings.",
    )
    owasp: str = Field(
        default=ASI06_REF,
        description="OWASP item this finding maps to, e.g. ASI06 or LLM02.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Combined detector confidence from 0 to 1, or None when unscored.",
    )

    def __str__(self) -> str:
        return f"{self.severity.value.upper():<8} {self.code} — {self.message}"


__all__ = ["Finding"]
