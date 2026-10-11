"""The rule catalogue: what each finding code means and how to fix it.

Every finding a scan reports carries a code such as `"secret_detected"`.
The code is the rule id, the same way Semgrep, Trivy, and Gitleaks key
their findings. This module describes each built-in rule once, so every
report format (HTML, Markdown, SARIF, JSON, terminal) shows the same title,
explanation, remediation steps, and references.

This is the only place a rule's severity, action, and message live. A
`SecurityCheck` lists the rules it can raise (`SecurityCheck.rules`) and
builds findings from them; the reporters look the same rules up here.

A custom check registers its own rules by listing them on the class, so
its findings render with a real title and remediation. A code nobody
registered still renders: `rule_for` builds a generic rule from it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .exceptions import ConfigurationError
from .models.enums import Action, Severity
from .owasp import (
    ASI06_ID,
    ASI06_TITLE,
    ASI06_URL,
    LLM01_ID,
    LLM01_TITLE,
    LLM01_URL,
    LLM02_ID,
    LLM02_TITLE,
    LLM02_URL,
)

REPO_URL = "https://github.com/jawadhussein462/mimvo"
RULES_DOC_URL = f"{REPO_URL}#what-it-finds"

_CWE_URL = "https://cwe.mitre.org/data/definitions/{}.html"


@dataclass(frozen=True)
class Reference:
    """A link a reader can follow to learn more about a rule.

    Attributes:
        label: Short text for the link, such as `"CWE-312"`.
        url: Where the link goes.
    """

    label: str
    url: str


@dataclass(frozen=True, kw_only=True)
class Rule:
    """One finding code: how it is raised and how it is explained.

    Example:
        A rule for a custom check::

            Rule(
                id="stale_memory",
                title="Stale memory",
                category="freshness",
                severity=Severity.LOW,
                action=Action.REVIEW,
                summary="The record has not been confirmed in a year.",
                remediation=("Confirm it with its owner or delete it.",),
            )

    Attributes:
        id: The finding code, such as `"secret_detected"`. Stable.
        title: Short name shown as the finding's heading.
        severity: Severity of a finding with this code. A low-confidence
            finding is reported one step lower.
        action: Recommended action for a finding with this code.
        summary: One sentence saying what was found, for rule docs.
        message: Sentence written on each finding. Defaults to `summary`.
        category: Which check raises it, such as `"secrets"`. `"custom"`
            for rules built for unregistered codes.
        remediation: Ordered steps to fix it.
        owasp_id: OWASP item id, such as `"LLM02"`. Defaults to ASI06.
        owasp_title: OWASP item title.
        owasp_url: Link to the OWASP item.
        cwe: CWE ids that describe the weakness, such as `("CWE-312",)`.
    """

    id: str
    title: str
    severity: Severity
    action: Action
    summary: str
    message: str = ""
    category: str = "custom"
    remediation: tuple[str, ...] = ()
    owasp_id: str = ASI06_ID
    owasp_title: str = ASI06_TITLE
    owasp_url: str = ASI06_URL
    cwe: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.message:
            object.__setattr__(self, "message", self.summary)

    @property
    def owasp(self) -> str:
        """The OWASP reference as one string, such as `"LLM02: Sensitive ..."`."""
        return f"{self.owasp_id}: {self.owasp_title}"

    @property
    def help_uri(self) -> str:
        """Documentation for this rule."""
        return RULES_DOC_URL

    @property
    def references(self) -> tuple[Reference, ...]:
        """OWASP first, then each CWE, then the Mimvo rule docs."""
        refs = [Reference(self.owasp, self.owasp_url)]
        refs.extend(Reference(cwe, _CWE_URL.format(cwe.split("-", 1)[1])) for cwe in self.cwe)
        refs.append(Reference("Mimvo rules", self.help_uri))
        return tuple(refs)


_OWASP = {
    "secrets": (LLM02_ID, LLM02_TITLE, LLM02_URL),
    "injection": (LLM01_ID, LLM01_TITLE, LLM01_URL),
    "poisoning": (ASI06_ID, ASI06_TITLE, ASI06_URL),
}
_DEFAULT_CWE = {"injection": ("CWE-1427",), "poisoning": ("CWE-345",)}


def _rule(
    code: str,
    category: str,
    *,
    title: str,
    message: str,
    severity: Severity,
    action: Action,
    summary: str,
    remediation: tuple[str, ...],
    cwe: tuple[str, ...] | None = None,
) -> Rule:
    owasp_id, owasp_title, owasp_url = _OWASP[category]
    return Rule(
        id=code,
        title=title,
        message=message,
        category=category,
        severity=severity,
        action=action,
        summary=summary,
        remediation=remediation,
        owasp_id=owasp_id,
        owasp_title=owasp_title,
        owasp_url=owasp_url,
        cwe=_DEFAULT_CWE.get(category, ()) if cwe is None else cwe,
    )


_REVIEW_POISON = (
    "Read the record and confirm with its owner whether the statement is true.",
    "If it is not, quarantine or delete it so the agent stops retrieving it.",
    "Find the write path that stored it and add a write-time check there.",
)

_RULES: tuple[Rule, ...] = (
    _rule(
        "secret_detected",
        "secrets",
        title="Leaked secret",
        message="Secret-like content detected. Delete the record and rotate the credential.",
        severity=Severity.CRITICAL,
        action=Action.DELETE,
        summary=(
            "The record contains a credential: an API key, token, password, private key, "
            "or connection string."
        ),
        remediation=(
            "Rotate or revoke the credential first. Deleting the record does not undo exposure.",
            "Delete the record from the store, and from any export or backup of it.",
            "Stop the write path from storing secrets, for example with a WriteGuard.",
        ),
        cwe=("CWE-312",),
    ),
    _rule(
        "pii_detected",
        "secrets",
        title="Personal data",
        message=(
            "Personal data detected; redact it or apply a retention rule. Do not treat this as a "
            "credential to rotate."
        ),
        severity=Severity.MEDIUM,
        action=Action.REVIEW,
        summary="The record contains personal data such as a name, email, or phone number.",
        remediation=(
            "Check whether the agent needs this data to do its job.",
            "Redact it, or apply your retention rule for personal data.",
            "Do not treat it as a credential to rotate.",
        ),
        cwe=("CWE-359",),
    ),
    _rule(
        "persistent_instruction",
        "injection",
        title="Hidden instruction",
        message="Content contains agent-directed instruction/injection phrases.",
        severity=Severity.HIGH,
        action=Action.REVIEW,
        summary=(
            "The record tries to override the agent's instructions, switch its persona, "
            "or hide something from the user. It will reach the prompt as trusted context."
        ),
        remediation=(
            "Read the record. Memory should hold facts and preferences, not orders to the agent.",
            "Delete it if it is an injection; it re-enters the prompt every time it is retrieved.",
            "Trace where it came from (a scraped page, a ticket, a tool result) and filter there.",
        ),
    ),
    _rule(
        "known_answer",
        "injection",
        title="Canary instruction dropped",
        message=(
            "The record stops a language model from following a canary instruction placed next to "
            "it (known-answer detection)."
        ),
        severity=Severity.MEDIUM,
        action=Action.REVIEW,
        summary=(
            "Placed next to a canary instruction, the record made a language model ignore it."
        ),
        remediation=(
            "Review the record for embedded instructions.",
            "Delete or quarantine it if it hijacks the model.",
        ),
    ),
    _rule(
        "embedding_injection",
        "injection",
        title="Injection embedding",
        message=(
            "The stored embedding is classified as prompt injection by an embedding-space linear "
            "model."
        ),
        severity=Severity.MEDIUM,
        action=Action.REVIEW,
        summary="A classifier on the stored vector labels the record as prompt injection.",
        remediation=(
            "Review the record text for embedded instructions.",
            "Calibrate the classifier threshold on your own data if hits look benign.",
        ),
    ),
    _rule(
        "memory_poisoning",
        "poisoning",
        title="Poisoned fact",
        message=(
            "Content matches a control-bypass phrase; a regex hit is a signal, not proof of a "
            "planted fact. Review before treating it as poison."
        ),
        severity=Severity.HIGH,
        action=Action.QUARANTINE,
        summary=(
            "The record claims a security control is off: authentication, MFA, approval, "
            "or review. An agent that retrieves it may act on it."
        ),
        remediation=(
            "Quarantine the record so retrieval stops returning it.",
            "Confirm with the control's owner whether the claim is true.",
            "Delete it if false, and find the source that wrote it.",
        ),
    ),
    _rule(
        "destination_redirect",
        "poisoning",
        title="Destination redirect",
        message=(
            "Content redirects payments or data to a new destination; confirm the change before it"
            " is remembered."
        ),
        severity=Severity.HIGH,
        action=Action.REVIEW,
        summary="The record routes payments or data to a new destination.",
        remediation=(
            "Confirm the change through a channel other than the agent's memory.",
            "Delete the record if nobody authorised the new destination.",
        ),
    ),
    _rule(
        "poisoning_cluster",
        "poisoning",
        title="Poisoning cluster",
        message=(
            "Content is one of several near-identical retrieved records, the multi-document "
            "poisoning pattern; confirm they are legitimate copies."
        ),
        severity=Severity.LOW,
        action=Action.REVIEW,
        summary=(
            "The record is one of several near-identical records, the pattern of "
            "multi-document poisoning."
        ),
        remediation=(
            "Compare the records in the cluster; legitimate copies of a template also match.",
            "Remove the copies if they were planted to outvote the truth.",
        ),
    ),
    _rule(
        "hub_record",
        "poisoning",
        title="Retrieval hub",
        message=(
            "This record appears in unusually many other records' nearest neighbours, the hubness "
            "pattern of a passage written to be retrieved for many queries."
        ),
        severity=Severity.LOW,
        action=Action.REVIEW,
        summary=(
            "The record is a nearest neighbour of unusually many others, so it is "
            "retrieved for many unrelated queries."
        ),
        remediation=(
            "Read the record; text written to match every query is a poisoning signal.",
            "Delete it if its content has no reason to be that central.",
        ),
    ),
    _rule(
        "adversarial_text",
        "poisoning",
        title="Adversarial text",
        message=(
            "A span of this content reads as machine-optimised rather than natural text (chunk-"
            "wise language-model perplexity)."
        ),
        severity=Severity.MEDIUM,
        action=Action.REVIEW,
        summary="Part of the record reads as machine-optimised rather than natural text.",
        remediation=(
            "Look for gibberish suffixes or token soup in the record.",
            "Delete it if the span was optimised to steer retrieval or generation.",
        ),
        cwe=("CWE-1039",),
    ),
    _rule(
        "embedding_mismatch",
        "poisoning",
        title="Tampered vector",
        message=(
            "The stored vector does not match a fresh embedding of the text; the vector may have "
            "been written directly or tampered with."
        ),
        severity=Severity.MEDIUM,
        action=Action.QUARANTINE,
        summary="The stored vector does not match a fresh embedding of the record text.",
        remediation=(
            "Quarantine the record; its vector may have been written directly.",
            "Re-embed it with the store's model, or delete it.",
            "Check who has write access to the vector column.",
        ),
    ),
    _rule(
        "temporal_contradiction",
        "poisoning",
        title="Contradicts older memory",
        message=(
            "A newer record contradicts older established neighbours; confirm the update is "
            "legitimate."
        ),
        severity=Severity.MEDIUM,
        action=Action.REVIEW,
        summary="A newer record contradicts the older records around it.",
        remediation=_REVIEW_POISON,
    ),
    _rule(
        "retrieval_flip",
        "poisoning",
        title="Answer flip",
        message=(
            "Removing this record changes the answer to probe questions generated from it; it may "
            "be the sole source of a planted fact."
        ),
        severity=Severity.MEDIUM,
        action=Action.REVIEW,
        summary="Removing the record changes the answers to questions about its topic.",
        remediation=_REVIEW_POISON,
    ),
)

RULES: dict[str, Rule] = {rule.id: rule for rule in _RULES}
"""Every registered rule by code: the built-ins plus those custom checks add."""


def register_rule(rule: Rule) -> Rule:
    """Add a rule to the catalogue so reports can describe its findings.

    `SecurityCheck` subclasses call this for each rule they list, so you
    rarely need it directly. A rule with the same code and category
    replaces the registered one, so redefining a check (in a notebook, or
    a custom check that replaces a built-in of the same name) works.

    Args:
        rule: The rule to add.

    Returns:
        `rule`, so the call can wrap a definition.

    Raises:
        ConfigurationError: A rule of another category already uses
            `rule.id`. Two checks cannot give one code two meanings.
    """
    existing = RULES.get(rule.id)
    if existing is not None and existing.category != rule.category:
        raise ConfigurationError(
            f"Rule {rule.id!r} is already registered by category {existing.category!r}; "
            f"category {rule.category!r} must pick another code."
        )
    RULES[rule.id] = rule
    return rule


def rules_in(category: str) -> dict[str, Rule]:
    """The registered rules of one category, by code.

    Args:
        category: Such as `"secrets"`, `"injection"`, or `"poisoning"`.

    Returns:
        `{code: rule}` in catalogue order.
    """
    return {code: rule for code, rule in RULES.items() if rule.category == category}


def rule_map(rules: Iterable[Rule]) -> dict[str, Rule]:
    """Key rules by code, for `SecurityCheck.rules`.

    Args:
        rules: Rules a check can raise.

    Returns:
        `{rule.id: rule}`.
    """
    return {rule.id: rule for rule in rules}


def rule_for(code: str) -> Rule:
    """Look up the rule for a finding code.

    Args:
        code: A finding code such as `"secret_detected"`.

    Returns:
        The catalogue entry, or a generic rule for codes from custom
        checks: the code as the title, medium severity, review, ASI06.
    """
    rule = RULES.get(code)
    if rule is not None:
        return rule
    return Rule(
        id=code,
        title=code.replace("_", " ").replace("-", " ").capitalize(),
        category="custom",
        severity=Severity.MEDIUM,
        action=Action.REVIEW,
        summary="Raised by a custom check.",
        remediation=("Review the record against the custom check's documentation.",),
        owasp_id=ASI06_ID,
        owasp_title=ASI06_TITLE,
        owasp_url=ASI06_URL,
    )


__all__ = [
    "REPO_URL",
    "RULES",
    "RULES_DOC_URL",
    "Reference",
    "Rule",
    "register_rule",
    "rule_for",
    "rule_map",
    "rules_in",
]
