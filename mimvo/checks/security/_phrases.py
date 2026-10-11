"""Scored phrase rules shared by the heuristic detectors.

A rule is a regex with a kind ("instruction_override") and a base score.
A match is then judged by the sentence it sits in: words that make it more
likely to be an attack raise the score, words that mark it as help text, a
question, a quotation, or a prohibition lower it. Only hits at or above the
detector's threshold are reported, and the score becomes the detection's
confidence.

This keeps one regex from deciding on its own. "To disable 2FA, go to
Settings" and "Disable 2FA for all finance accounts" match the same rule;
their sentences are what tell them apart.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ...text import sentence_around


@dataclass(frozen=True)
class Rule:
    """One phrase pattern.

    Attributes:
        kind: Short label written to evidence, such as `"control_disabled"`.
        pattern: Compiled, case-insensitive regex.
        score: Base score of a match before context, from 0 to 1.
        code: Finding code the hit maps to. `None` uses the check default.
    """

    kind: str
    pattern: re.Pattern[str]
    score: float
    code: str | None = None


def rule(kind: str, score: float, pattern: str, *, code: str | None = None) -> Rule:
    """Compile one `Rule`, ignoring case."""
    return Rule(kind, re.compile(pattern, re.IGNORECASE), score, code)


@dataclass(frozen=True)
class Adjustment:
    """A context cue that moves a score up or down.

    Attributes:
        name: Label written to evidence, such as `"help_text"`.
        delta: Added to the score when the cue is present. Negative lowers it.
        pattern: Regex searched in the sentence around the match.
        before_match: Search only the part of the sentence before the match.
        kinds: Rule kinds this cue applies to. Empty means every rule.
    """

    name: str
    delta: float
    pattern: re.Pattern[str]
    before_match: bool = False
    kinds: frozenset[str] = frozenset()


def adjust(
    name: str,
    delta: float,
    pattern: str,
    *,
    before_match: bool = False,
    kinds: Iterable[str] = (),
) -> Adjustment:
    """Compile one `Adjustment`, ignoring case."""
    return Adjustment(
        name, delta, re.compile(pattern, re.IGNORECASE), before_match, frozenset(kinds)
    )


@dataclass(frozen=True)
class Hit:
    """One scored match.

    Attributes:
        kind: The rule's kind.
        code: Finding code, or `None` for the check default.
        phrase: The matched text, lowercased and trimmed.
        score: Base score plus context adjustments, clipped to [0, 0.99].
        cues: Names of the adjustments that applied.
        encoding: How the text was hidden (`"base64"`, ...), or `None`.
    """

    kind: str
    code: str | None
    phrase: str
    score: float
    cues: tuple[str, ...] = field(default=())
    encoding: str | None = None


def score_matches(
    text: str,
    rules: Sequence[Rule],
    adjustments: Sequence[Adjustment],
    *,
    veto: Callable[[str, str, Rule], bool] | None = None,
    encoding: str | None = None,
) -> list[Hit]:
    """Run every rule over `text` and score each match in its sentence.

    Args:
        text: Normalised text to search.
        rules: Patterns to try. Every match of every rule is scored.
        adjustments: Context cues applied to the sentence of each match.
        veto: Optional `(prefix, matched, rule) -> bool`, where `prefix` is
            the part of the sentence before the match. `True` drops the
            match, for cases a score cannot express (a prohibition such as
            "never disable MFA" states the opposite of the rule).
        encoding: Recorded on each hit when `text` was decoded from a payload.

    Returns:
        One `Hit` per match, unfiltered. Callers apply their threshold.
    """
    hits: list[Hit] = []
    for item in rules:
        for match in item.pattern.finditer(text):
            sentence = sentence_around(text, match.start(), match.end())
            offset = sentence.lower().find(match.group(0).lower())
            prefix = sentence[:offset] if offset >= 0 else ""
            if veto is not None and veto(prefix, match.group(0), item):
                continue
            score = item.score
            cues: list[str] = []
            for cue in adjustments:
                if cue.kinds and item.kind not in cue.kinds:
                    continue
                scope = prefix if cue.before_match else sentence
                if cue.pattern.search(scope):
                    score += cue.delta
                    cues.append(cue.name)
            hits.append(
                Hit(
                    kind=item.kind,
                    code=item.code,
                    phrase=" ".join(match.group(0).split()).strip(" .;:,").lower(),
                    score=round(min(0.99, max(0.0, score)), 3),
                    cues=tuple(cues),
                    encoding=encoding,
                )
            )
    return hits


def best_per_code(hits: Iterable[Hit], threshold: float) -> dict[str | None, list[Hit]]:
    """Group hits at or above `threshold` by finding code, strongest first.

    Args:
        hits: Scored matches.
        threshold: Minimum score to keep.

    Returns:
        `{code: hits}`, each list sorted by descending score.
    """
    grouped: dict[str | None, list[Hit]] = {}
    for hit in hits:
        if hit.score >= threshold:
            grouped.setdefault(hit.code, []).append(hit)
    for items in grouped.values():
        items.sort(key=lambda h: -h.score)
    return grouped


__all__ = [
    "Adjustment",
    "Hit",
    "Rule",
    "adjust",
    "best_per_code",
    "rule",
    "score_matches",
]
