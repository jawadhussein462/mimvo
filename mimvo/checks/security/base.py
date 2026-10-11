"""Security checks: one check per problem, one detector per way of finding it.

A security check looks for one kind of problem (injection, poisoning, or
secrets). It runs one or more detectors. Each detector is a single way of
looking: a regex, a Hugging Face model, a hosted API, or a statistical filter.

How the classes fit together:

    MemoryCheck
    └── SecurityCheck
        ├── InjectionCheck
        ├── PoisoningCheck
        └── SecretsCheck

A detector only reports what it saw (`Detection`). The check turns those
reports into `Finding` objects. The engine then collects findings from every
check.

To add a new way of looking, add a detector class.
To add a new kind of problem, add a `SecurityCheck` subclass and list the
`Rule`s it can raise in `rules` (finding code → severity, action, message,
remediation). The rules are registered in `mimvo.rules`, so every report
describes the new findings too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from ...context import CheckContext
from ...exceptions import ConfigurationError
from ...models.enums import Category
from ...models.finding import Finding
from ...models.memory import MemoryCandidate
from ...rules import Rule, register_rule
from ...telemetry import get_logger
from ..base import MemoryCheck, Phase

_logger = get_logger(__name__)


@dataclass(frozen=True)
class Detection:
    """One thing a detector noticed in a memory.

    The detector does not decide what to do. It fills in this object and the
    parent check turns it into a finding.

    Attributes:
        detector: Name of the detector that produced this result. The check
            uses it to count votes and to record who agreed.
        code: Which finding to emit, for example `"api_key"`. `None` means
            "use the check's default code".
        score: How sure the detector is, usually between 0 and 1 (a model
            probability or a statistical score). `None` when this detector
            does not score its hits.
        evidence: Extra facts about the hit, such as pattern names or counts.
            Never put a raw secret or the full memory text here.
    """

    detector: str
    code: str | None = None
    score: float | None = None
    evidence: Mapping[str, object] = field(default_factory=dict)


class BatchError(Exception):
    """Some memories in a `detect_batch` call failed; the rest have results.

    Raise this from `detect_batch` so the check records each failure on the
    memory it belongs to and keeps every other memory's detections, instead
    of retrying the whole batch one memory at a time.

    Attributes:
        results: One entry per candidate, in order. `None` where that
            candidate failed.
        errors: Candidate index → the exception it raised.
    """

    def __init__(
        self, results: list[list[Detection] | None], errors: Mapping[int, BaseException]
    ) -> None:
        super().__init__(f"{len(errors)} of {len(results)} candidates failed")
        self.results = results
        self.errors = dict(errors)


@runtime_checkable
class Detector(Protocol):
    """One way of looking for a security problem.

    Anything with a `name` and a `detect` method can be used as a detector.
    A detector may also define `detect_batch(candidates, context)`, returning
    one list of detections per candidate; the check then calls it once per
    batch instead of `detect` once per memory (see `BaseDetector`).

    Attributes:
        name: Short name for this detector. It must be unique among the
            detectors on the same check, because the check uses it as a vote.
    """

    name: str

    def detect(self, candidate: MemoryCandidate, context: CheckContext) -> list[Detection]:
        """Inspect one memory and report every hit.

        Args:
            candidate: The memory being scanned. The text is `candidate.content`.
            context: Other information for this scan, such as the user's query
                (`context.query`) and nearby stored records (`context.existing`).
                A detector that only reads the text can ignore this.

        Returns:
            One `Detection` per hit. An empty list means this detector found
            nothing suspicious.
        """
        ...


class BaseDetector:
    """Starting point for a detector that only needs the memory text.

    Write `detect_text` and this class supplies `detect` for you: it pulls
    the text out of the candidate and calls `detect_text`.

    If the detector needs the query or the neighbouring records, override
    `detect` instead and read `context.query` or `context.existing`.

    Scans hand detectors memories in batches (`Config.batch_size`). Override
    `detect_batch` when one call over many memories is cheaper than many
    calls, such as a model forward pass. For a detector that waits on the
    network, set `concurrency` instead and the default `detect_batch` runs
    `detect` on that many threads.

    Attributes:
        name: Short name stamped onto every `Detection` this detector builds.
            Subclasses should set their own, for example `"heuristic"`.
        needs_corpus: Whether the detector reads other records
            (`context.existing` or the corpus). The engine runs text-only
            detectors while records stream in, and corpus detectors once
            every record has been read. A detector that only implements
            `detect_text` cannot see other records, so it may leave this
            unset. A detector that overrides `detect` must set it: `True`
            when it reads neighbours, `False` when it only reads the
            candidate (its embedding, say). Leaving it unset on a `detect`
            override is a `ConfigurationError` when the check is built.
        concurrency: How many memories the default `detect_batch` checks at
            once, on threads. `1` (the default) checks them in turn. Raise
            it only for a detector whose `detect` is thread-safe and spends
            its time waiting, such as a hosted API call.
    """

    name: str = "base"
    needs_corpus: bool | None = None
    concurrency: int = 1

    def detect(self, candidate: MemoryCandidate, context: CheckContext) -> list[Detection]:
        """Run this detector on one memory.

        The default uses only the memory text. `context` is accepted so this
        method matches the `Detector` protocol, and is otherwise unused.

        Args:
            candidate: The memory being scanned. Only `candidate.content` is read.
            context: Scan context. Ignored here. Override this method if you
                need the query or neighbouring records.

        Returns:
            Hits found in the memory text. An empty list means no hit.
        """
        return self.detect_text(candidate.content)

    def detect_batch(
        self, candidates: Sequence[MemoryCandidate], context: CheckContext
    ) -> list[list[Detection]]:
        """Run this detector on several memories.

        Args:
            candidates: The memories, in scan order.
            context: Scan context.

        Returns:
            One list of detections per candidate, in the same order.

        Raises:
            BatchError: At least one candidate's `detect` raised. It carries
                the other candidates' results.
        """
        results: list[list[Detection] | None] = [None] * len(candidates)
        errors: dict[int, BaseException] = {}

        def run(index: int) -> None:
            try:
                results[index] = self.detect(candidates[index], context)
            except Exception as exc:
                errors[index] = exc

        workers = min(max(self.concurrency, 1), len(candidates))
        if workers <= 1:
            for index in range(len(candidates)):
                run(index)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(run, range(len(candidates))))
        if errors:
            raise BatchError(results, errors)
        return [found or [] for found in results]

    def detect_text(self, text: str) -> list[Detection]:
        """Look for the problem in a plain string.

        Args:
            text: The memory content to inspect.

        Returns:
            Hits found in `text`. An empty list means no hit.

        Raises:
            NotImplementedError: Always, unless a subclass implements this.
        """
        raise NotImplementedError

    def hit(
        self,
        *,
        code: str | None = None,
        score: float | None = None,
        **evidence: object,
    ) -> Detection:
        """Build a `Detection` and attach this detector's name to it.

        Call this from `detect` or `detect_text` instead of constructing
        `Detection` yourself, so `detector` is always filled in.

        Args:
            code: Finding code for this hit, such as `"instruction_override"`.
                Leave it out to use the parent check's default code.
            score: Confidence for this hit, usually between 0 and 1. Leave it
                out when the method does not produce a score.
            **evidence: Extra facts, passed as keywords. Example:
                `hit(code="api_key", kinds=["openai"])`. Do not pass the secret
                itself or the full text.

        Returns:
            A `Detection` whose `detector` field is this detector's `name`.
        """
        return Detection(detector=self.name, code=code, score=score, evidence=evidence)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class SecurityCheck(MemoryCheck):
    """Run detectors on a memory and turn their hits into findings.

    A subclass handles one security problem. It must set:

    * `name` — label written on each finding, such as `"injection"`.
    * `default_code` — finding code used when a detector leaves `code` empty.
    * `rules` — every allowed finding code mapped to its `Rule` (severity,
      action, message). Build it with `rule_map(...)` or, for built-ins,
      `rules_in(<category>)`. Defining the subclass registers the rules.
    * `default_detectors` — detectors used when the caller does not pass any.

    These checks run while a store is scanned, so text that was saved by some
    other pipeline is still examined.

    `min_detectors` is a vote, counted separately for each finding code. With
    `min_detectors=2`, a code is reported only after two different detectors
    both raise it. Stacking noisy methods this way reports fewer false alarms
    and can miss a real hit that only one detector saw.

    Confidence. Each detector's best score on a code is combined as
    `1 - Π(1 - score)`: one detector at 0.7 gives 0.7, two at 0.7 give
    0.91. A finding whose confidence is below `low_confidence` is reported
    one severity step lower, so a borderline heuristic or a classifier just
    over its threshold does not rank with a clear-cut hit. Detections
    without a score do not contribute; a finding with no scored detection
    has `confidence=None` and keeps its rule's severity.

    Attributes:
        category: Always security for this family of checks.
        default_code: Code used when a detection has `code=None`.
        rules: Allowed codes and the rule each one is reported under.
        detectors: The detectors this instance will run, in order.
        min_detectors: How many different detectors must agree on a code
            before that code becomes a finding. `1` means any single hit
            is enough.
        low_confidence: Findings below this confidence drop one severity
            step. Default `0.6`.
    """

    category: ClassVar[Category] = Category.SECURITY
    default_code: ClassVar[str] = ""
    rules: ClassVar[Mapping[str, Rule]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register the subclass's rules and check that they are consistent.

        Raises:
            ConfigurationError: A key in `rules` differs from its rule's
                id, `default_code` is not in `rules`, or a rule code is
                already registered with a different definition.
        """
        super().__init_subclass__(**kwargs)
        for code, rule in cls.rules.items():
            if code != rule.id:
                raise ConfigurationError(
                    f"{cls.__name__}.rules maps {code!r} to rule {rule.id!r}; "
                    "keys must be rule ids."
                )
            register_rule(rule)
        if cls.rules and cls.default_code not in cls.rules:
            raise ConfigurationError(
                f"{cls.__name__}.default_code {cls.default_code!r} is not one of its rules."
            )

    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        *,
        min_detectors: int = 1,
        low_confidence: float = 0.6,
    ) -> None:
        """Choose the detectors this check will run.

        Args:
            detectors: Detectors to run on every memory. Pass `None` to use
                `default_detectors()`. An empty sequence is rejected: a check
                with nothing to run cannot scan.
            min_detectors: How many different detectors must agree on the same
                finding code before it is reported. `1` reports every hit.
                Must be at least 1 and no larger than the number of detectors.
            low_confidence: A finding whose combined confidence is below
                this is reported one severity step lower. `0` turns the
                step-down off. Must be within [0, 1].

        Raises:
            ConfigurationError: There are no detectors, two detectors share a
                name, `min_detectors` is below 1 or larger than the
                detector list, `low_confidence` is outside [0, 1], or a
                detector overrides `detect` without declaring
                `needs_corpus`.
        """
        resolved = list(detectors) if detectors is not None else self.default_detectors()
        if not resolved:
            raise ConfigurationError(f"Check {self.name!r} needs at least one detector.")
        names = [d.name for d in resolved]
        if len(set(names)) != len(names):
            raise ConfigurationError(
                f"Check {self.name!r}: detector names must be unique {names!r}."
            )
        if min_detectors < 1:
            raise ConfigurationError("min_detectors must be >= 1.")
        if min_detectors > len(resolved):
            raise ConfigurationError(
                f"min_detectors={min_detectors} exceeds the {len(resolved)} configured detector(s)."
            )
        if not 0.0 <= low_confidence <= 1.0:
            raise ConfigurationError("low_confidence must be within [0, 1].")
        for detector in resolved:
            corpus_pass = needs_corpus(detector)  # fails at construction, not mid-scan
            _logger.debug(
                "check {!r}: detector {!r} runs in the {} pass",
                self.name,
                detector.name,
                "corpus" if corpus_pass else "stream",
            )
        self.detectors: list[Detector] = resolved
        self.min_detectors = min_detectors
        self.low_confidence = low_confidence

    @classmethod
    def default_detectors(cls) -> list[Detector]:
        """Return the detectors used when the caller does not pass any.

        Built-in checks return detectors that are deterministic and work
        offline: no network call and no model download.

        Returns:
            The detectors to run. A subclass must override this and return
            at least one.

        Raises:
            NotImplementedError: Always, on this base class.
        """
        raise NotImplementedError

    # -- pipeline ----------------------------------------------------------------

    def check(self, candidate: MemoryCandidate, context: CheckContext) -> list[Finding]:
        """Scan one memory and return the findings that earned enough votes.

        Each detector is asked about `candidate`. Hits are grouped by finding
        code. A code is kept only when at least `min_detectors` different
        detectors reported it. Those hits are then merged into one finding.

        A detector that raises is recorded on `context.failures` (the scan
        report lists it under `errors` and is marked incomplete) and the
        other detectors' results are kept. A failure never becomes a
        finding: it says nothing about the record.

        Args:
            candidate: The stored memory to scan.
            context: Settings and neighbouring records for this scan.

        Returns:
            One `Finding` for each code that enough detectors agreed on,
            in the order the codes were first seen. Empty when nothing
            reached the vote threshold.

        Raises:
            ConfigurationError: A detector returned a `code` that is not a
                key in `rules`.
        """
        return self.decide(self.collect(candidate, context))

    # -- engine contract ---------------------------------------------------------

    def detectors_for(self, phase: Phase) -> list[Detector]:
        """The detectors that run in one pass.

        Args:
            phase: `STREAM` for text-only detectors, `CORPUS` for those
                that read other records (`needs_corpus`).

        Returns:
            Those detectors, in configured order.
        """
        corpus = phase is Phase.CORPUS
        return [d for d in self.detectors if needs_corpus(d) is corpus]

    def phases(self) -> frozenset[Phase]:
        """The passes at least one of this check's detectors runs in.

        Returns:
            A subset of `{STREAM, CORPUS}`. Votes are counted after the
            last one, over both.
        """
        return frozenset(phase for phase in Phase if self.detectors_for(phase))

    def observe(
        self, candidate: MemoryCandidate, context: CheckContext, phase: Phase
    ) -> dict[str, list[Detection]]:
        """Run this pass's detectors on one memory.

        Args:
            candidate: The memory being scanned.
            context: Scan context. Failing detectors are recorded on it.
            phase: Which detectors to run (see `detectors_for`).

        Returns:
            The pass's detections grouped by finding code.
        """
        return self.collect(candidate, context, self.detectors_for(phase))

    def conclude(self, observations: Sequence[object]) -> list[Finding]:
        """Vote over every pass's detections and build the findings.

        Args:
            observations: `observe` results for one record, in pass order.

        Returns:
            One finding per code that reached `min_detectors`.
        """
        by_code: dict[str, list[Detection]] = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            for code, detections in observation.items():
                by_code.setdefault(code, []).extend(detections)
        return self.decide(by_code)

    def observe_batch(
        self,
        candidates: Sequence[MemoryCandidate],
        context: CheckContext,
        phase: Phase,
    ) -> list[object]:
        """Run this pass's detectors on a batch of memories.

        Args:
            candidates: The memories, in scan order.
            context: Scan context. Failing detectors are recorded on it.
            phase: Which detectors to run (see `detectors_for`).

        Returns:
            One `{code: detections}` per candidate.
        """
        return list(self.collect_batch(candidates, context, self.detectors_for(phase)))

    def collect(
        self,
        candidate: MemoryCandidate,
        context: CheckContext,
        detectors: Sequence[Detector] | None = None,
    ) -> dict[str, list[Detection]]:
        """Run detectors on one memory and group their hits by finding code.

        Args:
            candidate: The stored memory to scan.
            context: Scan context. Failing detectors are recorded on it.
            detectors: Which detectors to run. `None` runs all of them.

        Returns:
            `{code: detections}` in first-seen order.

        Raises:
            ConfigurationError: A detector returned an unknown code.
        """
        return self.collect_batch([candidate], context, detectors)[0]

    def collect_batch(
        self,
        candidates: Sequence[MemoryCandidate],
        context: CheckContext,
        detectors: Sequence[Detector] | None = None,
    ) -> list[dict[str, list[Detection]]]:
        """Run detectors on several memories and group each one's hits by code.

        Each detector is asked once for the whole batch (`detect_batch`)
        when it supports that, otherwise once per memory. A detector that
        fails on some memories is recorded on `context.failures` against
        those memories only; the others keep its results.

        Args:
            candidates: The stored memories to scan, in order.
            context: Scan context. Failing detectors are recorded on it.
            detectors: Which detectors to run. `None` runs all of them.

        Returns:
            One `{code: detections}` per candidate, in first-seen order.

        Raises:
            ConfigurationError: A detector returned an unknown code.
        """
        grouped: list[dict[str, list[Detection]]] = [{} for _ in candidates]
        for detector in self.detectors if detectors is None else detectors:
            results, errors = _run_detector(detector, candidates, context)
            for index, exc in sorted(errors.items()):
                _logger.warning(
                    "detector {!r} in check {!r} raised {}",
                    detector.name,
                    self.name,
                    type(exc).__name__,
                )
                context.record_failure(
                    check=self.name,
                    detector=detector.name,
                    error=exc,
                    record_id=candidates[index].id,
                )
            for index, detections in enumerate(results):
                for detection in detections or []:
                    code = detection.code or self.default_code
                    if code not in self.rules:
                        raise ConfigurationError(
                            f"Detector {detector.name!r} emitted unknown code {code!r} for "
                            f"check {self.name!r}; known codes: {sorted(self.rules)}."
                        )
                    grouped[index].setdefault(code, []).append(detection)
        return grouped

    def decide(self, by_code: Mapping[str, Sequence[Detection]]) -> list[Finding]:
        """Apply the `min_detectors` vote and build one finding per code.

        Args:
            by_code: Detections grouped by finding code, from `collect`.

        Returns:
            Findings for the codes enough detectors agreed on.
        """
        findings: list[Finding] = []
        for code, detections in by_code.items():
            voters = {d.detector for d in detections}
            if len(voters) < self.min_detectors:
                continue
            findings.append(self.finding(code, detections))
        return findings

    def finding(self, code: str, detections: Sequence[Detection]) -> Finding:
        """Turn every detection that shares one code into a single finding.

        Severity, recommended action, and the user-facing message come from
        `rules[code]`. Evidence from the detections is combined, and their
        scores become the finding's confidence. Below `low_confidence` the
        severity drops one step.

        Args:
            code: Finding code to look up in `rules`, such as `"secret_detected"`.
            detections: Hits from one or more detectors that all used this
                code. Must be non-empty; their evidence is merged.

        Returns:
            One `Finding` for `code`. Its `check` field is this check's name.
        """
        rule = self.rules[code]
        confidence = combine_scores(detections)
        severity = rule.severity
        if confidence is not None and confidence < self.low_confidence:
            severity = severity.lower()
        return Finding(
            code=code,
            category=self.category,
            severity=severity,
            message=rule.message,
            evidence=merge_evidence(detections),
            check=self.name,
            recommended_action=rule.action,
            owasp=rule.owasp,
            confidence=confidence,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(detectors={self.detectors!r})"


def _run_detector(
    detector: Detector, candidates: Sequence[MemoryCandidate], context: CheckContext
) -> tuple[list[list[Detection] | None], dict[int, BaseException]]:
    """Ask one detector about a batch, keeping each failure on its own memory.

    Uses `detect_batch` when the detector has it and there is more than one
    candidate. If that call fails outright (rather than with a
    `BatchError`), the batch is retried one memory at a time, so the error
    is recorded against the memory that caused it.

    Returns:
        Per-candidate detections (`None` where it failed) and
        `{candidate index: exception}`.
    """
    batch = getattr(detector, "detect_batch", None)
    if callable(batch) and len(candidates) > 1:
        try:
            results = list(batch(candidates, context))
        except BatchError as exc:
            return list(exc.results), exc.errors
        except Exception as exc:
            _logger.debug(
                "detector {!r}: batch call raised {}; retrying one memory at a time",
                detector.name,
                type(exc).__name__,
            )
        else:
            if len(results) == len(candidates):
                return [list(found) for found in results], {}
            _logger.warning(
                "detector {!r}: detect_batch returned {} results for {} memories; "
                "retrying one memory at a time",
                detector.name,
                len(results),
                len(candidates),
            )
    out: list[list[Detection] | None] = []
    errors: dict[int, BaseException] = {}
    for index, candidate in enumerate(candidates):
        try:
            out.append(detector.detect(candidate, context))
        except Exception as exc:
            out.append(None)
            errors[index] = exc
    return out, errors


def combine_scores(detections: Sequence[Detection]) -> float | None:
    """Combine detector scores into one confidence.

    Each detector contributes its best score, clipped to [0, 1]. Agreement
    raises confidence: `1 - Π(1 - score)` over detectors.

    Args:
        detections: Hits that share one finding code.

    Returns:
        The combined confidence rounded to three places, or `None` when no
        detection carried a score.
    """
    best: dict[str, float] = {}
    for detection in detections:
        if detection.score is None:
            continue
        score = min(1.0, max(0.0, float(detection.score)))
        best[detection.detector] = max(best.get(detection.detector, 0.0), score)
    if not best:
        return None
    remaining = 1.0
    for score in best.values():
        remaining *= 1.0 - score
    return round(1.0 - remaining, 3)


def needs_corpus(detector: object) -> bool:
    """Whether a detector (or check) must wait until every record is read.

    Args:
        detector: A detector or a `MemoryCheck`.

    Returns:
        The object's own `needs_corpus` when it sets a bool. A
        `BaseDetector` that only implements `detect_text` is text-only
        (`False`): it cannot see other records. Any other object that does
        not declare the flag (a duck-typed `Detector` or a plain
        `MemoryCheck`) is `True`, so code that reads `context.existing`
        always sees the full scan.

    Raises:
        ConfigurationError: `detector` is a `BaseDetector` that overrides
            `detect` without declaring `needs_corpus`. Its author has to
            say which pass it belongs in.
    """
    flag = getattr(detector, "needs_corpus", None)
    if isinstance(flag, bool):
        return flag
    if isinstance(detector, BaseDetector):
        if type(detector).detect is BaseDetector.detect:
            return False
        raise ConfigurationError(
            f"{type(detector).__name__} overrides detect() but does not set needs_corpus. "
            "Set needs_corpus = True if it reads other records (context.existing, the "
            "corpus), or False if it only reads the candidate."
        )
    return True


def merge_evidence(detections: Sequence[Detection]) -> dict[str, object]:
    """Combine the evidence from several detections into one dictionary.

    Used when more than one detector reported the same finding code, so the
    report shows one finding with everyone's facts attached.

    List values are joined and duplicates are dropped. Any other value keeps
    the first one seen, so a later detector cannot overwrite it. Scores are
    collected separately, one per detector.

    Args:
        detections: The hits that belong to the same finding. Each one's
            `evidence` mapping is folded in.

    Returns:
        A dictionary stored on the finding. It always has:

        * `detectors` — sorted names of the detectors that voted.
        * `scores` — `{detector name: score}`, present only when at least
          one detection carried a score.

        Every other key comes from the detections' own evidence.
    """
    evidence: dict[str, object] = {"detectors": sorted({d.detector for d in detections})}
    scores = {d.detector: round(d.score, 4) for d in detections if d.score is not None}
    if scores:
        evidence["scores"] = scores
    for detection in detections:
        for key, value in detection.evidence.items():
            if key in ("detectors", "scores"):
                continue
            current = evidence.get(key)
            if current is None:
                evidence[key] = list(value) if isinstance(value, list | tuple | set) else value
            elif isinstance(current, list) and isinstance(value, list | tuple | set):
                merged = [*current, *(v for v in value if v not in current)]
                try:
                    evidence[key] = sorted(merged)
                except TypeError:
                    evidence[key] = merged
    return evidence


__all__ = [
    "BaseDetector",
    "BatchError",
    "Detection",
    "Detector",
    "SecurityCheck",
    "combine_scores",
    "merge_evidence",
    "needs_corpus",
]
