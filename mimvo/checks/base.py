"""The `MemoryCheck` base class and the contract the engine runs it through.

A check looks at one memory and returns findings. The simplest check sets
`name` and implements `check`. Adding a check does not require editing the
engine: the engine only uses the methods defined here.

A scan reads the store once, as a stream, then runs a second pass over the
packed corpus for work that compares records with each other. A check says
which passes it needs (`phases`), records what it saw on each pass
(`observe`), and turns those observations into findings once its last pass
is done (`conclude`). The defaults wrap `check`, so a plain check never
has to think about passes; `SecurityCheck` overrides them to split its
detectors across the passes and vote over both.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from ..context import CheckContext
from ..models.finding import Finding
from ..models.memory import MemoryCandidate


class Phase(StrEnum):
    """The two passes of a scan.

    Attributes:
        STREAM: While records are read, one at a time. Only the record
            itself (and records already read) can be seen.
        CORPUS: After every record has been read. `context.existing` and
            the corpus hold the whole scan.
    """

    STREAM = "stream"
    CORPUS = "corpus"


class MemoryCheck:
    """One named check over a single memory.

    Set `name` and implement `check`. The built-in security checks subclass
    `SecurityCheck`, which subclasses this.

    Attributes:
        name: Label stored on each finding this check emits. The client uses
            it to replace a default check: a custom check named `"injection"`
            takes the place of the built-in injection check.
        needs_corpus: Whether `check` reads other records. `True` (or
            unset) runs it once every record has been read, so
            `context.existing` is the whole scan; `False` runs it while
            records stream in.
    """

    name: str = "base"
    needs_corpus: bool | None = None

    def check(self, candidate: MemoryCandidate, context: CheckContext) -> list[Finding]:
        """Inspect one memory and return every problem found.

        Args:
            candidate: The memory being scanned. Read `candidate.content`.
            context: Scan settings, the clock, neighbouring records, and the
                optional retrieval query.

        Returns:
            Findings for this memory. An empty list means this check found
            nothing. Never return `None`.

        Raises:
            NotImplementedError: Always, unless a subclass implements this.
        """
        raise NotImplementedError

    # -- engine contract ---------------------------------------------------------

    def phases(self) -> frozenset[Phase]:
        """Which passes this check runs in.

        Returns:
            `{CORPUS}` unless `needs_corpus` is `False`, then `{STREAM}`.
            A check that returns both is observed on each pass and
            concluded after the corpus pass.
        """
        return frozenset({Phase.STREAM if self.needs_corpus is False else Phase.CORPUS})

    def observe(self, candidate: MemoryCandidate, context: CheckContext, phase: Phase) -> object:
        """Look at one memory during one pass.

        The engine keeps whatever this returns and hands every observation
        for the record to `conclude` after the check's last pass.

        Args:
            candidate: The memory being scanned.
            context: Scan context.
            phase: The pass being run. Always one of `phases()`.

        Returns:
            An observation only this check needs to understand. The default
            is the list of findings `check` returned. An empty (falsy)
            observation means "nothing seen": the engine may drop it rather
            than keep it until the corpus pass.
        """
        return self.check(candidate, context)

    def observe_batch(
        self,
        candidates: Sequence[MemoryCandidate],
        context: CheckContext,
        phase: Phase,
    ) -> list[object]:
        """Look at several memories during one pass.

        The engine hands each pass to checks in batches of
        `Config.batch_size` records, so a check whose work batches well,
        such as a model forward pass, can do it once per batch.

        Args:
            candidates: The memories, in scan order.
            context: Scan context.
            phase: The pass being run.

        Returns:
            One observation per candidate, in the same order. The default
            calls `observe` on each.
        """
        return [self.observe(candidate, context, phase) for candidate in candidates]

    def conclude(self, observations: Sequence[object]) -> list[Finding]:
        """Turn one record's observations into findings.

        Args:
            observations: What `observe` returned for this record, one per
                pass the check ran in, in pass order.

        Returns:
            The record's findings. The default joins the finding lists.
            The engine stamps this check's `name` on any finding whose
            `check` is unset.
        """
        findings: list[Finding] = []
        for observation in observations:
            if isinstance(observation, list):
                findings.extend(f for f in observation if isinstance(f, Finding))
        return findings


__all__ = ["MemoryCheck", "Phase"]
