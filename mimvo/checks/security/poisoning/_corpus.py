"""Shared neighbour selection for corpus poisoning detectors."""

from __future__ import annotations

from ....context import CheckContext
from ....models.enums import MemoryStatus
from ....models.memory import MemoryCandidate, MemoryRecord


def active_neighbours(candidate: MemoryCandidate, context: CheckContext) -> list[MemoryRecord]:
    """Return active records in this scan batch, excluding the candidate.

    Args:
        candidate: The memory being scored.
        context: Must include `context.existing`, the materialised batch.

    Returns:
        Neighbours TrustRAG, hubness, and NLI may compare against.
    """
    return [
        record
        for record in context.existing
        if record.status == MemoryStatus.ACTIVE
        and (candidate.id is None or record.id != candidate.id)
    ]


__all__ = ["active_neighbours"]
