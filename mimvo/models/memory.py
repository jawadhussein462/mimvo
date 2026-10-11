"""The two memory objects: `MemoryCandidate` during a scan, `MemoryRecord` in a store."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import MemoryStatus


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryCandidate(BaseModel):
    """One memory as a check sees it during a scan.

    Attributes:
        content: The text to inspect.
        metadata: Store fields copied onto the candidate. Checks may read
            them. Typical keys: `source`, `user`, `namespace`, timestamps.
            Do not put secrets here if a report will be forwarded.
        id: Record id from the store. `None` when the input was only a string.
        derived_from: Ids of records this one was built from, if the store
            tracks that.
        created_at: When the record was created. Defaults to now, in UTC.
        embedding: The vector stored with this memory, when the scanner
            fetched it. Needed by hubness, TrustRAG, and embedding-text
            consistency. `None` when the store did not return one.
    """

    model_config = ConfigDict(extra="forbid")

    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    id: str | None = Field(default=None, description="Record id, when the store has one.")
    derived_from: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    embedding: list[float] | None = Field(
        default=None,
        description="Stored embedding, when the scanner fetched it.",
    )


class MemoryRecord(BaseModel):
    """One memory as it sits in a store.

    Attributes:
        id: Store id. Required.
        content: The text that will be scanned.
        status: `active`, `revoked`, or `quarantined`. Defaults to active.
            TrustRAG skips neighbours that are not active.
        created_at: When the record was first stored. Defaults to now, in UTC.
            Prefer the store's timestamp when the scanner can read one.
        updated_at: When the record last changed. Defaults to now, in UTC.
        derived_from: Ids this record was built from.
        metadata: Extra store fields (source, user, namespace, provenance).
            Empty by default.
        embedding: The vector the store holds for this text. `None` when
            the scanner did not fetch embeddings.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    derived_from: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    embedding: list[float] | None = Field(
        default=None,
        description="Stored embedding, when the scanner fetched it.",
    )


__all__ = ["MemoryCandidate", "MemoryRecord"]
