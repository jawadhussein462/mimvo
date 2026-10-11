"""Hooks that run before a memory is written or after it is retrieved.

A store scan audits what is already saved. These helpers sit on the write
and retrieve paths so a bad record can be rejected before it lands, or
dropped before it is stuffed into a prompt.

`WriteGuard` is the write-path check. `RetrieveGuard` is the inference-time
filter (the role FilterRAG plays: it needs a query, so it does not run
on a query-free store scan).
"""

from __future__ import annotations

from .hook import RetrieveGuard, WriteDecision, WriteGuard

__all__ = ["RetrieveGuard", "WriteDecision", "WriteGuard"]
