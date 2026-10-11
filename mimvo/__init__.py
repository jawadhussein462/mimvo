"""Scan agent memory for poisoned facts, hidden instructions, and leaked secrets.

From the command line:

    mimvo scan chroma --path ./chroma_db --collection agent_memory
    mimvo scan jsonl export.jsonl --report report.html --json findings.json

From Python, create a `Mimvo` and call `scan`. The scan reads records and
returns a `ScanReport`. It does not change the store.
"""

from __future__ import annotations

from ._version import __version__
from .client import AsyncMimvo, Mimvo
from .config import Config
from .exceptions import (
    BackendError,
    ConfigurationError,
    MimvoError,
)
from .models import (
    Action,
    Category,
    Finding,
    MemoryCandidate,
    MemoryRecord,
    MemoryStatus,
    Risk,
    ScanError,
    ScanReport,
    Severity,
)

__all__ = [
    "Action",
    "AsyncMimvo",
    "BackendError",
    "Category",
    "Config",
    "ConfigurationError",
    "Finding",
    "MemoryCandidate",
    "MemoryRecord",
    "MemoryStatus",
    "Mimvo",
    "MimvoError",
    "Risk",
    "ScanError",
    "ScanReport",
    "Severity",
    "__version__",
]
