"""The two objects most people import: `Mimvo` and `AsyncMimvo`.

`Mimvo()` is ready to scan. It builds the default checks for you.
The sync class and the async class are separate, so `scan` is never
sometimes a coroutine and sometimes a normal function.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Iterable, Sequence
from typing import Any

from ._coerce import coerce_record
from .checks import default_checks
from .checks.base import MemoryCheck
from .config import Config
from .engine import Evaluator
from .exceptions import ConfigurationError
from .models.results import ScanReport
from .telemetry import Tracer


class _ClientBase:
    """Shared setup for the sync and async clients.

    Builds the config, the check list, and the engine. `scan` lives here
    because it does no I/O of its own: the caller supplies the records.
    """

    def __init__(
        self,
        *,
        fail_closed: bool | None = None,
        checks: Sequence[Any] | None = None,
        config: Config | None = None,
        tracer: Tracer | None = None,
        use_default_checks: bool = True,
    ) -> None:
        """Build a client and the checks it will run.

        Args:
            fail_closed: Whether an incomplete scan (a check or detector
                raised) counts as a failure: the CLI exits `2` and the guards
                block the affected records. A failure is never a finding
                either way; it is listed in `ScanReport.errors`. `None` keeps
                the value already on `config` (the default config uses `True`).
            checks: Extra checks, or replacements for a default check of the
                same name. Each item must be a `MemoryCheck`.
            config: Full settings object. Fields you also pass as arguments
                (`fail_closed`) override this object.
            tracer: Where scan spans are sent. `None` uses a tracer that
                records nothing.
            use_default_checks: When `True`, start from secrets, injection,
                and poisoning. When `False`, run only the checks you passed.

        Raises:
            ConfigurationError: An item in `checks` is not a `MemoryCheck`.
        """
        base_config = config or Config()
        updates: dict[str, Any] = {}
        if fail_closed is not None:
            updates["fail_closed"] = fail_closed
        self.config: Config = base_config.model_copy(update=updates) if updates else base_config

        # A user check named like a default ("injection", "secrets", ...) replaces
        # it, so a configured ``InjectionCheck(detectors=[...])`` slots into the
        # default pipeline instead of running alongside the heuristic one.
        resolved_checks: list[MemoryCheck] = list(default_checks()) if use_default_checks else []
        for chk in checks or []:
            if not isinstance(chk, MemoryCheck):
                raise ConfigurationError(f"{type(chk).__name__} is not a MemoryCheck.")
            position = next(
                (i for i, existing in enumerate(resolved_checks) if existing.name == chk.name),
                None,
            )
            if position is None:
                resolved_checks.append(chk)
            else:
                resolved_checks[position] = chk

        self._evaluator = Evaluator(
            config=self.config,
            checks=resolved_checks,
            tracer=tracer,
        )

    def _scan(self, source: Iterable[Any], *, query: str | None = None) -> ScanReport:
        """Find poisoned facts, hidden instructions, and leaked secrets.

        The store is not modified.

        Args:
            source: Where the records come from. Accepted shapes:

                * a scan source, which is read with `.records()`
                * an object with `.all()`, such as some vector-store wrappers
                * an iterable of `MemoryRecord` objects or dicts

                Records are streamed: each one is checked as it arrives and
                only a packed copy (text, metadata, 4-byte vector) is kept
                for the detectors that compare records with each other.
            query: The question that retrieved this batch, when you have one.
                Most checks ignore it. Cluster detectors can use it.

        Returns:
            A `ScanReport` with one entry per problem found. An empty
            `findings` list means nothing was flagged.

        Raises:
            ConfigurationError: `source` is not a scan source, an object with
                `.all()`, or an iterable of records.
        """
        records_fn = getattr(source, "records", None)
        all_fn = getattr(source, "all", None)
        if callable(records_fn):
            items: Iterable[Any] = records_fn()
        elif callable(all_fn):
            items = all_fn()
        elif isinstance(source, Iterable) and not isinstance(source, str | bytes | dict):
            items = source
        else:
            raise ConfigurationError(
                f"Cannot scan {type(source).__name__}: pass a scan source, "
                "an object with .all(), or an iterable of records."
            )
        records = (coerce_record(item) for item in items)
        return self._evaluator.scan(records, query=query)


class Mimvo(_ClientBase):
    """Synchronous client. Call `scan` and wait for the report.

    Example:
        Scan a list of records and print the summary::

            report = Mimvo().scan(records)
            print(report)
    """

    def scan(self, source: Iterable[Any], *, query: str | None = None) -> ScanReport:
        """Find poisoned facts, hidden instructions, and leaked secrets.

        The store is not modified.

        Args:
            source: Where the records come from. Accepted shapes:

                * a scan source, which is read with `.records()`
                * an object with `.all()`, such as some vector-store wrappers
                * an iterable of `MemoryRecord` objects or dicts

                Records are streamed: each one is checked as it arrives and
                only a packed copy (text, metadata, 4-byte vector) is kept
                for the detectors that compare records with each other.
            query: The question that retrieved this batch, when you have one.
                Most checks ignore it. Cluster detectors can use it.

        Returns:
            A `ScanReport` with one entry per problem found. An empty
            `findings` list means nothing was flagged.

        Raises:
            ConfigurationError: `source` is not a scan source, an object with
                `.all()`, or an iterable of records.
        """
        return self._scan(source, query=query)


class AsyncMimvo(_ClientBase):
    """Asynchronous client. Same arguments as `Mimvo`.

    `scan` runs the scan in a worker thread (`asyncio.to_thread`), so the
    event loop keeps serving other tasks while detectors run and while a
    scan source fetches pages from its store. Async iterables (an async
    generator, or a source whose `records()` is one) are read on the loop
    first, then scanned in the thread.
    """

    async def scan(
        self, source: Iterable[Any] | AsyncIterable[Any], *, query: str | None = None
    ) -> ScanReport:
        """Scan records and return the report. See `Mimvo.scan`.

        Args:
            source: A scan source, an object with `.all()`, an iterable of
                records or dicts, or an async iterable of them.
            query: The retrieval question for this batch, if you have one.

        Returns:
            The same `ScanReport` the synchronous client would return.
        """
        records_fn = getattr(source, "records", None)
        if callable(records_fn) and not isinstance(source, AsyncIterable):
            # Called once here; a sync iterator is handed to the thread as is.
            source = records_fn()
        if isinstance(source, AsyncIterable):
            source = [item async for item in source]
        return await asyncio.to_thread(self._scan, source, query=query)


__all__ = [
    "AsyncMimvo",
    "Mimvo",
]
