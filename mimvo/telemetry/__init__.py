"""Logging and tracing.

Logs go through loguru. Mimvo does not add or remove log sinks. The
application configures loguru.

Tracing is optional. Pass a tracer into `Mimvo`. The default tracer
does nothing. `otel_tracer()` in `mimvo.telemetry.otel` builds one
backed by OpenTelemetry.

Spans get counts and operation names. They do not get the memory text or
secret values.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# Span names.
SPAN_SCAN = "mimvo.scan"

# Attribute keys (safe metadata only).
ATTR_FINDING_COUNT = "mimvo.finding_count"
ATTR_OPERATION = "mimvo.operation"


def get_logger(name: str = "mimvo") -> Logger:
    """Return a loguru logger tagged with `name`.

    Args:
        name: Value stored on the `name` field of each log record. The
            default is `"mimvo"`.

    Returns:
        A loguru logger. This function does not add a log file or a console
        sink. The application does that.
    """
    return logger.bind(name=name)


@runtime_checkable
class Span(Protocol):
    """One tracing span. The engine calls `set_attribute` while a scan runs."""

    def set_attribute(self, key: str, value: object) -> None:
        """Attach one piece of metadata to this span.

        Args:
            key: Attribute name.
            value: A count or a short label. Not the memory text.
        """
        ...


@runtime_checkable
class Tracer(Protocol):
    """Starts spans. Pass an implementation to `Mimvo(tracer=...)`.

    `NullTracer` does nothing. `otel_tracer()` sends spans to OpenTelemetry.
    """

    def span(
        self, name: str, attributes: Mapping[str, object] | None = None
    ) -> AbstractContextManager[Span]:
        """Open a span for the duration of a `with` block.

        Args:
            name: Span name, such as `"mimvo.scan"`.
            attributes: Metadata set when the span opens. `None` means none.

        Returns:
            A context manager. Entering it yields a `Span`.
        """
        ...


class _NullSpan:
    def set_attribute(self, key: str, value: object) -> None:
        return None


class NullTracer:
    """Tracer used when the caller does not pass one.

    `span` yields an object whose `set_attribute` ignores every call.
    """

    @contextmanager
    def span(self, name: str, attributes: Mapping[str, object] | None = None) -> Iterator[Span]:
        """Yield a span that discards attributes.

        Args:
            name: Ignored. Accepted so this matches the `Tracer` protocol.
            attributes: Ignored.

        Returns:
            A context manager that yields a no-op span.
        """
        yield _NullSpan()


__all__ = [
    "ATTR_FINDING_COUNT",
    "ATTR_OPERATION",
    "SPAN_SCAN",
    "NullTracer",
    "Span",
    "Tracer",
    "get_logger",
]
