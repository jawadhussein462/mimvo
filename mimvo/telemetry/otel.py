"""Optional OpenTelemetry tracer.

Install with `pip install "mimvo[otel]"`, then pass the tracer in:

    Mimvo(tracer=otel_tracer())
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from ..exceptions import IntegrationError
from . import Span, Tracer


class OtelTracer:
    """Send Mimvo spans to an OpenTelemetry tracer.

    Attributes:
        _tracer: The OpenTelemetry tracer. Created from the global provider
            when you do not pass one in.
    """

    def __init__(self, tracer: object | None = None) -> None:
        """Wrap an OpenTelemetry tracer, or the process-global one.

        Args:
            tracer: An OpenTelemetry tracer. `None` uses
                `opentelemetry.trace.get_tracer("mimvo")`.

        Raises:
            IntegrationError: The `opentelemetry` package is not installed.
        """
        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise IntegrationError(
                "opentelemetry is not installed. Install with: pip install 'mimvo[otel]'"
            ) from exc
        # Typed as Any: the API is only known when the optional extra is installed.
        self._tracer: Any = tracer if tracer is not None else trace.get_tracer("mimvo")

    @contextmanager
    def span(self, name: str, attributes: Mapping[str, object] | None = None) -> Iterator[Span]:
        """Open an OpenTelemetry span for the duration of a `with` block.

        Args:
            name: Span name recorded by OpenTelemetry.
            attributes: Metadata copied onto the span when it opens.

        Returns:
            A context manager that yields the OpenTelemetry span.
        """
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span


def otel_tracer(tracer: object | None = None) -> Tracer:
    """Build a tracer you can pass to `Mimvo(tracer=...)`.

    Args:
        tracer: An existing OpenTelemetry tracer. `None` uses the
            process-global tracer named `"mimvo"`.

    Returns:
        An `OtelTracer`.

    Raises:
        IntegrationError: OpenTelemetry is not installed.
    """
    return OtelTracer(tracer)


__all__ = ["OtelTracer", "otel_tracer"]
