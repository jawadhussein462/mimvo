"""Errors Mimvo raises.

A scan returns a `ScanReport`. It does not raise because a memory looked
suspicious. These exceptions are for bad setup, a broken backend, or a
programming mistake.
"""

from __future__ import annotations


class MimvoError(Exception):
    """Base class for every error this library raises."""


class ConfigurationError(MimvoError):
    """The caller passed settings that cannot work.

    Raised for an empty detector list, a bad threshold, a missing API key,
    or a scan source that is not a source, a `.all()` object, or an iterable.
    """


class BackendError(MimvoError):
    """A store or hosted API failed while Mimvo was calling it.

    The message names the URL or operation. It does not include the memory
    text that was sent.
    """


class IntegrationError(MimvoError):
    """An optional extra is not installed or is set up wrong.

    Raised when a store client, a model library, or OpenTelemetry is
    needed and its extra is missing. The message names the
    `pip install "mimvo[<extra>]"` command.
    """


class CheckError(MimvoError):
    """A check crashed.

    The scan engine does not raise this: a crashing check or detector is
    recorded in `ScanReport.errors` and the scan is marked incomplete.
    `ScanReport.raise_for_errors()` raises it for callers who prefer an
    exception.
    """


__all__ = [
    "BackendError",
    "CheckError",
    "ConfigurationError",
    "IntegrationError",
    "MimvoError",
]
