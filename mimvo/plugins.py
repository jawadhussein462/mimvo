"""Find checks and scan sources other packages provide, through entry points.

A package adds to Mimvo by declaring entry points in its `pyproject.toml`,
the way pytest and flake8 plugins do:

    [project.entry-points."mimvo.checks"]
    freshness = "acme_mimvo.checks:FreshnessCheck"

    [project.entry-points."mimvo.sources"]
    weaviate = "acme_mimvo.sources:WeaviateScanSource"

Once that package is installed, `mimvo scan ... --check freshness` adds the
check and `mimvo scan plugin weaviate --option url=...` reads the store.
From Python, `load_check("freshness")` and `load_source("weaviate", ...)`
return the same objects. Detectors ship inside a check: a plugin check can
be a `SecurityCheck` with any detectors it likes.

Nothing is imported until a plugin is asked for by name, so a broken
plugin cannot break a scan that does not use it.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from .checks.base import MemoryCheck
from .exceptions import ConfigurationError, IntegrationError
from .sources.base import ScanSource

CHECKS_GROUP = "mimvo.checks"
SOURCES_GROUP = "mimvo.sources"


def _entry_points(group: str) -> dict[str, EntryPoint]:
    """Installed entry points in `group`, by name."""
    return {ep.name: ep for ep in entry_points(group=group)}


def available_checks() -> list[str]:
    """Names of the checks installed packages provide.

    Returns:
        Sorted entry-point names in the `mimvo.checks` group.
    """
    return sorted(_entry_points(CHECKS_GROUP))


def available_sources() -> list[str]:
    """Names of the scan sources installed packages provide.

    Returns:
        Sorted entry-point names in the `mimvo.sources` group.
    """
    return sorted(_entry_points(SOURCES_GROUP))


def _load(group: str, name: str, kind: str) -> tuple[EntryPoint, Any]:
    points = _entry_points(group)
    point = points.get(name)
    if point is None:
        known = ", ".join(sorted(points)) or "none installed"
        raise ConfigurationError(f"No {kind} plugin named {name!r} (available: {known}).")
    try:
        return point, point.load()
    except Exception as exc:
        raise IntegrationError(
            f"{kind} plugin {name!r} ({point.value}) failed to import: {exc}"
        ) from exc


def load_check(name: str) -> MemoryCheck:
    """Build the check a plugin registered under `name`.

    Args:
        name: Entry-point name in the `mimvo.checks` group.

    Returns:
        A check instance. A class or factory is called with no arguments;
        an instance is used as it is.

    Raises:
        ConfigurationError: No plugin has that name, or it does not give a
            `MemoryCheck`.
        IntegrationError: The plugin failed to import or to build.
    """
    point, target = _load(CHECKS_GROUP, name, "check")
    if not isinstance(target, MemoryCheck):
        if not callable(target):
            raise ConfigurationError(f"check plugin {name!r} ({point.value}) is not callable.")
        try:
            target = target()
        except Exception as exc:
            raise IntegrationError(f"check plugin {name!r} failed to build: {exc}") from exc
    if not isinstance(target, MemoryCheck):
        raise ConfigurationError(
            f"check plugin {name!r} ({point.value}) gave {type(target).__name__}, "
            "not a MemoryCheck."
        )
    return target


def load_source(name: str, options: Mapping[str, Any] | None = None) -> ScanSource:
    """Build the scan source a plugin registered under `name`.

    Args:
        name: Entry-point name in the `mimvo.sources` group.
        options: Keyword arguments for the source's constructor. The CLI
            passes each `--option key=value` as a string.

    Returns:
        An object with `records(batch_size=, sample=)`.

    Raises:
        ConfigurationError: No plugin has that name, the options do not
            fit its constructor, or it does not give a scan source.
        IntegrationError: The plugin failed to import.
    """
    point, target = _load(SOURCES_GROUP, name, "source")
    if not callable(target):
        raise ConfigurationError(f"source plugin {name!r} ({point.value}) is not callable.")
    try:
        source = target(**dict(options or {}))
    except TypeError as exc:
        raise ConfigurationError(f"source plugin {name!r}: {exc}") from exc
    if not isinstance(source, ScanSource):
        raise ConfigurationError(
            f"source plugin {name!r} ({point.value}) gave {type(source).__name__}, "
            "which has no records(batch_size=, sample=) method."
        )
    return source


__all__ = [
    "CHECKS_GROUP",
    "SOURCES_GROUP",
    "available_checks",
    "available_sources",
    "load_check",
    "load_source",
]
