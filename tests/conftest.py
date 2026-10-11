"""Shared pytest fixtures.

Object builders live in `tests/factories.py` so other tests can import
them without going through this file.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mimvo import Mimvo
from mimvo.context import CheckContext
from tests.factories import make_context


@pytest.fixture
def now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def guard() -> Mimvo:
    return Mimvo()


@pytest.fixture
def context() -> CheckContext:
    return make_context()
