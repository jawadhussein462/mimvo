"""Tests for coercion, enums, and serialization."""

from __future__ import annotations

import pytest

from mimvo import (
    Action,
    MemoryCandidate,
    MemoryRecord,
    Severity,
)
from mimvo._coerce import coerce_candidate
from mimvo.exceptions import ConfigurationError


def test_severity_and_action_ordering():
    assert Severity.CRITICAL.rank > Severity.HIGH.rank > Severity.INFO.rank
    assert Action.DELETE.precedence > Action.QUARANTINE.precedence > Action.REVIEW.precedence


def test_record_json_round_trip():
    r = MemoryRecord(id="m", content="c")
    dumped = r.model_dump_json()
    restored = MemoryRecord.model_validate_json(dumped)
    assert restored.id == r.id
    assert restored.content == "c"


def test_coerce_candidate_from_string_and_dict():
    from_str = coerce_candidate("hello")
    assert from_str.content == "hello"
    from_dict = coerce_candidate({"content": "hi"})
    assert from_dict.content == "hi"
    already = MemoryCandidate(content="x")
    assert coerce_candidate(already) is already


@pytest.mark.parametrize("bad", [123, 4.5, object()])
def test_coerce_candidate_rejects_bad_types(bad):
    with pytest.raises(ConfigurationError):
        coerce_candidate(bad)
