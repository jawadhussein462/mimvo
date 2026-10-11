"""Property tests: random models still match after a JSON round trip."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from mimvo import MemoryRecord
from mimvo.models.enums import MemoryStatus

_STATUS = st.sampled_from([s.value for s in MemoryStatus])


@given(
    content=st.text(max_size=80),
    status=_STATUS,
)
def test_record_json_round_trip(content, status):
    record = MemoryRecord(id="m", content=content, status=status)
    restored = MemoryRecord.model_validate_json(record.model_dump_json())
    assert restored.content == record.content
    assert restored.status == record.status
