"""Tests for the read-only scan sources: Chroma, Qdrant, pgvector, and Pinecone."""

from __future__ import annotations

import pytest

from mimvo.exceptions import ConfigurationError
from mimvo.sources.base import parse_datetime, quote_ident
from mimvo.sources.chroma import ChromaScanSource
from mimvo.sources.jsonl import JsonlScanSource
from mimvo.sources.pgvector import PgVectorScanSource
from mimvo.sources.pinecone import PineconeScanSource
from mimvo.sources.qdrant import QdrantScanSource
from tests.fakes import FakeChroma, FakePgConnection, FakePineconeIndex, FakeQdrant


def test_chroma_streams_and_samples():
    handle = FakeChroma()
    handle.upsert(
        ids=["a", "b", "c"],
        documents=["one", "two", "three"],
        metadatas=[{"source": "wiki", "user": "alice"}, {}, {}],
        embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    )
    src = ChromaScanSource(handle=handle, collection="mem")
    records = list(src.records(batch_size=2, sample=2))
    assert [r.id for r in records] == ["a", "b"]
    assert records[0].content == "one"
    assert records[0].embedding == [0.1, 0.2]
    assert records[0].metadata["source"] == "wiki"
    assert records[0].metadata["user"] == "alice"
    assert records[0].metadata["namespace"] == "mem"


def test_qdrant_is_read_only_scroll():
    client = FakeQdrant()
    client.upsert(
        "agent_memory",
        [
            {
                "id": "p1",
                "payload": {"content": "Alice prefers tea.", "user": "alice"},
                "vector": [1.0, 0.0],
            }
        ],
    )
    src = QdrantScanSource(client=client, collection="agent_memory")
    records = list(src.records())
    assert records[0].id == "p1" and "tea" in records[0].content
    assert records[0].embedding == [1.0, 0.0]
    assert records[0].metadata["user"] == "alice"


def test_pgvector_selects_and_honours_sample_and_read_only():
    conn = FakePgConnection(
        [
            ("m1", "Alice prefers tea."),
            ("m2", "Bob sits in Berlin."),
            ("m3", "Carol likes coffee."),
        ]
    )
    src = PgVectorScanSource(
        connection=conn,
        table="memories",
        text_column="content",
        id_column="id",
    )
    records = list(src.records(batch_size=2, sample=2))
    assert [r.id for r in records] == ["m1", "m2"]
    assert any("READ ONLY" in s.upper() for s in conn.statements)


def test_pgvector_rejects_unsafe_identifiers():
    with pytest.raises(ConfigurationError, match="identifier"):
        PgVectorScanSource(
            connection=FakePgConnection([]),
            table="memories; drop table memories",
            text_column="content",
        )
    assert quote_ident("public.memories") == '"public"."memories"'


def test_pinecone_lists_then_fetches_metadata():
    index = FakePineconeIndex(
        {
            "v1": {"metadata": {"text": "Alice prefers tea."}, "values": [0.2, 0.8]},
            "v2": {"metadata": {"content": "Bob sits in Berlin."}, "values": [0.1, 0.9]},
        }
    )
    src = PineconeScanSource(handle=index, namespace="prod")
    records = {r.id: r for r in src.records(batch_size=1, sample=2)}
    assert records["v1"].content == "Alice prefers tea."
    assert records["v2"].content == "Bob sits in Berlin."
    assert records["v1"].embedding == [0.2, 0.8]
    assert records["v1"].metadata["namespace"] == "prod"
    assert index.list_calls[0]["namespace"] == "prod"
    assert index.fetch_calls


def test_missing_connection_args():
    with pytest.raises(ConfigurationError):
        ChromaScanSource()
    with pytest.raises(ConfigurationError):
        QdrantScanSource(collection="x")
    with pytest.raises(ConfigurationError):
        PgVectorScanSource(table="t", text_column="c")
    with pytest.raises(ConfigurationError):
        PineconeScanSource()


def test_jsonl_keeps_embedding_metadata_and_timestamp(tmp_path):
    path = tmp_path / "mem.jsonl"
    path.write_text(
        '{"id": "m1", "content": "Alice prefers tea.", "embedding": [0.1, 0.2],'
        ' "created_at": "2024-01-02T03:04:05Z", "source": "wiki", "user": "ada"}\n',
        encoding="utf-8",
    )
    (record,) = list(JsonlScanSource(path).records())
    assert record.embedding == [0.1, 0.2]
    assert record.metadata["source"] == "wiki"
    assert record.metadata["user"] == "ada"
    assert record.created_at == parse_datetime("2024-01-02T03:04:05Z")
