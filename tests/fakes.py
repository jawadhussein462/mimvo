"""In-process stand-ins for store SDKs used by scan-source tests.

Each class copies the methods the scanner calls (`get`, `scroll`, `list`,
`fetch`) and keeps data in a dict. No real database is contacted.
"""

from __future__ import annotations


class FakeChroma:
    """Stand-in for a Chroma collection.

    `get` returns ids, documents, and metadatas. `add` skips ids that are
    already stored. `upsert` replaces them.
    """

    def __init__(self):
        self._d = {}

    def add(self, ids, documents, metadatas=None, embeddings=None):
        for i, mid in enumerate(ids):
            if mid in self._d:
                continue
            self._d[mid] = {
                "id": mid,
                "document": documents[i],
                "metadata": (metadatas or [None] * len(ids))[i],
                "embedding": (embeddings or [None] * len(ids))[i],
            }

    def upsert(self, ids, documents, metadatas=None, embeddings=None):
        for mid in ids:
            self._d.pop(mid, None)
        self.add(ids, documents, metadatas, embeddings)

    def get(self, ids=None, limit=None, offset=None, include=None):
        if ids is not None:
            items = [self._d[i] for i in ids if i in self._d]
        else:
            start = offset or 0
            items = list(self._d.values())[start : start + limit if limit else None]
        out = {
            "ids": [v["id"] for v in items],
            "documents": [v["document"] for v in items],
            "metadatas": [v["metadata"] for v in items],
        }
        if include is None or "embeddings" in include:
            out["embeddings"] = [v.get("embedding") for v in items]
        return out


def _pid(point):
    return str(point["id"] if isinstance(point, dict) else point.id)


class FakeQdrant:
    """Stand-in for a Qdrant client.

    `upsert` stores dict points. `scroll` returns one page and the next
    offset, or `None` when the page is the last.
    """

    def __init__(self):
        self._d = {}

    def upsert(self, collection_name, points):
        for point in points:
            payload = point["payload"] if isinstance(point, dict) else point.payload
            vector = (
                point.get("vector") if isinstance(point, dict) else getattr(point, "vector", None)
            )
            self._d[_pid(point)] = {
                "id": _pid(point),
                "payload": dict(payload),
                "vector": vector,
            }

    def scroll(self, collection_name, scroll_filter=None, limit=10, offset=None, **kw):
        items = list(self._d.values())
        start = offset or 0
        page = items[start : start + limit]
        nxt = start + limit if start + limit < len(items) else None
        return page, nxt


class FakePgCursor:
    def __init__(self, rows):
        self._all = list(rows)
        self._rows = list(rows)
        self._i = 0
        self.sql = None
        self.params = None

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params
        rows = list(self._all)
        if params:
            rows = rows[: int(params[0])]
        self._rows = rows
        self._i = 0

    def fetchmany(self, n):
        chunk = self._rows[self._i : self._i + n]
        self._i += n
        return chunk

    def __iter__(self):
        return iter(self._rows[self._i :])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakePgConnection:
    """Stand-in for a psycopg connection.

    `execute` records the SQL. `cursor` returns a cursor over the rows
    passed to the constructor.
    """

    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append(sql)

    def cursor(self, name=None):
        return FakePgCursor(self.rows)


class FakePineconeIndex:
    """Stand-in for a Pinecone index.

    `list` yields pages of ids. `fetch` returns the vectors for those ids.
    """

    def __init__(self, vectors):
        self._d = dict(vectors)
        self.list_calls = []
        self.fetch_calls = []

    def list(self, namespace="", limit=100):
        self.list_calls.append({"namespace": namespace, "limit": limit})
        ids = list(self._d)
        for i in range(0, max(len(ids), 1) if ids else 0, limit):
            yield ids[i : i + limit]

    def fetch(self, ids, namespace=""):
        self.fetch_calls.append(list(ids))
        return {"vectors": {i: self._d[i] for i in ids if i in self._d}}
