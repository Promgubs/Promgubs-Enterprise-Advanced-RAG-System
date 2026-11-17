"""Proves the access-control and versioning filters work inside the vector search.
Uses Qdrant's in-memory mode, so no server is needed (skipped if qdrant-client isn't installed)."""
import uuid

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("pydantic_settings")

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client import models as qm  # noqa: E402

from app.retrieval.vectorstore import FOREVER, VectorStore, build_filter  # noqa: E402

DIM = 4
HR_VEC = [1.0, 0.0, 0.0, 0.0]


def point(i, vec, roles, current=True, valid_to=None):
    return qm.PointStruct(
        id=str(uuid.UUID(int=i)),
        vector={"dense": vec, "sparse": qm.SparseVector(indices=[1, 2], values=[1.0, 1.0])},
        payload={
            "text": f"chunk {i}",
            "doc_id": f"d{i}",
            "allowed_roles": roles,
            "is_current": current,
            "valid_from": 0,
            "valid_to": valid_to if valid_to is not None else (FOREVER if current else 10),
        },
    )


@pytest.fixture
def store():
    s = VectorStore(QdrantClient(":memory:"), "test", DIM)
    s.ensure_collection()
    s.upsert(
        [
            point(1, HR_VEC, ["hr"]),  # exact match for the query vector, but HR only
            point(2, [0.9, 0.1, 0, 0], ["finance"]),
            point(3, [0.5, 0.5, 0, 0], ["all"]),
            point(4, HR_VEC, ["all"], current=False),  # superseded version
        ]
    )
    return s


def ids(points):
    return {p.payload["text"] for p in points}


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_user_never_sees_restricted_chunks(store, mode):
    flt = build_filter(roles=["finance", "all"])
    got = store.search(dense=HR_VEC, sparse=([1, 2], [1.0, 1.0]), mode=mode, limit=10, flt=flt)
    assert ids(got) == {"chunk 2", "chunk 3"}  # chunk 1 (HR) is the best vector match and still absent


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_no_roles_matches_nothing(store, mode):
    flt = build_filter(roles=[])
    assert store.search(dense=HR_VEC, sparse=([1, 2], [1.0, 1.0]), mode=mode, limit=10, flt=flt) == []


def test_admin_sees_all_current_but_not_superseded(store):
    flt = build_filter(roles=None)
    got = store.search(dense=HR_VEC, sparse=([1, 2], [1.0, 1.0]), mode="hybrid", limit=10, flt=flt)
    assert ids(got) == {"chunk 1", "chunk 2", "chunk 3"}


def test_as_of_returns_the_old_version(store):
    flt = build_filter(roles=None, as_of=5)  # chunk 4 was valid in [0, 10)
    got = store.search(dense=HR_VEC, sparse=([1, 2], [1.0, 1.0]), mode="dense", limit=10, flt=flt)
    assert "chunk 4" in ids(got)
    flt_later = build_filter(roles=None, as_of=50)
    got_later = store.search(dense=HR_VEC, sparse=([1, 2], [1.0, 1.0]), mode="dense", limit=10, flt=flt_later)
    assert "chunk 4" not in ids(got_later)


def test_payload_update_changes_access(store):
    store.set_payload({"allowed_roles": ["finance"]}, doc_id="d1")
    flt = build_filter(roles=["hr", "all"])
    got = store.search(dense=HR_VEC, sparse=([1, 2], [1.0, 1.0]), mode="dense", limit=10, flt=flt)
    assert "chunk 1" not in ids(got)
