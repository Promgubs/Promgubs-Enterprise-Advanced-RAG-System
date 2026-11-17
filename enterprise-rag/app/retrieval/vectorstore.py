"""Qdrant wrapper: one collection, named dense + sparse vectors, payload-based filtering."""
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client import models as qm

FOREVER = 4_102_444_800  # 2100-01-01 UTC: `valid_to` of the current version


def build_filter(*, roles: list[str] | None, as_of: int | None = None, doc_ids: list[str] | None = None) -> qm.Filter:
    """The access-control + versioning filter applied INSIDE the vector search.

    roles=None means unrestricted (admin). An empty list matches nothing, so a bug that
    loses a user's roles fails closed rather than open.
    as_of=None means "current versions only"; an epoch timestamp means "as the corpus stood then".
    """
    must: list[qm.Condition] = []
    if roles is not None:
        must.append(qm.FieldCondition(key="allowed_roles", match=qm.MatchAny(any=roles)))
    if as_of is None:
        must.append(qm.FieldCondition(key="is_current", match=qm.MatchValue(value=True)))
    else:
        must.append(qm.FieldCondition(key="valid_from", range=qm.Range(lte=as_of)))
        must.append(qm.FieldCondition(key="valid_to", range=qm.Range(gt=as_of)))
    if doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)))
    return qm.Filter(must=must)


def _selector(doc_id=None, version_id=None) -> qm.Filter:
    must = []
    if doc_id is not None:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=str(doc_id))))
    if version_id is not None:
        must.append(qm.FieldCondition(key="version_id", match=qm.MatchValue(value=str(version_id))))
    if not must:
        raise ValueError("refusing to select every point")
    return qm.Filter(must=must)


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client = client
        self.collection = collection
        self.dim = dim

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        # Payload indexes make the pre-filter cheap on large collections.
        for field, schema in [
            ("allowed_roles", qm.PayloadSchemaType.KEYWORD),
            ("doc_id", qm.PayloadSchemaType.KEYWORD),
            ("version_id", qm.PayloadSchemaType.KEYWORD),
            ("is_current", qm.PayloadSchemaType.BOOL),
            ("valid_from", qm.PayloadSchemaType.INTEGER),
            ("valid_to", qm.PayloadSchemaType.INTEGER),
        ]:
            self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)

    def upsert(self, points: list[qm.PointStruct], batch: int = 128) -> None:
        for i in range(0, len(points), batch):
            self.client.upsert(self.collection, points=points[i : i + batch], wait=True)

    def search(
        self,
        *,
        dense: list[float],
        sparse: tuple[list[int], list[float]],
        mode: str,
        limit: int,
        flt: qm.Filter,
    ) -> list[qm.ScoredPoint]:
        """mode: dense | sparse | hybrid. Hybrid = dense + sparse fused server-side with RRF.

        The filter is repeated on every prefetch branch, so restricted chunks are never candidates.
        """
        indices, values = sparse
        if mode != "dense" and not indices:
            mode = "dense"  # query had no lexical tokens
        if mode == "dense":
            res = self.client.query_points(
                self.collection, query=dense, using="dense", query_filter=flt, limit=limit, with_payload=True
            )
        elif mode == "sparse":
            res = self.client.query_points(
                self.collection,
                query=qm.SparseVector(indices=indices, values=values),
                using="sparse",
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )
        else:
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    qm.Prefetch(query=dense, using="dense", filter=flt, limit=limit),
                    qm.Prefetch(
                        query=qm.SparseVector(indices=indices, values=values), using="sparse", filter=flt, limit=limit
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        return res.points

    def set_payload(self, payload: dict, *, doc_id=None, version_id=None) -> None:
        self.client.set_payload(self.collection, payload=payload, points=_selector(doc_id, version_id), wait=True)

    def delete_version(self, version_id) -> None:
        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=_selector(version_id=version_id)))

    def delete_document(self, doc_id) -> None:
        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=_selector(doc_id=doc_id)))


@lru_cache
def get_store() -> VectorStore:
    from ..config import get_settings

    s = get_settings()
    client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)
    return VectorStore(client, s.qdrant_collection, s.embedding_dim)
