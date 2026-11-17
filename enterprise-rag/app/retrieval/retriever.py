"""Retrieval pipeline:

condense follow-up -> expand (rewrites + HyDE) -> embed (dense+sparse) -> ACL-filtered search per query
-> RRF across queries -> cross-encoder rerank -> Postgres ACL re-check -> expand to parent sections.
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import Document, ParentSection
from ..security import can_access, effective_roles, is_admin
from .embeddings import embed
from .fusion import rrf_fuse
from .query_rewrite import condense_question, expand_query
from .reranker import rerank
from .vectorstore import build_filter, get_store

log = logging.getLogger(__name__)


@dataclass
class RetrievalOptions:
    top_k: int = 6
    candidates: int = 50
    mode: str = "hybrid"  # dense | sparse | hybrid
    rerank: bool = True
    expand: bool = True
    as_of: datetime | None = None


@dataclass
class Hit:
    """A child chunk that survived retrieval."""

    point_id: str
    score: float
    payload: dict

    @property
    def text(self) -> str:
        return self.payload["text"]

    @property
    def rerank_text(self) -> str:
        head = " > ".join(x for x in (self.payload.get("title"), self.payload.get("section")) if x)
        return f"{head}\n{self.text}" if head else self.text


@dataclass
class Context:
    """A parent section handed to the LLM, with citation metadata."""

    cid: str  # "S1", "S2", ...
    document_id: uuid.UUID
    title: str
    file: str | None
    version: int
    location: str | None
    section: str | None
    url: str | None
    text: str
    snippets: list[str]
    score: float


@dataclass
class RetrievalResult:
    standalone_query: str
    queries: list[str]
    hits: list[Hit]
    contexts: list[Context]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _epoch(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def retrieve(db: Session, user, question: str, history: list[dict], opts: RetrievalOptions) -> RetrievalResult:
    settings = get_settings()
    store = get_store()
    timings: dict[str, float] = {}

    def lap(name: str, t0: float) -> None:
        timings[name] = round((time.perf_counter() - t0) * 1000, 1)

    # 1. query understanding
    t0 = time.perf_counter()
    standalone = condense_question(history, question) if history else question
    queries = [standalone]
    if opts.expand:
        exp = expand_query(standalone)
        queries += exp.rewrites
        if exp.hypothetical:
            queries.append(exp.hypothetical)
    lap("query_rewrite", t0)

    # 2. embed all variants in one batch
    t0 = time.perf_counter()
    embeddings = embed(queries)
    lap("embed", t0)

    # 3. search each variant, with access control and version filter applied inside the search
    t0 = time.perf_counter()
    roles = None if is_admin(user) else effective_roles(user)
    flt = build_filter(roles=roles, as_of=_epoch(opts.as_of))
    per_query = [
        store.search(
            dense=e.dense,
            sparse=(e.sparse_indices, e.sparse_values),
            mode=opts.mode,
            limit=opts.candidates,
            flt=flt,
        )
        for e in embeddings
    ]
    lap("search", t0)

    # 4. fuse across query variants
    by_id = {}
    for points in per_query:
        for p in points:
            by_id.setdefault(str(p.id), p)
    fused = rrf_fuse([[str(p.id) for p in points] for points in per_query])[: opts.candidates]
    candidates = [Hit(pid, score, by_id[pid].payload) for pid, score in fused]

    # 5. rerank
    t0 = time.perf_counter()
    if opts.rerank and len(candidates) > 1:
        for hit, score in zip(candidates, rerank(standalone, [c.rerank_text for c in candidates])):
            hit.score = score
        candidates.sort(key=lambda h: h.score, reverse=True)
    hits = candidates[: opts.top_k]
    lap("rerank", t0)

    # 6. Postgres is the authority on access. The Qdrant payload is a copy that could be stale
    #    (for example if a role change failed to propagate), so re-check before anything is shown.
    doc_ids = {uuid.UUID(h.payload["doc_id"]) for h in hits}
    docs = {d.id: d for d in db.scalars(select(Document).where(Document.id.in_(doc_ids)))} if doc_ids else {}
    allowed_hits = []
    for h in hits:
        doc = docs.get(uuid.UUID(h.payload["doc_id"]))
        if doc is not None and can_access(user, doc.allowed_roles):
            allowed_hits.append(h)
        else:
            log.warning("dropped hit %s: failed Postgres ACL re-check", h.point_id)
    hits = allowed_hits

    # 7. children -> parent sections (deduplicated, in rank order, within a character budget)
    t0 = time.perf_counter()
    parent_order: list[str] = []
    for h in hits:
        if h.payload["parent_id"] not in parent_order:
            parent_order.append(h.payload["parent_id"])
    parents = {}
    if parent_order:
        rows = db.scalars(select(ParentSection).where(ParentSection.id.in_([uuid.UUID(p) for p in parent_order])))
        parents = {str(p.id): p for p in rows}

    contexts: list[Context] = []
    used = 0
    for pid in parent_order:
        parent = parents.get(pid)
        if parent is None:
            continue
        if contexts and used + len(parent.text) > settings.context_char_budget:
            break
        matched = [h for h in hits if h.payload["parent_id"] == pid]
        first = matched[0]
        doc = docs[uuid.UUID(first.payload["doc_id"])]
        version = first.payload["version"]
        used += len(parent.text)
        contexts.append(
            Context(
                cid=f"S{len(contexts) + 1}",
                document_id=doc.id,
                title=doc.title,
                file=first.payload.get("file"),
                version=version,
                location=parent.location,
                section=parent.section or None,
                url=doc.source_uri if doc.source_type == "web" else f"/documents/{doc.id}/versions/{version}/file",
                text=parent.text,
                snippets=[h.text for h in matched],
                score=first.score,
            )
        )
    lap("parents", t0)

    return RetrievalResult(standalone, queries, hits, contexts, timings)
