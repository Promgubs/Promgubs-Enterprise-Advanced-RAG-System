"""Ingestion of one document version: parse -> chunk -> embed -> index -> atomically publish.

Publishing order (so a failure never leaves a document with no searchable version):
 1. upsert new points, invisible (valid_from == valid_to)
 2. in one Postgres transaction: store parents/chunks, mark the version current, supersede the old one
 3. before commit: flip the Qdrant payloads (new on first, then old off)
 4. commit; on any error the new points are deleted and the old version is restored
"""
import hashlib
import logging
import uuid

from qdrant_client import models as qm
from sqlalchemy import delete, select

from ..config import get_settings
from ..db import Chunk, Document, DocumentVersion, ParentSection, SessionLocal, utcnow
from ..retrieval.embeddings import embed
from ..retrieval.vectorstore import FOREVER, get_store
from .chunking import chunk_document
from .parsers import fetch_web, parse_file

log = logging.getLogger(__name__)


def _embed_text(title: str, section: str, child: str) -> str:
    """Prefix the heading path so a chunk that says 'the limit is 5' still knows what it is about."""
    head = " > ".join(x for x in (title, section) if x)
    return f"{head}\n{child}"


def _set_status(version_id: uuid.UUID, status: str, **fields) -> None:
    with SessionLocal() as db:
        ver = db.get(DocumentVersion, version_id)
        ver.status = status
        for k, v in fields.items():
            setattr(ver, k, v)
        db.commit()


def mark_failed(version_id: uuid.UUID, error: str) -> None:
    _set_status(version_id, "failed", error=error[:2000])


def process_version(version_id: uuid.UUID) -> None:
    s = get_settings()
    store = get_store()

    with SessionLocal() as db:
        ver = db.get(DocumentVersion, version_id)
        doc = db.get(Document, ver.document_id)
        doc_id, doc_title, source_type, source_uri = doc.id, doc.title, doc.source_type, doc.source_uri
        version_no, storage_path, filename = ver.version, ver.storage_path, ver.filename
        current_hash = db.scalar(
            select(DocumentVersion.content_hash).where(
                DocumentVersion.document_id == doc_id, DocumentVersion.is_current
            )
        )
        # idempotent retries: clear anything a previous attempt left behind
        db.execute(delete(ParentSection).where(ParentSection.version_id == version_id))
        db.commit()
    store.delete_version(version_id)

    parsed = fetch_web(source_uri) if source_type == "web" else parse_file(storage_path, source_type, doc_title)
    content_hash = hashlib.sha256("\n".join(p.text for p in parsed.pages).encode()).hexdigest()
    if content_hash == current_hash:
        log.info("version %s identical to current; skipping", version_id)
        _set_status(version_id, "unchanged", content_hash=content_hash, error=None)
        return

    parents = chunk_document(
        parsed.pages, child_chars=s.child_chunk_chars, overlap=s.child_overlap_chars, parent_max=s.parent_max_chars
    )
    flat = [(p, child) for p in parents for child in p.children]
    if not flat:
        raise ValueError("Document produced no text chunks")

    title = doc_title if source_type != "web" or doc_title != source_uri else parsed.title
    embeddings = embed([_embed_text(title, p.section, child) for p, child in flat])

    now = utcnow()
    ts = int(now.timestamp())
    parent_rows, chunk_rows, points = [], [], []
    i = 0
    for p in parents:
        parent_id = uuid.uuid4()
        parent_rows.append(
            ParentSection(
                id=parent_id, version_id=version_id, ordinal=p.ordinal, section=p.section, location=p.location, text=p.text
            )
        )
        for j, child in enumerate(p.children):
            chunk_id, e = uuid.uuid4(), embeddings[i]
            i += 1
            chunk_rows.append(Chunk(id=chunk_id, version_id=version_id, parent_id=parent_id, ordinal=j, text=child))
            points.append(
                qm.PointStruct(
                    id=str(chunk_id),
                    vector={
                        "dense": e.dense,
                        "sparse": qm.SparseVector(indices=e.sparse_indices, values=e.sparse_values),
                    },
                    payload={
                        "doc_id": str(doc_id),
                        "version_id": str(version_id),
                        "version": version_no,
                        "parent_id": str(parent_id),
                        "title": title,
                        "file": filename or title,
                        "section": p.section,
                        "location": p.location,
                        "text": child,
                        "allowed_roles": [],  # filled in at publish time (roles may change while we work)
                        "is_current": False,
                        "valid_from": ts,
                        "valid_to": ts,  # empty validity interval: invisible until published
                    },
                )
            )

    store.upsert(points)

    prev_id = None
    try:
        with SessionLocal() as db:
            # serialize concurrent publishes for this document
            locked_doc = db.scalar(select(Document).where(Document.id == doc_id).with_for_update())
            ver = db.get(DocumentVersion, version_id)
            prev = db.scalar(
                select(DocumentVersion).where(DocumentVersion.document_id == doc_id, DocumentVersion.is_current)
            )
            db.add_all(parent_rows)
            db.flush()
            db.add_all(chunk_rows)
            if prev is not None:
                prev_id = prev.id
                prev.is_current = False
                prev.superseded_at = now
                db.flush()  # free the unique "one current version" slot before taking it
            ver.is_current = True
            ver.status = "ready"
            ver.error = None
            ver.n_chunks = len(points)
            ver.content_hash = content_hash
            ver.activated_at = now
            if source_type == "web" and locked_doc.title == locked_doc.source_uri:
                locked_doc.title = parsed.title
            db.flush()

            store.set_payload(
                {"is_current": True, "valid_to": FOREVER, "allowed_roles": list(locked_doc.allowed_roles)},
                version_id=version_id,
            )
            if prev_id is not None:
                store.set_payload({"is_current": False, "valid_to": ts}, version_id=prev_id)
            db.commit()
    except Exception:
        store.delete_version(version_id)
        if prev_id is not None:
            store.set_payload({"is_current": True, "valid_to": FOREVER}, version_id=prev_id)
        raise
    log.info("published %s v%s with %d chunks", doc_id, version_no, len(points))
