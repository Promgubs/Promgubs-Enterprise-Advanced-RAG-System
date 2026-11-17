import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..config import get_settings
from ..db import Document, DocumentVersion, User, get_db
from ..ingestion.parsers import assert_public_url
from ..retrieval.vectorstore import get_store
from ..schemas import AccessUpdate, DocumentOut, WebIngest
from ..security import (
    ADMIN_ROLE,
    can_access,
    current_user,
    effective_roles,
    is_admin,
    require_admin,
    require_editor,
)

router = APIRouter(prefix="/documents", tags=["documents"])
settings = get_settings()

EXT_TYPES = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx"}
MAGIC = {"pdf": b"%PDF", "docx": b"PK\x03\x04", "xlsx": b"PK\x03\x04"}


# ---- helpers ----------------------------------------------------------------------------------


def _source_type(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in EXT_TYPES:
        raise HTTPException(415, "Supported files: .pdf, .docx, .xlsx")
    return EXT_TYPES[ext]


def _clean_roles(raw: list[str]) -> list[str]:
    return sorted({r.strip().lower() for r in raw if r and r.strip()})


def _resolve_roles(requested: list[str], user: User) -> list[str]:
    """Default to the uploader's own roles (fail closed). Non-admins can only grant roles they hold."""
    roles = _clean_roles(requested) or _clean_roles(user.roles) or [ADMIN_ROLE]
    if not is_admin(user) and not set(roles) <= set(effective_roles(user)):
        raise HTTPException(403, "You can only grant roles you hold")
    return roles


def _get_doc(db: Session, doc_id: uuid.UUID, user: User) -> Document:
    doc = db.scalar(select(Document).where(Document.id == doc_id).options(selectinload(Document.versions)))
    if doc is None or not can_access(user, doc.allowed_roles):
        raise HTTPException(404, "Document not found")  # 404, not 403: don't reveal that it exists
    return doc


def _require_manage(doc: Document, user: User) -> None:
    if not (is_admin(user) or doc.owner_id == user.id):
        raise HTTPException(403, "Only the owner or an admin can change this document")


def _create_version(db: Session, doc: Document, user: User, filename: str | None = None) -> DocumentVersion:
    db.execute(select(Document.id).where(Document.id == doc.id).with_for_update())  # serialize numbering
    last = db.scalar(select(func.max(DocumentVersion.version)).where(DocumentVersion.document_id == doc.id)) or 0
    ver = DocumentVersion(document_id=doc.id, version=last + 1, filename=filename, uploaded_by=user.id)
    db.add(ver)
    db.flush()
    return ver


def _store_upload(file: UploadFile, doc_id: uuid.UUID, version: int, source_type: str) -> str:
    ext = Path(file.filename or "").suffix.lower()
    dest_dir = Path(settings.storage_dir) / str(doc_id) / f"v{version}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"original{ext}"  # never use the client-supplied name on disk
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    with dest.open("wb") as out:
        head = file.file.read(4)
        if not head.startswith(MAGIC[source_type]):
            raise HTTPException(415, f"File content does not look like a .{source_type}")
        out.write(head)
        size = len(head)
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
            out.write(chunk)
    return str(dest)


# ---- endpoints --------------------------------------------------------------------------------


@router.post("", response_model=DocumentOut, status_code=202)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    allowed_roles: str = Form("", description="Comma-separated roles; defaults to your own roles"),
    db: Session = Depends(get_db),
    user: User = Depends(require_editor),
):
    """Upload a PDF, Word or Excel file. Returns immediately; the worker indexes it in the background."""
    source_type = _source_type(file.filename)
    roles = _resolve_roles(allowed_roles.split(","), user)
    doc = Document(
        title=(title or Path(file.filename).stem).strip(), source_type=source_type, allowed_roles=roles, owner_id=user.id
    )
    db.add(doc)
    db.flush()
    ver = _create_version(db, doc, user, filename=file.filename)
    try:
        ver.storage_path = _store_upload(file, doc.id, ver.version, source_type)
        db.commit()
    except Exception:
        db.rollback()
        shutil.rmtree(Path(settings.storage_dir) / str(doc.id), ignore_errors=True)
        raise
    return doc


@router.post("/web", response_model=DocumentOut, status_code=202)
def ingest_web(body: WebIngest, db: Session = Depends(get_db), user: User = Depends(require_editor)):
    try:
        assert_public_url(body.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    doc = Document(
        title=(body.title or body.url).strip(),
        source_type="web",
        source_uri=body.url,
        allowed_roles=_resolve_roles(body.allowed_roles, user),
        owner_id=user.id,
    )
    db.add(doc)
    db.flush()
    _create_version(db, doc, user)
    db.commit()
    return doc


@router.post("/{doc_id}/versions", response_model=DocumentOut, status_code=202)
def add_version(
    doc_id: uuid.UUID,
    file: UploadFile | None = File(None, description="Omit for web documents; the page is re-fetched"),
    db: Session = Depends(get_db),
    user: User = Depends(require_editor),
):
    """Upload a new version. Identical content is detected by the worker and ignored."""
    doc = _get_doc(db, doc_id, user)
    _require_manage(doc, user)
    if doc.source_type == "web":
        _create_version(db, doc, user)
        db.commit()
    else:
        if file is None:
            raise HTTPException(422, "A file is required for this document")
        if _source_type(file.filename) != doc.source_type:
            raise HTTPException(415, f"New version must be a .{doc.source_type} file")
        ver = _create_version(db, doc, user, filename=file.filename)
        try:
            ver.storage_path = _store_upload(file, doc.id, ver.version, doc.source_type)
            db.commit()
        except Exception:
            db.rollback()
            shutil.rmtree(Path(settings.storage_dir) / str(doc.id) / f"v{ver.version}", ignore_errors=True)
            raise
    db.expire(doc, ["versions"])
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    stmt = select(Document).options(selectinload(Document.versions)).order_by(Document.created_at.desc())
    if not is_admin(user):
        stmt = stmt.where(Document.allowed_roles.overlap(effective_roles(user)))
    return db.scalars(stmt.limit(limit).offset(offset)).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return _get_doc(db, doc_id, user)


@router.get("/{doc_id}/versions/{version}/file")
def download_version(
    doc_id: uuid.UUID, version: int, db: Session = Depends(get_db), user: User = Depends(current_user)
):
    doc = _get_doc(db, doc_id, user)
    ver = next((v for v in doc.versions if v.version == version), None)
    if ver is None or not ver.storage_path or not Path(ver.storage_path).is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(ver.storage_path, filename=ver.filename)


@router.patch("/{doc_id}/access", response_model=DocumentOut)
def update_access(
    doc_id: uuid.UUID, body: AccessUpdate, db: Session = Depends(get_db), user: User = Depends(require_editor)
):
    doc = _get_doc(db, doc_id, user)
    _require_manage(doc, user)
    doc.allowed_roles = _resolve_roles(body.allowed_roles, user)
    db.commit()
    # Propagate to the vector index. If this fails the Postgres re-check in the retriever still
    # protects narrowed access; widened access just takes effect after a retry.
    get_store().set_payload({"allowed_roles": doc.allowed_roles}, doc_id=doc.id)
    return doc


@router.delete("/{doc_id}", status_code=204, dependencies=[Depends(require_admin)])
def delete_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    get_store().delete_document(doc_id)  # vectors first, so nothing searchable outlives the row
    db.delete(doc)
    db.commit()
    shutil.rmtree(Path(settings.storage_dir) / str(doc_id), ignore_errors=True)
