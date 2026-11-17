"""PostgreSQL schema: users/roles, documents, versions, parent sections and chunks.

Postgres is the source of truth. Qdrant holds a searchable copy of each chunk plus
payload fields (allowed_roles, is_current, valid_from/valid_to) used for filtering.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=10, max_overflow=10)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_db():
    with SessionLocal() as session:
        yield session


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    roles: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(10))  # pdf | docx | xlsx | web
    source_uri: Mapped[str | None] = mapped_column(Text, default=None)  # web documents only
    allowed_roles: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        order_by="DocumentVersion.version",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentVersion(Base):
    """Also the job queue: the worker claims rows with status='pending'."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version"),
        Index("ix_versions_queue", "status", "created_at"),
        # at most one current version per document, enforced by the database
        Index("uq_one_current_version", "document_id", unique=True, postgresql_where=text("is_current")),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str | None] = mapped_column(Text, default=None)
    storage_path: Mapped[str | None] = mapped_column(Text, default=None)
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    status: Mapped[str] = mapped_column(String(12), default="pending")  # pending|processing|ready|unchanged|failed
    error: Mapped[str | None] = mapped_column(Text, default=None)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    document: Mapped[Document] = relationship(back_populates="versions")


class ParentSection(Base):
    """A heading-delimited section. Small child chunks are retrieved; the parent is sent to the LLM."""

    __tablename__ = "parent_sections"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(Text, default="")
    location: Mapped[str | None] = mapped_column(Text, default=None)
    text: Mapped[str] = mapped_column(Text)


class Chunk(Base):
    """Child chunk. `id` is also the Qdrant point id."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    parent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("parent_sections.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


def init_db() -> None:
    Base.metadata.create_all(engine)  # use Alembic migrations in production
