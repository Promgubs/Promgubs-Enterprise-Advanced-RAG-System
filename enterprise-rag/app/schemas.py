from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=10)
    roles: list[str] = Field(default_factory=list)


class RolesUpdate(BaseModel):
    roles: list[str]


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: str
    roles: list[str]


class VersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    version: int
    status: str
    is_current: bool
    filename: str | None
    n_chunks: int
    error: str | None
    created_at: datetime
    activated_at: datetime | None
    superseded_at: datetime | None


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    title: str
    source_type: str
    source_uri: str | None
    allowed_roles: list[str]
    created_at: datetime
    versions: list[VersionOut] = Field(default_factory=list)


class WebIngest(BaseModel):
    url: str
    title: str | None = None
    allowed_roles: list[str] = Field(default_factory=list)


class AccessUpdate(BaseModel):
    allowed_roles: list[str]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    top_k: int = Field(6, ge=1, le=20)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rerank: bool = True
    expand: bool = True
    as_of: datetime | None = Field(None, description="Answer as the corpus stood at this time (point-in-time query)")


class CitationOut(BaseModel):
    id: str
    document_id: UUID
    title: str
    file: str | None
    version: int
    location: str | None
    section: str | None
    url: str | None
    snippet: str


class QueryOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    grounded: bool
    standalone_query: str
    expanded_queries: list[str]
    timings_ms: dict[str, float]
