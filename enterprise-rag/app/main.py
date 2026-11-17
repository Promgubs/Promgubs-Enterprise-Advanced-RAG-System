import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import select

from .config import get_settings
from .db import SessionLocal, User, init_db
from .retrieval.vectorstore import get_store
from .routers import auth, documents, query
from .security import ADMIN_ROLE, hash_password

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("rag")


def _seed_admin() -> None:
    s = get_settings()
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.email == s.admin_email.lower())) is None:
            db.add(User(email=s.admin_email.lower(), password_hash=hash_password(s.admin_password), roles=[ADMIN_ROLE]))
            db.commit()
            log.info("seeded admin user %s", s.admin_email)


def _connect_qdrant(retries: int = 30) -> None:
    for attempt in range(retries):
        try:
            get_store().ensure_collection()
            return
        except Exception as exc:
            log.warning("waiting for Qdrant (%s)", exc)
            time.sleep(2)
    raise RuntimeError("Qdrant unavailable")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if s.app_env == "prod" and (s.jwt_secret == "replace-me" or s.admin_password.startswith("replace-me")):
        raise RuntimeError("Set JWT_SECRET and ADMIN_PASSWORD before running with APP_ENV=prod")
    init_db()
    _seed_admin()
    _connect_qdrant()
    if s.warmup_models:  # load BGE models now so the first request isn't slow
        from .retrieval.embeddings import embed
        from .retrieval.reranker import rerank

        embed(["warmup"])
        rerank("warmup", ["warmup"])
    yield


app = FastAPI(title="Enterprise RAG", version="1.0.0", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
