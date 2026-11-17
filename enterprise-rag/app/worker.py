"""Ingestion worker. Postgres is the queue: rows with status='pending' are claimed with
SELECT ... FOR UPDATE SKIP LOCKED, so any number of workers can run side by side."""
import logging
import time
from datetime import timedelta

from sqlalchemy import and_, or_, select

from .config import get_settings
from .db import DocumentVersion, SessionLocal, utcnow
from .ingestion.pipeline import mark_failed, process_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("worker")


def claim_next():
    s = get_settings()
    stale = utcnow() - timedelta(minutes=s.job_stale_minutes)  # reclaim jobs from crashed workers
    with SessionLocal() as db:
        ver = db.scalars(
            select(DocumentVersion)
            .where(
                or_(
                    DocumentVersion.status == "pending",
                    and_(DocumentVersion.status == "processing", DocumentVersion.locked_at < stale),
                )
            )
            .order_by(DocumentVersion.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).first()
        if ver is None:
            return None
        ver.status = "processing"
        ver.locked_at = utcnow()
        db.commit()
        return ver.id


def main() -> None:
    poll = get_settings().worker_poll_seconds
    log.info("worker started")
    while True:
        version_id = claim_next()
        if version_id is None:
            time.sleep(poll)
            continue
        log.info("processing %s", version_id)
        try:
            process_version(version_id)
        except Exception as exc:
            log.exception("failed %s", version_id)
            mark_failed(version_id, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
