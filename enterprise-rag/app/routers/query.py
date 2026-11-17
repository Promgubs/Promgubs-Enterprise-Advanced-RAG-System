import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import User, get_db
from ..generation.answer import generate_answer
from ..retrieval.retriever import RetrievalOptions, retrieve
from ..schemas import CitationOut, QueryIn, QueryOut
from ..security import current_user

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryOut)
def ask(body: QueryIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    history = [t.model_dump() for t in body.history]
    opts = RetrievalOptions(top_k=body.top_k, mode=body.mode, rerank=body.rerank, expand=body.expand, as_of=body.as_of)
    result = retrieve(db, user, body.question, history, opts)

    t0 = time.perf_counter()
    answer = generate_answer(body.question, history, result.contexts)
    result.timings_ms["generate"] = round((time.perf_counter() - t0) * 1000, 1)

    return QueryOut(
        answer=answer.text,
        citations=[
            CitationOut(
                id=c.cid,
                document_id=c.document_id,
                title=c.title,
                file=c.file,
                version=c.version,
                location=c.location,
                section=c.section,
                url=c.url,
                snippet=(c.snippets[0] if c.snippets else c.text)[:300],
            )
            for c in answer.citations
        ],
        grounded=answer.grounded,
        standalone_query=result.standalone_query,
        expanded_queries=result.queries[1:],
        timings_ms=result.timings_ms,
    )
