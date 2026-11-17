# Enterprise RAG

Search-engine-grade retrieval over PDF, Word, Excel and web documents, with hybrid search, reranking,
query expansion, role-based access, document versioning and cited answers.

```
upload ─► Postgres (queue) ─► worker: parse ─► chunk (parent/child) ─► BGE-M3 dense+sparse ─► Qdrant
                                                                                          │
question ─► condense follow-up ─► expand (rewrites + HyDE) ─► ACL-filtered hybrid search ─┘
        ─► RRF across variants ─► bge-reranker ─► Postgres ACL re-check ─► parent sections ─► LLM ─► validated [S#] citations
```

## Run it

```bash
cp .env.example .env        # set JWT_SECRET, ADMIN_PASSWORD, ANTHROPIC_API_KEY
docker compose up --build   # first boot downloads ~4.5 GB of BGE models; API docs at http://localhost:8000/docs
```

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login -d "username=admin@example.com&password=YOUR_PASSWORD" | jq -r .access_token)
AUTH="Authorization: Bearer $TOKEN"

# create users and roles (admin only)
curl -H "$AUTH" -H 'Content-Type: application/json' localhost:8000/users \
  -d '{"email":"hr@example.com","password":"a-long-password","roles":["hr","editor"]}'

# ingest
curl -H "$AUTH" -F file=@hr_policy.pdf -F allowed_roles=hr,legal localhost:8000/documents
curl -H "$AUTH" -H 'Content-Type: application/json' localhost:8000/documents/web \
  -d '{"url":"https://example.com/handbook","allowed_roles":["all"]}'
curl -H "$AUTH" -F file=@hr_policy_v2.pdf localhost:8000/documents/<doc_id>/versions   # new version

# ask (add "history" for follow-ups, "as_of": "2026-03-01T00:00:00Z" for point-in-time answers)
curl -H "$AUTH" -H 'Content-Type: application/json' localhost:8000/query \
  -d '{"question":"How many days of annual leave do we get?"}'
```

Local development: start Postgres and Qdrant (`docker compose up postgres qdrant`), then
`pip install torch && pip install -r requirements.txt`, `uvicorn app.main:app --reload`, and `python -m app.worker`.

## Design decisions

| Concern | Approach |
|---|---|
| Hybrid search | BGE-M3 gives dense and sparse vectors; Qdrant fuses both server-side with RRF. |
| Access control | `allowed_roles` filter runs **inside** every vector-search branch (pre-filter), then Postgres re-checks access before anything is returned. Empty role list matches nothing. Test: `tests/test_rbac_qdrant.py`. |
| Versioning | Every upload is a version. Chunks carry `is_current`, `valid_from`, `valid_to`; publishing flips them without deleting history. `as_of` queries use the validity interval. A partial unique index guarantees one current version per document. Unchanged content is detected by hash and skipped. |
| Chunking | Heading-aware sections → parent/child. Children (~900 chars) are searched, parents (up to ~3500 chars) go to the LLM. Excel rows become `Header: value; ...` lines with sheet and row-range locations. |
| Query understanding | Follow-ups are rewritten to standalone queries; each query gets 3 rewrites plus a HyDE passage; all variants are searched and fused. LLM failures fall back to the plain query. |
| Citations | Sources are numbered `S1..Sn`; the model cites inline; markers that don't match a provided source are stripped and reported. |
| Prompt injection | Retrieved text is delimited and declared untrusted in the system prompt. This reduces risk; it doesn't remove it. |
| Web ingestion | SSRF guard (public IPs only, every redirect hop re-checked). JS-rendered pages need Playwright. |
| Queue | Postgres `FOR UPDATE SKIP LOCKED`; run any number of workers; stale jobs are reclaimed. |

## Evaluation (the part worth showing)

1. Ingest a corpus, then write `eval/golden_set.jsonl` (see `eval/golden_set.example.jsonl`): 50 to 100 questions, each with
   the file and a distinctive phrase that must appear in a retrieved chunk.
2. `docker compose exec api python -m eval.run_eval --golden eval/golden_set.jsonl --k 5 --faithfulness`
3. Paste the table into this README:

| Configuration | Recall@5 | MRR | nDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| dense only | | | | | |
| keyword only | | | | | |
| hybrid (RRF) | | | | | |
| hybrid + rerank | | | | | |
| hybrid + rerank + expansion | | | | | |

## Tests

`pytest` runs chunking, fusion and parser tests with no services. `test_rbac_qdrant.py` runs against Qdrant's in-memory
mode and is skipped if `qdrant-client` isn't installed.

## Known gaps / next steps

- Scanned PDFs need OCR; legacy `.xls` and `.doc` aren't supported.
- Add Alembic migrations (the app uses `create_all`), rate limiting, an audit log of queries per user, and a reconciler job that compares Postgres and Qdrant.
- Pin versions after your first successful build (`pip freeze`); FlagEmbedding and transformers move quickly.
- Tune `candidates`, chunk sizes and a minimum rerank score against your golden set rather than by feel.
