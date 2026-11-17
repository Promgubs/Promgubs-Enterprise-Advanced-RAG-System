"""Retrieval evaluation and ablation.

    python -m eval.run_eval --golden eval/golden_set.jsonl --k 5 [--faithfulness]

Golden set (JSONL), one question per line:
  {"question": "...", "relevant": [{"file": "hr_policy.pdf", "contains": "distinctive phrase"}]}
A retrieved child chunk satisfies a `relevant` entry if its file matches and its text contains
the phrase (case-insensitive). Pick phrases that appear in only one place in the corpus.
Metrics are computed on child chunks, before parent expansion and before the LLM.
"""
import argparse
import json
import math
import re
import statistics
import time
from types import SimpleNamespace

from app.db import SessionLocal
from app.generation.answer import generate_answer
from app.generation.llm import get_llm
from app.retrieval.retriever import RetrievalOptions, retrieve

CONFIGS = {
    "dense only": dict(mode="dense", rerank=False, expand=False),
    "keyword only (BGE-M3 sparse)": dict(mode="sparse", rerank=False, expand=False),
    "hybrid (RRF)": dict(mode="hybrid", rerank=False, expand=False),
    "hybrid + rerank": dict(mode="hybrid", rerank=True, expand=False),
    "hybrid + rerank + expansion": dict(mode="hybrid", rerank=True, expand=True),
}


def satisfies(hit, rel: dict) -> bool:
    file_ok = rel["file"].lower() in {str(hit.payload.get("file", "")).lower(), str(hit.payload.get("title", "")).lower()}
    return file_ok and rel.get("contains", "").lower() in hit.text.lower()


def score_question(hits, relevants: list[dict], k: int) -> tuple[float, float, float]:
    """Returns (recall@k, reciprocal rank, nDCG@k). Each relevant entry can be 'found' only once."""
    remaining = set(range(len(relevants)))
    gains: list[int] = []
    first_rank = None
    for rank, hit in enumerate(hits[:k], start=1):
        found = next((i for i in sorted(remaining) if satisfies(hit, relevants[i])), None)
        if found is None:
            gains.append(0)
        else:
            remaining.discard(found)
            gains.append(1)
            first_rank = first_rank or rank
    recall = (len(relevants) - len(remaining)) / len(relevants)
    rr = 1 / first_rank if first_rank else 0.0
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevants), k)))
    return recall, rr, dcg / idcg


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1)]


JUDGE_SYSTEM = (
    "You grade whether an answer is supported by the provided sources. Split the answer into factual claims "
    'and count how many are directly supported. Reply with JSON only: {"supported": <int>, "total": <int>}.'
)


def faithfulness(question: str, answer_text: str, contexts) -> float | None:
    sources = "\n\n".join(f"[{c.cid}] {c.text}" for c in contexts)
    raw = get_llm().complete(JUDGE_SYSTEM, f"Sources:\n{sources}\n\nAnswer:\n{answer_text}", max_tokens=100)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    data = json.loads(match.group(0))
    return data["supported"] / data["total"] if data.get("total") else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--faithfulness", action="store_true", help="also LLM-judge answer faithfulness (costs tokens)")
    args = ap.parse_args()

    with open(args.golden) as f:
        golden = [json.loads(line) for line in f if line.strip()]
    admin = SimpleNamespace(roles=["admin"])  # evaluate retrieval quality without ACL effects

    rows = []
    with SessionLocal() as db:
        retrieve(db, admin, golden[0]["question"], [], RetrievalOptions(top_k=args.k, expand=False))  # warm up models
        for name, cfg in CONFIGS.items():
            metrics, latencies = [], []
            for item in golden:
                t0 = time.perf_counter()
                result = retrieve(db, admin, item["question"], [], RetrievalOptions(top_k=args.k, **cfg))
                latencies.append((time.perf_counter() - t0) * 1000)
                metrics.append(score_question(result.hits, item["relevant"], args.k))
            rows.append(
                (
                    name,
                    statistics.mean(m[0] for m in metrics),
                    statistics.mean(m[1] for m in metrics),
                    statistics.mean(m[2] for m in metrics),
                    percentile(latencies, 50),
                    percentile(latencies, 95),
                )
            )

        k = args.k
        print(f"\n{len(golden)} questions, k={k}\n")
        print(f"| Configuration | Recall@{k} | MRR | nDCG@{k} | p50 ms | p95 ms |")
        print("|---|---|---|---|---|---|")
        for name, rec, mrr, ndcg, p50, p95 in rows:
            print(f"| {name} | {rec:.3f} | {mrr:.3f} | {ndcg:.3f} | {p50:.0f} | {p95:.0f} |")

        if args.faithfulness:
            scores = []
            for item in golden:
                result = retrieve(db, admin, item["question"], [], RetrievalOptions(top_k=args.k))
                answer = generate_answer(item["question"], [], result.contexts)
                s = faithfulness(item["question"], answer.text, result.contexts) if result.contexts else None
                if s is not None:
                    scores.append(s)
            if scores:
                print(f"\nFaithfulness (hybrid + rerank + expansion, LLM judge): {statistics.mean(scores):.3f} over {len(scores)} answers")


if __name__ == "__main__":
    main()
