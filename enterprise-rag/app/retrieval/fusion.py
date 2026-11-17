from collections import defaultdict


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: merge several ranked id lists into one, best first.

    score(id) = sum over lists of 1 / (k + rank). Scale-free, so it can merge
    dense, sparse and multi-query result lists whose raw scores are not comparable.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
