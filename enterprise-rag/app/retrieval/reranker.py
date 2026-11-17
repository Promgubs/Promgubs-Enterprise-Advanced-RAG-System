import threading
from functools import lru_cache

from ..config import get_settings

_lock = threading.Lock()


@lru_cache
def _model():
    from FlagEmbedding import FlagReranker

    s = get_settings()
    return FlagReranker(s.reranker_model, use_fp16=s.use_fp16)


def rerank(query: str, passages: list[str]) -> list[float]:
    """Cross-encoder relevance scores in [0, 1], one per passage, same order."""
    if not passages:
        return []
    with _lock:
        scores = _model().compute_score([[query, p] for p in passages], normalize=True, batch_size=16, max_length=1024)
    if isinstance(scores, (int, float)):  # FlagReranker returns a bare float for a single pair
        scores = [scores]
    return [float(s) for s in scores]
