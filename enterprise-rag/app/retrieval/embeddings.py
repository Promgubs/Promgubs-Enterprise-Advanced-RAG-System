"""BGE-M3 produces dense and sparse (lexical-weight) vectors in one forward pass."""
import threading
from dataclasses import dataclass
from functools import lru_cache

from ..config import get_settings

_lock = threading.Lock()


@dataclass
class Embedded:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


@lru_cache
def _model():
    from FlagEmbedding import BGEM3FlagModel

    s = get_settings()
    return BGEM3FlagModel(s.embedding_model, use_fp16=s.use_fp16)


def embed(texts: list[str], batch_size: int = 16, max_length: int = 1024) -> list[Embedded]:
    if not texts:
        return []
    with _lock:  # one forward pass at a time; avoids GPU/CPU thrash under concurrent requests
        out = _model().encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
    results = []
    for dense, lexical in zip(out["dense_vecs"], out["lexical_weights"]):
        results.append(
            Embedded(
                dense=dense.tolist(),
                sparse_indices=[int(token_id) for token_id in lexical],
                sparse_values=[float(w) for w in lexical.values()],
            )
        )
    return results
