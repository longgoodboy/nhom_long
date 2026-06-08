"""Task 9 - hybrid retrieval pipeline with local fallback."""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, score_threshold: float = SCORE_THRESHOLD, use_reranking: bool = True) -> list[dict]:
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)
    merged = rerank_rrf([dense, sparse], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"
    final = rerank(query, merged, top_k=top_k, method=RERANK_METHOD) if use_reranking else merged[:top_k]
    if not final or float(final[0].get("score", 0.0)) < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        return fallback[:top_k]
    return final[:top_k]


if __name__ == "__main__":
    print(retrieve("hinh phat ma tuy", 3))
