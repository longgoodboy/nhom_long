"""
Task 9 - Retrieval pipeline.
"""

from __future__ import annotations

from .rag_utils import expand_query
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    expanded_query = expand_query(query)
    dense_results = semantic_search(expanded_query, top_k=max(top_k * 2, 5))
    sparse_results = lexical_search(expanded_query, top_k=max(top_k * 2, 5))

    merged = rerank_rrf([dense_results, sparse_results], top_k=max(top_k * 2, 5))
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(expanded_query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    if not final_results or float(final_results[0].get("score", 0.0)) < score_threshold:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    print(retrieve("hinh phat ma tuy", top_k=3))
