"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Score threshold cho cross-encoder rerank (range thường 0..10+ với MS-MARCO).
# < threshold → coi hybrid thất bại, fallback PageIndex.
SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"

# Lấy candidate pool RỘNG để strong BM25 hit không bị chìm khi RRF + rerank.
# top_k=5 → candidate=25 (đủ rộng nhưng cross-encoder vẫn chạy nhanh).
CANDIDATE_MULT = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Flow:
        1. semantic_search + lexical_search song song (mỗi nhánh top_k*CANDIDATE_MULT)
        2. RRF merge → giữ top_k*CANDIDATE_MULT candidates
        3. Cross-encoder rerank trên candidate pool rộng → trả về top_k
        4. Nếu top1 score < threshold → fallback PageIndex
    """
    candidate_k = top_k * CANDIDATE_MULT

    try:
        dense_results = semantic_search(query, top_k=candidate_k)
    except Exception as e:
        print(f"⚠ Semantic search failed: {e}")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=candidate_k)
    except Exception as e:
        print(f"⚠ Lexical search failed: {e}")
        sparse_results = []

    if not dense_results and not sparse_results:
        return pageindex_search(query, top_k=top_k)

    # Step 2 — RRF merge. Giữ TOÀN BỘ candidate pool (không cắt top_k*2)
    # để rerank ở step 3 thấy được mọi strong BM25 hit.
    merged = rerank_rrf([dense_results, sparse_results], top_k=candidate_k)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3 — Rerank
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except Exception as e:
            print(f"⚠ Reranking failed, using RRF order: {e}")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # Step 4 — Threshold fallback
    # Lưu ý: test_fallback_logic_exists truyền score_threshold=0.99 nên sẽ
    # chắc chắn fallback (giả định cross-encoder score < 0.99).
    if not final_results or final_results[0]["score"] < score_threshold:
        print(f"  ⚠ Hybrid score thấp ({final_results[0]['score'] if final_results else 'rỗng'}). Fallback → PageIndex")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception as e:
            print(f"  ⚠ PageIndex fallback failed: {e}")

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, top_k=3)
        for r in results:
            print(f"[{r['source']} - {r['score']:.3f}] {r['content'][:100]}...")

        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
