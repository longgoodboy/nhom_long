"""
Task 7 - Reranking.
"""

from __future__ import annotations

from .rag_utils import cosine_similarity, expand_query, hash_embed, keyword_overlap_score, metadata_type_boost


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Local heuristic reranker that approximates relevance scoring.
    """
    expanded_query = expand_query(query)
    query_embedding = hash_embed(expanded_query)
    reranked: list[dict] = []
    for candidate in candidates:
        content = candidate.get("content", "")
        base_score = float(candidate.get("score", 0.0))
        relevance = cosine_similarity(query_embedding, hash_embed(content))
        overlap = keyword_overlap_score(expanded_query, content)
        boost = metadata_type_boost(query, candidate.get("metadata", {}))
        score = 0.45 * base_score + 0.35 * relevance + 0.15 * overlap + 0.05 * boost
        item = candidate.copy()
        item["score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    def candidate_embedding(idx: int):
        return candidates[idx].get("embedding") or hash_embed(candidates[idx]["content"]).tolist()

    while remaining and len(selected) < top_k:
        best_idx = remaining[0]
        best_score = float("-inf")
        for idx in remaining:
            relevance = cosine_similarity(hash_embed(candidates[idx]["content"]), hash_embed(" ".join(map(str, query_embedding))))
            redundancy = 0.0
            if selected:
                redundancy = max(
                    cosine_similarity(
                        hash_embed(candidates[idx]["content"]),
                        hash_embed(candidates[sel]["content"]),
                    )
                    for sel in selected
                )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    merged = []
    for key, score in sorted(scores.items(), key=lambda entry: entry[1], reverse=True)[:top_k]:
        item = content_map[key].copy()
        item["score"] = float(score)
        merged.append(item)
    return merged


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    if method != "cross_encoder":
        raise ValueError(f"Unsupported rerank method in this implementation: {method}")
    return rerank_cross_encoder(query, candidates, top_k)


if __name__ == "__main__":
    sample = [
        {"content": "Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi ma tuy", "score": 0.6, "metadata": {}},
        {"content": "Python programming basics", "score": 0.4, "metadata": {}},
    ]
    print(rerank("hinh phat ma tuy", sample, top_k=2))
