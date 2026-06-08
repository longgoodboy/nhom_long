"""
Task 5 - Semantic search.
"""

from __future__ import annotations

from .rag_utils import (
    cosine_similarity,
    expand_query,
    hash_embed,
    keyword_overlap_score,
    load_chunk_corpus,
    metadata_type_boost,
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Lightweight dense retrieval using local hash embeddings.
    """
    chunks, embeddings = load_chunk_corpus()
    if not chunks:
        return []

    expanded_query = expand_query(query)
    query_embedding = hash_embed(expanded_query)
    results: list[dict] = []
    for chunk, embedding in zip(chunks, embeddings):
        dense_score = cosine_similarity(query_embedding, embedding)
        overlap_bonus = keyword_overlap_score(expanded_query, chunk["content"])
        boost = metadata_type_boost(query, chunk.get("metadata", {}))
        score = 0.7 * dense_score + 0.25 * overlap_bonus + boost
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for row in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{row['score']:.3f}] {row['metadata'].get('source')} :: {row['content'][:90]}")
