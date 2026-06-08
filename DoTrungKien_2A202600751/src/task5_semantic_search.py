"""
Task 5 - Semantic search module.

This module is compatible with the local JSON vector store created in Task 4.
It embeds the query with the same local hashing embedding and ranks chunks by
cosine similarity.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.common import safe_preview
from src.task4_chunking_indexing import CHUNKS_PATH, hash_embed, load_indexed_chunks, run_pipeline


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by vector similarity.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    if top_k <= 0:
        return []

    chunks = ensure_index()
    if not query.strip() or not chunks:
        return []

    query_embedding = hash_embed(query)
    results: list[dict] = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.get("embedding", []))
        if score <= 0:
            continue

        results.append(
            {
                "content": chunk["content"],
                "score": round(score, 6),
                "metadata": {
                    **chunk.get("metadata", {}),
                    "chunk_id": chunk.get("id"),
                    "retriever": "semantic",
                },
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def ensure_index() -> list[dict]:
    """Load local index, building it first if needed."""
    chunks = load_indexed_chunks()
    if chunks:
        return chunks

    if not CHUNKS_PATH.exists():
        run_pipeline()
    return load_indexed_chunks()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


if __name__ == "__main__":
    demo_results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for result in demo_results:
        source = result["metadata"].get("source", "unknown")
        preview = safe_preview(result["content"])
        print(f"[{result['score']:.3f}] {source}: {preview}...")
