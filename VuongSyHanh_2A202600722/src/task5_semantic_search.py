"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

VECTOR_STORE_FILE = Path(__file__).parent.parent / "data" / "vector_store.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.
    """
    if not VECTOR_STORE_FILE.exists():
        return []

    # Đọc chunks
    chunks = json.loads(VECTOR_STORE_FILE.read_text(encoding="utf-8"))
    if not chunks:
        return []

    # Sinh embedding cho query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query)

    # Tính cosine similarity (tích vô hướng do vector đã chuẩn hóa)
    results = []
    for chunk in chunks:
        emb = np.array(chunk["embedding"])
        score = float(np.dot(query_embedding, emb))
        results.append({
            "content": chunk["content"],
            "score": score,
            "metadata": chunk["metadata"]
        })

    # Sắp xếp giảm dần theo score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

