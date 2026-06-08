"""Task 5 - semantic search with an offline token-similarity fallback."""

import math
from collections import Counter
import unicodedata
from .task4_chunking_indexing import get_chunks



def _fix_mojibake(text: str) -> str:
    # Some starter tests/prompts display UTF-8 text decoded as cp1252; repair when possible.
    if "?" in text or "?" in text:
        try:
            return text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return text
    return text

def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

def _tokens(text: str) -> list[str]:
    text = _strip_accents(_fix_mojibake(text).lower())
    return [t.strip(".,;:!?()[]{}\"'`).????") for t in text.split() if t.strip(".,;:!?()[]{}\"'`).????")]

def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    q_vec = Counter(_tokens(query))
    results = []
    for chunk in get_chunks():
        score = _cosine(q_vec, Counter(_tokens(chunk["content"])))
        if score > 0:
            results.append({"content": chunk["content"], "score": float(score), "metadata": chunk.get("metadata", {})})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    print(semantic_search("hinh phat ma tuy", 3))
