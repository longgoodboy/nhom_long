"""Task 7 - reranking utilities with Jina API when configured and local fallback."""

from __future__ import annotations
import unicodedata
import os
import requests
from dotenv import load_dotenv

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")



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

def _tokens(text: str) -> set[str]:
    text = _strip_accents(_fix_mojibake(text).lower())
    return {t.strip(".,;:!?()[]{}\"'`).????") for t in text.split() if t.strip(".,;:!?()[]{}\"'`).????")}

def _local_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    q = _tokens(query)
    rescored = []
    for item in candidates:
        overlap = len(q & _tokens(item.get("content", ""))) / max(len(q), 1)
        score = 0.7 * float(item.get("score", 0.0)) + 0.3 * overlap
        new_item = item.copy()
        new_item["score"] = float(score)
        new_item.setdefault("metadata", {})["reranker"] = "local_overlap"
        rescored.append(new_item)
    rescored.sort(key=lambda r: r["score"], reverse=True)
    return rescored[:top_k]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank with Jina API when JINA_API_KEY exists; fallback locally otherwise."""
    if not candidates:
        return []
    if JINA_API_KEY:
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": JINA_RERANK_MODEL,
                    "query": query,
                    "documents": [c.get("content", "") for c in candidates],
                    "top_n": min(top_k, len(candidates)),
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            reranked = []
            for row in payload.get("results", []):
                idx = row.get("index")
                if idx is None or idx >= len(candidates):
                    continue
                item = candidates[idx].copy()
                item["score"] = float(row.get("relevance_score", item.get("score", 0.0)))
                item.setdefault("metadata", {})["reranker"] = "jina"
                reranked.append(item)
            if reranked:
                return reranked[:top_k]
        except Exception:
            pass
    return _local_rerank(query, candidates, top_k)


def rerank_mmr(query_embedding: list[float], candidates: list[dict], top_k: int = 5, lambda_param: float = 0.7) -> list[dict]:
    return sorted(candidates, key=lambda r: r.get("score", 0), reverse=True)[:top_k]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item
    results = []
    for content, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        merged = items[content].copy()
        merged["score"] = float(score)
        results.append(merged)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    if not candidates:
        return []
    if method in {"cross_encoder", "local", "rrf"}:
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr([], candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    print(rerank("ma tuy", [{"content": "ma tuy", "score": .2, "metadata": {}}]))
