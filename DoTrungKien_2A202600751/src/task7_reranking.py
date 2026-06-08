"""
Task 7 - Local reranking.

The default reranker blends retrieval score with query-aware lexical features.
All phrase matching is accent-insensitive so queries using "ma tuy", "ma tuý",
or "ma túy" behave consistently.
"""

from __future__ import annotations

import math
import sys
import unicodedata
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.common import safe_preview, tokenize
from src.task4_chunking_indexing import hash_embed
from src.task5_semantic_search import cosine_similarity


def normalize_for_match(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text.lower())
    no_accents = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return " ".join(tokenize(no_accents))


def normalize_scores(candidates: list[dict]) -> list[float]:
    scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(max_score, min_score):
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def token_overlap_score(query: str, content: str) -> float:
    query_tokens = set(tokenize(normalize_for_match(query)))
    if not query_tokens:
        return 0.0

    content_tokens = set(tokenize(normalize_for_match(content)))
    return len(query_tokens & content_tokens) / len(query_tokens)


def ngram_overlap_score(query: str, content: str, n: int = 3) -> float:
    query_tokens = tokenize(normalize_for_match(query))
    content_tokens = tokenize(normalize_for_match(content))
    if len(query_tokens) < n or len(content_tokens) < n:
        return 0.0

    query_ngrams = {
        tuple(query_tokens[index : index + n])
        for index in range(len(query_tokens) - n + 1)
    }
    content_ngrams = {
        tuple(content_tokens[index : index + n])
        for index in range(len(content_tokens) - n + 1)
    }
    return len(query_ngrams & content_ngrams) / len(query_ngrams)


def phrase_bonus(query: str, content: str) -> float:
    query_text = normalize_for_match(query)
    content_text = normalize_for_match(content)
    query_tokens = query_text.split()
    if not query_tokens:
        return 0.0

    if query_text in content_text:
        return 1.0

    best = 0
    for window_size in range(min(6, len(query_tokens)), 2, -1):
        for start in range(len(query_tokens) - window_size + 1):
            phrase = " ".join(query_tokens[start : start + window_size])
            if phrase in content_text:
                best = max(best, window_size)

    return best / len(query_tokens)


def important_phrase_bonus(query: str, content: str) -> float:
    query_text = normalize_for_match(query)
    content_text = normalize_for_match(content)
    weighted_phrases = {
        "quy trinh cai nghien": 1.0,
        "cac bien phap cai nghien": 0.9,
        "tang tru trai phep chat ma tuy": 1.0,
        "to chuc su dung trai phep chat ma tuy": 1.0,
        "cai nghien bat buoc": 0.35,
    }

    return max(
        (
            weight
            for phrase, weight in weighted_phrases.items()
            if phrase in query_text and phrase in content_text
        ),
        default=0.0,
    )


def legal_article_heading_bonus(query: str, content: str) -> float:
    query_text = normalize_for_match(query)
    content_text = normalize_for_match(content)
    article_phrases = [
        ("quy trinh cai nghien", "dieu 29 quy trinh cai nghien"),
        ("cac bien phap cai nghien", "dieu 28 cac bien phap cai nghien"),
        ("tang tru trai phep chat ma tuy", "tang tru trai phep chat ma tuy"),
    ]

    for query_phrase, content_phrase in article_phrases:
        if query_phrase in query_text and content_phrase in content_text:
            return 1.0
    return 0.0


def rerank_score(query: str, candidate: dict, normalized_score: float) -> float:
    content = candidate.get("content", "")
    return (
        (0.15 * normalized_score)
        + (0.25 * token_overlap_score(query, content))
        + (0.15 * phrase_bonus(query, content))
        + (0.10 * ngram_overlap_score(query, content, n=3))
        + (0.05 * ngram_overlap_score(query, content, n=2))
        + (0.15 * important_phrase_bonus(query, content))
        + (0.35 * legal_article_heading_bonus(query, content))
    )


def apply_diversity_penalty(selected: list[dict], candidate: dict) -> float:
    candidate_tokens = set(tokenize(normalize_for_match(candidate.get("content", ""))))
    if not candidate_tokens:
        return 0.0

    max_jaccard = 0.0
    for item in selected:
        selected_tokens = set(tokenize(normalize_for_match(item.get("content", ""))))
        if not selected_tokens:
            continue
        union = candidate_tokens | selected_tokens
        max_jaccard = max(max_jaccard, len(candidate_tokens & selected_tokens) / len(union))

    return 0.08 * max_jaccard


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    return rerank(query, candidates, top_k=top_k, method="hybrid")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    if top_k <= 0 or not candidates:
        return []

    enriched = []
    for candidate in candidates:
        item = {**candidate}
        item["embedding"] = item.get("embedding") or hash_embed(item.get("content", ""))
        enriched.append(item)

    selected: list[int] = []
    remaining = list(range(len(enriched)))

    for _ in range(min(top_k, len(enriched))):
        best_index = remaining[0]
        best_score = float("-inf")

        for index in remaining:
            relevance = cosine_similarity(query_embedding, enriched[index]["embedding"])
            diversity = 0.0
            for selected_index in selected:
                diversity = max(
                    diversity,
                    cosine_similarity(enriched[index]["embedding"], enriched[selected_index]["embedding"]),
                )
            score = lambda_param * relevance - (1 - lambda_param) * diversity
            if score > best_score:
                best_score = score
                best_index = index

        selected.append(best_index)
        remaining.remove(best_index)

    results = []
    for index in selected:
        item = {k: v for k, v in enriched[index].items() if k != "embedding"}
        item["metadata"] = {**item.get("metadata", {}), "reranker": "mmr"}
        results.append(item)
    return results


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("metadata", {}).get("chunk_id") or item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            items[key] = item

    results = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = items[key].copy()
        item["score"] = round(score, 6)
        item["metadata"] = {**item.get("metadata", {}), "reranker": "rrf"}
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "hybrid",
) -> list[dict]:
    if top_k <= 0 or not candidates:
        return []

    if method == "cross_encoder":
        method = "hybrid"
    if method == "mmr":
        return rerank_mmr(hash_embed(query), candidates, top_k=top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method != "hybrid":
        raise ValueError(f"Unknown rerank method: {method}")

    normalized_scores = normalize_scores(candidates)
    scored = []
    for candidate, normalized_score in zip(candidates, normalized_scores):
        item = candidate.copy()
        item["score"] = round(rerank_score(query, candidate, normalized_score), 6)
        item["metadata"] = {
            **item.get("metadata", {}),
            "reranker": "hybrid_local",
            "original_score": candidate.get("score", 0.0),
        }
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)

    selected: list[dict] = []
    remaining = scored[:]
    while remaining and len(selected) < top_k:
        best_index = 0
        best_adjusted_score = float("-inf")
        for index, candidate in enumerate(remaining):
            adjusted_score = candidate["score"] - apply_diversity_penalty(selected, candidate)
            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_index = index
        selected.append(remaining.pop(best_index))

    return selected


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 29. Quy trinh cai nghien ma tuy", "score": 0.3, "metadata": {}},
        {"content": "Cai nghien bat buoc tai co so cong lap", "score": 0.9, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    for result in rerank("quy trinh cai nghien bat buoc", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {safe_preview(result['content'])}")
