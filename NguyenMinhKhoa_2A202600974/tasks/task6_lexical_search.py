"""
Task 6 - Lexical search with BM25.
"""

from __future__ import annotations

from functools import lru_cache

from rank_bm25 import BM25Okapi

from .rag_utils import expand_query, load_chunk_corpus, metadata_type_boost, tokenize

CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]):
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


@lru_cache(maxsize=1)
def _get_index():
    global CORPUS
    CORPUS, _ = load_chunk_corpus()
    return build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    bm25 = _get_index()
    if not CORPUS:
        return []

    expanded_query = expand_query(query)
    tokenized_query = tokenize(expanded_query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results: list[dict] = []
    for idx in ranked_indices[:top_k]:
        boosted_score = float(scores[idx]) + metadata_type_boost(query, CORPUS[idx].get("metadata", {}))
        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": boosted_score,
                "metadata": CORPUS[idx].get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    for row in lexical_search("dieu 248 ma tuy", top_k=5):
        print(f"[{row['score']:.3f}] {row['metadata'].get('source')} :: {row['content'][:90]}")
