"""
Task 6 - Lexical search module using BM25.

The module searches the same local chunk index produced by Task 4. It uses
rank-bm25 when installed and falls back to a small pure-Python BM25
implementation otherwise.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Protocol


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.common import safe_preview, tokenize
from src.task4_chunking_indexing import load_indexed_chunks, run_pipeline


BM25_K1 = 1.5
BM25_B = 0.75


class BM25Index(Protocol):
    def get_scores(self, query_tokens: list[str]) -> list[float]:
        ...


class SimpleBM25:
    """Small Okapi BM25 implementation for dependency-light lexical retrieval."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = BM25_K1, b: float = BM25_B):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(document) for document in tokenized_corpus]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        self.term_frequencies = [Counter(document) for document in tokenized_corpus]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for document in self.tokenized_corpus:
            document_frequency.update(set(document))

        return {
            term: math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []

        for index, term_frequency in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[index]
            score = 0.0

            for token in query_tokens:
                tf = term_frequency.get(token, 0)
                if tf == 0:
                    continue

                idf = self.idf.get(token, 0.0)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1.0)
                )
                score += idf * (tf * (self.k1 + 1)) / denominator

            scores.append(score)

        return scores


_CORPUS_CACHE: list[dict] | None = None
_BM25_CACHE: BM25Index | None = None


def load_corpus() -> list[dict]:
    """Load chunks from the local Task 4 index, building it if needed."""
    global _CORPUS_CACHE

    if _CORPUS_CACHE is not None:
        return _CORPUS_CACHE

    chunks = load_indexed_chunks()
    if not chunks:
        run_pipeline()
        chunks = load_indexed_chunks()

    _CORPUS_CACHE = [
        {
            "content": chunk["content"],
            "metadata": {
                **chunk.get("metadata", {}),
                "chunk_id": chunk.get("id"),
            },
        }
        for chunk in chunks
        if chunk.get("content")
    ]
    return _CORPUS_CACHE


def build_bm25_index(corpus: list[dict]) -> BM25Index:
    """
    Build a BM25 index from corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(document["content"]) for document in corpus]

    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)
    except ImportError:
        return SimpleBM25(tokenized_corpus, k1=BM25_K1, b=BM25_B)


def get_bm25_index() -> tuple[list[dict], BM25Index]:
    """Return cached corpus and BM25 index."""
    global _BM25_CACHE

    corpus = load_corpus()
    if _BM25_CACHE is None:
        _BM25_CACHE = build_bm25_index(corpus)

    return corpus, _BM25_CACHE


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by exact keyword relevance using BM25.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    corpus, bm25 = get_bm25_index()
    if not corpus:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    results: list[dict] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue

        document = corpus[index]
        results.append(
            {
                "content": document["content"],
                "score": round(score, 6),
                "metadata": {
                    **document.get("metadata", {}),
                    "retriever": "lexical_bm25",
                },
            }
        )

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    demo_results = lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5)
    for result in demo_results:
        source = result["metadata"].get("source", "unknown")
        preview = safe_preview(result["content"])
        print(f"[{result['score']:.3f}] {source}: {preview}...")
