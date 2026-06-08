"""
Task 8 - PageIndex vectorless RAG fallback.

Real PageIndex usage requires an account and API key. This module provides:
    1. Optional hooks for a real PageIndex SDK/API when PAGEINDEX_API_KEY exists.
    2. A local vectorless fallback that searches markdown structure and BM25-style
       lexical evidence without using embeddings.

The fallback returns source='pageindex' so Task 9 can treat it as the vectorless
retrieval branch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.common import env, has_real_env, safe_preview
from src.task4_chunking_indexing import load_documents
from src.task6_lexical_search import lexical_search, tokenize


PAGEINDEX_API_KEY = env("PAGEINDEX_API_KEY")
PAGEINDEX_LOCAL_DIR = PROJECT_DIR / "data" / "pageindex"
PAGEINDEX_LOCAL_DOCS = PAGEINDEX_LOCAL_DIR / "documents.json"


def upload_documents() -> list[dict]:
    """
    Register markdown documents for PageIndex-style vectorless search.

    If a real PageIndex API key is configured, this function can be extended to
    upload through the SDK. For this lab, it writes a local document registry so
    fallback retrieval is reproducible.
    """
    PAGEINDEX_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_documents()

    payload = [
        {
            "content": document["content"],
            "metadata": document["metadata"],
        }
        for document in documents
    ]

    PAGEINDEX_LOCAL_DOCS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex when available, local fallback otherwise.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if top_k <= 0 or not query.strip():
        return []

    real_results = try_real_pageindex_search(query, top_k)
    if real_results:
        return real_results

    return local_vectorless_search(query, top_k)


def try_real_pageindex_search(query: str, top_k: int) -> list[dict]:
    """
    Best-effort PageIndex SDK query.

    The public SDK shape may change, so failures fall back silently to local
    vectorless search. This keeps the lab runnable without account setup.
    """
    if not has_real_env("PAGEINDEX_API_KEY"):
        return []

    try:
        from pageindex import PageIndex  # type: ignore

        client = PageIndex(api_key=PAGEINDEX_API_KEY)
        raw_results = client.query(query=query, top_k=top_k)
    except Exception:
        return []

    return normalize_pageindex_results(raw_results)


def normalize_pageindex_results(raw_results: Any) -> list[dict]:
    """Normalize possible SDK result shapes into the lab output contract."""
    results: list[dict] = []

    for item in raw_results or []:
        content = getattr(item, "text", None) or getattr(item, "content", None)
        score = getattr(item, "score", 0.0)
        metadata = getattr(item, "metadata", {}) or {}

        if isinstance(item, dict):
            content = item.get("text") or item.get("content")
            score = item.get("score", score)
            metadata = item.get("metadata", metadata)

        if not content:
            continue

        results.append(
            {
                "content": str(content),
                "score": round(float(score), 6),
                "metadata": metadata,
                "source": "pageindex",
            }
        )

    return results


def local_vectorless_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Local vectorless fallback.

    It combines BM25 chunk hits with markdown heading/source boosts. No dense
    embeddings are used here, which mirrors the role of PageIndex as a fallback
    when vector/hybrid retrieval is weak.
    """
    lexical_results = lexical_search(query, top_k=max(top_k * 4, 10))
    if not lexical_results:
        return search_whole_documents(query, top_k)

    reranked = []
    for result in lexical_results:
        heading_boost = structural_heading_score(query, result["content"])
        source_boost = 0.05 if result.get("metadata", {}).get("type") == "legal" else 0.0
        score = float(result["score"]) + heading_boost + source_boost
        reranked.append(
            {
                "content": result["content"],
                "score": round(score, 6),
                "metadata": {
                    **result.get("metadata", {}),
                    "retriever": "pageindex_local_vectorless",
                },
                "source": "pageindex",
            }
        )

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def search_whole_documents(query: str, top_k: int) -> list[dict]:
    """Fallback over whole markdown docs if the chunk BM25 index has no hit."""
    documents = load_local_documents()
    query_terms = set(tokenize(query))
    results: list[dict] = []

    for document in documents:
        content = document.get("content", "")
        content_terms = set(tokenize(content))
        if not query_terms or not content_terms:
            continue

        overlap = len(query_terms & content_terms) / len(query_terms)
        if overlap <= 0:
            continue

        results.append(
            {
                "content": first_relevant_section(content, query_terms),
                "score": round(overlap, 6),
                "metadata": {
                    **document.get("metadata", {}),
                    "retriever": "pageindex_local_document",
                },
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def load_local_documents() -> list[dict]:
    if PAGEINDEX_LOCAL_DOCS.exists():
        return json.loads(PAGEINDEX_LOCAL_DOCS.read_text(encoding="utf-8"))
    return upload_documents()


def structural_heading_score(query: str, content: str) -> float:
    """Boost chunks whose markdown headings match query terms."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0

    headings = re.findall(r"^#{1,6}\s+(.+)$", content, flags=re.MULTILINE)
    if not headings:
        headings = content.splitlines()[:2]

    heading_terms = set(tokenize(" ".join(headings)))
    if not heading_terms:
        return 0.0

    return 0.25 * (len(query_terms & heading_terms) / len(query_terms))


def first_relevant_section(content: str, query_terms: set[str], max_chars: int = 800) -> str:
    """Return the first paragraph with query overlap, or the document prefix."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content) if paragraph.strip()]

    for paragraph in paragraphs:
        if set(tokenize(paragraph)) & query_terms:
            return paragraph[:max_chars]

    return content[:max_chars]


if __name__ == "__main__":
    print(f"PageIndex API configured: {has_real_env('PAGEINDEX_API_KEY')}")
    results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
    for result in results:
        source_name = result.get("metadata", {}).get("source", "unknown")
        preview = safe_preview(result["content"])
        print(f"[{result['score']:.3f}] {source_name}: {preview}...")
