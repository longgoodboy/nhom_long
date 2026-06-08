"""
Task 8 - PageIndex-style vectorless fallback.
"""

from __future__ import annotations

from .task6_lexical_search import lexical_search


def upload_documents():
    """
    Placeholder for the real PageIndex upload step.
    """
    return {"status": "skipped", "reason": "local offline submission"}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Simulate a vectorless fallback by reusing lexical retrieval and marking the source.
    """
    results = lexical_search(query, top_k=top_k)
    fallback_results = []
    for item in results:
        fallback_item = item.copy()
        fallback_item["source"] = "pageindex"
        fallback_results.append(fallback_item)
    return fallback_results


if __name__ == "__main__":
    print(pageindex_search("ma tuy", top_k=3))
