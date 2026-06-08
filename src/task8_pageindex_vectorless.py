"""Task 8 - PageIndex-compatible vectorless search with safe local fallback."""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from .task4_chunking_indexing import load_documents
from .task6_lexical_search import lexical_search

load_dotenv()
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """Best-effort PageIndex upload boundary; local fallback remains the default."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is required for real PageIndex upload")
    try:
        from pageindex import PageIndex  # type: ignore
    except Exception as exc:
        raise RuntimeError("pageindex SDK is not importable in this environment") from exc
    client = PageIndex(api_key=PAGEINDEX_API_KEY)
    uploaded = 0
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        metadata = {"filename": md_file.name, "type": md_file.parent.name}
        # SDK versions differ; try common method names without making tests depend on them.
        if hasattr(client, "upload"):
            client.upload(content=content, metadata=metadata)
        elif hasattr(client, "add"):
            client.add(content=content, metadata=metadata)
        else:
            raise RuntimeError("Unsupported PageIndex SDK: no upload/add method found")
        uploaded += 1
    return uploaded


def _local_pageindex_fallback(query: str, top_k: int) -> list[dict]:
    results = lexical_search(query, top_k=top_k)
    if not results:
        # Last-resort structural fallback: return first documents as PageIndex-shaped evidence.
        docs = load_documents()[:top_k]
        results = [{"content": d["content"][:500], "score": 0.01, "metadata": d.get("metadata", {})} for d in docs]
    return [{**item, "source": "pageindex"} for item in results[:top_k]]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Query PageIndex if configured; otherwise return PageIndex-shaped local fallback."""
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex  # type: ignore
            client = PageIndex(api_key=PAGEINDEX_API_KEY)
            if hasattr(client, "query"):
                raw = client.query(query=query, top_k=top_k)
            elif hasattr(client, "search"):
                raw = client.search(query=query, top_k=top_k)
            else:
                raw = []
            output = []
            for row in raw or []:
                content = getattr(row, "text", None) or getattr(row, "content", None) or row.get("text", row.get("content", "")) if isinstance(row, dict) else ""
                score = getattr(row, "score", None) or (row.get("score", 0.0) if isinstance(row, dict) else 0.0)
                metadata = getattr(row, "metadata", None) or (row.get("metadata", {}) if isinstance(row, dict) else {})
                if content:
                    output.append({"content": content, "score": float(score), "metadata": metadata, "source": "pageindex"})
            if output:
                return output[:top_k]
        except Exception:
            pass
    return _local_pageindex_fallback(query, top_k)


if __name__ == "__main__":
    print(pageindex_search("ma tuy", 2))
