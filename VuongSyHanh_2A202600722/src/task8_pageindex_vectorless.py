"""
Task 8 — PageIndex Vectorless RAG.

PageIndex (https://pageindex.ai/) là RAG framework không dùng vector store:
LLM đọc cấu trúc (tree) của document và chọn các node liên quan trực tiếp.

Workflow:
    1. submit_document(pdf) → doc_id (async, chờ PageIndex parse + build tree)
    2. is_retrieval_ready(doc_id) → poll đến khi tree sẵn sàng
    3. submit_query(doc_id, query) → retrieval_id
    4. get_retrieval(retrieval_id) → list chunks + score

Cài đặt:
    pip install pageindex
    # API key: https://dash.pageindex.ai/api-keys
    # Thêm vào .env: PAGEINDEX_API_KEY=pi_xxxxx
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
print(f"PAGEINDEX_API_KEY: {'set' if PAGEINDEX_API_KEY else 'NOT set'}")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"
VECTOR_STORE_FILE = Path(__file__).parent.parent / "data" / "vector_store.json"

# Poll khi chờ PageIndex parse PDF (mỗi PDF mất ~1-3 phút)
PROCESSING_POLL_SEC = 10
PROCESSING_TIMEOUT_SEC = 600
# Poll khi chờ retrieval kết quả
RETRIEVAL_POLL_SEC = 2
RETRIEVAL_TIMEOUT_SEC = 60


# ---------------------------------------------------------------------------
# Document upload / cache
# ---------------------------------------------------------------------------

def _load_doc_cache() -> dict[str, str]:
    if DOC_CACHE_FILE.exists():
        return json.loads(DOC_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_doc_cache(cache: dict[str, str]) -> None:
    DOC_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _wait_until_ready(client, doc_id: str) -> bool:
    """Poll get_document đến khi status='completed'. Trả False nếu timeout/failed."""
    deadline = time.time() + PROCESSING_TIMEOUT_SEC
    while time.time() < deadline:
        info = client.get_document(doc_id)
        status = info.get("status", "")
        if status == "completed":
            return True
        if status in {"failed", "error"}:
            print(f"  ✗ PageIndex parse failed: {info}")
            return False
        time.sleep(PROCESSING_POLL_SEC)
    return False


def upload_documents() -> dict[str, str]:
    """Upload mọi PDF trong data/landing/legal/ lên PageIndex.

    Returns:
        Mapping {slug: doc_id}. Đã upload trước đó được dùng lại từ cache.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa set trong .env")
        return {}

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_doc_cache()

    pdfs = sorted(LEGAL_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"⚠ Không có PDF trong {LEGAL_DIR}")
        return cache

    # Free tier có credit giới hạn — upload nhỏ trước, dừng sớm nếu hết
    pdfs.sort(key=lambda p: p.stat().st_size)
    for pdf in pdfs:
        slug = pdf.stem
        if slug in cache:
            print(f"  ↻ Đã upload trước: {pdf.name} (doc_id={cache[slug]})")
            continue
        print(f"→ Upload {pdf.name} ({pdf.stat().st_size // 1024} KB)…")
        try:
            result = client.submit_document(str(pdf))
        except Exception as exc:
            msg = str(exc)
            if "InsufficientCredits" in msg:
                print(f"  ✗ Hết credit free tier — bỏ qua {pdf.name} và các file lớn hơn")
                break
            print(f"  ✗ Lỗi upload: {exc}")
            continue
        doc_id = result["doc_id"]
        print(f"  doc_id={doc_id} — chờ PageIndex parse…")
        if _wait_until_ready(client, doc_id):
            cache[slug] = doc_id
            _save_doc_cache(cache)
            print(f"  ✓ {slug} sẵn sàng")
        else:
            print(f"  ✗ Bỏ qua {slug} (timeout hoặc lỗi)")

    return cache


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _wait_for_retrieval(client, retrieval_id: str) -> dict | None:
    """Poll get_retrieval đến khi status='completed'. Trả None nếu timeout."""
    deadline = time.time() + RETRIEVAL_TIMEOUT_SEC
    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        status = result.get("status", "")
        if status == "completed":
            return result
        if status in {"failed", "error"}:
            return None
        time.sleep(RETRIEVAL_POLL_SEC)
    return None


def _normalize_retrieval_chunks(retrieval: dict, slug: str) -> list[dict]:
    """Bóc list chunks từ payload PageIndex và normalize sang format chung.

    PageIndex trả về `retrieved_nodes` (mỗi node là 1 phần của tree). Mỗi node
    có thể có `text` / `content`, `node_id`, `relevance_score` / `score`.
    """
    nodes = (
        retrieval.get("retrieved_nodes")
        or retrieval.get("nodes")
        or retrieval.get("results")
        or []
    )
    chunks = []
    for node in nodes:
        text = node.get("text") or node.get("content") or node.get("summary") or ""
        if not text:
            continue
        score = (
            node.get("relevance_score")
            or node.get("score")
            or 0.0
        )
        chunks.append({
            "content": text.strip(),
            "score": float(score),
            "metadata": {
                "source": f"{slug}.pdf",
                "node_id": node.get("node_id") or node.get("id"),
                "type": "legal",
            },
            "source": "pageindex",
        })
    return chunks


def _pageindex_search_remote(query: str, top_k: int) -> list[dict]:
    """Gọi PageIndex API thật trên mọi doc đã upload."""
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_doc_cache()
    if not cache:
        # Lần đầu chạy — upload luôn
        cache = upload_documents()
    if not cache:
        return []

    all_chunks: list[dict] = []
    for slug, doc_id in cache.items():
        try:
            submit = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submit["retrieval_id"]
            result = _wait_for_retrieval(client, retrieval_id)
            if result:
                all_chunks.extend(_normalize_retrieval_chunks(result, slug))
        except Exception as exc:
            print(f"  ⚠ {slug} query lỗi: {exc}")

    all_chunks.sort(key=lambda c: c["score"], reverse=True)
    return all_chunks[:top_k]


def _pageindex_search_fallback(query: str, top_k: int) -> list[dict]:
    """Fallback khi không có API key: dùng lexical_search nhưng đánh dấu nguồn."""
    try:
        from src.task6_lexical_search import lexical_search
        local = lexical_search(query, top_k=top_k)
    except Exception:
        local = []

    results = [{**r, "source": "pageindex"} for r in local]

    if not results and VECTOR_STORE_FILE.exists():
        chunks = json.loads(VECTOR_STORE_FILE.read_text(encoding="utf-8"))
        for i, chunk in enumerate(chunks[:top_k]):
            results.append({
                "content": chunk["content"],
                "score": 0.5 - i * 0.1,
                "metadata": chunk["metadata"],
                "source": "pageindex",
            })

    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval bằng PageIndex. Fallback nếu chưa có API key."""
    if PAGEINDEX_API_KEY:
        try:
            return _pageindex_search_remote(query, top_k)
        except Exception as exc:
            print(f"⚠ PageIndex query thất bại, dùng fallback nội bộ: {exc}")
    return _pageindex_search_fallback(query, top_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://dash.pageindex.ai/api-keys")
        raise SystemExit(1)

    print("=== Upload PDFs lên PageIndex ===")
    cache = upload_documents()
    print(f"\n{len(cache)} document(s) sẵn sàng truy vấn:")
    for slug, doc_id in cache.items():
        print(f"  • {slug} → {doc_id}")

    test_query = "Điều 249 Bộ luật Hình sự quy định hình phạt như thế nào về tội tàng trữ trái phép chất ma túy?"
    print(f"\n=== Test query ===\n{test_query}\n")
    for r in pageindex_search(test_query, top_k=5):
        print(f"[{r['score']:.3f}] {r['metadata']['source']}: {r['content'][:200]}")
