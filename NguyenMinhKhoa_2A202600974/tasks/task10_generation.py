"""
Task 10 - Generation with citations.
"""

from __future__ import annotations

import re

from .rag_utils import keyword_overlap_score, make_citation, query_intent
from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer in Vietnamese and keep every factual statement tied to a source."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for idx in range(0, len(chunks), 2):
        reordered.append(chunks[idx])
    start = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for idx in range(start, 0, -2):
        reordered.append(chunks[idx])
    return reordered


def format_context(chunks: list[dict]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        parts.append(
            f"[Document {idx} | Source: {metadata.get('source', f'source-{idx}')} | "
            f"Type: {metadata.get('type', 'unknown')}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_support_sentences(text: str, max_sentences: int = 2) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]
    if not sentences:
        return [text[:220].strip()]
    return sentences[:max_sentences]


def _rank_support_sentences(query: str, chunks: list[dict], limit: int = 3) -> list[tuple[str, str]]:
    scored: list[tuple[float, str, str]] = []
    for idx, chunk in enumerate(chunks, start=1):
        citation = make_citation(chunk.get("metadata", {}), index=idx)
        for sentence in _extract_support_sentences(chunk["content"], max_sentences=3):
            clean = sentence.replace("\n", " ").strip()
            if len(clean) < 30:
                continue
            score = keyword_overlap_score(query, clean) + float(chunk.get("score", 0.0))
            scored.append((score, clean, citation))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(sentence, citation) for _, sentence, citation in scored[:limit]]


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Toi khong the xac minh thong tin nay tu nguon hien co.",
            "sources": [],
            "retrieval_source": "none",
        }

    ordered = reorder_for_llm(chunks)
    _ = format_context(ordered)

    supports = _rank_support_sentences(query, chunks, limit=3)
    answer_parts = [f"{sentence} [{citation}]" for sentence, citation in supports]

    if not answer_parts:
        answer = "Toi khong the xac minh thong tin nay tu nguon hien co."
    else:
        intent = query_intent(query)
        if intent == "legal":
            prefix = "Thong tin phu hop nhat tu nguon phap ly hien co: "
        elif intent == "news":
            prefix = "Thong tin lien quan nhat tu cac bai bao hien co: "
        else:
            prefix = "Tong hop tu cac nguon hien co: "
        answer = prefix + " ".join(answer_parts)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    print(generate_with_citation("Hinh phat tang tru ma tuy?")["answer"])
