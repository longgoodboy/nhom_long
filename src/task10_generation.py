"""Task 10 - RAG generation with citations and offline extractive fallback."""

import os
from dotenv import load_dotenv
from .task9_retrieval_pipeline import retrieve

load_dotenv()
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer in Vietnamese using only provided context. Cite every factual claim. If evidence is insufficient, say so instead of guessing."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        md = chunk.get("metadata", {})
        source = md.get("source", f"Source {i}")
        doc_type = md.get("type", md.get("doc_type", "unknown"))
        title = md.get("title", source)
        parts.append(f"[Document {i} | Source: {source} | Title: {title} | Type: {doc_type}]\n{chunk.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def _citation(chunk: dict) -> str:
    md = chunk.get("metadata", {})
    return f"[{md.get('source', 'unknown source')}]"


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Khong tim thay du thong tin trong co so du lieu."
    lines = ["Duoi day la cau tra loi dua tren cac nguon tim thay:"]
    for chunk in chunks[:3]:
        snippet = " ".join(chunk.get("content", "").split())[:260]
        lines.append(f"- {snippet} {_citation(chunk)}")
    lines.append("Thong tin chi mang tinh tham khao, khong thay the tu van phap ly chuyen nghiep.")
    return "\n".join(lines)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            user_message = f"Context:\n{format_context(reordered)}\n\nQuestion: {query}"
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or _extractive_answer(query, reordered)
        except Exception:
            answer = _extractive_answer(query, reordered)
    else:
        answer = _extractive_answer(query, reordered)
    return {"answer": answer, "sources": chunks, "retrieval_source": chunks[0].get("source", "none") if chunks else "none"}


if __name__ == "__main__":
    print(generate_with_citation("Ketamine co phai chat ma tuy khong?"))
