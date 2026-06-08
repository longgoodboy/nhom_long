"""
Task 10 - Generation with citations using OpenAI.

This task retrieves context from the RAG pipeline, reorders it to reduce
lost-in-the-middle, and sends the final prompt to OpenAI. There is no local
generation fallback in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from openai import OpenAI

from src.common import env
from src.task9_retrieval_pipeline import retrieve


TOP_K = 5
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4o-mini")

# Keep sampling conservative for factual RAG answers.
TEMPERATURE = 0.3
TOP_P = 0.9

SYSTEM_PROMPT = """You are a Vietnamese RAG assistant.

Answer the user's question using only the provided context.
For every factual claim, include a citation in brackets using the source label
from the context, for example [luat-phong-chong-ma-tuy-2021, 2021].
If the context does not contain enough evidence, answer exactly:
I cannot verify this information

Rules:
- Do not use outside knowledge.
- Do not invent citations.
- Keep the answer concise and directly relevant to the question.
- Write the answer in Vietnamese unless the user asks otherwise."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to mitigate lost-in-the-middle.

    Example:
        [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks[:]

    front = [chunks[index] for index in range(0, len(chunks), 2)]
    back = [chunks[index] for index in range(len(chunks) - 1, 0, -1) if index % 2 == 1]
    return front + back


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into context with explicit source labels for citations.
    """
    context_parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"source-{index}").replace(".md", "")
        doc_type = metadata.get("type", "unknown")
        score = chunk.get("score", 0.0)

        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | Score: {score}]\n"
            f"{chunk.get('content', '')}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with OpenAI.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered_chunks = reorder_for_llm(chunks)
    context = format_context(reordered_chunks)
    answer = call_openai(query=query, context=context)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "context": context,
        "model": OPENAI_MODEL,
    }


def call_openai(query: str, context: str) -> str:
    """Call OpenAI Chat Completions with the configured model."""
    api_key = env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Please set it in .env.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}",
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    return response.choices[0].message.content or "I cannot verify this information"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    questions = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for question in questions:
        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print("=" * 70)
        result = generate_with_citation(question)
        print(f"\nModel: {result['model']}")
        print(f"A: {result['answer']}")
        print(f"[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
