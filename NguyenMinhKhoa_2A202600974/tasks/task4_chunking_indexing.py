"""
Task 4 - Chunking and indexing.
"""

from __future__ import annotations

from pathlib import Path

from .rag_utils import chunk_text, hash_embed, load_markdown_documents


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "local-hash-embedding"
EMBEDDING_DIM = 256

VECTOR_STORE = "in_memory"


def load_documents() -> list[dict]:
    """
    Read markdown files from data/standardized/.
    """
    return load_markdown_documents()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split each document into paragraph-aware chunks.
    """
    chunks: list[dict] = []
    for doc in documents:
        for idx, part in enumerate(chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append(
                {
                    "content": part,
                    "metadata": {**doc["metadata"], "chunk_index": idx},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add a lightweight local embedding to each chunk.
    """
    embedded: list[dict] = []
    for chunk in chunks:
        item = chunk.copy()
        item["embedding"] = hash_embed(chunk["content"], dim=EMBEDDING_DIM).tolist()
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict]):
    """
    Persist a small preview artifact so the indexing step has a tangible output.
    """
    index_dir = Path(__file__).resolve().parents[2] / "data" / "artifacts"
    index_dir.mkdir(parents=True, exist_ok=True)
    preview_path = index_dir / "chunk_index_preview.json"
    preview = [
        {
            "content": chunk["content"][:300],
            "metadata": chunk.get("metadata", {}),
            "embedding_dim": len(chunk.get("embedding", [])),
        }
        for chunk in chunks[:50]
    ]
    preview_path.write_text(str(preview), encoding="utf-8")
    return preview_path


def run_pipeline():
    docs = load_documents()
    chunks = chunk_documents(docs)
    embedded = embed_chunks(chunks)
    return index_to_vectorstore(embedded)


if __name__ == "__main__":
    output = run_pipeline()
    print(f"Indexed preview saved to: {output}")
