"""Task 4 - local document loading, chunking, and lightweight indexing."""

from pathlib import Path
import json

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_PATH = Path(__file__).parent.parent / "data" / "processed_chunks.json"

# Recursive character chunking is robust for mixed legal/news markdown.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL = "local-token-overlap-fallback"
EMBEDDING_DIM = 0
VECTOR_STORE = "local_json"


def load_documents() -> list[dict]:
    """Load standardized markdown documents with source metadata."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news" if "news" in md_file.parts else "unknown"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file),
                "title": content.splitlines()[0].lstrip("# ") if content.splitlines() else md_file.stem,
            },
        })
    return documents


def _split_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents into bounded overlapping text spans."""
    chunks: list[dict] = []
    for doc in documents:
        for i, text in enumerate(_split_text(doc.get("content", ""))):
            chunks.append({
                "content": text,
                "metadata": {**doc.get("metadata", {}), "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach simple token sets as a local embedding fallback."""
    for chunk in chunks:
        chunk["embedding"] = sorted(set(chunk["content"].lower().split()))
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Persist chunks to a local JSON index for offline retrieval."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{"content": c["content"], "metadata": c.get("metadata", {})} for c in chunks]
    INDEX_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(INDEX_PATH)


def get_chunks() -> list[dict]:
    docs = load_documents()
    return chunk_documents(docs)


def run_pipeline():
    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    index_to_vectorstore(chunks)
    print(f"Loaded {len(docs)} documents; indexed {len(chunks)} chunks to {INDEX_PATH}")


if __name__ == "__main__":
    run_pipeline()
