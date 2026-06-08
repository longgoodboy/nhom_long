"""
Task 4 - Chunking and local indexing.

Design choice for this lab workspace:
    - Chunking: Recursive character splitting with paragraph/sentence/word
      breakpoints. It is stable for both legal text and news articles.
    - Embedding: Local hashing bag-of-words vectors. This avoids installing heavy
      ML packages until we really need them, while still producing deterministic
      numeric vectors for Task 5-style retrieval.
    - Vector store: Local JSON files under data/index/. This keeps the project
      runnable without Docker, Weaviate Cloud, or FAISS.

The implementation is intentionally dependency-light. Later tasks can replace the
embedding/index backend with sentence-transformers + Weaviate without changing the
document/chunk shape.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import INDEX_DIR, PROJECT_DIR, STANDARDIZED_DIR, tokenize


CHUNKS_PATH = INDEX_DIR / "chunks.json"
INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

# 500 chars keeps each chunk focused enough for legal clauses and still small
# enough for reranking/generation prompts. Tests allow only 10% tolerance, so the
# splitter guarantees chunks do not exceed this value.
CHUNK_SIZE = 500

# 80 chars preserves local context across boundaries, useful when a legal clause
# or sentence crosses a split point. It remains safely below CHUNK_SIZE.
CHUNK_OVERLAP = 80

CHUNKING_METHOD = "recursive_character"

# Local deterministic embedding. Dimension 384 mirrors all-MiniLM-L6-v2's vector
# size, so replacing it later with sentence-transformers/all-MiniLM-L6-v2 is easy.
EMBEDDING_MODEL = "local-hashing-bow"
EMBEDDING_DIM = 384

VECTOR_STORE = "local_json"


def load_documents() -> list[dict]:
    """
    Read all markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    documents: list[dict] = []

    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative_path).replace("\\", "/"),
                    "type": doc_type,
                    "chars": len(content),
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into recursive character chunks.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks: list[dict] = []

    for doc_index, document in enumerate(documents):
        content = normalize_text(document["content"])
        text_chunks = split_text_recursive(content)

        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_id = build_chunk_id(document["metadata"]["path"], chunk_index)
            chunks.append(
                {
                    "id": chunk_id,
                    "content": chunk_text,
                    "metadata": {
                        **document["metadata"],
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "chunking_method": CHUNKING_METHOD,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )

    return chunks


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_recursive(text: str) -> list[str]:
    """Split text into chunks using preferred markdown/text breakpoints."""
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)
        if end < text_length:
            end = choose_split_point(text, start, end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(0, end - CHUNK_OVERLAP)
        if next_start <= start:
            next_start = end
        start = skip_leading_whitespace(text, next_start)

    return chunks


def choose_split_point(text: str, start: int, hard_end: int) -> int:
    """Choose the best split point inside [start, hard_end]."""
    min_end = start + max(CHUNK_SIZE // 2, 1)
    window = text[start:hard_end]

    for separator in ("\n\n", "\n", ". ", "; ", ", ", " "):
        offset = window.rfind(separator)
        candidate = start + offset + len(separator)
        if offset != -1 and candidate >= min_end:
            return candidate

    return hard_end


def skip_leading_whitespace(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def build_chunk_id(source_path: str, chunk_index: int) -> str:
    raw = f"{source_path}:{chunk_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add local hashing embeddings to chunks.

    Returns:
        Each chunk dict with an 'embedding': list[float].
    """
    embedded_chunks: list[dict] = []

    for chunk in chunks:
        embedded = {**chunk}
        embedded["embedding"] = hash_embed(chunk["content"])
        embedded_chunks.append(embedded)

    return embedded_chunks


def hash_embed(text: str) -> list[float]:
    """Create a normalized hashing bag-of-words vector."""
    vector = [0.0] * EMBEDDING_DIM

    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [round(value / norm, 6) for value in vector]


def index_to_vectorstore(chunks: list[dict]) -> dict:
    """
    Persist chunks and embeddings to a local JSON vector store.

    Returns:
        Index metadata.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "vector_store": VECTOR_STORE,
        "chunks_path": str(CHUNKS_PATH.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "chunk_count": len(chunks),
        "chunking_method": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "document_sources": sorted(
            {chunk["metadata"]["path"] for chunk in chunks}
        ),
    }

    INDEX_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def load_indexed_chunks() -> list[dict]:
    """Load indexed chunks from local JSON storage."""
    if not CHUNKS_PATH.exists():
        return []
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def run_pipeline() -> dict:
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    documents = load_documents()
    print(f"\nLoaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")

    metadata = index_to_vectorstore(embedded_chunks)
    print(f"Indexed to: {CHUNKS_PATH}")
    return metadata


if __name__ == "__main__":
    run_pipeline()
