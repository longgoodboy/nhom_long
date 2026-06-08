"""
Shared helpers for the Day 08 individual RAG assignment.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "la", "ve", "va", "cua", "cho", "trong", "tai", "theo", "mot", "nhung",
    "cac", "nhung", "nao", "gi", "ra", "sao", "voi", "tu", "den", "khi",
    "bi", "duoc", "co", "khong", "bao", "nhieu", "bao_nhieu", "quy_dinh",
}
QUERY_SYNONYMS = {
    "ma tuy": ["chat cam", "chat gay nghien", "chat ma tuy"],
    "tang tru": ["cat giau", "so huu trai phep"],
    "su dung": ["dung", "su dung trai phep"],
    "to chuc su dung": ["ru re su dung", "bo tri dia diem su dung"],
    "hinh phat": ["muc phat", "khung hinh phat", "xu ly"],
    "nghe si": ["ca si", "dien vien", "rapper", "nguoi noi tieng"],
    "cai nghien": ["cai nghien bat buoc", "co so cai nghien"],
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text.lower())
    ascii_text = strip_accents(normalized)
    tokens = [token for token in TOKEN_PATTERN.findall(ascii_text) if token]
    return tokens


def meaningful_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """
    Lightweight local embedding built from hashed unigrams and trigrams.
    """
    vector = np.zeros(dim, dtype=float)
    tokens = meaningful_tokens(text) or tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        idx = stable_hash(f"tok:{token}") % dim
        vector[idx] += 1.0

    joined = " ".join(tokens)
    for i in range(max(0, len(joined) - 2)):
        trigram = joined[i : i + 3]
        idx = stable_hash(f"tri:{trigram}") % dim
        vector[idx] += 0.2

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Paragraph-aware chunking with a sentence fallback.
    """
    clean = normalize_text(text)
    if not clean:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [clean]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = f"{overlap} {paragraph}".strip()
        else:
            sentences = re.split(r"(?<=[\.\!\?])\s+", paragraph)
            sentence_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence_chunk) + len(sentence) + 1 <= chunk_size:
                    sentence_chunk = f"{sentence_chunk} {sentence}".strip()
                else:
                    if sentence_chunk:
                        chunks.append(sentence_chunk)
                        overlap = sentence_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                        sentence_chunk = f"{overlap} {sentence}".strip()
                    else:
                        chunks.append(sentence[:chunk_size])
                        overlap = sentence[:chunk_size][-chunk_overlap:] if chunk_overlap > 0 else ""
                        sentence_chunk = overlap
            current = sentence_chunk.strip()

    if current:
        chunks.append(current)

    deduped: list[str] = []
    seen = set()
    for chunk in chunks:
        chunk = normalize_text(chunk)
        if chunk and chunk not in seen:
            deduped.append(chunk[:chunk_size])
            seen.add(chunk)
    return deduped


def load_markdown_documents() -> list[dict]:
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        doc_type = md_file.parent.name
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "path": str(md_file),
                },
            }
        )
    return documents


@lru_cache(maxsize=1)
def load_chunk_corpus() -> tuple[list[dict], list[np.ndarray]]:
    from .task4_chunking_indexing import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        EMBEDDING_DIM,
        chunk_documents,
        load_documents,
    )

    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings = [hash_embed(chunk["content"], dim=EMBEDDING_DIM) for chunk in chunks]
    return chunks, embeddings


def keyword_overlap_score(query: str, content: str) -> float:
    query_tokens = set(meaningful_tokens(query) or tokenize(query))
    content_tokens = set(meaningful_tokens(content) or tokenize(content))
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    coverage = len(query_tokens & content_tokens) / max(1, len(content_tokens))
    return 0.8 * overlap + 0.2 * min(coverage * 10, 1.0)


def make_citation(metadata: dict, index: int = 1) -> str:
    source = metadata.get("source", f"Document {index}")
    source = Path(source).stem.replace("-", " ").replace("_", " ").strip()
    return source or f"Document {index}"


def expand_query(query: str) -> str:
    lowered = strip_accents(query.lower())
    expansions: list[str] = []
    for phrase, synonyms in QUERY_SYNONYMS.items():
        if phrase in lowered:
            expansions.extend(synonyms)
    if "dieu " in lowered:
        expansions.append("bo luat hinh su")
    if "nghe si" in lowered or "ca si" in lowered or "dien vien" in lowered or "rapper" in lowered:
        expansions.append("showbiz viet")
    if "luat" in lowered or "nghi dinh" in lowered:
        expansions.append("van ban phap luat")
    return normalize_text(f"{query} {' '.join(expansions)}")


def query_intent(query: str) -> str:
    lowered = strip_accents(query.lower())
    if any(token in lowered for token in ["nghe si", "ca si", "dien vien", "rapper", "huu tin", "cong tri", "long nhat"]):
        return "news"
    if any(token in lowered for token in ["dieu ", "luat", "nghi dinh", "hinh phat", "cai nghien", "to chuc su dung", "tang tru"]):
        return "legal"
    return "mixed"


def metadata_type_boost(query: str, metadata: dict) -> float:
    intent = query_intent(query)
    doc_type = metadata.get("type", "")
    if intent == "mixed":
        return 0.03
    if intent == doc_type:
        return 0.08
    return -0.02
