"""Task 6 - lexical search using BM25 plus metadata-aware local ranking."""

from __future__ import annotations

import re
import unicodedata
from .task4_chunking_indexing import get_chunks

STOPWORDS = {
    "la", "cua", "va", "ve", "voi", "the", "nao", "duoc", "noi", "o", "nguon",
    "lien", "quan", "bi", "xu", "ly", "ra", "sao", "gi", "tu", "theo", "luat",
    "moi", "nhat", "trong", "cac", "nhung", "cho", "mot", "co", "phai", "chat",
}


def _fix_mojibake(text: str) -> str:
    # Repair common UTF-8-as-cp1252 mojibake without touching normal Vietnamese text.
    if "Ã" in text or "Â" in text:
        try:
            return text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return text
    return text


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _tokens(text: str) -> list[str]:
    text = _strip_accents(_fix_mojibake(text).lower())
    return re.findall(r"[a-z0-9]+", text)


def _signal_tokens(text: str) -> list[str]:
    return [tok for tok in _tokens(text) if tok not in STOPWORDS and (len(tok) > 2 or any(ch.isdigit() for ch in tok))]


def build_bm25_index(corpus: list[dict]):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([_signal_tokens(doc["content"]) for doc in corpus])


def _metadata_text(doc: dict) -> str:
    md = doc.get("metadata", {})
    return " ".join(str(md.get(k, "")) for k in ["title", "source", "type", "path"])


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    corpus = get_chunks()
    if not corpus:
        return []

    q_tokens = _signal_tokens(query)
    q_set = set(q_tokens)
    q_numbers = {tok for tok in q_tokens if any(ch.isdigit() for ch in tok)}
    if not q_tokens:
        return []

    doc_tokens = [_signal_tokens(doc["content"]) for doc in corpus]
    meta_tokens = [_signal_tokens(_metadata_text(doc)) for doc in corpus]
    overlap_scores = []
    for body, meta in zip(doc_tokens, meta_tokens):
        body_set = set(body)
        meta_set = set(meta)
        body_overlap = len(q_set & body_set)
        meta_overlap = len(q_set & meta_set)
        number_boost = 3 * len(q_numbers & (body_set | meta_set))
        phrase_boost = 1.5 if _strip_accents(query.lower())[:24] in _strip_accents((" ".join(meta)).lower()) else 0
        overlap_scores.append(body_overlap + 2.2 * meta_overlap + number_boost + phrase_boost)

    try:
        bm25 = build_bm25_index(corpus)
        raw_scores = [float(s) for s in bm25.get_scores(q_tokens)]
        min_score = min(raw_scores) if raw_scores else 0.0
        scores = [(score - min_score) + overlap for score, overlap in zip(raw_scores, overlap_scores)]
    except Exception:
        scores = overlap_scores

    ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
    results = []
    seen_sources: set[str] = set()
    for idx, score in ranked:
        if len(results) >= top_k:
            break
        if float(score) <= 0:
            continue
        metadata = corpus[idx].get("metadata", {})
        source_key = str(metadata.get("source", idx))
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        results.append({"content": corpus[idx]["content"], "score": float(score), "metadata": metadata})
    return results


if __name__ == "__main__":
    print(lexical_search("ma túy", 3))
