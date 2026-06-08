"""
Task 6 — Lexical Search Module (BM25)
"""

from pathlib import Path
from rank_bm25 import BM25Okapi


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

CORPUS = []
bm25 = None
tokenized_corpus = []


# =========================
# LOAD CORPUS
# =========================
def load_corpus():
    """
    Load toàn bộ markdown từ data/standardized/
    """
    corpus = []

    for file in STANDARDIZED_DIR.rglob("*.md"):
        content = file.read_text(encoding="utf-8").strip()

        if not content:
            continue

        corpus.append({
            "content": content,
            "metadata": {
                "source": str(file.relative_to(STANDARDIZED_DIR))
            }
        })

    return corpus


# =========================
# BUILD BM25 INDEX
# =========================
def build_bm25_index(corpus: list[dict]):
    """
    Xây BM25 index
    """
    global bm25, CORPUS, tokenized_corpus

    CORPUS = corpus

    tokenized_corpus = [
        doc["content"].lower().split()
        for doc in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# =========================
# LEXICAL SEARCH
# =========================
def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    BM25 search
    """
    global bm25, CORPUS

    if bm25 is None:
        corpus = load_corpus()
        build_bm25_index(corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # sort indices by score desc
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])

        if score <= 0:
            continue

        results.append({
            "content": CORPUS[idx]["content"],
            "score": score,
            "metadata": CORPUS[idx]["metadata"]
        })

    return results


# =========================
# TEST
# =========================
if __name__ == "__main__":
    results = lexical_search("tàng trữ trái phép chất ma tuý", top_k=5)

    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")