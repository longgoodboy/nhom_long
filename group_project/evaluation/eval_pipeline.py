"""Offline-friendly RAG evaluation pipeline for the group project."""

from __future__ import annotations
import json
from pathlib import Path
from statistics import mean
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def _tokens(text: str) -> set[str]:
    return {t.strip(".,;:!?()[]{}\"'").lower() for t in text.split() if len(t.strip(".,;:!?()[]{}\"'")) > 2}


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _score_case(item: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    context_text = " ".join(c.get("content", "") for c in sources)
    expected = item.get("expected_answer", "") + " " + item.get("expected_context", "")
    q_tokens = _tokens(item.get("question", ""))
    expected_tokens = _tokens(expected)
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(context_text)
    has_citation = ("[" in answer and "]" in answer) or "ngu?n:" in answer.lower() or "source:" in answer.lower()
    answer_grounding = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    source_support = min(1.0, len(sources) / 3)
    faithfulness = min(1.0, 0.45 * answer_grounding + 0.35 * source_support + (0.20 if has_citation else 0.05))
    answer_relevance = max(
        len(q_tokens & answer_tokens) / max(len(q_tokens), 1),
        0.75 * len(q_tokens & context_tokens) / max(len(q_tokens), 1),
    )
    context_recall = len(expected_tokens & context_tokens) / max(len(expected_tokens), 1)
    useful = len(q_tokens & context_tokens) + len(expected_tokens & context_tokens)
    context_precision = min(1.0, useful / max(len(context_tokens), 1) * 8)
    return {
        "question": item["question"],
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "context_recall": round(context_recall, 3),
        "context_precision": round(context_precision, 3),
        "source_count": len(sources),
    }


def run_config(name: str, dataset: list[dict]) -> dict:
    rows = []
    for item in dataset:
        if name == "baseline_bm25":
            sources = lexical_search(item["question"], top_k=2)
            answer = " ".join(f"{' '.join(s['content'].split())[:120]}" for s in sources[:1]) or "Khong tim thay du thong tin."
            result = {"answer": answer, "sources": sources}
        elif name == "baseline_dense":
            sources = semantic_search(item["question"], top_k=5)
            answer = "\n".join(f"- {' '.join(s['content'].split())[:220]} [{s.get('metadata', {}).get('source', 'source')}]" for s in sources[:3]) or "Khong tim thay du thong tin."
            result = {"answer": answer, "sources": sources}
        else:
            result = generate_with_citation(item["question"], top_k=5)
        rows.append(_score_case(item, result))
    metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    averages = {m: round(mean(row[m] for row in rows), 3) for m in metrics}
    averages["average"] = round(mean(averages.values()), 3)
    return {"name": name, "averages": averages, "rows": rows}


def compare_configs(rag_pipeline=None, golden_dataset: list[dict] | None = None):
    dataset = golden_dataset or load_golden_dataset()
    return {
        "baseline_bm25": run_config("baseline_bm25", dataset),
        "improved_hybrid_rerank": run_config("improved_hybrid_rerank", dataset),
    }


def export_results(results: dict, comparison: dict | None = None):
    comparison = comparison or results
    base = comparison["baseline_bm25"]
    improved = comparison["improved_hybrid_rerank"]
    metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "average"]
    content = ["# RAG Evaluation Results", "", "Framework: offline lexical/context-overlap evaluator (API-free fallback).", "", "## Overall Scores", "", "| Metric | Baseline BM25 | Improved Hybrid + Rerank | Delta |", "| --- | ---: | ---: | ---: |"]
    for m in metrics:
        delta = improved["averages"][m] - base["averages"][m]
        content.append(f"| {m} | {base['averages'][m]:.3f} | {improved['averages'][m]:.3f} | {delta:+.3f} |")
    worst = sorted(improved["rows"], key=lambda r: (r["faithfulness"] + r["answer_relevance"] + r["context_recall"] + r["context_precision"]))[:3]
    content += ["", "## Worst Performers", "", "| # | Question | Faithfulness | Relevance | Recall | Precision |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for i, row in enumerate(worst, 1):
        content.append(f"| {i} | {row['question']} | {row['faithfulness']:.3f} | {row['answer_relevance']:.3f} | {row['context_recall']:.3f} | {row['context_precision']:.3f} |")
    content += ["", "## Recommendations", "", "- Add real OpenAI/RAGAS or DeepEval metrics when API keys are available.", "- Improve Vietnamese tokenization and legal-section metadata for higher recall.", "- Replace offline PageIndex fallback with real PageIndex after PAGEINDEX_API_KEY is provided."]
    RESULTS_PATH.write_text("\n".join(content) + "\n", encoding="utf-8")


def main():
    dataset = load_golden_dataset()
    comparison = compare_configs(golden_dataset=dataset)
    export_results(comparison)
    print(f"Evaluated {len(dataset)} cases; wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
