# RAG Evaluation Results

Framework: offline lexical/context-overlap evaluator (API-free fallback).

## Overall Scores

| Metric | Baseline BM25 | Improved Hybrid + Rerank | Delta |
| --- | ---: | ---: | ---: |
| faithfulness | 0.707 | 0.719 | +0.012 |
| answer_relevance | 0.587 | 0.867 | +0.280 |
| context_recall | 0.504 | 0.554 | +0.050 |
| context_precision | 0.955 | 0.864 | -0.091 |
| average | 0.688 | 0.751 | +0.063 |

## Worst Performers

| # | Question | Faithfulness | Relevance | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Ketamine có phải chất ma túy không? | 0.705 | 0.450 | 0.150 | 0.289 |
| 2 | Vận chuyển ma túy bị phạt bao nhiêu năm? | 0.523 | 0.571 | 0.318 | 0.693 |
| 3 | Tin tức về nghệ sĩ liên quan ma túy cần đối chiếu gì? | 0.622 | 1.000 | 0.250 | 0.603 |

## Recommendations

- Add real OpenAI/RAGAS or DeepEval metrics when API keys are available.
- Improve Vietnamese tokenization and legal-section metadata for higher recall.
- Replace offline PageIndex fallback with real PageIndex after PAGEINDEX_API_KEY is provided.
