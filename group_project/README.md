# Group Project - Vietnam Drug Law RAG Assistant

## Product

Streamlit RAG chatbot for Vietnamese drug law and related news. The app answers legal/news questions with retrieved sources, citation cards, conversation memory, confidence badges, document explorer, keyword dashboard, and evaluation report.

## Architecture

```text
Streamlit UI
  -> Task 9 retrieve(query)
  -> semantic_search + lexical_search
  -> RRF merge + rerank
  -> PageIndex/API or local fallback
  -> Task 10 generate_with_citation
  -> citation cards + source preview
```

## How to run

```powershell
cd D:\vin_lab\Day08_RAG_pipeline_cohort2
pip install -r requirements.txt
pytest tests/ -v
python group_project\evaluation\eval_pipeline.py
streamlit run app.py
```

## Evaluation

Files:

- `group_project/evaluation/golden_dataset.json`
- `group_project/evaluation/eval_pipeline.py`
- `group_project/evaluation/results.md`

The evaluator compares:

- Baseline: BM25-only.
- Improved: hybrid retrieval + RRF/rerank + citation generation.

Metrics:

- Faithfulness.
- Answer relevance.
- Context recall.
- Context precision.

## Demo script

Ask:

1. `Ketamine co phai chat ma tuy khong?`
2. `Nguoi su dung ma tuy co bi xu ly hinh su khong?`
3. `Vu Cong Tri lien quan dieu luat nao?`
4. Follow-up: `Hanh vi do bi xu ly the nao?`

## Team members

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
| --- | --- | --- | --- |
| Nguyễn Hữu Đức | 2A202600683 | Task 1 — thu thập văn bản pháp luật (`src/task1_collect_legal_docs.py`, `data/landing/legal/`); Task 2 — crawl bài báo (`src/task2_crawl_news.py`, `data/landing/news/`) | Done |
| Nguyễn Duy Hưng | 2A202600578 | Task 3 — convert PDF/DOCX sang Markdown (`src/task3_convert_markdown.py`, `data/standardized/`); Task 4 — chunking + indexing (`src/task4_chunking_indexing.py`) | Done |
| Vương Sỹ Hạnh | 2A202600722 | Task 5 — semantic dense retrieval (`src/task5_semantic_search.py`); Task 6 — BM25 lexical retrieval (`src/task6_lexical_search.py`); tích hợp RAG pipeline + viết test (`tests/test_individual.py`) | Done |
| Đỗ Trung Kiên | 2A202600751 | Task 7 — reranking module (`src/task7_reranking.py`); Task 9 — hybrid retrieval pipeline với RRF (`src/task9_retrieval_pipeline.py`) | Done |
| Nguyễn Minh Khoa | 2A202600974 | Evaluation pipeline cho bài nhóm — golden dataset, metrics, A/B comparison, results report (`group_project/evaluation/`) | Done |
| Kiều Đức Long | 2A202600939 | Team lead + final integration: Streamlit UI (`app.py`), Task 8 PageIndex vectorless fallback (`src/task8_pageindex_vectorless.py`), Task 10 generation có citation (`src/task10_generation.py`), tài liệu dự án (`README.md`, `plan.md`, `steps.md`) | Done |


## API notes

- OpenAI is used when `OPENAI_API_KEY` is set.
- Jina reranker is used when `JINA_API_KEY` is set.
- PageIndex is attempted when `PAGEINDEX_API_KEY` is set.
- Safe local fallbacks keep the demo and tests working when external services are unavailable.
