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
| Kiều Đức Long | 2A202600939 | UI implementation, UX polishing, team lead, final integration | Done |
| Nguyễn Duy Hưng | 2A202600578 | Markdown conversion + chunking | Done |
| Vương Sỹ Hạnh | 2A202600722 | Semantic search + BM25 lexical retrieval | Done |
| Đỗ Trung Kiên | 2A202600751 | Reranking + hybrid RRF pipeline | Done |
| Nguyễn Minh Khoa | 2A202600974 | Evaluation pipeline + report + demo script | Done |
| Nguyễn Hữu Đức | 2A202600683 | Data collection + news/legal corpus | Done |


## API notes

- OpenAI is used when `OPENAI_API_KEY` is set.
- Jina reranker is used when `JINA_API_KEY` is set.
- PageIndex is attempted when `PAGEINDEX_API_KEY` is set.
- Safe local fallbacks keep the demo and tests working when external services are unavailable.
