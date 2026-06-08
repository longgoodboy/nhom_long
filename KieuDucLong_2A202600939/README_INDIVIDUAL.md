# Individual Submission - Kieu Duc Long 2A202600939

This folder contains the individual lab deliverables for the RAG pipeline tasks.

## Contents

- `src/task1_collect_legal_docs.py` to `src/task10_generation.py`: individual task implementations.
- `data/landing/`: raw legal/news source files used by collection and conversion tasks.
- `data/standardized/`: Markdown corpus used by chunking, retrieval, reranking, and generation.
- `tests/test_individual.py`: individual task test suite from the original repo.
- `requirements.txt`: dependencies.
- `.env.example`: environment variable template only. Real API keys are intentionally not included.

## How to validate from repo root

```powershell
pytest tests/ -q
```

## Notes

The original repo structure is kept unchanged so automated tests and the group Streamlit app still run normally. This folder is a clean copy for personal submission/review.
