# Steps - Checklist trien khai Vietnam Drug Law RAG Assistant

Tai lieu nay la checklist hanh dong de implement repo theo PRD va rubric cham diem. Lam theo thu tu tu Phase 0 den Phase 6 de toi da hoa diem va giam rui ro demo.

## Phase 0 - Setup va baseline check

### 0.1. Kiem tra moi truong

Lenh chay:

```bash
python --version
pip install -r requirements.txt
pytest tests/ -v
```

Done when:

- Python chay duoc.
- Dependencies cai duoc hoac loi cai dat duoc ghi lai.
- Biet ro test nao dang fail/pass truoc khi sua.

### 0.2. Tao `.env`

Lenh chay:

```bash
copy .env.example .env
```

Gia tri can dien neu co:

```env
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
JINA_API_KEY=<YOUR_JINA_API_KEY>
PAGEINDEX_API_KEY=<YOUR_PAGEINDEX_API_KEY>
```

Done when:

- `.env` ton tai local.
- Khong commit secret.
- Code co fallback khi cac key trong `.env` bi thieu.

### 0.3. Kiem tra data hien co

Lenh chay:

```bash
python -c "from pathlib import Path; print(len(list(Path('data/landing/legal').glob('*')))); print(len(list(Path('data/landing/news').glob('*'))))"
```

Done when:

- `data/landing/legal/` co it nhat 3 file PDF/DOC/DOCX, moi file > 1KB.
- `data/landing/news/` co it nhat 5 file JSON/HTML/MD/TXT, moi file co noi dung.
- News JSON co truong `url` de pass test metadata.

## Phase 1 - Data pipeline: Task 1 den Task 3

### 1.1. Hoan thien Task 1 legal docs

Can co trong `data/landing/legal/`:

- Bo luat Hinh su 2015.
- Luat Phong, chong ma tuy 2021.
- Luat Xu ly vi pham hanh chinh.
- Nghi dinh 105/2021.
- Nghi dinh 57/2022.

Done when:

- `pytest tests/test_individual.py::TestTask1 -v` pass.
- Ten file ro nghia, khong rong, uu tien lowercase snake/kebab case.

### 1.2. Hoan thien Task 2 news corpus

Can co trong `data/landing/news/`:

- Toi thieu 5 bai de pass tests.
- Uu tien 20-30 bai theo PRD de demo/retrieval tot hon.
- Moi JSON nen co `url`, `title`, `crawl_date`, `source`, `content` hoac `markdown`.

Done when:

- `pytest tests/test_individual.py::TestTask2 -v` pass.
- Co du bai ve mua ban, van chuyen, cai nghien, chinh sach, nghe si lien quan ma tuy.

### 1.3. Hoan thien Task 3 Markdown conversion

Implement/chay conversion sao cho output co cau truc:

```text
data/standardized/
  legal/*.md
  news/*.md
```

Done when:

- `pytest tests/test_individual.py::TestTask3 -v` pass.
- Markdown co noi dung > 200 chars.
- Legal va news deu duoc convert neu co the.

## Phase 2 - Retrieval core: Task 4 den Task 7

### 2.1. Implement Task 4 chunking/indexing

Yeu cau implementation:

- `load_documents()` doc tat ca `.md` trong `data/standardized/`.
- `chunk_documents(documents)` tao chunks voi `content` va `metadata`.
- `CHUNK_SIZE > 0`, `CHUNK_OVERLAP > 0`, `CHUNK_OVERLAP < CHUNK_SIZE`.
- Chunk khong vuot qua `CHUNK_SIZE * 1.1` trong tests.
- `embed_chunks()` va `index_to_vectorstore()` nen co fallback/cache neu vector DB that bai.

Recommended defaults:

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL = "BAAI/bge-m3"
```

Done when:

- `pytest tests/test_individual.py::TestTask4 -v` pass.
- Co comment ngan giai thich chunk size, overlap, embedding model.

### 2.2. Implement Task 5 semantic search

Yeu cau implementation:

- Function: `semantic_search(query: str, top_k: int = 10) -> list[dict]`.
- Tra ve toi da `top_k` results.
- Moi result co `content`, `score`, `metadata`.
- Results sorted by `score` descending.
- Neu embedding/vector store khong san sang, fallback bang lightweight similarity tren chunks local de tests/demo khong crash.

Done when:

- `pytest tests/test_individual.py::TestTask5 -v` pass.
- Query "phap luat ma tuy" tra ve ket qua co lien quan.

### 2.3. Implement Task 6 lexical BM25

Yeu cau implementation:

- Function: `lexical_search(query: str, top_k: int = 10) -> list[dict]`.
- Dung `rank-bm25` hoac fallback token overlap neu dependency loi.
- Score keyword match phai > 0 voi query "ma tuy".
- Results sorted descending.

Done when:

- `pytest tests/test_individual.py::TestTask6 -v` pass.
- BM25 co the giai thich trong demo: TF, IDF, length normalization.

### 2.4. Implement Task 7 reranking

Yeu cau implementation:

- `rerank(query, candidates, top_k=5)` khong crash voi dummy candidates.
- Ton trong `top_k`.
- Output co `score` va sorted descending.
- Implement `rerank_rrf(ranked_lists, top_k=5, k=60)` de merge semantic + lexical.
- Neu cross-encoder/API khong co, fallback local scoring/RRF.

Done when:

- `pytest tests/test_individual.py::TestTask7 -v` pass.
- Co the giai thich RRF/MMR/rerank trong demo.

## Phase 3 - End-to-end RAG: Task 8 den Task 10

### 3.1. Implement Task 8 PageIndex/vectorless fallback

Yeu cau implementation:

- Function: `pageindex_search(query: str, top_k: int = 5) -> list[dict]`.
- Neu PageIndex API key co san, query PageIndex.
- Neu khong co API key, fallback local search nhung gan `source = "pageindex"` cho ket qua fallback de dung interface.
- Khong crash khi PageIndex unavailable.

Done when:

- `pytest tests/test_individual.py::TestTask8 -v` pass hoac skip hop ly neu external service that bai.
- Fallback query obscure khong lam pipeline crash.

### 3.2. Implement Task 9 retrieval pipeline

Yeu cau implementation:

1. Goi `semantic_search(query, top_k * 2)`.
2. Goi `lexical_search(query, top_k * 2)`.
3. Merge bang RRF hoac weighted fusion.
4. Gan `source = "hybrid"` cho hybrid results.
5. Rerank neu `use_reranking = True`.
6. Neu khong co results hoac top score < `score_threshold`, fallback sang `pageindex_search`.
7. Tra ve toi da `top_k` results.

Done when:

- `pytest tests/test_individual.py::TestTask9 -v` pass.
- `retrieve("xyzabc123nonsense", top_k=3, score_threshold=0.99)` khong crash.

### 3.3. Implement Task 10 generation with citation

Yeu cau implementation:

- `reorder_for_llm(chunks)` giu chunk best o dau, phan bo chunk quan trong o dau/cuoi.
- `format_context(chunks)` dua source/type/chunk index vao context.
- `generate_with_citation(query, top_k=5)` tra dict co `answer`, `sources`, `retrieval_source`.
- Neu co OpenAI key, goi LLM voi temperature thap.
- Neu khong co key, tra loi extractive tu top chunks kem citation de test/demo khong crash.

Done when:

- `pytest tests/test_individual.py::TestTask10 -v` pass hoac skip chi khi external LLM loi.
- Answer co citation dang `[source]` hoac `[Nguon, Nam/Dieu]`.
- Neu context thieu, answer noi ro khong du thong tin.

## Phase 4 - Group Streamlit app va UX

### 4.1. Tao app Streamlit

Recommended file:

```text
app.py
```

Luon co cac thanh phan:

- Sidebar navigation: Chat, Legal Documents, News Articles, Evaluation Dashboard.
- Suggested questions.
- Chat history trong `st.session_state`.
- Status indicators khi retrieve/generate.
- Citation cards cho moi source.
- Expandable source preview.

Lenh chay:

```bash
streamlit run app.py
```

Done when:

- App mo duoc local.
- User hoi cau demo va nhan answer + sources.
- Source cards co title/source/snippet/score.

### 4.2. Conversation memory

Implementation rule:

- Luu 3-5 messages gan nhat.
- Voi follow-up, tao query mo rong bang latest user question + context ngan tu previous turn.
- Van retrieve lai evidence moi.

Done when:

- Flow demo chay duoc:
  - User: "Vu Cong Tri thi sao?"
  - User: "Hanh vi do bi xu ly the nao?"
  - Assistant hieu "hanh vi do" lien quan turn truoc va tra loi co source.

### 4.3. Bonus UX features

Implement theo thu tu uu tien:

1. Confidence badge High/Medium/Low.
2. Source highlight keyword overlap.
3. Document explorer cho legal/news.
4. Top keywords visualization.
5. Dark mode toggle.

Done when:

- It nhat 2 bonus UX features hoat dong trong demo.
- UI dung mau navy/white/gray, chuyen nghiep, de doc.

## Phase 5 - Evaluation pipeline

### 5.1. Golden dataset

File:

```text
group_project/evaluation/golden_dataset.json
```

Yeu cau:

- 15+ records.
- Moi record co `question`, `expected_answer`, `expected_context`.
- Bao phu legal-only, news-only, multi-document reasoning, follow-up style.

Done when:

- JSON valid.
- Co it nhat 15 Q&A.
- Cac cau demo bat buoc nam trong dataset.

### 5.2. Evaluation script

File:

```text
group_project/evaluation/eval_pipeline.py
```

Yeu cau:

- Chay pipeline tren golden dataset.
- Tinh 4 metrics: faithfulness, answer relevance, context recall, context precision.
- Chay 2 configs:
  - Baseline: dense-only hoac BM25-only, no rerank.
  - Improved: hybrid + RRF/rerank.
- Export results sang markdown/CSV neu co the.

Done when:

- Script chay duoc hoac co fallback/mock scoring ro rang neu thieu API eval.
- Ket qua co baseline vs improved.

### 5.3. Results report

File:

```text
group_project/evaluation/results.md
```

Noi dung bat buoc:

- Bang diem tung metric cho baseline va improved.
- Phan tich worst performers.
- Nguyen nhan loi.
- De xuat cai thien.
- Ngay chay eval va config/model su dung.

Done when:

- Reviewer doc file co the thay du 12 diem evaluation deliverable.

## Phase 6 - QA, demo, va nop bai

### 6.1. Automated tests

Lenh chay:

```bash
pytest tests/ -v
```

Neu fail tung task, debug bang:

```bash
pytest tests/test_individual.py::TestTask1 -v
pytest tests/test_individual.py::TestTask2 -v
pytest tests/test_individual.py::TestTask3 -v
pytest tests/test_individual.py::TestTask4 -v
pytest tests/test_individual.py::TestTask5 -v
pytest tests/test_individual.py::TestTask6 -v
pytest tests/test_individual.py::TestTask7 -v
pytest tests/test_individual.py::TestTask8 -v
pytest tests/test_individual.py::TestTask9 -v
pytest tests/test_individual.py::TestTask10 -v
```

Done when:

- Tat ca tests pass, hoac moi test fail co ly do va mitigation ro trong README.

### 6.2. Manual demo checklist

Chay app:

```bash
streamlit run app.py
```

Demo script:

1. Hoi: "Ketamine co phai chat ma tuy khong?"
2. Mo citation card lien quan Nghi dinh 57/2022.
3. Hoi: "Nguoi su dung ma tuy co bi xu ly hinh su khong?"
4. Chi ra legal citation va confidence badge.
5. Hoi: "Vu Cong Tri lien quan dieu luat nao?"
6. Chi ra multi-document reasoning: news + legal.
7. Hoi follow-up: "Hanh vi do bi xu ly the nao?"
8. Mo Evaluation Dashboard va giai thich baseline vs improved.

Done when:

- Demo khong crash.
- Tat ca answer co source.
- Khi thieu evidence, assistant noi khong du thong tin thay vi bia.

### 6.3. README va group README

Cap nhat README/group README de co:

- Kien truc he thong.
- Cach setup/chay test/chay app.
- Phan cong thanh vien voi placeholder neu chua co:
  - `<MEMBER_NAME> - <STUDENT_ID> - Retrieval`
  - `<MEMBER_NAME> - <STUDENT_ID> - Generation/UI`
  - `<MEMBER_NAME> - <STUDENT_ID> - Evaluation/Data`
- Link demo/deploy/video: `<DEMO_URL>`.
- Ket qua tests/evaluation moi nhat.

Done when:

- Reviewer co the clone repo, doc README, chay test/app theo huong dan.

### 6.4. Final scoring checklist

Truoc khi nop, check:

- [ ] Task 1: 3+ legal files.
- [ ] Task 2: 5+ news files va metadata `url`.
- [ ] Task 3: markdown standardized co content.
- [ ] Task 4: chunking/indexing pass tests.
- [ ] Task 5: semantic search sorted output.
- [ ] Task 6: BM25 lexical search sorted output.
- [ ] Task 7: rerank/RRF output co score.
- [ ] Task 8: PageIndex/fallback output co `source = "pageindex"`.
- [ ] Task 9: retrieve hybrid/fallback output co `source`.
- [ ] Task 10: generation dict co answer/citation/sources.
- [ ] Streamlit chatbot chay duoc.
- [ ] Conversation memory hoat dong.
- [ ] Source cards/source preview hoat dong.
- [ ] Evaluation golden dataset 15+ Q&A.
- [ ] Evaluation co 4 metrics.
- [ ] Evaluation co A/B comparison.
- [ ] `results.md` co worst performers.
- [ ] README/group README day du.
- [ ] Khong commit API keys.

## Suggested implementation order neu thoi gian gap

Neu chi con it thoi gian, lam theo thu tu uu tien nay:

1. Pass Task 1-10 automated tests.
2. Tao Streamlit chat toi thieu voi answer + citations.
3. Tao golden dataset 15 Q&A va results report.
4. Them confidence badge va source preview.
5. Them document explorer/top keywords/dark mode.
6. Lam dep README va demo script.
