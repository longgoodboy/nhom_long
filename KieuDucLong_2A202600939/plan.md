# Plan - Vietnam Drug Law RAG Assistant

## 1. Muc tieu san pham

Vietnam Drug Law RAG Assistant la he thong Legal RAG giup nguoi dung tra cuu quy dinh phap luat Viet Nam ve ma tuy va doi chieu voi tin tuc lien quan. San pham phai uu tien tinh dung nguon, tra loi co citation, giam hallucination, va co giao dien de dung cho ca nguoi khong co nen tang phap ly.

Doi tuong chinh:

- Sinh vien Luat: tra cuu nhanh dieu luat, muc phat, khai niem phap ly.
- Nguoi dan: hieu quy dinh ve su dung, tang tru, van chuyen, cai nghien ma tuy.
- Nha nghien cuu: so sanh tin tuc thuc te voi can cu phap luat va dieu khoan lien quan.

Cau hoi demo bat buoc:

- "Ketamine co phai chat ma tuy khong?"
- "Nguoi su dung ma tuy co bi xu ly hinh su khong?"
- "Vu Cong Tri lien quan dieu luat nao?"

## 2. Chien luoc dat diem toi da

Tong diem repo goc gom 3 phan: bai ca nhan 50 diem, bai nhom 30 diem, bonus/manual review 20 diem. Ke hoach trien khai uu tien theo thu tu sau:

1. Dam bao automated tests cho 10 task ca nhan chay qua bang `pytest tests/ -v`.
2. Xay dung chatbot Streamlit demo duoc end-to-end voi citation va source preview.
3. Hoan thien evaluation pipeline voi golden dataset 15+ Q&A, 4 metrics, A/B comparison, va bao cao worst performers.
4. Bo sung UX bonus: source highlight, document explorer, top keywords, dark mode, confidence badge.
5. Cap nhat README/bao cao demo de reviewer thay ro kien truc, phan cong, cach chay, cach cham diem.

Bang muc tieu diem:

| Hang muc | Diem toi da | Chien luoc |
| --- | ---: | --- |
| Task 1-3 data pipeline | 10 | Co du legal/news raw data va markdown standardized |
| Task 4-7 retrieval core | 25 | Chunking, semantic, BM25, RRF/rerank dung interface va sorted score |
| Task 8-10 E2E RAG | 15 | PageIndex/fallback, retrieve, generation citation, reorder context |
| Group chatbot | 18 | Streamlit chat, memory, citations, source docs, architecture README |
| Evaluation | 12 | Golden dataset, 4 metrics, A/B, results analysis |
| Bonus/manual | 20 | UX bonus, robust demo, cau hoi kho co fallback khong hallucinate |

## 3. Kien truc he thong

Luon giu pipeline ro rang, de demo va de giai thich:

```text
Raw legal PDFs + news JSON/HTML
  -> MarkItDown / markdown conversion
  -> data/standardized/legal + data/standardized/news
  -> chunking with metadata
  -> embeddings + vector store
  -> semantic_search(query)
  -> lexical_search(query) with BM25
  -> merge with RRF / weighted fusion
  -> rerank top candidates
  -> retrieve(query) with PageIndex/local fallback
  -> reorder_for_llm + format_context
  -> generate_with_citation
  -> Streamlit chat UI + citation cards + evaluation dashboard
```

Recommended technical choices:

- Frontend: Streamlit vi da co trong requirements va phu hop demo nhanh.
- Backend: Python modules trong `src/` theo dung task goc.
- Chunking: `RecursiveCharacterTextSplitter`, `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 50` de dap ung tests va tranh chunk qua dai.
- Embedding: uu tien `BAAI/bge-m3` cho tieng Viet; neu may yeu/API han che thi fallback local lightweight hoac cache embedding.
- Vector store: uu tien local-friendly implementation de tests pass; co the dung Chroma/FAISS/cache JSON neu Weaviate chua san sang.
- Lexical: BM25 voi `rank-bm25`, tokenize lower-case/split don gian de on dinh.
- Hybrid: RRF de hop nhat semantic + BM25, tranh phu thuoc mot retriever.
- Reranker: uu tien RRF/MMR local fallback; cross-encoder/Jina co the la bonus neu API key san sang.
- Generation: OpenAI neu co `OPENAI_API_KEY`; neu khong co key thi tra loi extractive tu context kem citation de demo/test khong crash.

## 4. Public interfaces bat buoc

Cac function sau phai ton tai va tra output dung format de automated tests cham diem:

| File | Interface | Output toi thieu |
| --- | --- | --- |
| `src/task4_chunking_indexing.py` | `load_documents()` | `list[dict]` co `content`, `metadata` |
| `src/task4_chunking_indexing.py` | `chunk_documents(documents)` | chunks co `content`, `metadata`, size hop le |
| `src/task5_semantic_search.py` | `semantic_search(query, top_k=10)` | sorted `list` co `content`, `score`, `metadata` |
| `src/task6_lexical_search.py` | `lexical_search(query, top_k=10)` | sorted `list` co `content`, `score`, `metadata` |
| `src/task7_reranking.py` | `rerank(query, candidates, top_k=5)` | sorted `list`, ton trong `top_k`, co `score` |
| `src/task8_pageindex_vectorless.py` | `pageindex_search(query, top_k=5)` | `list`, neu co result thi `source = "pageindex"` |
| `src/task9_retrieval_pipeline.py` | `retrieve(query, top_k=5, score_threshold=0.3)` | `list` co `content`, `score`, `source` la `hybrid` hoac `pageindex` |
| `src/task10_generation.py` | `reorder_for_llm(chunks)` | giu nguyen so chunk, chunk quan trong nhat o dau |
| `src/task10_generation.py` | `format_context(chunks)` | string co source label de citation |
| `src/task10_generation.py` | `generate_with_citation(query, top_k=5)` | dict co `answer`, `sources`, `retrieval_source` |

Nguyen tac output:

- Khong crash khi thieu API key; fallback bang local/extractive answer.
- Score phai la `float` va ket qua phai sorted descending.
- Metadata phai giu `source`, `type`/`doc_type`, `chunk_index` neu co.
- Citation phai dua tren source metadata, khong tu tao nguon.

## 5. Du lieu va corpus

Legal corpus toi thieu:

- Bo luat Hinh su 2015, uu tien cac dieu ve toi pham ma tuy.
- Luat Phong, chong ma tuy 2021.
- Luat Xu ly vi pham hanh chinh.
- Nghi dinh 105/2021.
- Nghi dinh 57/2022.

News corpus:

- 20-30 bai bao neu co the, toi thieu 5 de pass tests.
- Nhom chu de: mua ban, van chuyen, cai nghien, chinh sach, nghe si lien quan ma tuy.
- Moi bai nen co `url`, `title`, `published_date`/`crawl_date`, `source`, `content`/`markdown`.

Metadata chuan nen dung cho moi chunk:

```json
{
  "source": "ten_file_hoac_url",
  "title": "tieu de tai lieu",
  "type": "legal_or_news",
  "doc_id": "stable_id",
  "chunk_index": 0,
  "section": "Dieu/heading neu co",
  "url": "optional"
}
```

## 6. UI/UX plan

Streamlit app nen gom 3 khu vuc chinh:

- Sidebar:
  - Legal Documents.
  - News Articles.
  - Evaluation Dashboard.
  - Dark mode toggle.
  - Top keywords.
- Main chat:
  - Suggested questions tren trang chu.
  - Chat history kieu ChatGPT.
  - Search status: "Searching Legal Docs...", "Searching News...", "Reranking Results...", "Generating Answer...".
  - Assistant answer co citation inline.
  - Confidence badge: High/Medium/Low dua tren retrieval score va so source lien quan.
- Source/Citation area:
  - Citation cards hien title, source type, score, snippet.
  - Click/expand de xem chunk day du.
  - Highlight doan lien quan neu co keyword overlap voi query.

Conversation memory:

- Luu chat history trong `st.session_state`.
- Khi user hoi follow-up nhu "Hanh vi do bi xu ly the nao?", build standalone query tu 1-3 turn gan nhat.
- Van bat buoc retrieve lai context, khong chi dua vao memory de tra loi.

## 7. Evaluation plan

Framework uu tien: RAGAS hoac DeepEval. Neu API/eval model kho cai dat, van can co script va results report mau/ket qua da chay duoc trong moi truong co key.

Deliverables bat buoc:

- `group_project/evaluation/golden_dataset.json`: 15+ Q&A pairs.
- `group_project/evaluation/eval_pipeline.py`: script chay eval.
- `group_project/evaluation/results.md`: bang diem, A/B comparison, worst performers, de xuat cai tien.

Metrics bat buoc:

- Faithfulness.
- Answer Relevance.
- Context Recall.
- Context Precision.

A/B configs toi thieu:

- Baseline: dense-only hoac BM25-only, khong rerank.
- Improved: hybrid BM25 + semantic + RRF/rerank.

Bao cao nen co:

- Bang diem trung binh tung metric.
- Top 3 cau hoi kem nhat va ly do.
- Loi thuong gap: retrieve sai source, context thieu dieu luat, answer qua dai, citation chua sat.
- Ke hoach cai thien: query rewriting, metadata filter legal/news, reranker multilingual, source highlight.

## 8. Prompt va hallucination policy

Generation prompt phai theo quy tac:

- Chi dung context duoc cung cap.
- Moi khang dinh thuc te phai co citation.
- Neu context khong du, tra loi ro: "Khong tim thay du thong tin trong co so du lieu." hoac "Toi khong the xac minh thong tin nay tu nguon hien co."
- Khong tu dua loi khuyen phap ly ca nhan; nen them disclaimer ngan: "Thong tin chi mang tinh tham khao, khong thay the tu van phap ly chuyen nghiep."

Output format nen gom:

```text
Answer

Sources
- [Nguon, nam/dieu] snippet...

Confidence: High/Medium/Low
```

Confidence rule de demo:

- High: top score >= 0.70 va co it nhat 2 source lien quan.
- Medium: top score >= 0.40 hoac chi co 1 source tot.
- Low: top score < 0.40, can canh bao nguoi dung.

## 9. Rui ro va giam thieu

| Rui ro | Anh huong | Giam thieu |
| --- | --- | --- |
| Thieu API key OpenAI/Jina/PageIndex | App/test crash | Local fallback cho search, rerank, generation extractive |
| Encoding tieng Viet loi | Cau tra loi/xu ly keyword kem | Doc/ghi UTF-8, normalize text, tranh copy noi dung mojibake vao docs moi |
| Weaviate/Docker kho chay | Khong index duoc | Fallback local JSON/Chroma/FAISS hoac in-memory corpus cho tests |
| Latency > 5s | Demo kem | Cache documents/chunks/index, gioi han top_k, preload model |
| Citation sai hoac hallucination | Mat diem manual | Strict prompt, source cards, answer insufficient evidence khi thieu context |
| News/legal bi tron ngu canh | Multi-doc reasoning sai | Metadata filter va hien source type ro rang |

## 10. Acceptance criteria

Du an san sang nop/demo khi:

- `pytest tests/ -v` pass hoac cac loi con lai duoc ghi ro trong README/results.
- Streamlit app chay duoc local bang `streamlit run app.py`.
- 3 demo scenarios tra loi co citation va hien source cards.
- Follow-up question su dung conversation memory nhung van retrieve lai evidence.
- Evaluation dashboard/report co 4 metrics va A/B comparison.
- README/group README co kien truc, cach chay, phan cong, va link/video demo neu co.
- Khong co hardcoded secret/API key trong repo.

## 11. Placeholders can dien truoc khi nop

- Ten nhom: `<TEAM_NAME>`
- Thanh vien/MSSV: `<MEMBER_NAME> - <STUDENT_ID>`
- OpenAI API key: dat trong `.env`, khong commit.
- Jina/PageIndex API key: dat trong `.env`, khong commit.
- Demo URL hoac video: `<DEMO_URL>`
- Ket qua evaluation thuc te: `<EVAL_RESULTS_DATE>`
