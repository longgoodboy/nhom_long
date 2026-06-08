# Plan Để Hoàn Thành Bài Lab RAG Pipeline

## Tổng Quan
Bài lab bao gồm 2 phần chính:
1. **Bài Cá Nhân** (50% điểm): 10 tasks từ thu thập data đến generation có citation
2. **Bài Nhóm** (30% điểm): Xây dựng RAG Chatbot hoặc Evaluation Pipeline
3. **Bonus** (20% điểm): Thử thách LLM với câu hỏi khó

## Giai Đoạn 1: Hoàn Thành Bài Cá Nhân (5 tasks còn lại)

### Task 4: Chunking & Indexing
- [ ] Chọn chunking strategy: RecursiveCharacterTextSplitter (an toàn, phổ biến)
- [ ] Chọn embedding model: BAAI/bge-m3 (tốt cho tiếng Việt)
- [ ] Cài đặt dependencies: langchain-text-splitters, sentence-transformers, weaviate-client
- [ ] Implement chunking với chunk_size=512, overlap=50
- [ ] Index tất cả markdown files vào Weaviate
- [ ] Ghi rõ lựa chọn trong code comments

### Task 5: Semantic Search Module
- [ ] Implement semantic_search(query: str, top_k: int = 10) -> list[dict]
- [ ] Sử dụng embedding model từ Task 4
- [ ] Truy vấn Weaviate để lấy kết quả dense retrieval
- [ ] Trả về format: [{'content': str, 'score': float, 'metadata': dict}]
- [ ] Sắp xếp kết quả theo score giảm dần

### Task 6: Lexical Search Module (BM25)
- [ ] Cài đặt rank-bm25
- [ ] Implement lexical_search(query: str, top_k: int = 10) -> list[dict]
- [ ] Tokenize corpus và tạo BM25 index
- [ ] Trả về cùng format với semantic search
- [ ] Sắp xếp kết quả theo score BM25

### Task 7: Reranking Module
- [ ] Chọn Jina Reranker (jina-reranker-v2-base-multilingual) - tốt cho multilingual
- [ ] Cài đặt requests để gọi API
- [ ] Implement rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]
- [ ] Re-score và re-order candidates dựa trên relevance
- [ ] Lấy API key từ Jina AI (cần đăng ký miễn phí)

### Task 8: PageIndex Vectorless RAG
- [ ] Đăng ký tài khoản tại https://pageindex.ai/
- [ ] Cài đặt pageindex SDK
- [ ] Upload tài liệu lên PageIndex
- [ ] Implement pageindex_search(query: str, top_k: int = 5) -> list[dict]
- [ ] Thiết kế làm fallback khi hybrid search score < threshold

### Task 9: Retrieval Pipeline Hoàn Chỉnh
- [ ] Kết hợp semantic + lexical search
- [ ] Implement merge strategy (RRF hoặc weighted fusion)
- [ ] Áp dụng reranking trên kết quả merged
- [ ] Thêm logic fallback đến PageIndex khi score < threshold (0.3)
- [ ] Implement retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3)

### Task 10: Generation Có Citation
- [ ] Implement reorder_for_llm để tránh lost in the middle
- [ ] Tạo prompt template với instruction citaion
- [ ] Chọn LLM: Có thể sử dụng OpenAI GPT hoặc model lokal
- [ ] Implement generate_with_citation(query: str, context_chunks: list[dict]) -> str
- [ ] Đảm bảo output có format citation [Nguồn, Năm]
- [ ] Trả về "I cannot verify this information" khi không đủ evidence

## Giai Đoạn 2: Bài Nhóm (30% điểm)

### Lựa Chọn: Evaluation Pipeline (ít triển khai UI hơn chatbot)
- [ ] Chọn framework: RAGAS (industry standard cho RAG evaluation)
- [ ] Tạo golden dataset với 15+ cặp Q&A về pháp luật ma tuý
- [ ] Chạy evaluation với 4 metrics: faithfulness, answer relevancy, context recall, context precision
- [ ] Thiết kế A/B test: so sánh cấu hình có reranking vs không reranking
- [ ] Viết eval_pipeline.py để tự động chạy evaluation
- [ ] Tạo results.md với bảng điểm và phân tích worst performers

## Giai Đoạn 3: Bonus (20% điểm)
- [ ] Thử thách LLM với câu hỏi pháp luật ma tuý phức tạp mà không có trong dataset
- [ ] Ví dụ: "Luật hiện hành quy địnhอย่างไร về xử phạt người mua để dùng ma tuý Synthesis?"
- [ ] Mục tiêu là khiến LLM trả về "I cannot verify this information"

## Lịch Triển Khai
- Ngày 1: Hoàn thành Task 1-3 (nếu chưa làm)
- Ngày 2-3: Hoàn thành Task 4-7
- Ngày 4: Hoàn thành Task 8-10
- Ngày 5-6: Làm bài nhóm (evaluation pipeline)
- Ngày 7: Kiểm tra, sửa lỗi, chuẩn bị demo

## Lưu Ý Kỹ Thuật
- Đảm bảo所有API keys được lưu trong .env file
- Sử dụng virtual environment để tránh xung đột dependencies
- Test từng task riêng sau khi hoàn thành
- Đừng quên commit code thường xuyên
- Chuẩn bị môi trường chạy demo trước ngày présentation

## File Cần Tạo/Sửa
- `src/task4_chunking_indexing.py` - Task 4
- `src/task5_semantic_search.py` - Task 5
- `src/task6_lexical_search.py` - Task 6
- `src/task7_reranking.py` - Task 7
- `src/task8_pageindex_vectorless.py` - Task 8
- `src/task9_retrieval_pipeline.py` - Task 9
- `src/task10_generation.py` - Task 10
- `group_project/evaluation/golden_dataset.json` - Bài nhóm
- `group_project/evaluation/eval_pipeline.py` - Bài nhóm
- `group_project/evaluation/results.md` - Bài nhóm