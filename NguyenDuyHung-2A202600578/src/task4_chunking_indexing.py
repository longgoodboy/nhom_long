"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn RecursiveCharacterTextSplitter: an toàn, phổ biến, hoạt động tốt với đa dạng nội dung
CHUNK_SIZE = 512        # Kích thước_chunk trung bình, cân bằng giữa context và hiệu suất
CHUNK_OVERLAP = 50      # Overlap 10% để giữ mất mát thông tin giữa các chunks
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn BAAI/bge-m3: model embedding multilingual state-of-the-art, tốt cho tiếng Việt
EMBEDDING_MODEL = "BAAI/bge-m3"  # Dimension 1024, hiệu suất cao trên các tác vụ truy vấn đa ngôn ngữ
EMBEDDING_DIM = 1024

# Chọn Weaviate: hỗ trợ hybrid search (dense + BM25) built-in, dễ mở rộng
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Xác định loại tài liệu dựa trên đường dẫn
        if "legal" in str(md_file):
            doc_type = "legal"
        elif "news" in str(md_file):
            doc_type = "news"
        else:
            doc_type = "unknown"

        documents.append({
            "content": content,
            "metadata": {"source": str(md_file.relative_to(STANDARDIZED_DIR)), "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    # Kết nối đến Weaviate local instance
    client = weaviate.connect_to_local()

    # Kiểm tra xem collection đã tồn tại chưa
    collection_name = "DrugLawDocs"
    try:
        # Thử tạo collection mới
        client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),  # Chúng ta sẽ cung cấp vector riêng
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        print(f"Created new collection: {collection_name}")
    except Exception as e:
        # Collection có thể đã tồn tại
        print(f"Collection {collection_name} may already exist: {e}")
        # Thử sử dụng collection hiện có
        pass

    # Lấy collection
    collection = client.collections.get(collection_name)

    # Xóa tất cả dữ liệu cũ (tùy chọn - để bắt đầu mới)
    # collection.data.delete_many()

    # Insert chunks
    print(f"Indexing {len(chunks)} chunks to Weaviate...")
    with collection.batch.dynamic() as batch:
        for i, chunk in enumerate(chunks):
            batch.add_object(
                properties={
                    "content": chunk["content"],
                    "source": chunk["metadata"]["source"],
                    "doc_type": chunk["metadata"]["type"],
                    "chunk_index": chunk["metadata"]["chunk_index"]
                },
                vector=chunk["embedding"]
            )
            # Hiển thị tiến độ mỗi 50 chunks
            if (i + 1) % 50 == 0:
                print(f"  Indexed {i + 1}/{len(chunks)} chunks")

    print(f"✓ Successfully indexed {len(chunks)} chunks to Weaviate")
    client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
