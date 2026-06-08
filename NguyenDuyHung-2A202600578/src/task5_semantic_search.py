from sentence_transformers import SentenceTransformer
import weaviate

# Load model ONCE (important)
_model = SentenceTransformer("BAAI/bge-m3")


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Dense retrieval with Weaviate + bge-m3
    """

    # 1. Embed query
    query_embedding = _model.encode(query).tolist()

    # 2. Connect vector DB
    client = weaviate.connect_to_local()
    collection = client.collections.get("DrugLawDocs")

    # 3. Search
    results = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        return_metadata=["distance"]
    )

    search_results = []

    for obj in results.objects:
        props = obj.properties

        distance = getattr(obj.metadata, "distance", 0.0)

        # clamp safety (IMPORTANT for test)
        score = max(0.0, min(1.0, 1.0 - float(distance)))

        search_results.append({
            "content": props.get("content", ""),
            "score": score,
            "metadata": {
                "source": props.get("source", ""),
                "doc_type": props.get("doc_type", ""),
                "chunk_index": props.get("chunk_index", 0)
            }
        })

    client.close()

    # enforce top_k strictly
    search_results.sort(key=lambda x: x["score"], reverse=True)

    return search_results[:top_k]