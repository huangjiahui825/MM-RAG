from qdrant_client import QdrantClient

import config
from embeddings.image_embedding import ImageEmbedding


def image_search(query_image_path: str, top_k: int = 5) -> list:
    client = QdrantClient(path=config.QDRANT_PATH)
    embed_model = ImageEmbedding(
        api_key=config.API_KEY,
        api_base=config.API_BASE,
        model_name=config.IMAGE_MODEL,
    )

    query_vector = embed_model.embed(query_image_path)
    hits = client.search(
        collection_name=config.IMAGE_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
    )

    results = []
    for hit in hits:
        results.append({
            "file_name": hit.payload.get("file_name", ""),
            "file_path": hit.payload.get("file_path", ""),
            "score": hit.score,
        })
    return results


if __name__ == "__main__":
    query_path = "./data/online_query_materials/images/query.jpg"
    results = image_search(query_path, top_k=3)
    for i, r in enumerate(results):
        print(f"[{i+1}] score={r['score']:.4f} | {r['file_name']}")
