from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

import config
from embeddings.text_embedding import TextEmbedding


def get_text_retriever(top_k: int = 5):
    client = QdrantClient(path=config.QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.TEXT_COLLECTION,
    )
    embed_model = TextEmbedding(
        api_key=config.API_KEY,
        api_base=config.API_BASE,
        model_name=config.TEXT_MODEL,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index.as_retriever(similarity_top_k=top_k)


def text_search(query: str, top_k: int = 5) -> list:
    retriever = get_text_retriever(top_k=top_k)
    nodes = retriever.retrieve(query)
    results = []
    for node in nodes:
        results.append({
            "text": node.get_content(),
            "score": node.score,
            "source": node.metadata.get("file_name", ""),
        })
    return results


if __name__ == "__main__":
    query = "橱柜推荐"
    results = text_search(query, top_k=3)
    for i, r in enumerate(results):
        print(f"[{i+1}] score={r['score']:.4f} | {r['source']}")
        print(f"     {r['text'][:100]}")
