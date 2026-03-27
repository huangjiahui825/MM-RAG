from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config
from embeddings.text_embedding import TextEmbedding


def build_text_index(texts_dir: str = "./data/offline_materials/texts"):
    # 初始化 Qdrant 本地存储
    client = QdrantClient(path=config.QDRANT_PATH)

    if not client.collection_exists(config.TEXT_COLLECTION):
        client.create_collection(
            collection_name=config.TEXT_COLLECTION,
            vectors_config=VectorParams(size=config.TEXT_DIM, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.TEXT_COLLECTION,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 加载文本文件
    documents = SimpleDirectoryReader(texts_dir).load_data()
    print(f"加载文档数: {len(documents)}")

    # 分块 + embedding + 入库
    embed_model = TextEmbedding(
        api_key=config.API_KEY,
        api_base=config.API_BASE,
        model_name=config.TEXT_MODEL,
    )
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            embed_model,
        ],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents, show_progress=True)
    print(f"入库 chunk 数: {len(nodes)}")


if __name__ == "__main__":
    build_text_index()
