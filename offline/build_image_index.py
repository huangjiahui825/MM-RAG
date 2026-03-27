from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from embeddings.image_embedding import ImageEmbedding

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_image_index(images_dir: str = "./data/offline_materials/images"):
    # 初始化 Qdrant 本地存储
    client = QdrantClient(path=config.QDRANT_PATH)

    if not client.collection_exists(config.IMAGE_COLLECTION):
        client.create_collection(
            collection_name=config.IMAGE_COLLECTION,
            vectors_config=VectorParams(size=config.IMAGE_DIM, distance=Distance.COSINE),
        )

    embed_model = ImageEmbedding(
        api_key=config.API_KEY,
        api_base=config.API_BASE,
        model_name=config.IMAGE_MODEL,
    )

    image_paths = [
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    print(f"发现图片数: {len(image_paths)}")

    points = []
    for idx, path in enumerate(image_paths):
        print(f"  embedding [{idx+1}/{len(image_paths)}]: {path.name}")
        vector = embed_model.embed(str(path))
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={"file_name": path.name, "file_path": str(path)},
        ))

    client.upsert(collection_name=config.IMAGE_COLLECTION, points=points)
    print(f"入库图片数: {len(points)}")


if __name__ == "__main__":
    build_image_index()
