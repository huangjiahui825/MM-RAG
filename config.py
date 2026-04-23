import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")
API_BASE = "https://oneapi.qunhequnhe.com/v1"

RERANK_API_KEY = os.getenv("RERANK_API_KEY")

QDRANT_PATH = "./qdrant_storage"

LLM_MODEL = 'qwen3.5-plus'

OFFLINE_MATERIALS_JSON = "data/offline_materials/offline_materials.json"
ONLINE_QUERY_TEXT_IMAGES_JSON = "data(test)/online_querys/online_query.json"
OUTPUT_PATH = "./output/multimodal_rag/answer.md"

MULTIMODAL_MODEL = "qwen3-vl-embedding"
MULTIMODAL_DIM = 2560

# MULTIMODAL_MODEL = "doubao-embedding-vision"
# MULTIMODAL_DIM = 1024

RERANK_MODEL = "qwen3-vl-rerank"

TEXT_COLLECTION = "text_collection"
IMAGE_COLLECTION = "image_collection"

RERANKER_MODEL_PATH = "./qwen/Qwen3-VL-Reranker-8B"