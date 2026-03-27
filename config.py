import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["API_KEY"]
API_BASE = "https://oneapi.qunhequnhe.com/v1"

TEXT_MODEL = "bge-m3-1024d"
TEXT_DIM = 1024

IMAGE_MODEL = "img-embedding-cnclip-yuntu-ft-768d"
IMAGE_DIM = 768

TEXT_COLLECTION = "text_collection"
IMAGE_COLLECTION = "image_collection"

QDRANT_PATH = "./qdrant_storage"
