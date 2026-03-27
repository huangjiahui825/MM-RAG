import base64
import mimetypes
from pathlib import Path
from typing import List
import httpx


class ImageEmbedding:
    def __init__(self, api_key: str, api_base: str, model_name: str):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name

    def _to_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None:
            mime_type = "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def embed(self, image_path: str) -> List[float]:
        data_url = self._to_data_url(image_path)
        response = httpx.post(
            url=f"{self.api_base}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "input": data_url,
                "encoding_format": "float",
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
