from typing import List
import httpx
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field


class TextEmbedding(BaseEmbedding):
    api_key: str = Field(description="API key")
    api_base: str = Field(description="API base URL")

    def _call_api(self, text: str) -> List[float]:
        response = httpx.post(
            url=f"{self.api_base}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._call_api(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._call_api(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
