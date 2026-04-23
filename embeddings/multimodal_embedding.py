import httpx
import config
import time

class MultimodalEmbedding:
    def __init__(self):
        self.api_key = config.EMBEDDING_API_KEY
        self.api_base = config.API_BASE
        self.model = config.MULTIMODAL_MODEL

    def _get_single_embedding(self, input_item, max_retries=3):
        """内部辅助方法：获取单个输入的向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": [input_item], # 包装成列表
            "encoding_format": "float",
            "dimensions": config.MULTIMODAL_DIM
        }
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(f"{self.api_base}/embeddings", headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["data"][0]["embedding"]
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt # 指数退避
                    print(f"API 报错: {e}, 正在进行第 {attempt+1} 次重试，等待 {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e

    def embed(self, text: str = None, image_url: str = None):
        """
        分别获取文本和图片的向量，支持双路召回。
        """
        results = {}
        
        # 1. 获取文本向量
        if text:
            results["text_vector"] = self._get_single_embedding({
                "text": text,
                "type": "text"
            })
            
        # 2. 获取图片向量
        if image_url:
            results["image_vector"] = self._get_single_embedding({
                "image_url": {"url": image_url},
                "type": "image_url"
            })
            
        return results