import httpx
import json
from typing import Any
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
import config
from prompt import VLM_SELECT_ASSETIDS_PROMPT


class QwenLLM(CustomLLM):
    context_window: int = 400000
    num_output: int = 2048
    model_name: str = config.LLM_MODEL

    @property
    def metadata(self) -> LLMMetadata:
        """获取 LLM 元数据。"""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """同步完成接口。"""
        headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{config.API_BASE}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """流式完成接口（如果不需要可以简单实现）。"""
        response = self.complete(prompt, **kwargs)
        yield response

def get_llm():
    return QwenLLM()

def extract_filter_criteria(query_text: str) -> dict:
    """从用户查询中提取尺寸(x, y, z)"""
    prompt = f"""
    你是一个专业的家居助手。请从用户的描述中提取出他想要寻找的“尺寸信息（长、宽、高）”。
    
    用户描述："{query_text}"
    
    请仅返回 JSON 格式，包含以下字段：
    - target_name: 家具的名称（如“沙发”、“餐桌”、“床”等），如果没有明确提到则为 null。
    - target_x: 长度（数值，单位默认毫米），如果没有提到则为 null。
    - target_y: 宽度/深度（数值，单位默认毫米），如果没有提到则为 null。

    注意：
    1.如果用户提供的尺寸单位是厘米(cm)或米(m)，请自动换算为毫米(mm)的数值。
    2.禁止提取高度信息。
    3.尺寸第一个数值默认为长度，第二个数值默认为宽度。如：尺寸1700×2600mm，则长度为1700mm，宽度为2600mm。
    
    示例输出：{{"target_name": "沙发", "target_x": 2000, "target_y": 800}}
    """
    llm = get_llm()
    response = llm.complete(prompt)
    try:
        # 简单清理并解析 JSON
        content = response.text.strip().replace("```json", "").replace("```", "")
        extracted = json.loads(content)
            
        return extracted
    except:
        return {"target_name": None, "target_x": None, "target_y": None}

def vlm_select_assetids(all_queries_data: list) -> list:
    """Use VLM to select final assetids per query."""
    if not all_queries_data:
        return []

    query_order = [int(data["query_id"]) for data in all_queries_data]
    results_by_query = {
        int(data["query_id"]): {"query_index": int(data["query_id"]), "assetid": ""}
        for data in all_queries_data
    }

    queries_for_vlm = [data for data in all_queries_data if data.get("retrieved_images")]
    if not queries_for_vlm:
        return [results_by_query[qid] for qid in query_order]

    llm = get_llm()

    all_queries_context = ""
    for data in queries_for_vlm:
        all_queries_context += f"#### [Query {data['query_id']}]\n"
        all_queries_context += f"- Query Text: {data['query_text']}\n"
        all_queries_context += f"- Query Image URL: {data['query_image']}\n"
        all_queries_context += "- Retrieved Images:\n"
        for img in data["retrieved_images"]:
            all_queries_context += (
                f"  - assetid: {img['assetid']}, name: {img['name']}, "
                f"description: {img['description']}, URL: {img['url']}\n"
            )
        all_queries_context += "\n"

    full_prompt = VLM_SELECT_ASSETIDS_PROMPT.format(
        all_queries_context=all_queries_context
    )

    print("Calling VLM to select final assetids...")
    response = llm.complete(full_prompt)

    try:
        content = response.text.strip().replace("```json", "").replace("```", "").strip()
        results = json.loads(content)
        if not isinstance(results, list):
            raise ValueError("VLM output is not a JSON list")

        for item in results:
            qidx = int(item["query_index"])
            if qidx not in results_by_query:
                continue

            raw_assetid = item.get("assetid")
            if raw_assetid is None:
                normalized_assetid = ""
            else:
                normalized_assetid = str(raw_assetid).strip()
                if normalized_assetid.lower() in {"none", "null"}:
                    normalized_assetid = ""

            results_by_query[qidx]["assetid"] = normalized_assetid

        return [results_by_query[qid] for qid in query_order]
    except Exception as e:
        print(f"[VLM parse error] raw output: {response.text}")
        raise RuntimeError(f"Failed to parse VLM output: {e}")
