import httpx
import config

def rerank(query_text: str, query_image_url: str, candidates: list, top_k: int = 3) -> list:
    """
    使用 qwen3-vl-rerank API 对检索候选进行重排。
    """
    if not candidates:
        return []

    # 1. 构造 documents列表
    documents = []
    for hit in candidates:
        documents.append({
            "text": hit.payload.get("description", ""),
            
            "type": "image_url",
            "image_url": {"url": hit.payload.get("url")}
        })

    # 2. 构造请求 Payload
    payload = {
        "model": config.RERANK_MODEL,
        "query": {
            "text": query_text,
            "image": query_image_url
        },
        "documents": documents,
        "top_n": len(documents)
    }

    headers = {
        "Authorization": f"Bearer {config.RERANK_API_KEY}",
        "Content-Type": "application/json",
    }

    # 3. 调用 API
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{config.API_BASE}/rerank", 
                headers=headers, 
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # API 返回格式示例: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            api_results = data.get("results", [])
    except Exception as e:
        print(f"[Reranker] API 调用失败: {e}")
        # 如果 API 失败，降级返回原始检索结果的前 top_k 个
        return [{
            "assetid": hit.payload.get("assetid"),
            "name": hit.payload.get("name", ""),
            "url": hit.payload.get("url", ""),
            "description": hit.payload.get("description", ""),
            "retrieval_score": round(hit.score, 4),
            "reranker_score": 0.0,
        } for hit in candidates[:top_k]]

    # 4. 映射分数并组装结果
    # 创建一个索引到分数的映射表
    score_map = {res["index"]: res["relevance_score"] for res in api_results}

    final_results = []
    for i, hit in enumerate(candidates):
        final_results.append({
            "assetid": hit.payload.get("assetid"), # 注意：这里统一使用 assetid
            "name": hit.payload.get("name", ""),
            "url": hit.payload.get("url", ""),
            "description": hit.payload.get("description", ""),
            "retrieval_score": round(hit.score, 4),
            "reranker_score": round(score_map.get(i, 0.0), 4),
        })

    # 5. 按 reranker_score 降序排序
    final_results.sort(key=lambda x: x["reranker_score"], reverse=True)
    
    return final_results[:top_k]