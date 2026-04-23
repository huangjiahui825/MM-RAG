from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from online.multimodal_search import multimodal_search
from reranker import rerank
from generation import vlm_select_assetids
from fastapi.middleware.cors import CORSMiddleware
from crypt import encrypt


app = FastAPI()

# 跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有http方法
    allow_headers=["*"],  # 允许所有请求头
)


class SearchRequest(BaseModel):
    description: str
    url: str
    name: str
    size_x: Optional[float] = None
    size_y: Optional[float] = None

class SearchResponse(BaseModel):
    query_index: int
    assetid: str

@app.post("/search", response_model=List[SearchResponse])
def search_api(requests: List[SearchRequest]):

    all_queries_data = []
    
    for i, req in enumerate(requests):
        filters = {
            "target_name": req.name,
            "target_x": req.size_x,
            "target_y": req.size_y
        }
        
        hits = multimodal_search(
            text_query=req.description,
            image_query_url=req.url,
            top_k=20,
            filters=filters
        )
        
        reranked_results = rerank(
            query_text=req.description,
            query_image_url=req.url,
            candidates=hits,
            top_k=10
        )
        
        # 严格对齐 main.py 的数据结构
        all_queries_data.append({
            "query_id": i + 1,
            "query_text": req.description,
            "query_image": req.url,
            "retrieved_images": reranked_results
        })

    
    # 将所有查询和候选池统一交给 VLM 进行组合搭配
    final_assetids = vlm_select_assetids(all_queries_data)
    for item in final_assetids:
        raw_assetid = item.get("assetid")
        if raw_assetid is None:
            item["assetid"] = ""
            continue

        assetid_text = str(raw_assetid).strip()
        if not assetid_text or assetid_text.lower() in {"none", "null"}:
            item["assetid"] = ""
            continue

        try:
            # Only numeric assetid should be encrypted.
            item["assetid"] = encrypt(int(assetid_text))
        except (TypeError, ValueError):
            item["assetid"] = ""
    # 打印在终端便于查看
    print(f"\n========================================")
    print(f"VLM 最终组合搭配的 assetids: {final_assetids}")
    print(f"========================================\n")
    
    return final_assetids

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
