from pathlib import Path
import json
from online.multimodal_search import multimodal_search
from offline.build_multimodal_index import build_multimodal_index
from generation import vlm_select_assetids
from reranker import rerank
from crypt import encrypt
import config
import time


def run_offline():
    print("=" * 50)
    print(f"离线阶段：构建多模态索引 ({config.MULTIMODAL_DIM}维)")
    print("=" * 50)

    build_multimodal_index()

    print("\n离线阶段完成。")

def run_online():
    with open(config.ONLINE_QUERY_TEXT_IMAGES_JSON, 'r', encoding='utf-8') as f:
        query_data = json.load(f)

    all_results = []
    latencies = []

    for i, item in enumerate(query_data):
        
        # 4.23 新增：
        print(f"\n--- 查询 {i+1}/{len(query_data)} ---")
        text_query = item['query_text']
        image_query_url = item['query_image']
        filter_criteria = {
            "target_name": item.get('name'),
            "target_x": item.get('size_x'),
            "target_y": item.get('size_y')
        }
        print(f"使用的过滤条件：{filter_criteria}")
        start_time = time.time()

        
        
        hits = multimodal_search(text_query, image_query_url, top_k=20, filters=filter_criteria)
        latency = time.time() - start_time
        latencies.append(latency)
        reranked_results = rerank(text_query, image_query_url, hits, top_k=10)
        retrieved_ids = [res['assetid'] for res in reranked_results]

        reranker_scores = [res.get('reranker_score', 0.0) for res in reranked_results]

        print(f"\n--- 查询 {i+1}/{len(query_data)} ---")
        print(f"提取条件：{filter_criteria}")
        print(f"检索结果 IDs: {retrieved_ids}")
        print(f"精排分数: {reranker_scores}")
        print(f"召回阶段耗时: {latency:.4f}s")

        # 当前走 VLM + Prompt 路线
        current_query_results = {
            "query_id": i + 1,
            "query_text": text_query,
            "query_image": image_query_url,
            "retrieved_images": reranked_results
        }
        all_results.append(current_query_results)

    if latencies:
        avg_recall_latency = sum(latencies) / len(latencies)
        print(f"\n" + "="*30)
        print(f"平均召回时延: {avg_recall_latency:.4f}s")
        
    final_assetids = vlm_select_assetids(all_results)
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

    print(f"\n========================================")
    print(f"VLM 最终组合搭配的 assetids: {final_assetids}")
    print(f"========================================\n")

if __name__ == "__main__":
    # run_offline()
    run_online()
