from qdrant_client import QdrantClient
import config
import os
import json
from embeddings.multimodal_embedding import MultimodalEmbedding
from online.RRF import rrf_fusion
from qdrant_client.http import models

# 4.23 新增：
# 加载真分类表
# CATEGORY_MAPPING = {}
# mapping_file_path = os.path.join(os.path.dirname(__file__), '../true_classification_table/label_prodcatid.json')
# try:
#     with open(mapping_file_path, 'r', encoding='utf-8') as f:
#         CATEGORY_MAPPING = json.load(f)
#     print(f"成功加载分类映射表，共 {len(CATEGORY_MAPPING)} 条记录。")
# except Exception as e:
#     print(f"加载分类映射表失败，请检查路径: {e}")



def _execute_search(client, query_vectors, query_filter, top_k):
    """内部函数：执行 Pool A, B, C 的检索并返回 RRF 融合结果"""
    # Pool A: 文召回图
    pool_a = client.query_points(
        collection_name=config.IMAGE_COLLECTION,
        query=query_vectors["text_vector"],
        query_filter=query_filter,
        limit=top_k
    ).points
    
    # Pool B: 图召回图
    pool_b = client.query_points(
        collection_name=config.IMAGE_COLLECTION,
        query=query_vectors["image_vector"],
        query_filter=query_filter,
        limit=top_k
    ).points

    # Pool C: 文召回文
    pool_c = client.query_points(
        collection_name=config.TEXT_COLLECTION,
        query=query_vectors["text_vector"],
        query_filter=query_filter,
        limit=top_k
    ).points
    
    return rrf_fusion([pool_a, pool_b, pool_c])

def build_filter(name=None, size_x=None, size_y=None):
    """构建过滤条件的辅助函数"""
    must_conditions = []

    # 4.23 注销：
    # if prod_cat_id and isinstance(prod_cat_id, list) and len(prod_cat_id) > 0:
    #     must_conditions.append(models.FieldCondition(
    #         key="true_classification_id",
    #         match=models.MatchAny(any=prod_cat_id)
    #     ))
    # elif prod_cat_id is not None and not isinstance(prod_cat_id, list):
    #     must_conditions.append(models.FieldCondition(
    #         key="true_classification_id",
    #         match=models.MatchValue(value=prod_cat_id)
    #     ))

    # 4.23 新增：
    # 新增 name 过滤 (使用 MatchText 进行全文匹配)
    if name:
        must_conditions.append(models.FieldCondition(
            key="name",
            match=models.MatchText(text=name)
        ))


    # 增加 10% 容差的尺寸过滤
    if size_x is not None:
        try:
            val_x = float(size_x)
            must_conditions.append(models.FieldCondition(
                key="size_x", range=models.Range(gte=val_x * 0.90, lte=val_x * 1.10)
            ))
        except (ValueError, TypeError):
            pass
    if size_y is not None:
        try:
            val_y = float(size_y)
            must_conditions.append(models.FieldCondition(
                key="size_y", range=models.Range(gte=val_y * 0.90, lte=val_y * 1.10)
            ))
        except (ValueError, TypeError):
            pass

    
    return models.Filter(must=must_conditions) if must_conditions else None

# 4.23 新增：
def multimodal_search(text_query: str, image_query_url: str, top_k: int = 3, filters: dict = None):
    embedder = MultimodalEmbedding()
    client = QdrantClient(path=config.QDRANT_PATH)
    
    # 获取查询向量
    query_vectors = embedder.embed(text=text_query, image_url=image_query_url)
    query_label = filters.get("label") or filters.get("target_name") if filters else None
    
    # 4.23 注销：
    # prod_cat_ids = None
    # if query_label:
    #     # 尝试从字典中获取映射列表 (转为字符串匹配)
    #     prod_cat_ids = CATEGORY_MAPPING.get(str(query_label))
        
    #     if prod_cat_ids:
    #         print(f"[Mapping] 成功将 '{query_label}' 映射为 prodCatIds: {prod_cat_ids}")
    #     else:
    #         print(f"[Mapping] 未找到 '{query_label}' 的映射，降级为无分类过滤。")

    size_x = filters.get('target_x') if filters else None
    size_y = filters.get('target_y') if filters else None

    final_results = []
    seen_ids = set()
    def add_results(new_results):
        for res in new_results:
            if res.id not in seen_ids:
                final_results.append(res)
                seen_ids.add(res.id)
                if len(final_results) >= top_k:
                    break

    # 尺寸+分类 (10%容差)
    # 4.23 注销：
    # print(f"\n[Search] Trying Level 1 Filter: Size=({size_x}, {size_y}), prodCatIds={prod_cat_ids}")
    # 4.23 新增：
    print(f"\n[Search] Trying Level 1 Filter: Size=({size_x}, {size_y}), name={query_label}")


    q_filter = build_filter(
        # 4.23 注销：
        # prod_cat_id=prod_cat_ids,
        # 4.23 新增;
        name=query_label,
        size_x=size_x, 
        size_y=size_y
    )
    results = _execute_search(client, query_vectors, q_filter, top_k)
    add_results(results)
    print(f"[Search] Search finished: Found {len(final_results)} results")
    return final_results
