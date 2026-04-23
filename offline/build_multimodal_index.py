import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, TextIndexParams, TokenizerType
import config
from embeddings.multimodal_embedding import MultimodalEmbedding

CHECKPOINT_FILE = "label/indexing_checkpoint.json"

def build_multimodal_index():
    client = QdrantClient(path=config.QDRANT_PATH)
    embedder = MultimodalEmbedding()

    # 1. 构建两个向量数据库
    for col in [config.TEXT_COLLECTION, config.IMAGE_COLLECTION]:

        if not client.collection_exists(col):
            print(f"创建集合: {col}")

            client.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=config.MULTIMODAL_DIM, distance=Distance.COSINE),
            )

            # 4.22 注销：
            client.create_payload_index(
                collection_name=col,
                field_name="name",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                ),
            )

            # 4.22 新增：
            # client.create_payload_index(
            #     collection_name=col,
            #     field_name="true_classification_id",
            #     field_schema="integer",
            #     )

            # client.create_payload_index(
            #     collection_name=col,
            #     field_name="true_classification_name",
            #     field_schema="keyword",
            #     )
            
            # client.create_payload_index(
            #     collection_name=col,
            #     field_name="tag_ids",
            #     field_schema="integer",
            #     )
            
            # client.create_payload_index(
            #     collection_name=col,
            #     field_name="tag_names",
            #     field_schema="keyword",
            #     )



            # 为尺寸字段创建数值型索引，加速 Range 过滤
            client.create_payload_index(
                collection_name=col,
                field_name="size_x",
                field_schema="float",
            )
            client.create_payload_index(
                collection_name=col,
                field_name="size_y",
                field_schema="float",
            )
            client.create_payload_index(
                collection_name=col,
                field_name="size_z",
                field_schema="float",
            )

    # 2. 读取断点
    start_index = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            start_index = json.load(f).get("last_index", 0)
            print(f"检测到断点，从第 {start_index} 个素材开始继续处理...")

    materials_path = "label/preprocessed_data.json"


    with open(materials_path, "r", encoding="utf-8-sig") as f:
        materials = json.load(f)
    
    batch_size = 100
    total = len(materials)

    # 定义处理单个素材的内部函数
    def process_single_item(i, item):
        item_name = item['visualcat']['name']
        print(f"--- [{i+1}/{total}] 正在处理: {item_name} ---")
        vectors = embedder.embed(text=item.get('description', "") or " ", image_url=item['url'])
        
        size_info = item.get("size", {})
        
        # 4.22 注销：
        # true_cls = item.get("true_classification", {}) if isinstance(item.get("true_classification"), dict) else {}
        # tags = item.get("tag", []) if isinstance(item.get("tag"), list) else []
        # tag_ids = [t.get("tagId") for t in tags if isinstance(t, dict) and t.get("tagId") is not None]
        # tag_names = [t.get("tagName") for t in tags if isinstance(t, dict) and t.get("tagName")]



        payload = {


            # 4.22 新增：
            # "true_classification_id": true_cls.get("prodCatId"),
            # "true_classification_name": true_cls.get("name", ""),
            # "tag_ids": tag_ids,
            # "tag_names": tag_names,




            "assetid": item["assetid"],
            "name": item["visualcat"]["name"],
            "url": item["url"],
            "description": item.get("description", ""),
            "style": item.get("style", ""),
            "color": item.get("color", ""),
            "material": item.get("material", ""),
            "category_path": item["visualcat"].get("path", ""),
            "size_x": size_info.get("x"),
            "size_y": size_info.get("y"),
            "size_z": size_info.get("z"),
            "asset_type": item.get("asset_type", ""),
            "tag_id": item.get("tag_id", ""),
            "tag_type": item.get("tag_type", ""),
            "version": item.get("version", ""),
            "itemType": item.get("itemType", ""),
            "ItemID": item.get("ItemID", ""),
            "brandName": item.get("brandName", ""),
            # 嵌套字典安全获取 (brand)
            "brand_id": item.get("brand", {}).get("brandId", "") if isinstance(item.get("brand"), dict) else "",
            "brand_name": item.get("brand", {}).get("name", "") if isinstance(item.get("brand"), dict) else "",
            "custom_material": item.get("custom_material", ""),
            "product_image": item.get("product_image", ""),
            "brandgood_size": item.get("brandgood_size", ""),
            "model_number": item.get("model_number", ""),
            # 嵌套字典安全获取 (series)
            "series_id": item.get("series", {}).get("tagId", "") if isinstance(item.get("series"), dict) else "",
            "series_name": item.get("series", {}).get("name", "") if isinstance(item.get("series"), dict) else "",
            "material_bag": item.get("material_bag", ""),
            "material_type": item.get("material_type", ""),
            "custom_base_material": item.get("custom_base_material", "")
        }
        
        t_point = PointStruct(id=i, vector=vectors["text_vector"], payload=payload)
        i_point = PointStruct(id=i, vector=vectors["image_vector"], payload=payload)
        return t_point, i_point

    for batch_start in range(start_index, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = materials[batch_start:batch_end]
        
        text_points = []
        image_points = []
        
        print(f"\n>>> 开始并发处理批次: {batch_start} 到 {batch_end - 1}")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交任务
            futures = {
                executor.submit(process_single_item, batch_start + idx, item): (batch_start + idx)
                for idx, item in enumerate(batch_items)
            }
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    t_point, i_point = future.result()
                    text_points.append(t_point)
                    image_points.append(i_point)
                except Exception as e:
                    idx = futures[future]
                    print(f"处理第 {idx} 个素材时发生错误: {e}")
        
        # 批量入库并保存断点
        if text_points and image_points:
            client.upsert(collection_name=config.TEXT_COLLECTION, points=text_points)
            client.upsert(collection_name=config.IMAGE_COLLECTION, points=image_points)
                
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"last_index": batch_end}, f)
                
            print(f">>> 已成功入库并保存断点: {batch_end}")

if __name__ == "__main__":
    build_multimodal_index()