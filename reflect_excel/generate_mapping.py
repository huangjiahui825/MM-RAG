import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# 1. 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-0IxPIR7GK7AeHWDRC8Dd668775584922Aa5c6378Aa8b912d",
    base_url="https://oneapi.qunhequnhe.com/v1"
)

def get_embeddings(texts, model="text-embedding-v3", batch_size=10):
    """批量获取文本的 Embedding 向量，分批请求以避免超出 API 限制"""
    # 过滤掉空字符串，并确保所有输入都是字符串
    valid_texts = [str(t).strip() for t in texts if str(t).strip()]
    
    if not valid_texts:
        return []
        
    all_embeddings = []
    
    # 将文本列表按 batch_size 分批
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        try:
            response = client.embeddings.create(input=batch, model=model)
            all_embeddings.extend([data.embedding for data in response.data])
        except Exception as e:
            print(f"获取第 {i} 到 {i + len(batch)} 条数据的 Embedding 时出错: {e}")
            raise e
            
    return all_embeddings


def main():
    # 2. 读取 Excel 数据
    file_path = 'reflect_excel/test.xlsx'
    print(f"正在读取数据: {file_path}")
    df = pd.read_excel(file_path)

    # 提取上游分类 (A, B列) 并去重、去空
    # 注意：这里假设 A列是 '家具类别ID', B列是 '名字'
    upstream_df = df[['家具类别ID', '名字']].dropna(subset=['家具类别ID', '名字']).drop_duplicates(subset=['家具类别ID'])
    upstream_df['家具类别ID'] = upstream_df['家具类别ID'].astype(int)
    upstream_df['名字'] = upstream_df['名字'].astype(str).str.strip()
    upstream_df = upstream_df[upstream_df['名字'] != '']
    upstream_names = upstream_df['名字'].tolist()

    # 提取本地分类及其扩展字段
    local_columns = [
        'prodcatid', 'name', 'localeid', 'parentid', 'type', 'roomtypeids', 
        'keyword', 'picurl', 'lastmodified', 'created', 'sort', 'visible', 
        'layoutmodelid', 'icon', 'weight', 'soft', 'sysdictdataids', 
        'prodtype', 'condition'
    ]
    
    # 确保这些列在 Excel 中存在
    existing_local_cols = [col for col in local_columns if col in df.columns]
    local_df = df[existing_local_cols].dropna(subset=['prodcatid', 'name']).drop_duplicates(subset=['prodcatid'])
    local_df['prodcatid'] = local_df['prodcatid'].astype(int)
    local_df['name'] = local_df['name'].astype(str).str.strip()
    local_df = local_df[local_df['name'] != '']
    local_names = local_df['name'].tolist()

    print(f"提取到上游分类 {len(upstream_names)} 个，本地分类 {len(local_names)} 个。")

    if not upstream_names or not local_names:
        print("错误：提取到的分类名称为空，请检查 Excel 文件内容。")
        return

    # 3. 获取 Embeddings
    print("正在调用 API 获取上游分类的 Embeddings...")
    upstream_embeddings = get_embeddings(upstream_names)

    print("正在调用 API 获取本地分类的 Embeddings...")
    local_embeddings = get_embeddings(local_names)

    # 4. 计算余弦相似度
    print("正在计算相似度矩阵...")
    similarity_matrix = cosine_similarity(upstream_embeddings, local_embeddings)

    # 5. 筛选并生成映射表
    THRESHOLD = 0.7  # 相似度阈值
    mapping_data = []

    for i, up_row in enumerate(upstream_df.itertuples(index=False)):
        for j, loc_row in enumerate(local_df.itertuples(index=False)):
            sim_score = similarity_matrix[i][j]
            
            if sim_score >= THRESHOLD:
                # 基础映射信息
                res = {
                    'value': up_row.家具类别ID,
                    'label': up_row.名字,
                    'prodcatid': loc_row.prodcatid,
                    'name': loc_row.name,
                    'similarity': round(sim_score, 4)
                }
                
                # 动态添加本地分类的所有扩展字段
                # loc_row 是一个 namedtuple，我们可以将其转换为字典
                loc_dict = loc_row._asdict()
                for key, value in loc_dict.items():
                    if key not in ['prodcatid', 'name']:
                        # 映射一些名称差异
                        output_key = key
                        if key == 'roomtypeids': output_key = 'roomtypeid'
                        if key == 'sysdictdataids': output_key = 'stsdictdataids'
                        res[output_key] = value
                
                mapping_data.append(res)

    # 6. 导出结果
    result_df = pd.DataFrame(mapping_data)
    
    if not result_df.empty:
        result_df = result_df.sort_values(by='similarity', ascending=False)
        output_path = 'reflect_excel/mapping_result.xlsx'
        result_df.to_excel(output_path, index=False)
        print(f"映射表已生成！共找到 {len(result_df)} 对映射关系，已保存至 {output_path}")
    else:
        print("没有找到高于阈值的映射关系，请尝试调低 THRESHOLD。")

if __name__ == "__main__":
    main()
