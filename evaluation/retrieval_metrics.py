def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """Top-K 结果中命中的相关商品数 / 总相关商品数"""
    if not relevant_ids:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    hits = len(set(retrieved_top_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """Top-K 结果中相关商品数 / K"""
    retrieved_top_k = retrieved_ids[:k]
    hits = len(set(retrieved_top_k) & set(relevant_ids))
    return hits / k


def mrr(retrieved_ids: list, relevant_ids: list) -> float:
    """第一个相关商品的排名倒数"""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved_ids: list, relevant_ids: list) -> float:
    """所有相关商品命中位置的 Precision 均值"""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant_ids)


def evaluate_retrieval(retrieved_ids: list, relevant_ids: list, k: int = 3) -> dict:
    """计算全部检索评估指标，返回字典"""
    return {
        f"Recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"Precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        "MRR": mrr(retrieved_ids, relevant_ids),
        # "MAP": average_precision(retrieved_ids, relevant_ids),
    }
