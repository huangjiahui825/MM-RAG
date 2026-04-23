def rrf_fusion(results_list, k=60):
    scores = {}
    point_map = {}

    for hits in results_list:
        for rank, hit in enumerate(hits):
            point_id = hit.id
            point_map[point_id] = hit
            # 多路融合
            scores[point_id] = scores.get(point_id, 0) + 1.0 / (k + rank)


        # 按得分排序
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final_results = []
    for p_id, score in sorted_ids:
        hit = point_map[p_id]
        hit.score = score
        final_results.append(hit)
    return final_results