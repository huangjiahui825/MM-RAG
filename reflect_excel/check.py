import os
import time
import argparse
from typing import Dict, Tuple

import pandas as pd
from openai import OpenAI


def make_client() -> OpenAI:
    api_key = "sk-0IxPIR7GK7AeHWDRC8Dd668775584922Aa5c6378Aa8b912d"
    if not api_key:
        raise RuntimeError("未找到 OPENAI_API_KEY 或 LLM_API_KEY 环境变量。")
    base_url = os.getenv("OPENAI_BASE_URL", "https://oneapi.qunhequnhe.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def detect_col(df: pd.DataFrame, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def ai_is_related(
    client: OpenAI,
    model: str,
    label: str,
    name: str,
    cache: Dict[Tuple[str, str], bool],
) -> bool:
    key = (label.strip(), name.strip())
    if key in cache:
        return cache[key]

    l = key[0].lower()
    n = key[1].lower()
    if l == n or l in n or n in l:
        cache[key] = True
        return True

    system_prompt = (
        "你是家具品类清洗助手。"
        "判断两个名称是否属于同类/近义家具品类。"
        "例如：冰箱 vs 空调 => NO；梳妆台 vs 化妆台 => YES。"
        "只输出 YES 或 NO。"
    )
    user_prompt = f"label: {label}\nname: {name}"

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    txt = (resp.choices[0].message.content or "").strip().upper()
    related = txt.startswith("YES")
    cache[key] = related
    return related


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="reflect_excel/mapping_result_copy.xlsx")
    parser.add_argument("--model", default="qwen3-max")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"文件不存在: {args.file}")

    df = pd.read_excel(args.file)
    if df.empty:
        print("文件为空，无需处理。")
        return

    value_col = detect_col(df, ["value", "Value", "id"], default="value")
    label_col = detect_col(df, ["label", "Label", "家具名称", "上游名称"], default="label")
    name_col = detect_col(df, ["name", "Name", "本地名称", "品类名称"], default="name")
    sim_col = detect_col(df, ["similarity", "Similarity", "score", "Score"], default=None)

    for col in [value_col, label_col, name_col]:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}，当前列: {list(df.columns)}")

    client = make_client()
    cache: Dict[Tuple[str, str], bool] = {}

    # 逐行打标：是否相关
    related_flags = []
    total = len(df)
    for i, row in df.iterrows():
        label = str(row.get(label_col, "")).strip()
        name = str(row.get(name_col, "")).strip()

        if not label or not name:
            # 缺失信息，先标记为不相关，后续由 value 兜底保留
            related_flags.append(False)
        else:
            related_flags.append(ai_is_related(client, args.model, label, name, cache))

        if args.sleep > 0:
            time.sleep(args.sleep)

        if (len(related_flags) % 50 == 0) or (len(related_flags) == total):
            print(f"[{len(related_flags)}/{total}] 已完成判定")

    df["__related"] = related_flags

    # 按 value 分组过滤，且每个 value 至少保留 1 条
    kept_parts = []
    fallback_count = 0
    removed_count = 0

    for value, g in df.groupby(value_col, dropna=False, sort=False):
        g_related = g[g["__related"] == True]

        if not g_related.empty:
            kept = g_related.copy()
        else:
            # 兜底：该 value 全被判不相关，也至少保留 1 条
            fallback_count += 1
            if sim_col and sim_col in g.columns:
                sim_series = pd.to_numeric(g[sim_col], errors="coerce")
                if sim_series.notna().any():
                    keep_idx = sim_series.idxmax()
                else:
                    keep_idx = g.index[0]
            else:
                keep_idx = g.index[0]
            kept = g.loc[[keep_idx]].copy()

        removed_count += (len(g) - len(kept))
        kept_parts.append(kept)

    result_df = pd.concat(kept_parts, axis=0).sort_index()
    result_df = result_df.drop(columns=["__related"], errors="ignore")

    # 覆盖保存回原文件
    result_df.to_excel(args.file, index=False)

    print("\n清洗完成")
    print(f"原始行数: {len(df)}")
    print(f"保留行数: {len(result_df)}")
    print(f"删除行数: {removed_count}")
    print(f"触发兜底(每个value至少1条)次数: {fallback_count}")
    print(f"已保存: {args.file}")


if __name__ == "__main__":
    main()
