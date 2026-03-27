from offline.build_text_index import build_text_index
from offline.build_image_index import build_image_index
from online.text_search import text_search
from online.image_search import image_search


def run_offline():
    print("=" * 50)
    print("离线阶段：建索引")
    print("=" * 50)

    print("\n[1/2] 文本建索引...")
    build_text_index()

    print("\n[2/2] 图片建索引...")
    build_image_index()

    print("\n离线阶段完成。")


def run_online():
    print("\n" + "=" * 50)
    print("在线阶段：检索")
    print("=" * 50)

    # 文本检索
    text_query = "橱柜推荐"
    print(f"\n[文本检索] query: 「{text_query}」")
    text_results = text_search(text_query, top_k=3)
    for i, r in enumerate(text_results):
        print(f"  [{i+1}] score={r['score']:.4f} | {r['source']}")
        print(f"       {r['text'][:80]}...")

    # 图片检索
    image_query = "./data/online_query_materials/images/query.jpg"
    print(f"\n[图片检索] query: {image_query}")
    image_results = image_search(image_query, top_k=3)
    for i, r in enumerate(image_results):
        print(f"  [{i+1}] score={r['score']:.4f} | {r['file_name']}")


if __name__ == "__main__":
    run_offline()
    run_online()
