import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv()
from agent.utils.db import product_collection, policy_collection

SEARCH_CASES = [
    {"id": "S01", "collection": "products", "query": "laptop gaming Acer Nitro hiệu năng cao", "keyword_in_result": "Nitro", "k": 3},
    {"id": "S02", "collection": "products", "query": "MacBook Air M4 màn hình Retina", "keyword_in_result": "MacBook", "k": 3},
    {"id": "S03", "collection": "products", "query": "máy tính bảng iPad học sinh sinh viên", "keyword_in_result": "iPad", "k": 3},
    {"id": "S04", "collection": "products", "query": "màn hình Samsung ViewFinity 4K đồ họa", "keyword_in_result": "Samsung", "k": 3},
    {"id": "S05", "collection": "products", "query": "micro thu âm không dây Boya chất lượng cao", "keyword_in_result": "Boya", "k": 3},
    {"id": "S06", "collection": "policies", "query": "chính sách hoàn tiền khi trả hàng lỗi", "keyword_in_result": "hoàn tiền", "k": 3},
    {"id": "S07", "collection": "policies", "query": "các phương thức thanh toán được hỗ trợ", "keyword_in_result": "thanh toán", "k": 3},
    {"id": "S08", "collection": "policies", "query": "chính sách đổi trả hàng", "keyword_in_result": "đổi", "k": 3},
    {"id": "S09", "collection": "policies", "query": "bảo hành sản phẩm bao lâu", "keyword_in_result": "bảo hành", "k": 3},
    {"id": "S10", "collection": "policies", "query": "phí vận chuyển giao hàng toàn quốc", "keyword_in_result": "vận chuyển", "k": 3},
]


def search(collection_name, query, k):
    col = product_collection if collection_name == "products" else policy_collection
    k = min(max(k, 1), 10)
    results = col.query(query_texts=[query], n_results=k)
    docs = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    return [{"text": doc, "distance": distances[i] if i < len(distances) else None} for i, doc in enumerate(docs)]


def run_tests():
    hit1 = hit3 = 0
    results = []
    print("\n" + "=" * 65)
    print("SEMANTIC SEARCH TEST")
    print("=" * 65)

    for tc in SEARCH_CASES:
        try:
            docs = search(tc["collection"], tc["query"], tc["k"])
        except Exception as e:
            print(f"[{tc['id']}] EXCEPTION: {e}")
            results.append({"id": tc["id"], "hit@1": False, "hit@3": False})
            continue

        kw = tc["keyword_in_result"].lower()
        texts = [d["text"].lower() for d in docs]
        h1 = len(texts) > 0 and kw in texts[0]
        h3 = any(kw in t for t in texts[:3])
        if h1:
            hit1 += 1
        if h3:
            hit3 += 1

        status = "PASS" if h3 else "FAIL"
        print(f"[{tc['id']}] {status}  \"{tc['query'][:50]}\"")
        print(f"         -> Hit@1 {'PASS' if h1 else 'FAIL'}  Hit@3 {'PASS' if h3 else 'FAIL'}")
        if not h3 and docs:
            print(f"         -> top result: \"{docs[0]['text'][:80]}...\"")
        results.append({"id": tc["id"], "hit@1": h1, "hit@3": h3})

    total = len(SEARCH_CASES)
    print("=" * 65)
    print(f"Hit@1: {hit1}/{total} ({100*hit1//total}%)")
    print(f"Hit@3: {hit3}/{total} ({100*hit3//total}%)")
    print("=" * 65 + "\n")
    return results


if __name__ == "__main__":
    run_tests()