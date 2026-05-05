import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv()
from agent.utils.db import get_pg_conn, release_pg_conn

PRODUCT_CASES = [
    {"id": "R01", "input": "realme 11 8GB 128GB", "should_match": True, "expected_name_contains": "realme 11"},
    {"id": "R02", "input": "Samsung Galaxy S25 Ultra 512GB", "should_match": True, "expected_name_contains": "Galaxy S25"},
    {"id": "R03", "input": "MacBook Air M4 15 inch 2025", "should_match": True, "expected_name_contains": "MacBook Air M4"},
    {"id": "R04", "input": "Laptop ASUS Vivobook S14 M3407KA", "should_match": True, "expected_name_contains": "Vivobook S14"},
    {"id": "R05", "input": "Laptop Acer Gaming Nitro Lite 16", "should_match": True, "expected_name_contains": "Nitro Lite 16"},
    {"id": "R06", "input": "Acer Nitro 5 gaming", "should_match": True, "expected_name_contains": "Nitro 5"},
    {"id": "R07", "input": "MacBook Air M4 13 inch", "should_match": True, "expected_name_contains": "MacBook Air"},
    {"id": "R08", "input": "Galaxy S25 Ultra", "should_match": True, "expected_name_contains": "Galaxy S25"},
    {"id": "R09", "input": "ASUS Vivobook S14", "should_match": True, "expected_name_contains": "Vivobook S14"},
    {"id": "R10", "input": "Nubia Pad Pro", "should_match": True, "expected_name_contains": "Nubia Pad Pro"},
    {"id": "R11", "input": "Samsung", "should_match": False, "expected_name_contains": None},
    {"id": "R12", "input": "iPhone", "should_match": False, "expected_name_contains": None},
    {"id": "R13", "input": "Laptop XYZ ProMax 9000", "should_match": False, "expected_name_contains": None},
    {"id": "R14", "input": "sản phẩm không tồn tại abc123", "should_match": False, "expected_name_contains": None},
]


def resolve_product(raw_name: str):
    clean_name = raw_name.strip()

    if len(clean_name.split()) < 2:
        return None

    conn = get_pg_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT ps_product_id, name, similarity(name, %s) AS sim
            FROM product
            WHERE similarity(name, %s) > 0.3
            ORDER BY sim DESC
            LIMIT 1
            """,
            (clean_name, clean_name),
        )
        row = cur.fetchone()
        if not row:
            return None
        product_id, matched_name, sim = row
        return {"product_id": str(product_id), "name": matched_name, "similarity": sim}
    finally:
        cur.close()
        release_pg_conn(conn)


def run_tests():
    passed = 0
    results = []
    print("\n" + "=" * 65)
    print("PRODUCT RESOLUTION TEST  (similarity/pg_trgm, threshold=0.3)")
    print("=" * 65)

    for tc in PRODUCT_CASES:
        try:
            result = resolve_product(tc["input"])
        except Exception as e:
            print(f"[{tc['id']}] EXCEPTION: {e}")
            results.append({"id": tc["id"], "passed": False})
            continue

        errors = []
        if tc["should_match"] and result is None:
            errors.append("Expected a match but got None")
        elif not tc["should_match"] and result is not None:
            errors.append(f"Expected no match but got '{result['name']}' (sim={result.get('similarity', '?'):.2f})")
        elif tc["should_match"] and result and tc["expected_name_contains"]:
            if tc["expected_name_contains"].lower() not in result["name"].lower():
                errors.append(f"Matched '{result['name']}' but expected to contain '{tc['expected_name_contains']}'")

        if errors:
            matched = f"-> '{result['name']}' (sim={result.get('similarity', 0):.2f})" if result else "-> no match"
            print(f"[{tc['id']}] FAIL  \"{tc['input']}\" {matched}")
            for e in errors:
                print(f"         -> {e}")
            results.append({"id": tc["id"], "passed": False})
        else:
            matched = f"-> '{result['name']}' (sim={result.get('similarity', 0):.2f})" if result else "-> correctly no match"
            print(f"[{tc['id']}] PASS  \"{tc['input']}\" {matched}")
            passed += 1
            results.append({"id": tc["id"], "passed": True})

    total = len(PRODUCT_CASES)
    print("=" * 65)
    print(f"RESULT: {passed}/{total} passed ({100*passed//total}%)")
    print("=" * 65 + "\n")
    return results


if __name__ == "__main__":
    run_tests()