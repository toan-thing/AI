import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from agent.utils.nodes import parse_runnable, ParseOutput
from langchain_core.messages import HumanMessage

DELAY_BETWEEN_CALLS = 10
MAX_RETRIES = 5

TEST_CASES = [
    {"id": "P01", "message": "Tôi muốn mua laptop",                     "expected": {"category": "laptop"}},
    {"id": "P02", "message": "Cho tôi xem điện thoại",                  "expected": {"category": "mobile"}},
    {"id": "P03", "message": "Tivi nào tốt",                            "expected": {"category": "tivi"}},
    {"id": "P04", "message": "Máy tính bảng cho trẻ em",                "expected": {"category": "tablet"}},
    {"id": "P05", "message": "Mua máy in cho văn phòng",                "expected": {"category": "may-in"}},
    {"id": "P06", "message": "Laptop Acer giá sinh viên",               "expected": {"category": "laptop", "brand": "Acer"}},
    {"id": "P07", "message": "Điện thoại Samsung mới nhất",             "expected": {"category": "mobile", "brand": "Samsung"}},
    {"id": "P08", "message": "Laptop gaming MSI",                       "expected": {"category": "laptop", "brand": "MSI"}},
    {"id": "P09", "message": "Laptop dưới 15 triệu",                    "expected": {"category": "laptop", "price_max": 15000000}},
    {"id": "P10", "message": "Điện thoại từ 10 đến 20 triệu",           "expected": {"category": "mobile", "price_min": 10000000, "price_max": 20000000}},
    {"id": "P11", "message": "Laptop gaming trên 25 triệu",             "expected": {"category": "laptop", "price_min": 25000000}},
    {"id": "P12", "message": "Laptop RAM 16GB",                         "expected": {"category": "laptop", "spec": {"ram": {"value": 16, "op": "gte"}}}},
    {"id": "P13", "message": "Laptop RAM 32GB SSD 512GB",               "expected": {"category": "laptop", "spec": {"ram": {"value": 32, "op": "gte"}, "storage": {"value": 512, "op": "gte"}}}},
    {"id": "P14", "message": "Laptop phản hồi dưới 5ms",                "expected": {"category": "laptop", "spec": {"response_time": {"value": 5, "op": "lte"}}}},
    {"id": "P15", "message": "Điện thoại pin hơn 5000mAh",              "expected": {"category": "mobile", "spec": {"battery": {"value": 5000, "op": "gte"}}}},
    {"id": "P16", "message": "Laptop dưới 20 triệu RAM 16GB màn hình 15 inch",  "expected": {"category": "laptop", "price_max": 20000000, "spec": {"ram": {"value": 16, "op": "gte"}, "screen_size": {"value": 15, "op": "eq"}}}},
    {"id": "P17", "message": "Samsung Galaxy S25 Ultra có tốt không",   "expected": {"mentioned_products": ["Samsung Galaxy S25 Ultra"]}},
    {"id": "P18", "message": "So sánh MacBook Air M4 và ASUS Vivobook S14", "expected": {"mentioned_products": ["MacBook Air M4", "ASUS Vivobook S14"]}},
    {"id": "P19", "message": "laptop tốt nhất hiện nay",                "expected": {"category": "laptop", "brand": None, "price_min": None, "price_max": None}},
    {"id": "P20", "message": "điện thoại pin trâu",                     "expected": {"category": "mobile", "spec": {"battery": {"value": None, "op": None}}}},
]


def check_spec(got_spec, expected_spec):
    for key, exp in expected_spec.items():
        got = got_spec.get(key)
        if exp.get("value") is None:
            if got is not None and got.get("value") is not None:
                return False, f"spec.{key} should be null but got value={got.get('value')}"
            continue
        if got is None:
            return False, f"spec.{key} missing"
        if got.get("value") != exp.get("value"):
            return False, f"spec.{key}.value expected={exp['value']} got={got.get('value')}"
        if got.get("op") != exp.get("op"):
            return False, f"spec.{key}.op expected={exp['op']} got={got.get('op')}"
    return True, ""


def invoke_with_retry(message):
    for attempt in range(MAX_RETRIES):
        try:
            result = parse_runnable.invoke({"messages": [HumanMessage(content=message)]})
            return result.model_dump()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(w in err for w in ["quota", "rate", "429", "resource exhausted"])
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 15
                print(f"         Rate limited, waiting {wait}s before retry {attempt + 2}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise


def run_tests():
    passed = 0
    results = []
    print("\n" + "=" * 65)
    print("PARSE NODE TEST")
    print("=" * 65)

    for i, tc in enumerate(TEST_CASES):
        if i > 0:
            time.sleep(DELAY_BETWEEN_CALLS)

        try:
            rd = invoke_with_retry(tc["message"])
        except Exception as e:
            print(f"[{tc['id']}] EXCEPTION: {e}")
            results.append({"id": tc["id"], "passed": False})
            continue

        errors = []
        for field, exp in tc["expected"].items():
            if field == "spec":
                ok, msg = check_spec(rd.get("spec", {}), exp)
                if not ok:
                    errors.append(msg)
            elif field == "mentioned_products":
                got_lower = [x.lower() for x in rd.get("mentioned_products", [])]
                for name in exp:
                    if not any(name.lower() in g for g in got_lower):
                        errors.append(f"mentioned_products missing '{name}', got {rd.get('mentioned_products')}")
            else:
                got = rd.get(field)
                if isinstance(exp, str):
                    ok = got is not None and exp.lower() in got.lower()
                else:
                    ok = got == exp
                if not ok:
                    errors.append(f"{field}: expected={exp!r} got={got!r}")

        if errors:
            print(f"[{tc['id']}] FAIL  \"{tc['message'][:58]}\"")
            for e in errors:
                print(f"         -> {e}")
            results.append({"id": tc["id"], "passed": False})
        else:
            print(f"[{tc['id']}] PASS  \"{tc['message'][:58]}\"")
            passed += 1
            results.append({"id": tc["id"], "passed": True})

    total = len(TEST_CASES)
    print("=" * 65)
    print(f"RESULT: {passed}/{total} passed ({100*passed//total}%)")
    print("=" * 65 + "\n")
    return results


if __name__ == "__main__":
    run_tests()