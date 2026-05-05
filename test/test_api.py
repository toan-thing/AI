import sys, os, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests

BASE_URL = "http://localhost:8000/api/agent"

REAL_CUSTOMER_ID = "test-customer-001"
REAL_VARIANT_ID = "VAR-0686-01"



def new_sid():
    return str(uuid.uuid4())


def check(label, passed, detail=""):
    mark = "PASS" if passed else "FAIL"
    line = f"  {mark} {label}"
    if detail:
        line += f"  ->  {detail}"
    print(line)
    return passed


def sep(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def safe_json(r):
    try:
        return r.json()
    except Exception:
        return {}


def test_health():
    sep("GET /health")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    ok = True
    ok &= check("Status 200", r.status_code == 200)
    if not ok:
        return ok
    d = safe_json(r)
    ok &= check("Has 'status'", "status" in d)
    ok &= check("status == 'ok'", d.get("status") == "ok")
    ok &= check("Has 'sessions'", "sessions" in d)
    return ok


def test_chat_basic():
    sep("POST /chat -- basic")
    sid = new_sid()
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": sid, "message": "xin chào"},
        timeout=60,
    )
    ok = True
    ok &= check("Status 200", r.status_code == 200)
    if not ok:
        return ok
    d = safe_json(r)
    ok &= check("Has 'session_id'", "session_id" in d)
    ok &= check("session_id matches", d.get("session_id") == sid)
    ok &= check("Has 'reply'", "reply" in d)
    ok &= check("reply non-empty", bool(d.get("reply", "").strip()), repr(d.get("reply", "")[:60]))
    return ok


def test_chat_with_customer_id():
    sep("POST /chat -- with customer_id (personalization)")
    sid = new_sid()
    r = requests.post(
        f"{BASE_URL}/chat",
        json={
            "session_id": sid,
            "message": "cho tôi xem điện thoại",
            "customer_id": REAL_CUSTOMER_ID,
        },
        timeout=120,
    )
    ok = True
    ok &= check("Status 200", r.status_code == 200)
    if not ok:
        return ok
    ok &= check("Has 'reply'", "reply" in safe_json(r))
    return ok


def test_state_fields():
    sep("GET /state/{session_id} -- correct fields after chat")
    sid = new_sid()
    requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": sid, "message": "laptop Dell dưới 20 triệu"},
        timeout=60,
    )

    r = requests.get(f"{BASE_URL}/state/{sid}", timeout=10)
    ok = True
    ok &= check("Status 200", r.status_code == 200)
    if not ok:
        return ok
    d = safe_json(r)
    for field in [
        "user_id",
        "category",
        "brand",
        "series",
        "color",
        "price_min",
        "price_max",
        "spec",
        "mentioned_products",
        "resolved_products",
        "messages_count",
    ]:
        ok &= check(f"Has '{field}'", field in d)
    ok &= check("category == 'laptop'", d.get("category") == "laptop", str(d.get("category")))
    ok &= check("price_max == 20000000", d.get("price_max") == 20000000, str(d.get("price_max")))
    ok &= check(
        "messages_count >= 2",
        (d.get("messages_count") or 0) >= 2,
        str(d.get("messages_count")),
    )
    return ok


def test_state_not_found():
    sep("GET /state/{session_id} -- non-existent -> 404")
    r = requests.get(f"{BASE_URL}/state/nonexistent-{new_sid()}", timeout=10)
    return check("Status 404", r.status_code == 404)


def test_session_context():
    sep("POST /chat -- context persists across turns")
    sid = new_sid()
    requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": sid, "message": "laptop dưới 20 triệu"},
        timeout=60,
    )
    r2 = requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": sid, "message": "thêm điều kiện RAM 16GB"},
        timeout=60,
    )
    state = safe_json(requests.get(f"{BASE_URL}/state/{sid}", timeout=10))

    ok = True
    ok &= check("Turn 2 status 200", r2.status_code == 200)
    ok &= check("Category still laptop", state.get("category") == "laptop", str(state.get("category")))
    ok &= check(
        "price_max still 20000000",
        state.get("price_max") == 20000000,
        str(state.get("price_max")),
    )
    ok &= check(
        "messages_count >= 4",
        (state.get("messages_count") or 0) >= 4,
        str(state.get("messages_count")),
    )
    return ok


def test_reset():
    sep("POST /reset/{session_id}")
    sid = new_sid()
    requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": sid, "message": "laptop"},
        timeout=60,
    )

    before = requests.get(f"{BASE_URL}/state/{sid}", timeout=10)
    ok = check("Session exists before reset", before.status_code == 200)

    reset_r = requests.post(f"{BASE_URL}/reset/{sid}", timeout=10)
    ok &= check("Reset status 200", reset_r.status_code == 200)
    if reset_r.status_code == 200:
        ok &= check("Reset returns ok", safe_json(reset_r).get("status") == "reset")

    after = requests.get(f"{BASE_URL}/state/{sid}", timeout=10)
    ok &= check("Session gone after reset", after.status_code == 404)
    return ok


def test_purchase_success():
    sep("POST /purchase -- valid variant")
    r = requests.post(
        f"{BASE_URL}/purchase",
        json={
            "customer_id": REAL_CUSTOMER_ID,
            "variant_id": REAL_VARIANT_ID,
            "amount": 1,
        },
        timeout=10,
    )
    ok = True
    ok &= check("Status 200", r.status_code == 200, f"got {r.status_code}: {r.text[:100]}")
    if not ok:
        return ok
    ok &= check("status == 'ok'", safe_json(r).get("status") == "ok")
    return ok


def test_purchase_not_found():
    sep("POST /purchase -- nonexistent variant -> 404")
    r = requests.post(
        f"{BASE_URL}/purchase",
        json={
            "customer_id": REAL_CUSTOMER_ID,
            "variant_id": "nonexistent-variant-id-xyz9999",
            "amount": 1,
        },
        timeout=10,
    )
    return check("Status 404", r.status_code == 404, f"got {r.status_code}")


def test_chat_missing_session_id():
    sep("POST /chat -- missing session_id -> 422")
    r = requests.post(f"{BASE_URL}/chat", json={"message": "xin chào"}, timeout=10)
    return check("Status 422 (validation error)", r.status_code == 422, f"got {r.status_code}")


def test_purchase_missing_field():
    sep("POST /purchase -- missing required field -> 422")
    r = requests.post(
        f"{BASE_URL}/purchase",
        json={"customer_id": REAL_CUSTOMER_ID},
        timeout=10,
    )
    return check("Status 422 (validation error)", r.status_code == 422, f"got {r.status_code}")


TESTS = [
    ("GET /health", test_health),
    ("POST /chat -- basic", test_chat_basic),
    ("POST /chat -- with customer_id", test_chat_with_customer_id),
    ("GET /state -- correct fields", test_state_fields),
    ("GET /state -- 404 not found", test_state_not_found),
    ("POST /chat -- context across turns", test_session_context),
    ("POST /reset", test_reset),
    ("POST /purchase -- success", test_purchase_success),
    ("POST /purchase -- 404 not found", test_purchase_not_found),
    ("POST /chat -- 422 missing field", test_chat_missing_session_id),
    ("POST /purchase -- 422 missing field", test_purchase_missing_field),
]


def run_tests():
    print("\n" + "=" * 60)
    print("  API ENDPOINT VERIFICATION")
    print(f"  Target: {BASE_URL}")
    print("=" * 60)

    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except Exception as e:
        print(f"\nCannot reach server at {BASE_URL}\n   {e}")
        sys.exit(1)

    summary = []
    for name, fn in TESTS:
        try:
            passed = fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            passed = False
        summary.append((name, passed))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Endpoint':<45} {'Result'}")
    print("  " + "-" * 52)
    total_passed = 0
    for name, passed in summary:
        if passed:
            total_passed += 1
        mark = "PASS" if passed else "FAIL"
        print(f"  {name:<45} {mark}")
    print("  " + "-" * 52)
    print(f"  Total: {total_passed}/{len(summary)} passed")
    print(f"{'='*60}\n")

    return [{"name": n, "passed": p} for n, p in summary]


if __name__ == "__main__":
    run_tests()