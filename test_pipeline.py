"""Test the new prediction pipeline endpoint."""
import requests
import json

URL = "http://127.0.0.1:5000/predict"

def test(label, data, expect_status=200):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    print(f"Input: {json.dumps(data)}")
    r = requests.post(URL, json=data)
    print(f"Status: {r.status_code}")
    print(f"Response:\n{json.dumps(r.json(), indent=2, ensure_ascii=False)}")
    assert r.status_code == expect_status, f"Expected {expect_status}, got {r.status_code}"
    print(">>> PASSED")

# Test 1: Normal daytime transaction
test("Normal daytime transaction", {
    "transaction_id": "TXN-001",
    "amount": 150.50,
    "date": "2026-04-02",
    "time": "14:30"
})

# Test 2: Night + Weekend + High amount
test("Night + Weekend + High amount", {
    "transaction_id": "TXN-002",
    "amount": 7500.00,
    "date": "2026-04-05",  # Saturday
    "time": "03:00"
})

# Test 3: Micro-transaction (card testing)
test("Micro-transaction", {
    "transaction_id": "TXN-003",
    "amount": 0.50,
    "date": "2026-04-01",
    "time": "02:15"
})

# Test 4: Validation - empty fields
test("Validation: empty fields", {
    "transaction_id": "",
    "amount": "",
    "date": "",
    "time": ""
}, expect_status=422)

# Test 5: Validation - missing all keys
test("Validation: missing fields", {
    "something": "else"
}, expect_status=422)

print(f"\n{'='*60}")
print("ALL TESTS PASSED >>>")
print(f"{'='*60}")
