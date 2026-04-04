"""Quick test of the improved feature engineering pipeline."""
from services.feature_engineering import run_pipeline
from routes.predict import model

test_cases = [
    {"label": "Normal $150 2PM", "data": {"transaction_id": "TXN-001", "amount": "150", "date": "2026-04-04", "time": "14:00"}},
    {"label": "Normal $500 noon", "data": {"transaction_id": "RANDOM-XYZ", "amount": "500", "date": "2026-04-04", "time": "12:00"}},
    {"label": "Night $25000 1130PM", "data": {"transaction_id": "NIGHT-TXN", "amount": "25000", "date": "2026-04-05", "time": "23:30"}},
    {"label": "Suspicious $99999 3AM", "data": {"transaction_id": "TXN-FRAUD", "amount": "99999", "date": "2026-04-04", "time": "03:00"}},
    {"label": "Micro $0.50 230AM", "data": {"transaction_id": "MICRO-001", "amount": "0.50", "date": "2026-04-04", "time": "02:30"}},
    {"label": "Weekend $8000 1AM", "data": {"transaction_id": "WKND-001", "amount": "8000", "date": "2026-04-05", "time": "01:00"}},
    {"label": "Small $50 10AM", "data": {"transaction_id": "SAFE-001", "amount": "50", "date": "2026-04-03", "time": "10:00"}},
    {"label": "Random $12345 4PM", "data": {"transaction_id": "ABCXYZ123", "amount": "12345", "date": "2026-01-15", "time": "16:00"}},
]

lines = []
for tc in test_cases:
    result = run_pipeline(tc["data"])
    prob = float(model.predict_proba(result["model_input"])[0][1])
    label = "FRAUD" if prob > 0.5 else "Legit"
    lines.append(f"{tc['label']}: {prob:.4f} ({label})")

with open("test_results.txt", "w") as f:
    f.write("\n".join(lines))

print("Done - check test_results.txt")
