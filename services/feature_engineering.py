"""
Feature Engineering Service
---------------------------
Converts raw transaction fields (transaction_id, amount, date, time)
into ML-ready features for the XGBoost fraud detection model.

Pipeline: Validation → Feature Engineering → Vector Conversion → Scaling
"""

import hashlib
import math
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ── Constants ────────────────────────────────────────────────────────────────

HIGH_AMOUNT_THRESHOLD = 5000.0
NIGHT_START_HOUR = 22  # 10 PM
NIGHT_END_HOUR = 5     # 5 AM
TOTAL_MODEL_FEATURES = 30  # XGBoost model expects 30 features

# Pre-fitted scaler with reasonable defaults for the 8 engineered features
# In production, this would be fitted on training data and saved via joblib
_scaler = StandardScaler()
_scaler.mean_ = np.array([
    4.5,    # log_amount (mean of log values)
    12.0,   # hour (midday)
    15.0,   # day (mid-month)
    6.5,    # month (mid-year)
    0.286,  # weekend_flag (~2/7 days)
    0.292,  # night_flag (~7/24 hours)
    0.1,    # high_amount_flag (10% of txns)
    0.5,    # transaction_hash (normalized 0-1)
])
_scaler.scale_ = np.array([
    2.0,    # log_amount
    6.9,    # hour
    8.8,    # day
    3.5,    # month
    0.452,  # weekend_flag
    0.455,  # night_flag
    0.3,    # high_amount_flag
    0.29,   # transaction_hash
])
_scaler.var_ = _scaler.scale_ ** 2
_scaler.n_features_in_ = 8
_scaler.n_samples_seen_ = np.array([100000] * 8)


# ── 1) Input Validation ─────────────────────────────────────────────────────

def validate_input(data: dict) -> list[str]:
    """
    Validates raw transaction input fields.
    Returns a list of error messages (empty list = valid).
    """
    errors = []

    if not data:
        return ["Request body is empty."]

    # Transaction ID
    txn_id = data.get("transaction_id", "")
    if not txn_id or not str(txn_id).strip():
        errors.append("Transaction ID is required.")

    # Amount
    amount = data.get("amount")
    if amount is None or amount == "":
        errors.append("Amount is required.")
    else:
        try:
            amount_val = float(amount)
            if amount_val < 0:
                errors.append("Amount must be a positive number.")
        except (ValueError, TypeError):
            errors.append("Amount must be a valid numeric value.")

    # Date
    date_str = data.get("date", "")
    if not date_str or not str(date_str).strip():
        errors.append("Date is required.")
    else:
        try:
            datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
        except ValueError:
            errors.append("Date must be in YYYY-MM-DD format.")

    # Time
    time_str = data.get("time", "")
    if not time_str or not str(time_str).strip():
        errors.append("Time is required.")
    else:
        try:
            datetime.strptime(str(time_str).strip(), "%H:%M")
        except ValueError:
            errors.append("Time must be in HH:MM format.")

    return errors


# ── 2) Feature Engineering ───────────────────────────────────────────────────

def engineer_features(data: dict) -> dict:
    """
    Converts raw transaction fields into meaningful ML-ready features.

    Input:  { transaction_id, amount, date, time }
    Output: dict of engineered feature names → values
    """
    amount = float(data["amount"])
    date_str = str(data["date"]).strip()
    time_str = str(data["time"]).strip()
    txn_id = str(data["transaction_id"]).strip()

    # Parse date & time
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    time_obj = datetime.strptime(time_str, "%H:%M")

    hour = time_obj.hour
    day = date_obj.day
    month = date_obj.month
    weekday = date_obj.weekday()  # 0=Monday, 6=Sunday

    # ── Engineered features ──

    # Log-transformed amount (add 1 to handle amount=0)
    log_amount = math.log1p(amount)

    # Weekend flag (Saturday=5, Sunday=6)
    weekend_flag = 1.0 if weekday >= 5 else 0.0

    # Night transaction flag (10 PM – 5 AM)
    night_flag = 1.0 if (hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR) else 0.0

    # High amount flag
    high_amount_flag = 1.0 if amount > HIGH_AMOUNT_THRESHOLD else 0.0

    # Transaction hash → normalized numeric value (0-1)
    hash_digest = hashlib.sha256(txn_id.encode()).hexdigest()
    transaction_hash = int(hash_digest[:8], 16) / 0xFFFFFFFF

    return {
        "log_amount": round(log_amount, 6),
        "hour": float(hour),
        "day": float(day),
        "month": float(month),
        "weekend_flag": weekend_flag,
        "night_flag": night_flag,
        "high_amount_flag": high_amount_flag,
        "transaction_hash": round(transaction_hash, 6),
    }


# ── 3) Numeric Vector Conversion ────────────────────────────────────────────

def build_feature_vector(engineered: dict) -> np.ndarray:
    """
    Converts engineered features dict into an ordered numeric array.
    Returns shape (1, 8) numpy array.
    """
    feature_order = [
        "log_amount",
        "hour",
        "day",
        "month",
        "weekend_flag",
        "night_flag",
        "high_amount_flag",
        "transaction_hash",
    ]
    values = [engineered[key] for key in feature_order]
    return np.array(values, dtype=np.float64).reshape(1, -1)


# ── 4) Scaling ───────────────────────────────────────────────────────────────

def scale_features(vector: np.ndarray) -> np.ndarray:
    """
    Applies StandardScaler to the 8-feature vector.
    Returns scaled array of same shape.
    """
    return _scaler.transform(vector)


# ── 5) Pad to Model Dimensions ──────────────────────────────────────────────

def pad_to_model_input(scaled_vector: np.ndarray) -> np.ndarray:
    """
    The XGBoost model expects 30 features (Time, V1-V28, Amount).
    We map our 8 engineered features into the 30-slot vector:
      - Slot 0 (Time)   ← hour (scaled)
      - Slot 29 (Amount) ← log_amount (scaled)
      - Slots 1-8       ← remaining engineered features
      - Slots 9-28      ← zeros

    This allows the model to run; predictions are approximate.
    """
    model_input = np.zeros((1, TOTAL_MODEL_FEATURES), dtype=np.float64)

    scaled = scaled_vector.flatten()

    # Map engineered features into the model's feature slots
    model_input[0, 0] = scaled[1]   # hour → Time slot
    model_input[0, 29] = scaled[0]  # log_amount → Amount slot
    model_input[0, 1] = scaled[2]   # day
    model_input[0, 2] = scaled[3]   # month
    model_input[0, 3] = scaled[4]   # weekend_flag
    model_input[0, 4] = scaled[5]   # night_flag
    model_input[0, 5] = scaled[6]   # high_amount_flag
    model_input[0, 6] = scaled[7]   # transaction_hash

    return model_input


# ── 6) Risk Insights Generator ──────────────────────────────────────────────

def generate_risk_insights(engineered: dict, amount: float) -> list[str]:
    """
    Generates human-readable risk insights from the engineered features.
    """
    insights = []

    if engineered["night_flag"] == 1.0:
        insights.append("⚠️ Transaction occurred during high-risk hours (10 PM – 5 AM)")

    if engineered["weekend_flag"] == 1.0:
        insights.append("📅 Weekend transaction detected — fraud rates are elevated")

    if engineered["high_amount_flag"] == 1.0:
        insights.append(f"💰 High-value transaction (${amount:,.2f} exceeds ${HIGH_AMOUNT_THRESHOLD:,.0f} threshold)")

    if amount < 1.0:
        insights.append("🔍 Micro-transaction detected — common in card-testing fraud")

    hour = int(engineered["hour"])
    if 2 <= hour <= 4:
        insights.append("🌙 Deep-night transaction (2 AM – 4 AM) — extremely unusual activity window")

    if not insights:
        insights.append("✅ No unusual patterns detected in transaction attributes")

    return insights


# ── Full Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(data: dict) -> dict:
    """
    Executes the complete feature engineering pipeline.

    Input:  raw transaction dict
    Output: {
        "model_input": np.ndarray ready for prediction,
        "engineered_features": dict of named features,
        "risk_insights": list of insight strings,
    }
    """
    # Step 1: Validate
    errors = validate_input(data)
    if errors:
        raise ValueError(errors)

    # Step 2: Feature engineering
    engineered = engineer_features(data)

    # Step 3: Numeric vector conversion
    vector = build_feature_vector(engineered)

    # Step 4: Scale
    scaled = scale_features(vector)

    # Step 5: Pad to match model expectations
    model_input = pad_to_model_input(scaled)

    # Step 6: Risk insights
    amount = float(data["amount"])
    risk_insights = generate_risk_insights(engineered, amount)

    return {
        "model_input": model_input,
        "engineered_features": engineered,
        "risk_insights": risk_insights,
    }
