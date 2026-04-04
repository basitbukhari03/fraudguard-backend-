"""
Feature Engineering Service
---------------------------
Converts raw transaction fields (transaction_id, amount, date, time)
into ML-ready features for the XGBoost fraud detection model.

The XGBoost model was trained on credit card fraud data with 30 features:
  Time, V1-V28 (PCA components), Amount

Since we only receive (txn_id, amount, date, time), we engineer risk
signals and map them into the feature positions the model actually uses
to separate fraud from legitimate transactions.

Pipeline: Validation → Risk Analysis → Feature Synthesis → Model Input
"""

import hashlib
import math
import numpy as np
from datetime import datetime

# ── Constants ────────────────────────────────────────────────────────────────

HIGH_AMOUNT_THRESHOLD = 5000.0
VERY_HIGH_AMOUNT = 20000.0
EXTREME_AMOUNT = 50000.0
NIGHT_START_HOUR = 22  # 10 PM
NIGHT_END_HOUR = 5     # 5 AM
DEEP_NIGHT_START = 1   # 1 AM
DEEP_NIGHT_END = 4     # 4 AM
TOTAL_MODEL_FEATURES = 30


# ── 1) Input Validation ─────────────────────────────────────────────────────

def validate_input(data: dict) -> list[str]:
    """
    Validates raw transaction input fields.
    Returns a list of error messages (empty list = valid).
    """
    errors = []

    if not data:
        return ["Request body is empty."]

    txn_id = data.get("transaction_id", "")
    if not txn_id or not str(txn_id).strip():
        errors.append("Transaction ID is required.")

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

    date_str = data.get("date", "")
    if not date_str or not str(date_str).strip():
        errors.append("Date is required.")
    else:
        try:
            datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
        except ValueError:
            errors.append("Date must be in YYYY-MM-DD format.")

    time_str = data.get("time", "")
    if not time_str or not str(time_str).strip():
        errors.append("Time is required.")
    else:
        try:
            datetime.strptime(str(time_str).strip(), "%H:%M")
        except ValueError:
            errors.append("Time must be in HH:MM format.")

    return errors


# ── 2) Risk Score Calculation ────────────────────────────────────────────────

def calculate_risk_score(amount: float, hour: int, day: int, month: int,
                         weekday: int, txn_hash: float) -> float:
    """
    Calculates a composite risk score (0.0 - 1.0) based on multiple factors.
    Higher score = more suspicious transaction.
    """
    score = 0.0

    # ── Amount-based risk (0 - 0.40) ──
    if amount < 1.0:
        # Micro-transactions: card testing pattern
        score += 0.25
    elif amount > EXTREME_AMOUNT:
        score += 0.40
    elif amount > VERY_HIGH_AMOUNT:
        score += 0.30
    elif amount > HIGH_AMOUNT_THRESHOLD:
        score += 0.20
    elif amount > 2000.0:
        score += 0.08

    # ── Time-based risk (0 - 0.30) ──
    if DEEP_NIGHT_START <= hour <= DEEP_NIGHT_END:
        # 1 AM - 4 AM: highest risk window
        score += 0.30
    elif hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR:
        # 10 PM - 5 AM: elevated risk
        score += 0.18
    elif hour < 6:
        score += 0.12

    # ── Day-of-week risk (0 - 0.10) ──
    if weekday >= 5:  # Weekend
        score += 0.10
    elif weekday == 0:  # Monday (high fraud day)
        score += 0.05

    # ── Unusual day-of-month (0 - 0.05) ──
    if day == 1 or day >= 28:
        score += 0.05  # Start/end of month

    # ── Transaction hash adds variance (0 - 0.15) ──
    # This makes predictions more diverse — same risk factors produce
    # slightly different results for different transaction IDs
    hash_variance = (txn_hash - 0.5) * 0.15
    score += hash_variance

    return max(0.0, min(1.0, score))


# ── 3) Feature Synthesis ────────────────────────────────────────────────────

def synthesize_model_features(amount: float, hour: int, day: int,
                               month: int, weekday: int,
                               risk_score: float, txn_hash: float) -> np.ndarray:
    """
    Creates a 30-feature vector that maps our risk signals into the
    feature positions the XGBoost model uses for fraud detection.

    Key insight: In the original credit card fraud dataset, fraud
    transactions typically show:
      - Strong negative values in V14, V12, V10, V3
      - Elevated positive values in V4, V11
      - Higher absolute values in V1-V7
      - Specific patterns in Amount and Time

    We use the risk_score and individual factors to generate appropriate
    values in these positions.
    """
    features = np.zeros(TOTAL_MODEL_FEATURES, dtype=np.float64)

    log_amount = math.log1p(amount)
    is_night = 1.0 if (hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR) else 0.0
    is_weekend = 1.0 if weekday >= 5 else 0.0
    is_high_amount = 1.0 if amount > HIGH_AMOUNT_THRESHOLD else 0.0
    is_micro = 1.0 if amount < 1.0 else 0.0

    # ── Slot 0: Time ──
    # Original dataset: Time is seconds elapsed (0 - 172800)
    # Map hour to a reasonable range
    features[0] = hour * 3600.0 + day * 60.0

    # ── Slot 29: Amount ──
    features[29] = amount

    # ── V1 (slot 1): Strong general fraud indicator ──
    # Fraud transactions tend to have large negative V1
    if risk_score > 0.5:
        features[1] = -2.0 - risk_score * 3.0 + txn_hash * 0.5
    elif risk_score > 0.25:
        features[1] = -0.5 - risk_score * 1.5
    else:
        features[1] = 1.5 - risk_score * 2.0

    # ── V2 (slot 2): Amount-correlated component ──
    features[2] = (log_amount - 4.5) * 0.8 + (is_night * -1.2)

    # ── V3 (slot 3): Strong fraud separator ──
    # Fraud: typically large negative V3
    if risk_score > 0.4:
        features[3] = -3.0 * risk_score - is_high_amount * 2.0
    else:
        features[3] = 1.0 - risk_score * 3.0

    # ── V4 (slot 4): Fraud-positive component ──
    # Fraud transactions tend to have positive V4
    if risk_score > 0.3:
        features[4] = 2.0 * risk_score + is_night * 1.5
    else:
        features[4] = -0.5 + risk_score * 2.0

    # ── V5 (slot 5) ──
    features[5] = -1.5 * risk_score + is_weekend * 0.3

    # ── V6 (slot 6) ──
    features[6] = -0.8 * risk_score + txn_hash * 0.2

    # ── V7 (slot 7): Amount interaction ──
    features[7] = (log_amount - 5.0) * 0.5 * (1.0 + risk_score)

    # ── V8-V9 (slots 8-9) ──
    features[8] = -0.3 * risk_score + is_micro * 1.5
    features[9] = -0.5 * risk_score + (month - 6) * 0.05

    # ── V10 (slot 10): Important fraud indicator ──
    # Fraud: strongly negative V10
    if risk_score > 0.4:
        features[10] = -4.0 * risk_score
    else:
        features[10] = 0.2 - risk_score * 1.5

    # ── V11 (slot 11): Positive in fraud ──
    features[11] = 2.5 * risk_score + is_night * 0.8

    # ── V12 (slot 12): Strong negative in fraud ──
    if risk_score > 0.35:
        features[12] = -3.5 * risk_score - is_high_amount * 1.5
    else:
        features[12] = 0.5 - risk_score * 2.0

    # ── V13 (slot 13) ──
    features[13] = -0.3 * risk_score

    # ── V14 (slot 14): MOST important fraud indicator ──
    # Strongly negative for fraud in most credit card models
    if risk_score > 0.3:
        features[14] = -5.0 * risk_score - is_high_amount * 2.0
    elif risk_score > 0.15:
        features[14] = -1.0 * risk_score
    else:
        features[14] = 0.5 - risk_score

    # ── V15 (slot 15) ──
    features[15] = 0.3 * risk_score

    # ── V16-V17 (slots 16-17): Moderate importance ──
    features[16] = -2.0 * risk_score + txn_hash * 0.3
    features[17] = -1.5 * risk_score + is_night * 0.5

    # ── V18-V28 (slots 18-28): Lower importance, add noise ──
    for i in range(18, 29):
        # Generate deterministic but varied values
        slot_hash = (txn_hash * (i + 1)) % 1.0
        noise = (slot_hash - 0.5) * 0.3
        features[i] = -risk_score * 0.5 * (1.0 + noise)

    return features.reshape(1, -1)


# ── 4) Engineered Features for Display ──────────────────────────────────────

def engineer_features_for_display(data: dict) -> dict:
    """
    Creates the human-readable engineered features dict for the API response.
    """
    amount = float(data["amount"])
    date_str = str(data["date"]).strip()
    time_str = str(data["time"]).strip()
    txn_id = str(data["transaction_id"]).strip()

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    time_obj = datetime.strptime(time_str, "%H:%M")

    hour = time_obj.hour
    day = date_obj.day
    month = date_obj.month
    weekday = date_obj.weekday()

    log_amount = math.log1p(amount)
    weekend_flag = 1.0 if weekday >= 5 else 0.0
    night_flag = 1.0 if (hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR) else 0.0
    high_amount_flag = 1.0 if amount > HIGH_AMOUNT_THRESHOLD else 0.0

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


# ── 5) Risk Insights Generator ──────────────────────────────────────────────

def generate_risk_insights(engineered: dict, amount: float,
                           risk_score: float) -> list[str]:
    """
    Generates human-readable risk insights from the engineered features.
    """
    insights = []

    if engineered["night_flag"] == 1.0:
        hour = int(engineered["hour"])
        if DEEP_NIGHT_START <= hour <= DEEP_NIGHT_END:
            insights.append(
                "🚨 Transaction during deep-night hours (1 AM – 4 AM) "
                "— highest fraud risk window"
            )
        else:
            insights.append(
                "⚠️ Transaction occurred during high-risk hours (10 PM – 5 AM)"
            )

    if engineered["weekend_flag"] == 1.0:
        insights.append(
            "📅 Weekend transaction detected — fraud rates are elevated"
        )

    if amount > EXTREME_AMOUNT:
        insights.append(
            f"🚨 Extremely high-value transaction "
            f"(${amount:,.2f} exceeds ${EXTREME_AMOUNT:,.0f})"
        )
    elif engineered["high_amount_flag"] == 1.0:
        insights.append(
            f"💰 High-value transaction "
            f"(${amount:,.2f} exceeds ${HIGH_AMOUNT_THRESHOLD:,.0f} threshold)"
        )

    if amount < 1.0:
        insights.append(
            "🔍 Micro-transaction detected — common in card-testing fraud"
        )

    if risk_score > 0.6:
        insights.append(
            "🔴 Multiple high-risk factors combined — elevated fraud probability"
        )
    elif risk_score > 0.35:
        insights.append(
            "🟡 Moderate risk factors detected — review recommended"
        )

    if not insights:
        insights.append(
            "✅ No unusual patterns detected in transaction attributes"
        )

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

    # Step 2: Parse inputs
    amount = float(data["amount"])
    date_str = str(data["date"]).strip()
    time_str = str(data["time"]).strip()
    txn_id = str(data["transaction_id"]).strip()

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    time_obj = datetime.strptime(time_str, "%H:%M")

    hour = time_obj.hour
    day = date_obj.day
    month = date_obj.month
    weekday = date_obj.weekday()

    # Transaction hash for variance
    hash_digest = hashlib.sha256(txn_id.encode()).hexdigest()
    txn_hash = int(hash_digest[:8], 16) / 0xFFFFFFFF

    # Step 3: Calculate composite risk score
    risk_score = calculate_risk_score(
        amount, hour, day, month, weekday, txn_hash
    )

    # Step 4: Synthesize 30-feature model input
    model_input = synthesize_model_features(
        amount, hour, day, month, weekday, risk_score, txn_hash
    )

    # Step 5: Human-readable features
    engineered = engineer_features_for_display(data)

    # Step 6: Risk insights
    risk_insights = generate_risk_insights(engineered, amount, risk_score)

    return {
        "model_input": model_input,
        "engineered_features": engineered,
        "risk_insights": risk_insights,
    }
