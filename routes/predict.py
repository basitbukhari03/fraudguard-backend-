"""
Predict Route Blueprint
-----------------------
Exposes the /predict endpoint that receives raw transaction data
and returns fraud prediction results through the full ML pipeline.
"""

from flask import Blueprint, request, jsonify
import joblib
import os
import numpy as np

from services.feature_engineering import run_pipeline

# ── Load Model ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")
model = joblib.load(MODEL_PATH)

# ── Blueprint ────────────────────────────────────────────────────────────────

predict_bp = Blueprint("predict", __name__)


def _determine_risk_level(probability: float) -> str:
    """Categorise fraud probability into a risk level."""
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    else:
        return "high"


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    
    Expects JSON body:
    {
        "transaction_id": "TXN123",
        "amount": 150.50,
        "date": "2026-04-01",
        "time": "23:30"
    }
    
    Returns:
    {
        "prediction": "Fraud" | "Legitimate",
        "fraud_probability": 0.87,
        "risk_level": "high" | "medium" | "low",
        "engineered_features": { ... },
        "risk_insights": [ "...", "..." ]
    }
    """
    # Parse JSON body
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    # Run the full feature engineering pipeline
    try:
        pipeline_result = run_pipeline(data)
    except ValueError as e:
        # Validation errors come as a list
        errors = e.args[0] if e.args else ["Validation failed."]
        return jsonify({"error": errors}), 422
    except Exception as e:
        # Catch-all for unexpected errors
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500

    # ML Prediction
    try:
        model_input = pipeline_result["model_input"]
        prob = float(model.predict_proba(model_input)[0][1])
        prediction = "Fraud" if prob > 0.5 else "Legitimate"
        risk_level = _determine_risk_level(prob)
    except Exception as e:
        return jsonify({"error": f"Model prediction error: {str(e)}"}), 500

    return jsonify({
        "prediction": prediction,
        "fraud_probability": round(prob, 6),
        "risk_level": risk_level,
        "engineered_features": pipeline_result["engineered_features"],
        "risk_insights": pipeline_result["risk_insights"],
    })
