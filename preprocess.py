import numpy as np
import joblib

# Load scaler (agar training ke waqt scaler save kiya tha)
# Agar scaler nahi hai to is file ka scaler part skip kar sakte ho
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None


def preprocess_input(data: dict):
    """
    Takes raw transaction data from API (JSON)
    Returns preprocessed numpy array ready for model prediction
    """

    # Order MUST match training dataset
    feature_order = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
        "V28", "Amount"
    ]

    try:
        values = [float(data[feature]) for feature in feature_order]
    except KeyError as e:
        raise ValueError(f"Missing feature: {e}")

    X = np.array(values).reshape(1, -1)

    # Apply scaling if scaler exists
    if scaler is not None:
        X = scaler.transform(X)

    return X
