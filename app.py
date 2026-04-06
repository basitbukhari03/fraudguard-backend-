"""
Fraud Detection API — Main Application
---------------------------------------
Flask entry point with Blueprint registration, CORS, and Resend email support.
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from routes.predict import predict_bp
from routes.auth import auth_bp

app = Flask(__name__)

# ── Resend API Key (for email verification) ──────────────────────
app.config["RESEND_API_KEY"] = os.environ.get("RESEND_API_KEY", "")

# CORS: allow local dev, production domain, Vercel, and mobile app
CORS(app, origins=[
    "http://localhost:8080",
    "http://localhost:5173",
    "https://fraudguard.live",
    "https://www.fraudguard.live",
    "https://fraudguard-frontend.vercel.app",
], supports_credentials=True)

# Register route blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(auth_bp)


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Fraud Detection API",
        "version": "4.0",
        "endpoints": {
            "POST /predict": "Analyze a transaction for fraud",
            "POST /auth/register": "Create a new user account (sends OTP)",
            "POST /auth/verify": "Verify email with OTP code",
            "POST /auth/resend-code": "Resend verification code",
            "POST /auth/login": "Login with email and password",
            "GET /auth/me": "Get current user info (requires JWT)"
        }
    })


@app.route("/debug/mail")
def debug_mail():
    api_key = os.environ.get("RESEND_API_KEY", "")
    result = {
        "resend_api_key_set": bool(api_key),
        "resend_api_key_preview": api_key[:8] + "***" if api_key else "NOT SET",
    }
    # Test Resend API
    try:
        import requests
        r = requests.get(
            "https://api.resend.com/domains",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        result["resend_connection"] = f"Status {r.status_code}: {r.text[:100]}"
    except Exception as e:
        result["resend_connection"] = f"FAILED: {str(e)}"
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
