"""
Fraud Detection API — Main Application
---------------------------------------
Flask entry point with Blueprint registration, CORS, and Mail support.
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from flask_mail import Mail
from routes.predict import predict_bp
from routes.auth import auth_bp

app = Flask(__name__)

# ── Mail Configuration (Gmail SMTP) ──────────────────────────────
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD", "")
app.config["MAIL_DEFAULT_SENDER"] = ("FraudGuard", os.environ.get("MAIL_USERNAME", "noreply@fraudguard.live"))
app.config["MAIL_TIMEOUT"] = 10  # 10 second timeout for SMTP

mail = Mail(app)

# Make mail accessible to blueprints
app.extensions["mail"] = mail

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
        "version": "3.2",
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
    username = os.environ.get("MAIL_USERNAME", "")
    password = os.environ.get("MAIL_PASSWORD", "")
    result = {
        "mail_username_set": bool(username),
        "mail_username_value": username[:3] + "***" if username else "NOT SET",
        "mail_password_set": bool(password),
        "mail_password_length": len(password) if password else 0,
        "mail_server": app.config.get("MAIL_SERVER"),
        "mail_port": app.config.get("MAIL_PORT"),
        "mail_use_tls": app.config.get("MAIL_USE_TLS"),
    }
    # Try SMTP connection
    try:
        import smtplib
        server = smtplib.SMTP(app.config["MAIL_SERVER"], app.config["MAIL_PORT"], timeout=10)
        server.starttls()
        server.login(username, password)
        server.quit()
        result["smtp_connection"] = "SUCCESS"
    except Exception as e:
        result["smtp_connection"] = f"FAILED: {str(e)}"
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
