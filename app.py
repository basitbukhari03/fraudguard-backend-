"""
Fraud Detection API — Main Application
---------------------------------------
Flask entry point with Blueprint registration and CORS support.
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from routes.predict import predict_bp

app = Flask(__name__)

# CORS: allow local dev, production domain, and Vercel previews
CORS(app, origins=[
    "http://localhost:8080",
    "http://localhost:5173",
    "https://fraudguard.live",
    "https://www.fraudguard.live",
    "https://fraudguard-frontend.vercel.app",
], supports_credentials=True)

# Register route blueprints
app.register_blueprint(predict_bp)


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Fraud Detection API",
        "version": "2.0",
        "endpoints": {
            "POST /predict": "Analyze a transaction for fraud"
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
