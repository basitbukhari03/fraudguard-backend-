"""
Auth Route Blueprint
---------------------
Exposes /auth/register, /auth/login, and /auth/me endpoints
for user authentication with MongoDB + JWT.
"""

import os
import datetime
from flask import Blueprint, request, jsonify
import bcrypt
import jwt
from services.database import users_collection

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# JWT Secret — set in Render environment variables
JWT_SECRET = os.environ.get("JWT_SECRET", "fraudguard-secret-key-change-in-production")
JWT_EXPIRY_HOURS = 72  # Token valid for 3 days


def _generate_token(user_id: str, email: str, name: str) -> str:
    """Generate a JWT token for the user."""
    payload = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    POST /auth/register
    Body: { "name": "...", "email": "...", "password": "..." }
    Returns: { "token": "...", "user": { "name": "...", "email": "..." } }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    # Validation
    errors = []
    if not name:
        errors.append("Name is required.")
    if not email or "@" not in email:
        errors.append("Valid email is required.")
    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters.")
    if errors:
        return jsonify({"error": errors}), 422

    # Hash password
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    # Insert user
    try:
        result = users_collection.insert_one({
            "name": name,
            "email": email,
            "password": hashed,
            "created_at": datetime.datetime.utcnow(),
        })
    except Exception as e:
        if "duplicate key" in str(e).lower() or "E11000" in str(e):
            return jsonify({"error": "An account with this email already exists."}), 409
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

    # Generate token
    token = _generate_token(str(result.inserted_id), email, name)

    return jsonify({
        "token": token,
        "user": {"name": name, "email": email}
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    POST /auth/login
    Body: { "email": "...", "password": "..." }
    Returns: { "token": "...", "user": { "name": "...", "email": "..." } }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 422

    # Find user
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    # Verify password
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return jsonify({"error": "Invalid email or password."}), 401

    # Generate token
    token = _generate_token(str(user["_id"]), email, user["name"])

    return jsonify({
        "token": token,
        "user": {"name": user["name"], "email": user["email"]}
    })


@auth_bp.route("/me", methods=["GET"])
def me():
    """
    GET /auth/me
    Header: Authorization: Bearer <token>
    Returns: { "user": { "name": "...", "email": "..." } }
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid authorization header."}), 401

    token = auth_header.split(" ", 1)[1]

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token has expired. Please log in again."}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token."}), 401

    return jsonify({
        "user": {
            "name": payload["name"],
            "email": payload["email"],
        }
    })
