import os
import re
import random
import datetime
import threading
import requests as http_requests
from flask import Blueprint, request, jsonify, current_app
import bcrypt
import jwt
from services.database import users_collection, verification_codes

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# JWT Secret — set in Render environment variables
JWT_SECRET = os.environ.get("JWT_SECRET", "fraudguard-secret-key-change-in-production")
JWT_EXPIRY_HOURS = 72  # Token valid for 3 days

# ── Allowed Email Domains ─────────────────────────────────────────
ALLOWED_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "yahoo.co.uk",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "icloud.com",
    "protonmail.com",
    "proton.me",
    "aol.com",
    "mail.com",
    "zoho.com",
    "yandex.com",
    "gmx.com",
    "gmx.net",
}

# ── Email Format Regex ────────────────────────────────────────────
EMAIL_REGEX = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)


def _validate_email(email: str) -> str | None:
    """Validate email format and domain. Returns error message or None."""
    if not email or not EMAIL_REGEX.match(email):
        return "Please enter a valid email address."

    domain = email.split("@")[1].lower()
    if domain not in ALLOWED_DOMAINS:
        return (
            f"Email domain '@{domain}' is not supported. "
            f"Please use Gmail, Outlook, Yahoo, iCloud, or ProtonMail."
        )
    return None


def _generate_otp() -> str:
    """Generate a 6-digit OTP code."""
    return str(random.randint(100000, 999999))


def _send_verification_email(email: str, name: str, code: str):
    """Send OTP verification email using Resend HTTP API in background thread."""
    api_key = current_app.config.get("RESEND_API_KEY", "")
    if not api_key:
        print("[MAIL] RESEND_API_KEY not set — skipping email.")
        return

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto;
                background: #0A0E1A; color: #FFFFFF; padding: 40px; border-radius: 16px;">
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #6C63FF; margin: 0;">FraudGuard</h1>
            <p style="color: #94A3B8; font-size: 14px;">AI-Powered Fraud Detection</p>
        </div>

        <p style="color: #E2E8F0;">Hi <strong>{name}</strong>,</p>
        <p style="color: #94A3B8;">Your verification code is:</p>

        <div style="text-align: center; margin: 30px 0;">
            <div style="display: inline-block; background: linear-gradient(135deg, #6C63FF, #8B83FF);
                        padding: 16px 40px; border-radius: 12px; letter-spacing: 8px;">
                <span style="font-size: 32px; font-weight: bold; color: #FFFFFF;">{code}</span>
            </div>
        </div>

        <p style="color: #94A3B8; font-size: 13px;">
            This code expires in <strong>10 minutes</strong>.<br>
            If you didn't create a FraudGuard account, you can ignore this email.
        </p>

        <hr style="border: 1px solid #1E2642; margin: 30px 0;">
        <p style="color: #64748B; font-size: 11px; text-align: center;">
            &copy; FraudGuard — Powered by XGBoost ML Pipeline
        </p>
    </div>
    """

    # Send in background thread so request doesn't hang
    def send_async():
        try:
            r = http_requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": "FraudGuard <noreply@fraudguard.live>",
                    "to": [email],
                    "subject": f"FraudGuard — Your Verification Code: {code}",
                    "html": html_body,
                },
                timeout=15,
            )
            if r.status_code == 200:
                print(f"[MAIL] ✅ Verification email sent to {email}")
            else:
                print(f"[MAIL] ❌ Resend API error {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[MAIL] ❌ Failed to send email to {email}: {str(e)}")

    thread = threading.Thread(target=send_async)
    thread.daemon = True
    thread.start()
    print(f"[MAIL] 📧 Email queued for {email}")


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


# ── REGISTER ─────────────────────────────────────────────────────

@auth_bp.route("/register", methods=["POST"])
def register():
    """
    POST /auth/register
    Body: { "name": "...", "email": "...", "password": "..." }
    Returns: { "message": "Verification code sent to email." }
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

    email_error = _validate_email(email)
    if email_error:
        errors.append(email_error)

    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters.")

    if errors:
        return jsonify({"error": errors}), 422

    # Check if already registered and verified
    existing = users_collection.find_one({"email": email})
    if existing:
        if existing.get("verified", False):
            return jsonify({"error": "An account with this email already exists."}), 409
        else:
            # Not verified yet — update password and resend code
            hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
            users_collection.update_one(
                {"email": email},
                {"$set": {"name": name, "password": hashed}}
            )
            # Generate and send new OTP
            code = _generate_otp()
            verification_codes.delete_many({"email": email})
            verification_codes.insert_one({
                "email": email,
                "code": code,
                "created_at": datetime.datetime.utcnow(),
            })
            _send_verification_email(email, name, code)
            return jsonify({
                "message": "Verification code sent to your email.",
                "email": email,
            }), 200

    # Hash password
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    # Insert user (unverified)
    try:
        users_collection.insert_one({
            "name": name,
            "email": email,
            "password": hashed,
            "verified": False,
            "created_at": datetime.datetime.utcnow(),
        })
    except Exception as e:
        if "duplicate key" in str(e).lower() or "E11000" in str(e):
            return jsonify({"error": "An account with this email already exists."}), 409
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

    # Generate OTP and save
    code = _generate_otp()
    verification_codes.insert_one({
        "email": email,
        "code": code,
        "created_at": datetime.datetime.utcnow(),
    })

    # Send OTP email (async — runs in background thread)
    _send_verification_email(email, name, code)

    return jsonify({
        "message": "Verification code sent to your email.",
        "email": email,
    }), 201


# ── VERIFY EMAIL ──────────────────────────────────────────────────

@auth_bp.route("/verify", methods=["POST"])
def verify():
    """
    POST /auth/verify
    Body: { "email": "...", "code": "..." }
    Returns: { "token": "...", "user": { "name": "...", "email": "..." } }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    email = data.get("email", "").strip().lower()
    code = data.get("code", "").strip()

    if not email or not code:
        return jsonify({"error": "Email and verification code are required."}), 422

    # Find the OTP record
    record = verification_codes.find_one({"email": email, "code": code})
    if not record:
        return jsonify({"error": "Invalid or expired verification code."}), 400

    # Mark user as verified
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found."}), 404

    users_collection.update_one(
        {"email": email},
        {"$set": {"verified": True}}
    )

    # Remove used verification codes
    verification_codes.delete_many({"email": email})

    # Generate token
    token = _generate_token(str(user["_id"]), email, user["name"])

    return jsonify({
        "token": token,
        "user": {"name": user["name"], "email": user["email"]}
    }), 200


# ── RESEND CODE ───────────────────────────────────────────────────

@auth_bp.route("/resend-code", methods=["POST"])
def resend_code():
    """
    POST /auth/resend-code
    Body: { "email": "..." }
    Returns: { "message": "New verification code sent." }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Email is required."}), 422

    # Find user
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "No account found with this email."}), 404

    if user.get("verified", False):
        return jsonify({"error": "This email is already verified."}), 400

    # Generate new code
    code = _generate_otp()
    verification_codes.delete_many({"email": email})
    verification_codes.insert_one({
        "email": email,
        "code": code,
        "created_at": datetime.datetime.utcnow(),
    })

    # Send email (async)
    _send_verification_email(email, user["name"], code)

    return jsonify({"message": "New verification code sent to your email."}), 200


# ── LOGIN ─────────────────────────────────────────────────────────

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

    # Check if verified
    if not user.get("verified", False):
        return jsonify({
            "error": "Email not verified. Please check your inbox for the verification code.",
            "unverified": True,
            "email": email,
        }), 403

    # Verify password
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return jsonify({"error": "Invalid email or password."}), 401

    # Generate token
    token = _generate_token(str(user["_id"]), email, user["name"])

    return jsonify({
        "token": token,
        "user": {"name": user["name"], "email": user["email"]}
    })


# ── ME ────────────────────────────────────────────────────────────

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
