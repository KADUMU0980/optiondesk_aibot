"""
/api/auth — POST
Validates Upstox token by calling /user/profile.
Token is sent in request body and returned to client to store in localStorage.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

UPSTOX_BASE = "https://api.upstox.com/v2"


@app.route("/api/auth", methods=["POST", "OPTIONS"])
def auth():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body = request.get_json(silent=True) or {}
    token = body.get("token", "").strip()

    if not token:
        return jsonify({"success": False, "error": "Token required"}), 400

    try:
        r = requests.get(
            f"{UPSTOX_BASE}/user/profile",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=10,
        )
        d = r.json()
        if d.get("status") == "success":
            prof = d.get("data", {})
            return jsonify({"success": True, "user": prof.get("user_name", "User")})
        return jsonify({"success": False, "error": str(d.get("errors", "Invalid token"))}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Vercel serverless entrypoint
handler = app
