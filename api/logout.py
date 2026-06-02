"""
/api/logout — POST (stateless, just returns success — client clears localStorage)
"""
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/api/logout", methods=["POST", "OPTIONS"])
def logout():
    return jsonify({"success": True})


handler = app
