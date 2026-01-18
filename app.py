# app.py
from __future__ import annotations

import os
from typing import Any, Dict

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# Allow all origins for a public demo. Restrict in production.
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "llama3-8b-8192").strip()
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions").strip()

DEFAULT_TIMEOUT_SECONDS = 20


@app.get("/")
def home():
    return jsonify({"status": "ok", "service": "ChatAI World API", "provider": "groq"}), 200


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # Explicitly handle preflight for some clients/proxies.
    if request.method == "OPTIONS":
        return ("", 204)

    if not GROQ_API_KEY:
        return jsonify({"reply": "❌ GROQ_API_KEY mancante su Render"}), 500

    data: Dict[str, Any] = request.get_json(silent=True) or {}
    user_text = str(data.get("message", "")).strip()
    session_id = str(data.get("session_id", "")).strip()

    if not user_text:
        return jsonify({"reply": "⚠️ Scrivi un messaggio.", "session_id": session_id}), 400

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Sei ChatAI World, assistente utile e chiaro."},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.7,
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return jsonify({"reply": "⏳ Timeout chiamando Groq. Riprova.", "session_id": session_id}), 504
    except requests.RequestException as e:
        return jsonify({"reply": f"❌ Errore rete: {e}", "session_id": session_id}), 502

    # Non-200: prova a mostrare un messaggio utile senza crashare.
    if not resp.ok:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:300]
        return (
            jsonify(
                {
                    "reply": f"❌ Groq HTTP {resp.status_code}: {detail}".strip(),
                    "session_id": session_id,
                }
            ),
            502,
        )

    try:
        data_out = resp.json()
        reply = data_out["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"reply": f"❌ Risposta Groq non valida: {e}", "session_id": session_id}), 502

    return jsonify({"reply": reply, "session_id": session_id}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
