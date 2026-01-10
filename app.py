# app.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GROQ_API_KEY_ENV = "GROQ_API_KEY"
MODEL_ENV = "MODEL"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 20.0

GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))


@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva con Groq!"


def _get_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if not value:
        return None
    value = value.strip()
    return value or None


def _groq_chat(groq_key: str, model: str, prompt: str) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=GROQ_TIMEOUT_SECONDS)

    if resp.status_code != 200:
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text[:500]}
        print(f"[GROQ] status={resp.status_code} payload={payload}")
        return None

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip() if isinstance(content, str) else None
    except Exception as e:
        print(f"[GROQ] JSON parse error: {e} body={resp.text[:500]}")
        return None


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify({"reply": "⚠️ Messaggio vuoto."})

    groq_key = _get_env(GROQ_API_KEY_ENV)
    model = os.getenv(MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL

    print(f"[ENV] GROQ={'OK' if groq_key else 'MISSING'} MODEL={model}")

    if not groq_key:
        return jsonify({"reply": "❌ Manca GROQ_API_KEY su Render."})

    reply = None
    try:
        reply = _groq_chat(groq_key, model, user_message)
    except requests.RequestException as e:
        print(f"[GROQ] Request error: {e}")

    if not reply:
        return jsonify(
            {
                "reply": "❌ Nessuna AI disponibile al momento (controlla Logs su Render per 401/429/timeout).",
            }
        )

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
