# app.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_MODEL = os.getenv("HF_MODEL", "google/gemma-7b-it")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

HF_TIMEOUT_SECONDS = float(os.getenv("HF_TIMEOUT_SECONDS", "20"))
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))


@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva con AI!"


def _get_env_key(primary_name: str, *aliases: str) -> Optional[str]:
    """
    Returns the first non-empty environment variable value among primary and aliases.
    """
    for name in (primary_name, *aliases):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _hf_generate(hf_key: str, prompt: str) -> Optional[str]:
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {hf_key}"}

    # HuggingFace Inference API sometimes returns non-200 JSON with details
    resp = requests.post(
        url,
        headers=headers,
        json={"inputs": prompt},
        timeout=HF_TIMEOUT_SECONDS,
    )

    if resp.status_code != 200:
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text[:500]}
        print(f"[HF] status={resp.status_code} payload={payload}")
        return None

    try:
        output = resp.json()
    except Exception as e:
        print(f"[HF] JSON parse error: {e} body={resp.text[:500]}")
        return None

    # Expected: list of {"generated_text": "..."}
    if isinstance(output, list) and output:
        item = output[0]
        if isinstance(item, dict):
            text = item.get("generated_text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    print(f"[HF] Unexpected output shape: {type(output)} value={str(output)[:500]}")
    return None


def _groq_chat(groq_key: str, prompt: str) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=GROQ_TIMEOUT_SECONDS)

    if resp.status_code != 200:
        # Important: surface real error (401/429 etc.)
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text[:500]}
        print(f"[GROQ] status={resp.status_code} payload={payload}")
        return None

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[GROQ] JSON parse error: {e} body={resp.text[:500]}")
        return None


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify({"reply": "⚠️ Messaggio vuoto."})

    # Accept both correct and your previous naming (alias)
    hf_key = _get_env_key("HUGGINGFACE_API_KEY", "HF_API_KEY")
    groq_key = _get_env_key("GROQ_API_KEY")

    # Debug visibility in Render logs (NO secrets printed)
    print(
        f"[ENV] HF={'OK' if hf_key else 'MISSING'} "
        f"GROQ={'OK' if groq_key else 'MISSING'} "
        f"HF_MODEL={HF_MODEL} GROQ_MODEL={GROQ_MODEL}"
    )

    reply: Optional[str] = None

    # 1) Try HuggingFace (if key exists)
    if hf_key:
        try:
            reply = _hf_generate(hf_key, user_message)
        except requests.RequestException as e:
            print(f"[HF] Request error: {e}")

    # 2) Fallback to Groq
    if not reply and groq_key:
        try:
            reply = _groq_chat(groq_key, user_message)
        except requests.RequestException as e:
            print(f"[GROQ] Request error: {e}")

    # 3) If still nothing, return actionable error
    if not reply:
        # If keys missing, tell exactly which one
        if not hf_key and not groq_key:
            return jsonify(
                {
                    "reply": "❌ Nessuna AI disponibile: mancano GROQ_API_KEY e HUGGINGFACE_API_KEY su Render.",
                }
            )
        if groq_key is None and hf_key is not None:
            return jsonify(
                {
                    "reply": "❌ HuggingFace non ha risposto e manca GROQ_API_KEY per il fallback.",
                }
            )
        return jsonify(
            {
                "reply": "❌ Nessuna AI disponibile al momento (controlla Logs su Render per 401/429/timeout).",
            }
        )

    return jsonify({"reply": reply})


if __name__ == "__main__":
    # Render sets PORT; local dev uses 5000
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
