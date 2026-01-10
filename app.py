# app.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ======================
# CONFIG
# ======================
GROQ_API_KEY_ENV = "GROQ_API_KEY"
MODEL_ENV = "MODEL"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_MEMORY_MESSAGES = 50  # 👈 memoria reale

# ======================
# MEMORY (per IP)
# ======================
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}


# ======================
# UTILS
# ======================
def _get_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _sanitize_reply(text: str) -> str:
    s = text
    s = re.sub(r"```(?:\w+)?\n([\s\S]*?)```", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"\*(.*?)\*", r"\1", s)
    s = re.sub(r"_(.*?)_", r"\1", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s{0,3}>\s?", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*[-•]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)
    s = s.replace("*", "")
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva con Groq + memoria!"


def _groq_chat(
    groq_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_seconds: float,
) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)

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
        return _sanitize_reply(content) if isinstance(content, str) else None
    except Exception as e:
        print(f"[GROQ] JSON parse error: {e}")
        return None


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify({"reply": "⚠️ Messaggio vuoto."})

    groq_key = _get_env(GROQ_API_KEY_ENV)
    model = (_get_env(MODEL_ENV) or DEFAULT_MODEL)
    timeout_seconds = float(os.getenv("GROQ_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))

    if not groq_key:
        return jsonify({"reply": "❌ Manca GROQ_API_KEY su Render."})

    # 🔑 Identifica utente (IP)
    user_id = request.remote_addr or "anonymous"

    # Inizializza memoria
    if user_id not in CHAT_MEMORY:
        CHAT_MEMORY[user_id] = [
            {
                "role": "system",
                "content": (
                    "Sei ChatAI World. Rispondi in italiano, in testo semplice, "
                    "senza Markdown, con risposte chiare e naturali."
                ),
            }
        ]

    memory = CHAT_MEMORY[user_id]

    # Aggiungi messaggio utente
    memory.append({"role": "user", "content": user_message})

    # Tieni solo ultimi 50 messaggi (escluso system)
    system_msg = memory[0]
    trimmed = [system_msg] + memory[-MAX_MEMORY_MESSAGES * 2 :]

    print(f"[MEMORY] IP={user_id} messages={len(trimmed)} model={model}")

    reply = _groq_chat(groq_key, model, trimmed, timeout_seconds)

    if not reply:
        return jsonify({"reply": "❌ Nessuna AI disponibile (controlla Logs Render)."})

    # Salva risposta AI
    trimmed.append({"role": "assistant", "content": reply})
    CHAT_MEMORY[user_id] = trimmed

    return jsonify({"reply": reply})


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
