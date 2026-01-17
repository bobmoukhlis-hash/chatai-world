from __future__ import annotations

import os
import re
import requests
from typing import Any, Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")
TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "25"))

MAX_MEMORY_MESSAGES = 40  # 40 turni
CHAT_MEMORY: Dict[str, List[Dict[str, Any]]] = {}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------- UTILS ----------------
def sanitize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def system_prompt(lang: str, mode: str) -> str:
    base = (
        "Sei ChatAI World, un assistente AI utile e affidabile.\n"
        "Regole:\n"
        "- Risposte chiare e semplici\n"
        "- Niente markdown\n"
        "- Aiuta sempre l'utente\n"
    )

    if lang:
        base += f"Rispondi nella lingua dell'utente ({lang}).\n"

    mode = (mode or "general").lower()
    if mode == "study":
        base += "Modalità Studio: spiega passo passo con esempi.\n"
    elif mode == "code":
        base += "Modalità Coding: fornisci codice funzionante e spiegazioni brevi.\n"
    elif mode == "content":
        base += "Modalità Social: crea contenuti pronti, titoli e CTA.\n"
    elif mode == "translate":
        base += "Modalità Traduzione: traduci in modo naturale.\n"

    return base


def get_memory(session_id: str, system_msg: str):
    if session_id not in CHAT_MEMORY:
        CHAT_MEMORY[session_id] = [{"role": "system", "content": system_msg}]
    else:
        CHAT_MEMORY[session_id][0] = {"role": "system", "content": system_msg}

    CHAT_MEMORY[session_id] = (
        CHAT_MEMORY[session_id][:1]
        + CHAT_MEMORY[session_id][1:][-MAX_MEMORY_MESSAGES * 2 :]
    )
    return CHAT_MEMORY[session_id]


def call_openrouter(messages: list) -> str | None:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chatai-world.onrender.com",
        "X-Title": "ChatAI World",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    r = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
    )

    if r.status_code != 200:
        print("OpenRouter error:", r.status_code, r.text[:300])
        return None

    data = r.json()
    return sanitize(data["choices"][0]["message"]["content"])


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva (OpenRouter)"


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    sid = str(data.get("session_id", "")).strip()
    if sid:
        CHAT_MEMORY.pop(sid, None)
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    if not OPENROUTER_API_KEY:
        return jsonify({"reply": "❌ OPENROUTER_API_KEY mancante su Render"})

    data = request.get_json(silent=True) or {}

    session_id = str(data.get("session_id", "default"))
    user_text = str(data.get("message", "")).strip()
    image_data = data.get("image_data")
    lang = str(data.get("preferred_lang", ""))
    mode = str(data.get("mode", "general"))

    if image_data and not user_text:
        return jsonify({
            "reply": "📷 Ho ricevuto la foto. Al momento posso solo lavorare sul testo. Descrivimi cosa vedi."
        })

    if not user_text:
        return jsonify({"reply": "⚠️ Scrivi un messaggio."})

    sys_msg = system_prompt(lang, mode)
    memory = get_memory(session_id, sys_msg)

    memory.append({"role": "user", "content": user_text})

    reply = call_openrouter(memory)
    if not reply:
        return jsonify({"reply": "❌ Errore AI. Controlla i log su Render."})

    memory.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
