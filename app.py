from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
MODEL_ENV = "MODEL"

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_MEMORY_MESSAGES = 50

CHAT_MEMORY: Dict[str, List[Dict[str, Any]]] = {}

# ---------------- UTILS ----------------
def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else None


def _sanitize_reply(text: str) -> str:
    s = text
    s = re.sub(r"```(?:\w+)?\n([\s\S]*?)```", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _build_system_prompt(lang: str, mode: str) -> str:
    return (
        "Sei ChatAI World, un assistente AI utile.\n"
        f"Rispondi nella lingua dell'utente ({lang}).\n"
        "Niente markdown.\n"
        "Testo semplice.\n"
        f"Modalità: {mode}\n"
    )


def _get_or_init_memory(session_id: str, system_prompt: str):
    mem = CHAT_MEMORY.get(session_id)
    if not mem:
        mem = [{"role": "system", "content": system_prompt}]
        CHAT_MEMORY[session_id] = mem
    else:
        mem[0]["content"] = system_prompt
    return mem


def _trim_memory(messages):
    return [messages[0]] + messages[-(MAX_MEMORY_MESSAGES * 2):]


def _openrouter_chat(api_key, model, messages, timeout):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chatai-world.onrender.com",
        "X-Title": "ChatAI World",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        print("OPENROUTER ERROR:", r.text)
        return None

    return _sanitize_reply(r.json()["choices"][0]["message"]["content"])


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva (OpenRouter)"


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    CHAT_MEMORY.pop(data.get("session_id"), None)
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    session_id = str(data.get("session_id", "default"))
    message = str(data.get("message", "")).strip()
    image_data = data.get("image_data")
    lang = data.get("preferred_lang", "it")
    mode = data.get("mode", "general")

    if image_data and not message:
        return jsonify({
            "reply": "📷 Foto ricevuta. Posso descriverla se mi spieghi cosa mostra."
        })

    if not message:
        return jsonify({"reply": "⚠️ Scrivi un messaggio."})

    api_key = _get_env(OPENROUTER_API_KEY_ENV)
    if not api_key:
        return jsonify({"reply": "❌ OPENROUTER_API_KEY mancante su Render."})

    model = _get_env(MODEL_ENV) or DEFAULT_MODEL
    timeout = float(os.getenv("TIMEOUT", DEFAULT_TIMEOUT_SECONDS))

    system_prompt = _build_system_prompt(lang, mode)
    memory = _get_or_init_memory(session_id, system_prompt)

    memory.append({"role": "user", "content": message})
    memory = _trim_memory(memory)

    reply = _openrouter_chat(api_key, model, memory, timeout)
    if not reply:
        return jsonify({"reply": "❌ Errore AI."})

    memory.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
