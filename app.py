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

# ================= CONFIG =================
GROQ_API_KEY_ENV = "GROQ_API_KEY"
MODEL_ENV = "MODEL"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
TIMEOUT_SECONDS = 25
MAX_TURNS = 50  # 50 messaggi utente + 50 AI

# memoria in RAM (per session_id)
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}


# ================= UTILS =================
def get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v else None


def clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = text.replace("**", "").replace("*", "")
    return text.strip()


def system_prompt(lang: str, mode: str) -> str:
    base = (
        "Sei ChatAI World, assistente AI potente per studio, coding, social media e traduzioni.\n"
        "Rispondi nella lingua dell'utente.\n"
        "NON usare markdown, niente **, niente elenchi pesanti.\n"
        "Risposte chiare, utili e dirette.\n"
    )

    if mode == "study":
        base += "Modalità STUDIO: spiega passo passo, come un tutor.\n"
    elif mode == "code":
        base += "Modalità CODING: scrivi codice completo HTML CSS JS.\n"
    elif mode == "content":
        base += "Modalità SOCIAL: idee e script per YouTube TikTok.\n"
    elif mode == "translate":
        base += "Modalità TRADUZIONE: traduci correttamente.\n"

    if lang:
        base += f"Lingua preferita dispositivo: {lang}\n"

    return base


def trim_memory(mem: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system = mem[0]
    rest = mem[1:]
    rest = rest[-(MAX_TURNS * 2):]
    return [system] + rest


def groq_chat(api_key: str, model: str, messages: List[Dict[str, str]]) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        if r.status_code != 200:
            print("[GROQ ERROR]", r.text)
            return None

        data = r.json()
        return clean_text(data["choices"][0]["message"]["content"])
    except Exception as e:
        print("[GROQ EXCEPTION]", e)
        return None


# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva (Groq only)"


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if session_id:
        CHAT_MEMORY.pop(session_id, None)
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    session_id = str(data.get("session_id", "")).strip() or "default"
    user_text = str(data.get("message", "")).strip()
    image_data = data.get("image_data")  # <-- QUI
    preferred_lang = str(data.get("preferred_lang", "")).strip()
    mode = str(data.get("mode", "general")).strip()

    # 📷 SOLO FOTO (nessun testo)
    if image_data and not user_text:
        return jsonify({
            "reply": "📷 Ho ricevuto la foto. Al momento non posso analizzarla visivamente. Descrivimi cosa vedi e ti aiuterò."
        })

    api_key = get_env(GROQ_API_KEY_ENV)
    if not api_key:
        return jsonify({"reply": "Errore: manca GROQ_API_KEY su Render."})

    model = get_env(MODEL_ENV) or DEFAULT_MODEL

    # inizializza memoria
    if session_id not in CHAT_MEMORY:
        CHAT_MEMORY[session_id] = [{
            STATS = {
    "messages": 0,
    "languages": {},
    "modes": {},
    "sessions": set()
        }
            "role": "system",
            "content": system_prompt(preferred_lang, mode)
        }]
STATS["messages"] += 1
STATS["sessions"].add(session_id)

lang = preferred_lang.split("-")[0] if preferred_lang else "unknown"
STATS["languages"][lang] = STATS["languages"].get(lang, 0) + 1
STATS["modes"][mode] = STATS["modes"].get(mode, 0) + 1
    memory = CHAT_MEMORY[session_id]
    memory[0]["content"] = system_prompt(preferred_lang, mode)

    memory.append({"role": "user", "content": user_text})
    memory = trim_memory(memory)

    reply = groq_chat(api_key, model, memory)
    if not reply:
        return jsonify({"reply": "Errore AI (controlla Render logs)."})

    memory.append({"role": "assistant", "content": reply})
    CHAT_MEMORY[session_id] = trim_memory(memory)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
@app.route("/admin", methods=["GET"])
def admin():
    key = request.args.get("key")
    if key != os.getenv("ADMIN_KEY", "secret"):
        return "Accesso negato", 403

    return jsonify({
        "messages": STATS["messages"],
        "sessions": len(STATS["sessions"]),
        "languages": STATS["languages"],
        "modes": STATS["modes"],
    })
