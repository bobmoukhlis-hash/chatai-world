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
GROQ_API_KEY_ENV = "GROQ_API_KEY"
MODEL_ENV = "MODEL"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 25.0
MAX_MEMORY_MESSAGES = 50  # 50 turni (user+assistant)

# Memoria in RAM (per session_id)
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
    s = re.sub(r"\*(.*?)\*", r"\1", s)
    s = re.sub(r"_(.*?)_", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _lang_hint(preferred_lang: str) -> str:
    if not preferred_lang:
        return "Rispondi nella lingua dell'utente."
    return (
        f"La lingua preferita del dispositivo è '{preferred_lang}'. "
        "Rispondi nella lingua dell'utente; se è ambiguo, usa quella lingua."
    )


def _mode_rules(mode: str) -> str:
    m = (mode or "general").lower()
    if m == "study":
        return (
            "MODALITÀ STUDIO:\n"
            "- Spiega passo-passo.\n"
            "- Usa esempi semplici.\n"
            "- Proponi esercizi con soluzione.\n"
        )
    if m == "code":
        return (
            "MODALITÀ CODING:\n"
            "- Fornisci codice completo e funzionante.\n"
            "- Spiega brevemente come usarlo.\n"
        )
    if m == "content":
        return (
            "MODALITÀ SOCIAL:\n"
            "- Idee forti.\n"
            "- Script pronti (intro, corpo, CTA).\n"
        )
    if m == "translate":
        return (
            "MODALITÀ TRADUZIONE:\n"
            "- Traduzione naturale e corretta.\n"
        )
    return "MODALITÀ GENERALE:\n- Risposte chiare e utili.\n"


def _build_system_prompt(preferred_lang: str, mode: str) -> str:
    return (
        "Sei ChatAI World, un assistente AI utile per studio, coding e contenuti.\n"
        f"{_lang_hint(preferred_lang)}\n"
        "REGOLE:\n"
        "- Niente markdown.\n"
        "- Testo semplice.\n"
        "- Aiuta sempre l'utente.\n\n"
        f"{_mode_rules(mode)}"
    )


def _get_or_init_memory(session_id: str, system_prompt: str) -> List[Dict[str, Any]]:
    mem = CHAT_MEMORY.get(session_id)
    if not mem:
        mem = [{"role": "system", "content": system_prompt}]
        CHAT_MEMORY[session_id] = mem
    else:
        mem[0] = {"role": "system", "content": system_prompt}
    return mem


def _trim_memory(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_msg = messages[0]
    tail = messages[1:]
    tail = tail[-(MAX_MEMORY_MESSAGES * 2):]
    return [system_msg] + tail


def _groq_chat(
    groq_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_seconds: float,
) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)
    if resp.status_code != 200:
        print("[GROQ ERROR]", resp.status_code, resp.text[:300])
        return None

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return _sanitize_reply(content)


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva (Groq)"


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "")).strip()
    if session_id:
        CHAT_MEMORY.pop(session_id, None)
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    session_id = str(data.get("session_id", "")).strip() or "default"
    user_text = str(data.get("message", "")).strip()
    image_data = data.get("image_data")
    preferred_lang = str(data.get("preferred_lang", "")).strip()
    mode = str(data.get("mode", "general")).strip()

    # 📷 SOLO FOTO
    if image_data and not user_text:
        return jsonify({
            "reply": "📷 Ho ricevuto la foto. Non posso analizzarla visivamente. Descrivimi cosa vedi."
        })

    if not user_text:
        return jsonify({"reply": "⚠️ Scrivi un messaggio."})

    groq_key = _get_env(GROQ_API_KEY_ENV)
    if not groq_key:
        return jsonify({"reply": "❌ Manca GROQ_API_KEY su Render."})

    model = _get_env(MODEL_ENV) or DEFAULT_MODEL
    timeout_seconds = float(os.getenv("GROQ_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))

    system_prompt = _build_system_prompt(preferred_lang, mode)
    memory = _get_or_init_memory(session_id, system_prompt)

    memory.append({"role": "user", "content": user_text})
    memory = _trim_memory(memory)

    reply = _groq_chat(groq_key, model, memory, timeout_seconds)
    if not reply:
        return jsonify({"reply": "❌ Errore AI (controlla Logs Render)."})

    memory.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
