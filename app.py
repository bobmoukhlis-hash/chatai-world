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

# ===== Config =====
GROQ_API_KEY_ENV = "GROQ_API_KEY"
MODEL_ENV = "MODEL"
VISION_MODEL_ENV = "VISION_MODEL"  # opzionale per foto

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 25.0
MAX_MEMORY_MESSAGES = 50  # memoria (coppie user/assistant)

# Memoria in RAM per session_id
CHAT_MEMORY: Dict[str, List[Dict[str, Any]]] = {}


def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else None


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


@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva con Groq + mic/voce/foto!"


def _build_system_prompt() -> str:
    return (
        "Sei ChatAI World, un assistente AI multilingua.\n"
        "- Rileva automaticamente la lingua dell'utente.\n"
        "- Rispondi SEMPRE nella stessa lingua dell'utente.\n"
        "- Supporta italiano, inglese, arabo e tutte le altre lingue del mondo.\n"
        "- Usa testo semplice, naturale, senza Markdown.\n"
        "- Se l'utente scrive in arabo, rispondi in arabo.\n"
        "- Se scrive in inglese, rispondi in inglese.\n"
        "- Se scrive in italiano, rispondi in italiano.\n"
        "- Mantieni risposte chiare, educate e utili."
    )


def _get_or_init_memory(session_id: str) -> List[Dict[str, Any]]:
    if session_id not in CHAT_MEMORY:
        CHAT_MEMORY[session_id] = [{"role": "system", "content": _build_system_prompt()}]
    return CHAT_MEMORY[session_id]


def _trim_memory(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # preserva system + ultimi 2*MAX_MEMORY_MESSAGES messaggi
    system_msg = messages[0]
    tail = messages[1:]
    tail = tail[-(MAX_MEMORY_MESSAGES * 2) :]
    return [system_msg] + tail


def _groq_chat(
    groq_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_seconds: float,
) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.7}

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
        print(f"[GROQ] JSON parse error: {e} body={resp.text[:500]}")
        return None


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "")).strip()
    if session_id and session_id in CHAT_MEMORY:
        CHAT_MEMORY.pop(session_id, None)
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "")).strip() or "default"
    user_text = str(data.get("message", "")).strip()
    image_data = data.get("image_data")  # DataURL (string) o None

    groq_key = _get_env(GROQ_API_KEY_ENV)
    if not groq_key:
        return jsonify({"reply": "❌ Manca GROQ_API_KEY su Render."})

    model = (_get_env(MODEL_ENV) or DEFAULT_MODEL)
    vision_model = _get_env(VISION_MODEL_ENV)  # opzionale
    timeout_seconds = float(os.getenv("GROQ_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))

    memory = _get_or_init_memory(session_id)
    memory = _trim_memory(memory)

    # Costruisci messaggio user: testo + (eventuale) immagine
    if image_data:
        if not vision_model:
            return jsonify(
                {
                    "reply": "📷 Foto ricevuta, ma la visione non è attiva. Imposta VISION_MODEL su Render per analizzare immagini.",
                }
            )
        # Formato “vision” stile OpenAI: content come array (text + image_url)
        user_content: List[Dict[str, Any]] = []
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        user_content.append({"type": "image_url", "image_url": {"url": image_data}})
        memory.append({"role": "user", "content": user_content})
        use_model = vision_model
    else:
        if not user_text:
            return jsonify({"reply": "⚠️ Scrivi un messaggio o carica una foto."})
        memory.append({"role": "user", "content": user_text})
        use_model = model

    memory = _trim_memory(memory)
    print(f"[ENV] session={session_id} model={use_model} msgs={len(memory)} img={'YES' if image_data else 'NO'}")

    reply = _groq_chat(groq_key, use_model, memory, timeout_seconds)
    if not reply:
        return jsonify({"reply": "❌ Nessuna AI disponibile (controlla Logs Render per 401/429/timeout)."} )

    memory.append({"role": "assistant", "content": reply})
    CHAT_MEMORY[session_id] = _trim_memory(memory)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
