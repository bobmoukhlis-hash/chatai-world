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

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 25.0
MAX_MEMORY_MESSAGES = 50  # keeps last 50 user+assistant turns (approx via cap below)

# In-memory per session_id
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


def _lang_hint(preferred_lang: str) -> str:
    """
    preferred_lang examples: 'it-IT', 'en-US', 'ar-SA', 'fr-FR'
    We instruct: respond in user's language; if unsure, use preferred_lang.
    """
    pl = (preferred_lang or "").strip()
    if not pl:
        return "Rispondi nella lingua dell'utente."
    return (
        f"La lingua preferita del dispositivo è '{pl}'. "
        "Rispondi nella lingua dell'utente; se è ambiguo, usa la lingua preferita."
    )


def _mode_rules(mode: str) -> str:
    m = (mode or "general").strip().lower()
    if m == "study":
        return (
            "MODALITÀ STUDIO:\n"
            "- Spiega come un tutor: passi chiari, esempi brevi.\n"
            "- Se serve: fai domande per capire il livello.\n"
            "- Dai mini-esercizi + soluzione.\n"
            "- Per università: includi definizioni, formule (testo), dimostrazioni semplici.\n"
        )
    if m == "code":
        return (
            "MODALITÀ CODING:\n"
            "- Scrivi codice completo e funzionante (HTML/CSS/JS), niente pezzi mancanti.\n"
            "- Spiega poco ma chiaro: cosa fa, come eseguirlo.\n"
            "- Se debug: indica errore → causa → fix.\n"
            "- Segui best practices e sicurezza base.\n"
        )
    if m == "content":
        return (
            "MODALITÀ SOCIAL (YouTube/TikTok/Reels):\n"
            "- Proponi idee forti (hook nei primi 2 secondi).\n"
            "- Dai uno script pronto (intro, corpo, CTA), durata 15s/30s/60s.\n"
            "- Suggerisci titolo, caption, hashtag, struttura video.\n"
        )
    if m == "translate":
        return (
            "MODALITÀ TRADUZIONE:\n"
            "- Traduci fedelmente ma naturale.\n"
            "- Se richiesto: versioni formale/informale.\n"
            "- Correggi grammatica e proponi alternative.\n"
        )
    return (
        "MODALITÀ GENERALE:\n"
        "- Risposte utili, chiare, brevi.\n"
        "- Se l'utente chiede studio/coding/social/traduzione, adattati automaticamente.\n"
    )


def _build_system_prompt(preferred_lang: str, mode: str) -> str:
    return (
        "Sei ChatAI World: assistente AI multilingua super utile per studiare, programmare e creare contenuti.\n"
        f"{_lang_hint(preferred_lang)}\n"
        "REGOLE:\n"
        "- Rispondi SENZA Markdown (niente **, niente elenchi lunghissimi). Testo semplice.\n"
        "- Se l'utente chiede HTML/CSS/JS: fornisci codice completo pronto all'uso.\n"
        "- Se l'utente studia: spiega passo-passo e proponi esercizi.\n"
        "- Se l'utente crea contenuti: dai idee e script pronti per social.\n"
        "- Se l'utente chiede traduzioni: traduci e correggi.\n"
        "- Se una richiesta è pericolosa/illegale, rifiuta e proponi alternativa sicura.\n\n"
        f"{_mode_rules(mode)}"
    )


def _get_or_init_memory(session_id: str, system_prompt: str) -> List[Dict[str, Any]]:
    mem = CHAT_MEMORY.get(session_id)
    if not mem:
        mem = [{"role": "system", "content": system_prompt}]
        CHAT_MEMORY[session_id] = mem
    else:
        # keep system message updated to current mode/lang
        mem[0] = {"role": "system", "content": system_prompt}
    return mem


def _trim_memory(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # messages = [system] + history
    system_msg = messages[0]
    tail = messages[1:]
    # keep last ~100 messages (50 turns ≈ user+assistant => 100 msgs)
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


@app.route("/", methods=["GET"])
def home():
    return "🌍 ChatAI World API attiva (Groq + multi-lingua + modalità + memoria)!"


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
    preferred_lang = str(data.get("preferred_lang", "")).strip()  # from phone/browser
    mode = str(data.get("mode", "general")).strip()

    if not user_text:
        return jsonify({"reply": "⚠️ Scrivi un messaggio."})

    groq_key = _get_env(GROQ_API_KEY_ENV)
    if not groq_key:
        return jsonify({"reply": "❌ Manca GROQ_API_KEY su Render."})

    model = _get_env(MODEL_ENV) or DEFAULT_MODEL
    timeout_seconds = float(os.getenv("GROQ_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))

    system_prompt = _build_system_prompt(preferred_lang, mode)
    memory = _get_or_init_memory(session_id, system_prompt)
    memory.append({"role": "user", "content": user_text})
    memory = _trim_memory(memory)

    print(f"[CHAT] session={session_id} lang={preferred_lang or '-'} mode={mode} model={model} msgs={len(memory)}")

    reply = _groq_chat(groq_key, model, memory, timeout_seconds)
    if not reply:
        return jsonify({"reply": "❌ Nessuna AI disponibile (controlla Logs Render per 401/429/timeout)."} )

    memory.append({"role": "assistant", "content": reply})
    CHAT_MEMORY[session_id] = _trim_memory(memory)

    return jsonify({"reply": reply})
    

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
