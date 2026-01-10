from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# 🔑 Chiavi API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

@app.route("/")
def home():
    return "🌍 ChatAI World API attiva!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    # --- Step 1: Chiamata Groq ---
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "mixtral-8x7b",
        "messages": [
            {"role": "system", "content": "Sei un assistente intelligente che conosce persone, eventi, cultura e notizie globali."},
            {"role": "user", "content": user_message}
        ]
    }

    groq_response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload
    ).json()

    reply = groq_response["choices"][0]["message"]["content"]

    # --- Step 2: (Opzionale) arricchimento Hugging Face ---
    if "chi è" in user_message.lower() or "cerca" in user_message.lower():
        hf_headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        hf_resp = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            headers=hf_headers, json={"inputs": user_message}
        ).json()
        extra = hf_resp[0]["generated_text"] if isinstance(hf_resp, list) else ""
        reply += f"\n\n🔎 Info aggiuntiva: {extra}"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
