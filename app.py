from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "🌍 ChatAI World API attiva con AI!"

@app.route('/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"reply": "⚠️ Messaggio vuoto."})

    reply = f"🌍 Hai detto: {user_message}"
    return jsonify({"reply": reply})

    # 🔑 Leggi le chiavi dalle variabili d'ambiente su Render
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # ✅ Prova prima Hugging Face
    if hf_key:
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/google/gemma-7b-it",
                headers={"Authorization": f"Bearer {hf_key}"},
                json={"inputs": user_message}
            )
            if response.status_code == 200:
                output = response.json()
                if isinstance(output, list) and len(output) > 0:
                    reply = output[0].get("generated_text", "🤖 Nessuna risposta generata.")
                else:
                    reply = "⚙️ Nessuna risposta valida dal modello."
                return jsonify({"reply": reply})
        except Exception as e:
            print("Errore Hugging Face:", e)

    # 🔄 In caso di fallback su Groq
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [{"role": "user", "content": user_message}]
                }
            )
            if response.status_code == 200:
                data = response.json()
                reply = data["choices"][0]["message"]["content"]
                return jsonify({"reply": reply})
        except Exception as e:
            print("Errore Groq:", e)

    return jsonify({"reply": "❌ Nessuna AI disponibile al momento."})


if if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
