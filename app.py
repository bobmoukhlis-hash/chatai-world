from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 🔥 permette a GitHub Pages di inviare richieste al backend

@app.route('/')
def home():
    return "🌍 ChatAI World API attiva!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower()

    if not user_message:
        return jsonify({"reply": "⚠️ Messaggio vuoto."})

    if "ciao" in user_message:
        reply = "👋 Ciao! Come posso aiutarti oggi?"
    elif "mondo" in user_message:
        reply = "🌍 Il mondo sarà pieno di cambiamenti, innovazioni e nuove scoperte!"
    else:
        reply = f"🤖 Hai detto: {user_message}"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
