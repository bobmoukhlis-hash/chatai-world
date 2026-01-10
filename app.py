from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ permette al sito GitHub di comunicare con il backend

app = Flask(__name__)
CORS(app)  # ✅ abilita tutte le origini (incluso GitHub Pages)

@app.route('/')
def home():
    return "🌍 ChatAI World API attiva!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower()

    if not user_message:
        return jsonify({"reply": "⚠️ Messaggio vuoto."})

    # 🔹 Logica semplice di test
    if "ciao" in user_message:
        reply = "👋 Ciao! Come posso aiutarti oggi?"
    elif "mondo" in user_message:
        reply = "🌍 Il mondo è un posto affascinante, in continua evoluzione!"
    else:
        reply = f"🤖 Hai detto: {user_message}"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
