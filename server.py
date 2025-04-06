
from flask import Flask, request, jsonify
import random

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    coin = data.get('coin')
    timeframe = data.get('timeframe', '1h')
    
    # Simulazione risposta
    response = {
        "coin": coin,
        "entry_price": round(random.uniform(100, 1000), 2),
        "stop_loss": round(random.uniform(90, 99), 2),
        "take_profit_1": round(random.uniform(110, 120), 2),
        "take_profit_2": round(random.uniform(120, 130), 2),
        "take_profit_3": round(random.uniform(130, 140), 2),
        "trend": random.choice(["Uptrend", "Downtrend", "Neutral"]),
        "rsi": round(random.uniform(30, 70), 2),
        "trade_score": random.randint(60, 95),
        "note": "Simulated analysis."
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
