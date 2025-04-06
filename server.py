from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BITGET_BASE_URL = 'https://api.bitget.com/api/v2/market'

def get_candles(symbol, timeframe='1H', limit=150):
    url = f"{BITGET_BASE_URL}/candles"
    params = {
        "symbol": symbol,
        "period": timeframe.lower(),
        "limit": limit
    }
    print(f"[DEBUG] Requesting: {url} with params: {params}")
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()['data']
        print(f"[DEBUG] Response data: {data[:2]} ...")  # Mostra solo i primi 2
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        return df
    else:
        print(f"[ERROR] Failed fetching candles: {response.status_code}")
        return None

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        print("[DEBUG] OPTIONS request received")
        return jsonify({"message": "CORS OK"}), 200

    try:
        data = request.get_json(force=True)
        print(f"[DEBUG] Received data: {data}")
    except Exception as e:
        print(f"[ERROR] Error parsing JSON: {str(e)}")
        return jsonify({"error": "Invalid JSON received"}), 400

    if not data:
        print("[ERROR] No data received")
        return jsonify({"error": "No data received"}), 400

    symbol = data.get('symbol', 'BTCUSDT')
    timeframe = data.get('timeframe', '1h')

    df = get_candles(symbol, timeframe)
    if df is None or len(df) < 50:
        print("[ERROR] Unable to fetch or insufficient data")
        return jsonify({"error": "Unable to fetch or insufficient data"}), 400

    df = calculate_indicators(df)
    last = df.iloc[-1]
    
    trend = "Uptrend" if last['sma50'] > last['sma200'] else "Downtrend" if last['sma50'] < last['sma200'] else "Neutral"
    rsi = round(last['rsi'], 2)
    atr = round(last['atr'], 2)
    
    entry = round(last['close'], 2)
    stop_loss = round(entry - (atr * 2), 2)
    take_profit_1 = round(entry + (atr * 2), 2)
    take_profit_2 = round(entry + (atr * 3), 2)
    take_profit_3 = round(entry + (atr * 4), 2)
    
    risk_reward = round((take_profit_1 - entry) / (entry - stop_loss), 2) if (entry - stop_loss) != 0 else 0
    
    score = 0
    score += 25 if trend == "Uptrend" else 0
    score += 25 if 40 <= rsi <= 60 else 15
    score += 25 if risk_reward >= 2 else 15
    score += 15 if atr < entry * 0.03 else 5
    score += 10
    
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "entry_price": entry,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "take_profit_3": take_profit_3,
        "trend": trend,
        "rsi": rsi,
        "atr": atr,
        "risk_reward_ratio": risk_reward,
        "trade_score": score,
        "note": "Trend {} - RSI {:.2f} - R/R {:.2f}".format(trend, rsi, risk_reward)
    }
    print(f"[DEBUG] Sending response: {result}")
    return jsonify(result)

def calculate_indicators(df):
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift()))
    df['atr'] = df['tr'].rolling(window=14).mean()
    return df

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"[DEBUG] Server starting on port {port}")
    app.run(host='0.0.0.0', port=port)
