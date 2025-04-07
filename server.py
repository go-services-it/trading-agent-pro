import os
import requests
import pandas as pd
import numpy as np
import time
import hmac
import hashlib
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# URL base API Bitget
BITGET_BASE_URL = 'https://api.bitget.com/api/v2/mix/market'

# API Keys da Environment Variables
API_KEY = os.getenv("BITGET_API_KEY")
API_SECRET = os.getenv("BITGET_API_SECRET")
API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE")

def sign_request(timestamp, method, request_path, body, secret):
    message = str(timestamp) + method + request_path + (body or "")
    mac = hmac.new(secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def get_candles(symbol='BTCUSDT', granularity='1H', limit=220):
    url_path = f"/api/v2/mix/market/candles"
    full_url = f"{BITGET_BASE_URL}/candles"
    
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": limit,
        "productType": "usdt-futures"
    }

    timestamp = str(int(time.time() * 1000))
    method = "GET"
    body = ""

    query_string = f"symbol={symbol}&granularity={granularity}&limit={limit}"
    request_path_with_params = f"{url_path}?{query_string}"

    signature = sign_request(timestamp, method, request_path_with_params, body, API_SECRET)

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": API_PASSPHRASE,
        "Content-Type": "application/json"
    }

    print(f"[DEBUG] Requesting URL: {full_url}")
    print(f"[DEBUG] Requesting Params: {params}")

    try:
        response = requests.get(full_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        parsed_response = response.json()

        if parsed_response.get('code') == '00000':
            data = parsed_response.get('data')
            if not data:
                print(f"[WARN] Nessun dato candela restituito per {symbol}/{granularity}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            df.sort_index(inplace=True)
            print(f"[DEBUG] DataFrame creato con {len(df)} righe.")
            return df

        else:
            print(f"[ERROR] Bitget API error: {parsed_response.get('msg')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Errore richiesta HTTP verso Bitget: {str(e)}")
        if e.response is not None:
            print(f"[ERROR] Response status: {e.response.status_code}")
            print(f"[ERROR] Response text: {e.response.text}")
        return None

def calculate_indicators(df):
    if df is None or df.empty:
        return None

    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'].fillna(method='bfill', inplace=True)

    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift())
    low_close_prev = abs(df['low'] - df['close'].shift())
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14).mean()
    df['atr'].fillna(method='bfill', inplace=True)

    print("[DEBUG] Indicatori calcolati.")
    return df

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
@cross_origin()
def analyze():
    if request.method == 'OPTIONS':
        print("[DEBUG] Ricevuta richiesta OPTIONS")
        return '', 200

    print("[DEBUG] Ricevuta richiesta POST /analyze")
    try:
        data = request.get_json(force=True)
        if not data:
            raise ValueError("Payload JSON vuoto ricevuto")
        print(f"[DEBUG] Dati JSON ricevuti: {data}")
    except Exception as e:
        print(f"[ERROR] Errore nel parsing del JSON: {str(e)}")
        return jsonify({"error": "Payload JSON non valido o mancante"}), 400

    symbol = data.get('symbol', 'BTCUSDT')
    granularity = data.get('granularity', '1H')

    df = get_candles(symbol, granularity)
    if df is None or len(df) < 200:
        return jsonify({"error": "Dati insufficienti o errore nel recupero dati da Bitget."}), 400

    df = calculate_indicators(df)
    if df is None or df.empty:
        return jsonify({"error": "Errore durante il calcolo degli indicatori."}), 500

    last = df.iloc[-1]

    trend = "Uptrend" if last['sma50'] > last['sma200'] else "Downtrend" if last['sma50'] < last['sma200'] else "Neutral"
    rsi = round(last['rsi'], 2)
    atr = round(last['atr'], 2) if not pd.isna(last['atr']) else 0
    entry = round(last['close'], 2)

    if atr > 0:
        stop_loss = round(entry - (atr * 2), 2)
        take_profit_1 = round(entry + (atr * 2), 2)
        take_profit_2 = round(entry + (atr * 3), 2)
        take_profit_3 = round(entry + (atr * 4), 2)
        risk_reward = round((take_profit_1 - entry) / (entry - stop_loss), 2) if (entry - stop_loss) != 0 else 0
    else:
        stop_loss = entry
        take_profit_1 = entry
        take_profit_2 = entry
        take_profit_3 = entry
        risk_reward = 0

    relative_atr = (atr / entry) * 100 if entry > 0 else 0

    score = 0
    score += 25 if trend == "Uptrend" else 0
    score += 25 if 30 < rsi < 70 else 10
    score += 25 if risk_reward >= 1.5 else 10
    score += 20 if relative_atr < 3 else 5
    score = min(score, 100)

    result = {
        "symbol": symbol,
        "granularity": granularity,
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
        "note": f"Trend: {trend}, RSI: {rsi:.2f}, R/R(TP1): {risk_reward:.2f}, ATR: {atr:.2f} ({relative_atr:.2f}%)"
    }

    print(f"[DEBUG] Analisi completata. Invio risposta: {result}")
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"[DEBUG] Avvio server Flask su http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
