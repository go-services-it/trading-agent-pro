import os
import requests
import pandas as pd
import numpy as np
import time
import hmac
import hashlib
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# URL base API Bitget
BITGET_BASE_URL = 'https://api.bitget.com/api/v2/mix/market'

# Carica API Keys (puoi anche usare variabili d'ambiente)
API_KEY = os.getenv("BITGET_API_KEY", "bg_34f91d51ffde824b8ea384319a807ac4")
API_SECRET = os.getenv("BITGET_API_SECRET", "0f1ed2d25d7bfac1ea8e9883f12bc4160e483deba93021e353e1f27c12fd70a1")
API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE", "BiTD4sh8o4rd")

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
        "limit": limit
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

            print(f"[DEBUG] Primo dato candela ricevuto: {data[0]}")

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

# Tutto il resto del file (calculate_indicators, Flask endpoints) resta invariato.

# ... [segue il resto del tuo server.py originale]

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"[DEBUG] Avvio server Flask di sviluppo su http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
