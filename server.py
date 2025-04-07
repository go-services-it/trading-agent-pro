import os
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
# Configurazione CORS permissiva, considera restrizioni maggiori in produzione
CORS(app, resources={r"/*": {"origins": "*"}})

# URL base per l'API v2 Market di Bitget
# BITGET_BASE_URL = 'https://api.bitget.com/api/v2/market' Mercato Spot
BITGET_BASE_URL = 'https://api.bitget.com/api/v2/mix/market' # Mercato Futures
#, productType='USDT-FUTURES'

def get_candles(symbol='BTCUSDT', granularity='1H', limit=220): # Aumentato limite per SMA200
    """
    Recupera i dati delle candele dall'API Bitget v2.
    """
    url = f"{BITGET_BASE_URL}/candles"

    # <<< VERIFICA QUI >>> Controlla la documentazione API Bitget v2 per i valori esatti di 'granularity'
    # Esempi comuni: '1min', '5min', '1H', '4H', '1D', '1W'
    # Assicurati che il valore 'granularity' passato a questa funzione sia uno di quelli accettati.
    # Rimosso .lower() assumendo che l'API voglia formati come '1H'. Modifica se necessario.
    params = {
        "symbol": symbol,
        "granularity": granularity, # Il parametro si chiama 'granularity' nella v2
        "limit": limit,
        "productType" : productType
    }
    print(f"[DEBUG] Requesting URL: {url}")
    print(f"[DEBUG] Requesting Params: {params}")

    try:
        response = requests.get(url, params=params, timeout=10) # Aggiunto timeout
        response.raise_for_status() # Solleva eccezioni per errori HTTP (4xx, 5xx)

        parsed_response = response.json()
        # print(f"[DEBUG] Raw API Response: {parsed_response}") # Decommenta per vedere tutta la risposta grezza

        # <<< VERIFICA QUI >>> Controlla il codice di successo specifico di Bitget v2 (di solito '00000')
        bitget_success_code = '00000'

        if parsed_response.get('code') == bitget_success_code:
            data = parsed_response.get('data') # Usa .get() per sicurezza

            if data is None or not isinstance(data, list):
                print(f"[ERROR] Chiave 'data' mancante o non è una lista nella risposta API (anche se codice successo)")
                return None

            if not data:
                print(f"[WARN] Nessun dato candela restituito dall'API per {symbol} / {granularity}")
                # Restituisce un DataFrame vuoto per coerenza
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Stampa il primo elemento per verifica struttura
            print(f"[DEBUG] Primo dato candela ricevuto: {data[0]}")
            # <<< VERIFICA QUI >>> Assicurati che l'ordine dei dati in data[0] corrisponda alle colonne:
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            try:
                df = pd.DataFrame(data, columns=expected_columns)
                # Converti timestamp (assumendo sia millisecondi) in Datetime e imposta come indice
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                # Converti le altre colonne in numerico
                for col in ['open', 'high', 'low', 'close', 'volume']:
                     # errors='coerce' trasformerà valori non numerici in NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # Rimuovi righe con eventuali NaN dovuti a conversione fallita
                df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
                # Ordina per data (l'API dovrebbe già farlo, ma è una sicurezza)
                df.sort_index(inplace=True)
                print(f"[DEBUG] DataFrame creato con {len(df)} righe.")
                return df
            except Exception as e:
                 print(f"[ERROR] Errore durante la creazione o conversione del DataFrame: {str(e)}")
                 return None

        else:
            # Errore logico restituito dall'API stessa (es. simbolo non valido)
            error_msg = parsed_response.get('msg', 'Errore API sconosciuto')
            print(f"[ERROR] L'API Bitget ha restituito un errore: Code={parsed_response.get('code')}, Msg='{error_msg}'")
            return None

    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout durante la richiesta all'API Bitget: {url}")
        return None
    except requests.exceptions.RequestException as e:
        # Gestisce errori di connessione, HTTP status codes non 2xx (grazie a raise_for_status), etc.
        print(f"[ERROR] Errore richiesta HTTP verso Bitget: {str(e)}")
        if e.response is not None:
             print(f"[ERROR] Response status: {e.response.status_code}")
             print(f"[ERROR] Response text: {e.response.text}")
        return None
    except ValueError: # Errore nel decodificare JSON
        print(f"[ERROR] Impossibile decodificare la risposta JSON dall'API.")
        # Stampa l'inizio della risposta per capire cosa è arrivato
        if response is not None:
             print(f"[ERROR] Testo risposta ricevuto: {response.text[:200]}...")
        return None
    except Exception as e: # Altri errori imprevisti
        print(f"[ERROR] Errore imprevisto in get_candles: {str(e)}")
        return None

def calculate_indicators(df):
    """
    Calcola gli indicatori tecnici sul DataFrame.
    """
    if df is None or df.empty:
        return None # O restituisci df vuoto se preferisci

    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Usa Exponential Moving Average (EMA) per RSI per una risposta più rapida (comune)
    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()

    # Gestione divisione per zero per RS
    rs = avg_gain / avg_loss.replace(0, np.nan) # Sostituisci 0 con NaN per evitare divisione
    df['rsi'] = 100 - (100 / (1 + rs))
    # Riempi eventuali NaN iniziali nell'RSI
    df['rsi'].fillna(method='bfill', inplace=True) # Backfill per riempire i NaN iniziali

    # Calcolo ATR
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift())
    low_close_prev = abs(df['low'] - df['close'].shift())
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14).mean() # Usa EMA per ATR (comune)
    # Riempi eventuali NaN iniziali nell'ATR
    df['atr'].fillna(method='bfill', inplace=True)

    print("[DEBUG] Indicatori calcolati.")
    return df

@app.route('/')
def home():
    """Serve la pagina HTML principale."""
    # Assicurati che 'index.html' sia nella stessa directory di server.py o specifica il path corretto
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """
    Endpoint per ricevere simbolo/granularity, ottenere dati, calcolare indicatori e restituire analisi.
    """
    if request.method == 'OPTIONS':
        # Gestione richiesta pre-flight CORS
        print("[DEBUG] Ricevuta richiesta OPTIONS")
        # Risposta standard per pre-flight, i browser la gestiscono
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        return jsonify({"message": "CORS OK"}), 200, headers

    print("[DEBUG] Ricevuta richiesta POST /analyze")
    try:
        # Forza l'interpretazione come JSON anche se Content-Type non è perfetto
        data = request.get_json(force=True)
        if not data:
             raise ValueError("Payload JSON vuoto ricevuto")
        print(f"[DEBUG] Dati JSON ricevuti: {data}")
    except Exception as e:
        print(f"[ERROR] Errore nel parsing del JSON in input: {str(e)}")
        return jsonify({"error": "Payload JSON non valido o mancante"}), 400

    # Estrai simbolo e granularity dal payload JSON, con valori di default
    symbol = data.get('symbol', 'BTCUSDT') # <<< VERIFICA QUI: Assicurati che il formato sia corretto per l'API (es. BTCUSDT_SPBL?)
    granularity = data.get('granularity', '1H') # <<< VERIFICA QUI: Assicurati che il formato sia corretto (es. '1H', '4h'?)

    # Ottieni i dati delle candele
    df = get_candles(symbol, granularity)

    # Controlla se abbiamo dati sufficienti DOPO averli ottenuti
    # Necessari almeno 200 periodi per SMA200 e ~14 per RSI/ATR senza NaN iniziali
    required_length = 200
    if df is None or len(df) < required_length:
        print(f"[ERROR] Impossibile ottenere dati sufficienti ({len(df) if df is not None else 0} righe) da Bitget per {symbol}/{granularity}. Richiesti almeno {required_length}.")
        error_msg = "Dati insufficienti o errore nel recupero dati da Bitget."
        if df is not None and len(df) > 0: # Se abbiamo alcuni dati, forse l'API ne ha restituiti pochi
             error_msg = f"Dati insufficienti ({len(df)} righe) per l'analisi. Controlla simbolo/granularity o prova più tardi."

        return jsonify({"error": error_msg}), 400 # Usiamo 400 Bad Request o 503 Service Unavailable? 400 è ok se input forse errato.

    # Calcola gli indicatori
    df = calculate_indicators(df)

    # Controlla se gli indicatori sono stati calcolati (potrebbero essere None se df era vuoto)
    if df is None:
         print("[ERROR] Calcolo indicatori fallito.")
         return jsonify({"error": "Errore durante il calcolo degli indicatori."}), 500

    # Estrai l'ultimo dato disponibile (più recente)
    # Assicurati che ci siano dati dopo il dropna e i calcoli
    if df.empty:
         print("[ERROR] DataFrame vuoto dopo calcolo indicatori.")
         return jsonify({"error": "Nessun dato valido disponibile dopo il calcolo degli indicatori."}), 500

    last = df.iloc[-1]

    # Controlla se i valori chiave sono NaN (potrebbe succedere se ci sono pochi dati nonostante il check iniziale)
    if pd.isna(last['sma50']) or pd.isna(last['sma200']) or pd.isna(last['rsi']) or pd.isna(last['atr']):
        print("[WARN] Valori NaN trovati negli indicatori dell'ultima candela. Dati insufficienti?")
        # Potresti restituire un errore o un risultato parziale
        return jsonify({"error": "Dati insufficienti per calcolare tutti gli indicatori sull'ultima candela."}), 400

    # Logica di analisi (come nel tuo codice originale)
    trend = "Uptrend" if last['sma50'] > last['sma200'] else "Downtrend" if last['sma50'] < last['sma200'] else "Neutral"
    rsi = round(last['rsi'], 2)
    atr = round(last['atr'], 2) if not pd.isna(last['atr']) else 0 # Gestisci ATR nullo

    entry = round(last['close'], 2)

    # Calcoli Stop Loss e Take Profit basati su ATR
    # Assicurati che ATR non sia zero o nullo per evitare divisioni per zero o SL/TP uguali all'entry
    if atr > 0:
        stop_loss = round(entry - (atr * 2), 2)
        take_profit_1 = round(entry + (atr * 2), 2)
        take_profit_2 = round(entry + (atr * 3), 2)
        take_profit_3 = round(entry + (atr * 4), 2)
        risk_reward = round((take_profit_1 - entry) / (entry - stop_loss), 2) if (entry - stop_loss) != 0 else 0
    else:
        # Se ATR è 0 o NaN, imposta valori di default o segnala impossibilità di calcolo
        stop_loss = entry # O None
        take_profit_1 = entry # O None
        take_profit_2 = entry # O None
        take_profit_3 = entry # O None
        risk_reward = 0
        print("[WARN] ATR è zero o nullo, impossibile calcolare SL/TP basati su ATR.")


    # Calcolo Score (come nel tuo codice originale)
    score = 0
    score += 25 if trend == "Uptrend" else 0 # Penalizza downtrend/neutral
    score += 25 if 30 < rsi < 70 else 10 # Zona RSI "normale"
    # Considera un punteggio diverso per ipervenduto/ipercomprato se rilevante per la strategia
    # score += 15 if rsi <= 30 else 0 # Esempio per ipervenduto
    # score += 15 if rsi >= 70 else 0 # Esempio per ipercomprato
    score += 25 if risk_reward >= 1.5 else 10 # R/R minimo 1.5
    # Normalizza ATR rispetto al prezzo per valutare volatilità relativa
    relative_atr = (atr / entry) * 100 if entry > 0 else 0
    score += 20 if relative_atr < 3 else 5 # Meno del 3% di volatilità (ATR/prezzo)
    score = min(score, 100) # Assicura che il punteggio non superi 100

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
    # Questa parte viene eseguita solo quando avvii lo script direttamente (es. python server.py)
    # Non viene usata da Gunicorn su Render, ma è utile per test locali.
    # Render imposta la variabile PORT, Gunicorn la usa (o usa quella specificata nel Procfile).
    port = int(os.environ.get('PORT', 5001)) # Cambiato porta default per evitare conflitti comuni
    print(f"[DEBUG] Avvio server Flask di sviluppo su http://0.0.0.0:{port}")
    # debug=True ricarica automaticamente il server alle modifiche, utile in sviluppo locale
    # Non usare debug=True in produzione! Gunicorn gestisce i worker.
    app.run(host='0.0.0.0', port=port, debug=True)