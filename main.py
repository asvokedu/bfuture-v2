import os
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from utils import calculate_technical_indicators, generate_label
from collect_data import get_binance_klines, get_all_usdt_symbols
import joblib
from collections import defaultdict
import subprocess

MODEL_PATH = 'models'
INTERVAL = '1h'
SYMBOLS = get_all_usdt_symbols(use_futures=True)
DATASET_PATH = 'training_data/training_dataset.csv'
PREDICTION_LOG = 'predicted_signals.csv'

prediction_results = defaultdict(list)

KLINE_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
]

# ⏱️ Fungsi countdown timer
def countdown_timer(seconds):
    try:
        for remaining in range(int(seconds), 0, -1):
            print(f"\r⏱️  Mulai analisis berikutnya dalam {remaining:>4}s...", end='', flush=True)
            time.sleep(1)
        print("\r✅ Mulai analisis sekarang!               ")
    except KeyboardInterrupt:
        print("\n⏹️  Timer dibatalkan oleh user.")

def run_generate_and_train():
    if not os.path.exists(DATASET_PATH):
        print("📥 Mengambil data awal untuk pelatihan...")
        subprocess.run(['python3', 'generate_training_dataset.py'])
    print("🧠 Melatih model...")
    subprocess.run(['python3', 'train_model.py'])

def analyze_symbol(symbol):
    try:
        end_time = datetime.utcnow()
        start_time = end_time - pd.Timedelta(hours=100)

        raw_klines = get_binance_klines(symbol, INTERVAL, start_time, end_time, use_futures=True)
        df = pd.DataFrame(raw_klines, columns=KLINE_COLUMNS)
        if df.empty:
            print(f"⚠️ Data kosong untuk {symbol}, dilewati.")
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.astype({
            'open': float, 'high': float, 'low': float,
            'close': float, 'volume': float
        })
        df['symbol'] = symbol

        df = calculate_technical_indicators(df)

        if df.empty or df.isnull().any().any():
            print(f"⚠️ Data tidak cukup untuk {symbol}, dilewati.")
            return

        latest_data = df.iloc[-1:][['rsi', 'macd', 'signal_line', 'support', 'resistance']]
        model_file = os.path.join(MODEL_PATH, f"{symbol.replace('/', '')}_model.pkl")
        encoder_file = os.path.join(MODEL_PATH, f"{symbol.replace('/', '')}_label_encoder.pkl")

        if not os.path.exists(model_file) or not os.path.exists(encoder_file):
            print(f"⚠️ Model atau encoder tidak ditemukan untuk {symbol}.")
            return

        model = joblib.load(model_file)
        label_encoder = joblib.load(encoder_file)

        prediction_encoded = model.predict(latest_data)[0]
        prediction = label_encoder.inverse_transform([int(prediction_encoded)])[0]

        close_price = df.iloc[-1]['close']
        rsi = latest_data['rsi'].values[0]

        prediction_results[prediction].append({
            'symbol': symbol,
            'price': close_price,
            'rsi': rsi
        })

        log_data = pd.DataFrame([{
            'timestamp': df.iloc[-1]['timestamp'],
            'symbol': symbol,
            'price': close_price,
            'predicted_label': prediction,
            'future_price': None
        }])
        if os.path.exists(PREDICTION_LOG):
            log_data.to_csv(PREDICTION_LOG, mode='a', header=False, index=False)
        else:
            log_data.to_csv(PREDICTION_LOG, index=False)

    except Exception as e:
        print(f"❌ Gagal menganalisis {symbol}: {e}")

def print_grouped_predictions():
    print("\n📈 Hasil Prediksi Kelompok:")
    for label, entries in prediction_results.items():
        print(f"\n📌 {label.upper()} ({len(entries)} aset)")
        for item in sorted(entries, key=lambda x: x['rsi'], reverse=True):
            print(f" - {item['symbol']} | Harga: {item['price']:.4f} | RSI: {item['rsi']:.2f}")
    print("\n==============================\n")

def run_analysis():
    global prediction_results
    prediction_results.clear()

    print("🔁 Memulai analisis real-time...")
    threads = []
    for symbol in SYMBOLS:
        thread = threading.Thread(target=analyze_symbol, args=(symbol,))
        thread.start()
        threads.append(thread)
        time.sleep(0.2)

    for thread in threads:
        thread.join()

    print_grouped_predictions()

def update_training_data(symbol):
    try:
        now = datetime.utcnow()
        start = now - pd.Timedelta(hours=6)

        raw_klines = get_binance_klines(symbol, '1h', start, now, use_futures=True)
        df = pd.DataFrame(raw_klines, columns=KLINE_COLUMNS)
        if df.empty:
            print(f"⚠️ Data kosong untuk {symbol}, dilewati.")
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.astype({
            'open': float, 'high': float, 'low': float,
            'close': float, 'volume': float
        })
        df['symbol'] = symbol

        df = calculate_technical_indicators(df)
        df = generate_label(df)
        df = df[['timestamp', 'symbol', 'rsi', 'macd', 'signal_line', 'support', 'resistance', 'label']].dropna()

        if df.empty:
            return

        if os.path.exists(DATASET_PATH):
            df_existing = pd.read_csv(DATASET_PATH)
            df_combined = pd.concat([df_existing, df], ignore_index=True)
            df_combined.drop_duplicates(subset=['timestamp', 'symbol'], keep='last', inplace=True)
        else:
            df_combined = df

        df_combined.to_csv(DATASET_PATH, index=False)
        print(f"📥 Dataset diperbarui: {symbol}")
    except Exception as e:
        print(f"❌ Gagal update dataset {symbol}: {e}")

def update_all_training_data():
    print("🔄 Memperbarui dataset pelatihan dari candle terbaru...")
    for symbol in SYMBOLS:
        update_training_data(symbol)
        time.sleep(0.1)

def update_future_prices():
    if not os.path.exists(PREDICTION_LOG):
        return

    df_pred = pd.read_csv(PREDICTION_LOG, parse_dates=['timestamp'])
    df_pred_missing = df_pred[df_pred['future_price'].isna()]

    if df_pred_missing.empty:
        return

    updated_rows = []

    for _, row in df_pred_missing.iterrows():
        symbol = row['symbol']
        ts = row['timestamp']
        future_ts = ts + pd.Timedelta(hours=1)

        try:
            raw = get_binance_klines(symbol, '1h', ts, future_ts + pd.Timedelta(hours=1), use_futures=True)
            df_future = pd.DataFrame(raw, columns=KLINE_COLUMNS)
            if df_future.empty:
                continue

            df_future['timestamp'] = pd.to_datetime(df_future['timestamp'], unit='ms')
            df_future = df_future[['timestamp', 'close']].astype({'close': float})

            future_row = df_future[df_future['timestamp'] >= future_ts].head(1)
            if not future_row.empty:
                future_price = future_row.iloc[0]['close']
                df_pred.loc[(df_pred['symbol'] == symbol) & (df_pred['timestamp'] == ts), 'future_price'] = future_price
                updated_rows.append((symbol, ts, future_price))
        except Exception as e:
            print(f"❌ Gagal update future_price {symbol} @ {ts}: {e}")

    if updated_rows:
        df_pred.to_csv(PREDICTION_LOG, index=False)
        for s, t, p in updated_rows:
            print(f"✅ future_price terisi: {s} @ {t} -> {p:.4f}")

# ⏰ Fungsi sinkronisasi ke awal candle dengan countdown
def wait_until_next_candle(interval_minutes=60):
    now = datetime.utcnow()
    next_candle = (now.replace(second=0, microsecond=0) +
                   timedelta(minutes=interval_minutes - now.minute % interval_minutes))
    wait_seconds = (next_candle - now).total_seconds()
    print(f"⏳ Menunggu {int(wait_seconds)} detik sampai candle baru...")
    countdown_timer(wait_seconds)

if __name__ == "__main__":
    run_generate_and_train()
    while True:
        wait_until_next_candle(interval_minutes=60)
        update_all_training_data()
        update_future_prices()
        run_analysis()
