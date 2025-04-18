import os
import time
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collect_data import get_all_usdt_symbols, get_binance_klines
from utils import calculate_technical_indicators, generate_label

# Konfigurasi
INTERVAL = "1h"
FUTURES = True  # Set True untuk Futures, False untuk Spot
MIN_CANDLE_COUNT = 720  # Minimum 30 hari data untuk 1h interval
MAX_WORKERS = 10  # Jumlah thread maksimum

# Waktu pengambilan data
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=365)

dataset = []

def fetch_and_process_symbol(symbol):
    try:
        print(f"🔍 Mengambil data untuk {symbol}...")
        raw_klines = get_binance_klines(symbol, INTERVAL, start_time, end_time, use_futures=FUTURES)

        if len(raw_klines) < MIN_CANDLE_COUNT:
            raise ValueError("Data tidak mencukupi (kurang dari 30 hari)")

        df = pd.DataFrame(raw_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            '_1', '_2', '_3', '_4', '_5', '_6'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df = df[['timestamp', 'symbol', 'close', 'volume']].astype({'close': float, 'volume': float})

        df = calculate_technical_indicators(df)
        df = generate_label(df)
        return df

    except Exception as e:
        print(f"❌ Gagal mengambil data untuk {symbol}: {e}")
        return None

def main():
    print(f"📥 Mengambil data historis 1 tahun dari Binance {'Futures' if FUTURES else 'Spot'}...")
    symbols = get_all_usdt_symbols(use_futures=FUTURES)

    global dataset
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_and_process_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                dataset.append(result)

    if dataset:
        new_data = pd.concat(dataset, ignore_index=True)
        os.makedirs("training_data", exist_ok=True)
        file_path = "training_data/training_dataset.csv"

        if os.path.exists(file_path):
            print("📂 Membaca dataset lama untuk digabungkan...")
            old_data = pd.read_csv(file_path, parse_dates=["timestamp"])
            combined_data = pd.concat([old_data, new_data], ignore_index=True)
            combined_data.drop_duplicates(subset=["timestamp", "symbol"], keep="last", inplace=True)
        else:
            print("📁 Dataset lama tidak ditemukan, membuat file baru...")
            combined_data = new_data

        combined_data.sort_values(by=["timestamp", "symbol"], inplace=True)
        combined_data.to_csv(file_path, index=False)
        print("✅ Dataset pelatihan diperbarui dan disimpan di training_data/training_dataset.csv")
    else:
        print("⚠️ Tidak ada data yang berhasil dikumpulkan.")

if __name__ == "__main__":
    main()
