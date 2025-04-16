# generate_training_dataset.py
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from collect_data import get_all_usdt_symbols, get_binance_klines
from utils import calculate_technical_indicators, generate_label

# Konfigurasi
INTERVAL = "1h"
FUTURES = True  # Set True untuk Futures, False untuk Spot
MIN_CANDLE_COUNT = 720  # Minimum 30 hari data untuk 1h interval

# Waktu pengambilan data
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=365)

dataset = []

def main():
    print(f"📥 Mengambil data historis 1 tahun dari Binance {'Futures' if FUTURES else 'Spot'}...")

    symbols = get_all_usdt_symbols(use_futures=FUTURES)

    for symbol in symbols:
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

            dataset.append(df)

        except Exception as e:
            print(f"❌ Gagal mengambil data untuk {symbol}: {e}")
            continue

    if dataset:
        final_df = pd.concat(dataset, ignore_index=True)
        os.makedirs("training_data", exist_ok=True)
        final_df.to_csv("training_data/training_dataset.csv", index=False)
        print("✅ Dataset pelatihan disimpan di training_data/training_dataset.csv")
    else:
        print("⚠️ Tidak ada data yang berhasil dikumpulkan.")

if __name__ == "__main__":
    main()
