# backtest_predictions.py

import pandas as pd
from datetime import timedelta
import argparse

CSV_PATH = 'predicted_signals.csv'

def calculate_returns(df):
    df = df[df['predicted_label'] == 'AGGRESSIVE BUY'].copy()

    if df.empty:
        print("❌ Tidak ada sinyal AGGRESSIVE BUY ditemukan.")
        return

    df['return_pct'] = (df['future_price'] - df['price']) / df['price'] * 100
    total_trades = len(df)
    winning_trades = len(df[df['return_pct'] > 0])
    losing_trades = len(df[df['return_pct'] <= 0])
    avg_return = df['return_pct'].mean()
    cumulative_return = ((df['future_price'] / df['price']).prod() - 1) * 100

    print(f"\n📊 Hasil Backtest Sinyal AGGRESSIVE BUY ({hours_back} jam terakhir):")
    print(f"- Total sinyal    : {total_trades}")
    print(f"- Menang          : {winning_trades} ({winning_trades/total_trades:.2%})")
    print(f"- Kalah           : {losing_trades} ({losing_trades/total_trades:.2%})")
    print(f"- Rata-rata return: {avg_return:.2f}% per sinyal")
    print(f"- Return kumulatif: {cumulative_return:.2f}%\n")

def main(hours_back):
    try:
        df = pd.read_csv(CSV_PATH)

        if not {'timestamp', 'price', 'future_price', 'predicted_label'}.issubset(df.columns):
            print("❌ Kolom yang dibutuhkan tidak ditemukan dalam CSV.")
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        latest_time = df['timestamp'].max()
        time_threshold = latest_time - timedelta(hours=hours_back)
        df_filtered = df[df['timestamp'] >= time_threshold]

        if df_filtered.empty:
            print(f"⚠️ Tidak ada data dalam {hours_back} jam terakhir.")
            return

        calculate_returns(df_filtered)

    except FileNotFoundError:
        print(f"❌ File {CSV_PATH} tidak ditemukan.")
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest sinyal berdasarkan file predicted_signals.csv")
    parser.add_argument('--hours', type=int, default=1, help='Jumlah jam terakhir untuk dianalisis')
    args = parser.parse_args()
    hours_back = args.hours
    main(hours_back)
