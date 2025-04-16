# ==== utils.py ====
import pandas as pd
import numpy as np
import ta

def calculate_technical_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['signal_line'] = macd.macd_signal()
    df['support'] = df['close'].rolling(window=20).min()
    df['resistance'] = df['close'].rolling(window=20).max()
    df.dropna(inplace=True)
    return df

def generate_label(df, reward_threshold=0.0075, risk_threshold=0.004, n_future=3):
    """
    Labeling cerdas berdasarkan reward-to-risk untuk timeframe pendek.

    Parameters:
    - df: DataFrame dengan kolom 'close'
    - reward_threshold: target kenaikan (%), misal 0.75%
    - risk_threshold: batas penurunan (%), misal 0.4%
    - n_future: jumlah candle ke depan yang diamati (misal 3 = 3 jam)

    Returns:
    - df: DataFrame dengan kolom 'label' berisi 'AGGRESSIVE BUY', 'SELL', 'WAIT'
    """
    df = df.copy()
    labels = []
    close_prices = df["close"].values

    for i in range(len(close_prices)):
        if i + n_future >= len(close_prices):
            labels.append("WAIT")
            continue

        current_price = close_prices[i]
        future_prices = close_prices[i + 1:i + 1 + n_future]

        max_gain = (max(future_prices) - current_price) / current_price
        max_loss = (current_price - min(future_prices)) / current_price

        if max_gain >= reward_threshold and max_loss <= risk_threshold:
            labels.append("AGGRESSIVE BUY")
        elif max_loss > risk_threshold:
            labels.append("SELL")
        else:
            labels.append("WAIT")

    df["label"] = labels
    return df
