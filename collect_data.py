# collect_data.py
import requests

def get_all_usdt_symbols(use_futures=True):
    """
    Mengambil semua simbol trading USDT dari Binance.
    use_futures=True untuk Binance Futures (default),
    use_futures=False untuk spot.
    """
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo" if use_futures else "https://api.binance.com/api/v3/exchangeInfo"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        symbols = []
        for s in data['symbols']:
            if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
                symbols.append(s['symbol'])

        return symbols
    except Exception as e:
        print(f"❌ Gagal mengambil simbol dari Binance: {e}")
        return []

def get_binance_klines(symbol, interval, start_time, end_time, use_futures=True):
    """
    Mengambil data kline (candlestick) Binance Spot atau Futures.
    Otomatis lakukan pagination jika data > 1000 candle.
    """
    base_url = "https://fapi.binance.com" if use_futures else "https://api.binance.com"
    url = f"{base_url}/fapi/v1/klines" if use_futures else f"{base_url}/api/v3/klines"

    limit = 1000
    data = []

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break

            data.extend(klines)
            start_ts = klines[-1][0] + 1
        except requests.exceptions.RequestException as e:
            raise Exception(f"📡 Gagal mengambil data Binance untuk {symbol}: {e}")

    return data
