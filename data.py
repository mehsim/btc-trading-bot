import time
import requests
import pandas as pd

def get_history(symbol="BTCUSDT", interval="15", limit=1000, pages=1):
    url = "https://api.bybit.com/v5/market/kline"
    all_data = []
    
    current_end = None
    
    for page in range(pages):
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": str(interval),
            "limit": limit
        }
        if current_end is not None:
            params["end"] = current_end
            
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching page {page + 1}: Received status code {response.status_code}")
                print(f"Response content: {response.text[:300]}")
                break
            res = response.json()
            if "result" in res and "list" in res["result"] and len(res["result"]["list"]) > 0:
                batch = res["result"]["list"]
                all_data.extend(batch)
                
                # Get the oldest timestamp (last element in the descending list)
                oldest_ts = int(float(batch[-1][0]))
                current_end = oldest_ts - 1
            else:
                break
        except Exception as e:
            print(f"Error fetching page {page + 1}: {e}")
            break
            
        if pages > 1:
            time.sleep(0.1)
            
    if not all_data:
        print(f"Bybit klines returned no data. Attempting Binance API fallback for {symbol}...")
        try:
            binance_interval = "1h"
            if str(interval) == "15":
                binance_interval = "15m"
            elif str(interval) == "60":
                binance_interval = "1h"
            elif str(interval) == "240":
                binance_interval = "4h"
            elif str(interval).upper() == "D":
                binance_interval = "1d"

            binance_url = "https://api.binance.com/api/v3/klines"
            binance_params = {
                "symbol": symbol.upper(),
                "interval": binance_interval,
                "limit": min(limit * pages, 1000)
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(binance_url, params=binance_params, headers=headers, timeout=10)
            if resp.status_code == 200:
                binance_data = resp.json()
                for item in binance_data:
                    all_data.append([
                        float(item[0]), # timestamp
                        float(item[1]), # open
                        float(item[2]), # high
                        float(item[3]), # low
                        float(item[4]), # close
                        float(item[5]), # volume
                        float(item[7])  # turnover (quote asset volume)
                    ])
                print(f"Successfully loaded {len(binance_data)} candles from Binance API fallback.")
            else:
                print(f"Binance fallback failed with HTTP {resp.status_code}")
        except Exception as ex:
            print(f"Error fetching Binance fallback klines: {ex}")

    if not all_data:
        print(f"Bybit & Binance failed. Attempting Kraken API fallback for {symbol}...")
        try:
            kraken_interval = 60
            if str(interval) == "15":
                kraken_interval = 15
            elif str(interval) == "60":
                kraken_interval = 60
            elif str(interval) == "240":
                kraken_interval = 240
            elif str(interval).upper() == "D":
                kraken_interval = 1440

            kraken_url = "https://api.kraken.com/0/public/OHLC"
            kraken_params = {
                "pair": "XBTUSD",
                "interval": kraken_interval
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(kraken_url, params=kraken_params, headers=headers, timeout=10)
            if resp.status_code == 200:
                res = resp.json()
                if "result" in res and len(res["result"]) > 0:
                    pair_key = [k for k in res["result"].keys() if k != "last"][0]
                    kraken_data = res["result"][pair_key]
                    
                    # Take the last 'limit' candles (Kraken returns 720 by default)
                    candles_to_take = kraken_data[-limit:] if len(kraken_data) > limit else kraken_data
                    for item in candles_to_take:
                        all_data.append([
                            float(item[0]) * 1000, # timestamp in ms
                            float(item[1]),        # open
                            float(item[2]),        # high
                            float(item[3]),        # low
                            float(item[4]),        # close
                            float(item[6]),        # volume
                            float(item[6]) * float(item[4]) # turnover
                        ])
                    print(f"Successfully loaded {len(candles_to_take)} candles from Kraken API fallback.")
                else:
                    print(f"Kraken returned empty results or error: {res.get('error')}")
            else:
                print(f"Kraken fallback failed with HTTP {resp.status_code}")
        except Exception as ex:
            print(f"Error fetching Kraken fallback klines: {ex}")

    if not all_data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df = df.astype(float)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df.reset_index(drop=True, inplace=True)

    return df

def get_bybit_oi_history(symbol="BTCUSDT", interval="15", start_ts_ms=None, end_ts_ms=None):
    url = "https://api.bybit.com/v5/market/open-interest"
    interval_time = "1h"
    if str(interval) == "5":
        interval_time = "5min"
    elif str(interval) == "15":
        interval_time = "15min"
    elif str(interval) == "60":
        interval_time = "1h"
        
    oi_data = []
    cursor = None
    for page in range(100):
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": interval_time,
            "limit": 200
        }
        if cursor:
            params["cursor"] = cursor
            
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                break
            res = resp.json()
            if "result" in res and "list" in res["result"] and len(res["result"]["list"]) > 0:
                batch = res["result"]["list"]
                oi_data.extend(batch)
                
                oldest_ts = int(batch[-1]["timestamp"])
                if start_ts_ms and oldest_ts < int(start_ts_ms):
                    break
                
                cursor = res["result"].get("nextPageCursor")
                if not cursor:
                    break
            else:
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching OI history: {e}")
            break
            
    if not oi_data:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
        
    df_oi = pd.DataFrame(oi_data)
    df_oi["timestamp"] = df_oi["timestamp"].astype(float)
    df_oi["open_interest"] = df_oi["openInterest"].astype(float)
    df_oi = df_oi[["timestamp", "open_interest"]].sort_values("timestamp").reset_index(drop=True)
    return df_oi

def get_bybit_funding_history(symbol="BTCUSDT", start_ts_ms=None, end_ts_ms=None):
    url = "https://api.bybit.com/v5/market/funding/history"
    funding_data = []
    current_end = int(end_ts_ms) if end_ts_ms else int(time.time() * 1000)
    for page in range(50):
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": 200,
            "endTime": current_end
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                break
            res = resp.json()
            if "result" in res and "list" in res["result"] and len(res["result"]["list"]) > 0:
                batch = res["result"]["list"]
                funding_data.extend(batch)
                
                oldest_ts = int(batch[-1]["fundingRateTimestamp"])
                if start_ts_ms and oldest_ts < int(start_ts_ms):
                    break
                
                current_end = oldest_ts - 1
            else:
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching Funding history: {e}")
            break
            
    if not funding_data:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
        
    df_funding = pd.DataFrame(funding_data)
    df_funding["timestamp"] = df_funding["fundingRateTimestamp"].astype(float)
    df_funding["funding_rate"] = df_funding["fundingRate"].astype(float)
    df_funding = df_funding[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)
    return df_funding

def get_fear_and_greed_history():
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            res = resp.json()
            if "data" in res and len(res["data"]) > 0:
                fng_data = []
                for item in res["data"]:
                    fng_data.append({
                        "timestamp": float(item["timestamp"]) * 1000,
                        "fear_greed": float(item["value"])
                    })
                df_fng = pd.DataFrame(fng_data)
                df_fng = df_fng.sort_values("timestamp").reset_index(drop=True)
                return df_fng
    except Exception as e:
        print(f"Error fetching Fear & Greed history: {e}")
    return pd.DataFrame(columns=["timestamp", "fear_greed"])

def merge_derivatives_sentiment_features(df, symbol, interval):
    if df.empty:
        df["open_interest"] = 0.0
        df["funding_rate"] = 0.0
        df["fear_greed"] = 50.0
        return df

    min_ts = int(df["timestamp"].min())
    max_ts = int(df["timestamp"].max())
    
    # Fetch
    df_oi = get_bybit_oi_history(symbol=symbol, interval=interval, start_ts_ms=min_ts, end_ts_ms=max_ts)
    df_funding = get_bybit_funding_history(symbol=symbol, start_ts_ms=min_ts, end_ts_ms=max_ts)
    df_fng = get_fear_and_greed_history()
    
    # Merge open interest
    if not df_oi.empty:
        df_oi = df_oi.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_oi, on="timestamp", direction="backward")
    else:
        df["open_interest"] = 0.0
        
    # Merge funding rate
    if not df_funding.empty:
        df_funding = df_funding.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_funding, on="timestamp", direction="backward")
    else:
        df["funding_rate"] = 0.0
        
    # Merge fear and greed
    if not df_fng.empty:
        df_fng = df_fng.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_fng, on="timestamp", direction="backward")
    else:
        df["fear_greed"] = 50.0
        
    # Clean up NaNs
    df["open_interest"] = df["open_interest"].ffill().bfill().fillna(0.0)
    df["funding_rate"] = df["funding_rate"].ffill().bfill().fillna(0.0)
    df["fear_greed"] = df["fear_greed"].ffill().bfill().fillna(50.0)
    
    return df