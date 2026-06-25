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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df = df.astype(float)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df.reset_index(drop=True, inplace=True)

    return df