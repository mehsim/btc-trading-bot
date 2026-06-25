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
            res = requests.get(url, params=params).json()
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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df = df.astype(float)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df.reset_index(drop=True, inplace=True)

    return df