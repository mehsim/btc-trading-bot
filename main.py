import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dotenv import load_dotenv
load_dotenv()

import websocket
import json
import requests
import pandas as pd
import numpy as np
import joblib
import threading
import time
from datetime import datetime

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
from data import get_history
import xml.etree.ElementTree as ET
from flask import Flask, jsonify, render_template

# ==========================================
# WEB DASHBOARD CONFIGURATION & STATE
# ==========================================
app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True
bot_logs = []
logs_lock = threading.Lock()

bot_state = {
    "live_price": None,
    "last_update": 0.0,
    "active_trade": None,
    "latest_prediction": None,
    "confluence_results": None,
    "regime": "Unknown",
    "adx": 0.0,
    "status": "Initializing",
    "calibration": {
        "p95": 0.0,
        "max_conf": 0.0,
        "mean": 54.81
    },
    "trade_history": [],
    "prediction_history": []
}

def save_history():
    data = {
        "trade_history": bot_state["trade_history"],
        "prediction_history": bot_state["prediction_history"]
    }
    try:
        with open("dashboard_history.json", "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving history to disk: {e}")

def load_history():
    if os.path.exists("dashboard_history.json"):
        try:
            with open("dashboard_history.json", "r") as f:
                data = json.load(f)
                bot_state["trade_history"] = data.get("trade_history", [])
                bot_state["prediction_history"] = data.get("prediction_history", [])
                print(f"Loaded {len(bot_state['trade_history'])} trades and {len(bot_state['prediction_history'])} predictions from dashboard_history.json")
        except Exception as e:
            print(f"Error loading history from disk: {e}")

# Thread-safe print wrapper to redirect logs to dashboard log panel
_print = print
def print(*args, **kwargs):
    _print(*args, **kwargs)
    if "file" not in kwargs or kwargs["file"] is None:
        msg = " ".join(map(str, args))
        timestamp = datetime.now().strftime("%H:%M:%S")
        lines = msg.split('\n')
        with logs_lock:
            for line in lines:
                if line.strip(): # ignore empty lines in console
                    bot_logs.append(f"[{timestamp}] {line}")
            # Keep history to 200 lines
            if len(bot_logs) > 200:
                bot_logs[:] = bot_logs[-200:]

@app.route("/api/status")
def get_status():
    state_copy = bot_state.copy()
    with logs_lock:
        state_copy["logs"] = list(bot_logs)
    return jsonify(state_copy)

@app.route("/api/terminate", methods=["POST"])
def terminate_bot():
    import os
    import signal
    print("[System] Terminate request received from web dashboard. Shutting down gracefully...")
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"status": "terminating"})

@app.route("/")
def index():
    return render_template("index.html")

def run_flask():
    import logging
    import os
    # Mute default werkzeug request logs to prevent console pollution
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# =========================
# CONFIGURATION
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "60"

# =========================
from xgboost import XGBClassifier, XGBRegressor
models_trending = {
    "trend": XGBClassifier(),
    "price": XGBRegressor()
}
models_trending["trend"].load_model("xgb_trending_trend.json")
models_trending["price"].load_model("xgb_trending_price.json")

models_ranging = {
    "trend": XGBClassifier(),
    "price": XGBRegressor()
}
models_ranging["trend"].load_model("xgb_ranging_trend.json")
models_ranging["price"].load_model("xgb_ranging_price.json")

# =========================
# WEB SOCKET FOR LIVE PRICE
# =========================
live_price = None
last_ws_update_time = 0.0

def on_message(ws, message):
    global live_price, last_ws_update_time
    try:
        data = json.loads(message)
        if "data" in data and isinstance(data["data"], dict):
            price = data["data"].get("lastPrice")
            if price:
                live_price = float(price)
                last_ws_update_time = time.time()
                bot_state["live_price"] = live_price
                bot_state["last_update"] = last_ws_update_time
    except Exception:
        pass

def on_open(ws):
    print(f"Connected to Bybit WebSocket for {SYMBOL}")
    ws.send(json.dumps({
        "op": "subscribe",
        "args": [f"tickers.{SYMBOL}"]
    }))

def on_close(ws, close_status_code, close_msg):
    pass

def start_ws():
    url = "wss://stream.bybit.com/v5/public/spot"
    while True:
        try:
            ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_close=on_close
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception:
            pass
        time.sleep(3)

# WebSocket thread is started inside if __name__ == "__main__" block at the bottom


# =========================
# FEATURE ENGINE
# =========================
features = [
    "RSI", "MACD_diff", "MFI", "ATR_norm",
    "close_to_EMA9", "close_to_EMA21", "close_to_EMA50", "close_to_EMA200", "EMA9_to_EMA21", 
    "BB_pct", "BB_width", "return_5m", "volatility_10m", "volume_ratio",
    "high_low_ratio", "open_close_ratio", "RSI_diff", "MACD_diff_diff", "ROC_5", "ROC_10",
    "ADX", "ADX_pos", "ADX_neg", "close_to_VWAP",
    "btc_return_5m", "btc_return_5m_lag1", "btc_return_5m_lag2", "btc_return_5m_lag3",
    "RSI_24", "ROC_24", "volatility_24h"
]
for lag in [1, 2, 3, 4, 5]:
    features.append(f"return_5m_lag{lag}")
for lag in [1, 2, 3]:
    features.append(f"volume_ratio_lag{lag}")
for lag in [1, 2]:
    features.append(f"RSI_lag{lag}")
    features.append(f"MACD_diff_lag{lag}")
    features.append(f"BB_pct_lag{lag}")

def add_features(df):
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["MACD_diff"] = macd.macd_diff() / (df["close"] + 1e-8)
    
    df["EMA_9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["EMA_21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    df["EMA_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["EMA_200"] = EMAIndicator(df["close"], window=200).ema_indicator()
    
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()
    
    df["MFI"] = MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14).money_flow_index()
    
    # ATR Volatility normalized
    atr_ind = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["ATR_norm"] = atr_ind.average_true_range() / (df["close"] + 1e-8)
    
    # Normalized scale-invariant features
    df["close_to_EMA9"] = df["close"] / df["EMA_9"] - 1.0
    df["close_to_EMA21"] = df["close"] / df["EMA_21"] - 1.0
    df["close_to_EMA50"] = df["close"] / df["EMA_50"] - 1.0
    df["close_to_EMA200"] = df["close"] / df["EMA_200"] - 1.0
    df["EMA9_to_EMA21"] = df["EMA_9"] / df["EMA_21"] - 1.0
    df["BB_pct"] = (df["close"] - df["BB_low"]) / (df["BB_high"] - df["BB_low"] + 1e-8)
    df["BB_width"] = (df["BB_high"] - df["BB_low"]) / df["BB_mid"]
    
    df["return_5m"] = df["close"].pct_change(1)
    df["volatility_10m"] = df["return_5m"].rolling(10).std()
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-8)
    
    # Additional engineered features
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
    df["open_close_ratio"] = (df["close"] - df["open"]) / df["open"]
    df["RSI_diff"] = df["RSI"].diff()
    df["MACD_diff_diff"] = df["MACD_diff"].diff()
    
    # ADX Indicator
    adx_ind = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX"] = adx_ind.adx()
    df["ADX_pos"] = adx_ind.adx_pos()
    df["ADX_neg"] = adx_ind.adx_neg()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    rolling_pv = (typical_price * df["volume"]).rolling(window=168, min_periods=1).sum()
    rolling_v = df["volume"].rolling(window=168, min_periods=1).sum()
    df["VWAP"] = rolling_pv / (rolling_v + 1e-8)
    df["close_to_VWAP"] = df["close"] / df["VWAP"] - 1.0
    
    # Momentum (Rate of Change)
    df["ROC_5"] = df["close"].pct_change(5)
    df["ROC_10"] = df["close"].pct_change(10)
    
    # BTC correlation return and lags
    df["btc_return_5m"] = df["close_btc"].pct_change(1)
    for lag in [1, 2, 3]:
        df[f"btc_return_5m_lag{lag}"] = df["btc_return_5m"].shift(lag)
        
    # Autoregressive target coin lags
    for lag in [1, 2, 3, 4, 5]:
        df[f"return_5m_lag{lag}"] = df["return_5m"].shift(lag)
    for lag in [1, 2, 3]:
        df[f"volume_ratio_lag{lag}"] = df["volume_ratio"].shift(lag)
    for lag in [1, 2]:
        df[f"RSI_lag{lag}"] = df["RSI"].shift(lag)
        df[f"MACD_diff_lag{lag}"] = df["MACD_diff"].shift(lag)
        df[f"BB_pct_lag{lag}"] = df["BB_pct"].shift(lag)
        
    # Macro-technical indicators
    df["RSI_24"] = RSIIndicator(df["close"], window=24).rsi()
    df["ROC_24"] = df["close"].pct_change(24)
    df["volatility_24h"] = df["return_5m"].rolling(24).std()
        
    df.dropna(inplace=True)
    return df

def build_df(current_price):
    try:
        # Fetch target coin data (limit 300 to satisfy EMA 200)
        df_target = get_history(symbol=SYMBOL, interval=INTERVAL, limit=300)
        if df_target is not None and len(df_target) > 0:
            df_target.loc[df_target.index[-1], "close"] = current_price
            
            if SYMBOL == "BTCUSDT":
                df = df_target.copy()
                df["close_btc"] = df["close"]
            else:
                # Fetch BTCUSDT data
                df_btc = get_history(symbol="BTCUSDT", interval=INTERVAL, limit=300)
                if df_btc is not None and len(df_btc) > 0:
                    df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                    # Merge target coin and BTCUSDT on timestamp
                    df = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
                else:
                    return None
            
            if len(df) > 0:
                df = add_features(df)
                return df
    except Exception as e:
        print(f"Error fetching candle data: {e}")
    return None

def get_local_time_str(t):
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

def evaluate_predictions(df_completed):
    if not bot_state["prediction_history"]:
        return

    # Create a map of timestamp to close price for quick lookup
    ts_map = {}
    for _, row in df_completed.iterrows():
        ts_map[int(row["timestamp"])] = float(row["close"])

    for pred in bot_state["prediction_history"]:
        if not pred["evaluation"]["evaluated"]:
            # Check 1 hour later (1 hour = 3600 seconds = 3,600,000 milliseconds)
            target_ts = int(pred["candle_timestamp"]) + 3600000
            if target_ts in ts_map:
                exit_price = ts_map[target_ts]
                ref_price = pred["ref_price"]
                change = exit_price - ref_price
                change_pct = (change / ref_price) * 100
                direction = pred["direction"]
                
                # Check success
                success = (change > 0 and direction == "Bullish") or (change < 0 and direction == "Bearish")
                
                pred["evaluation"] = {
                    "evaluated": True,
                    "exit_price": float(exit_price),
                    "change": float(change),
                    "change_pct": float(change_pct),
                    "success": bool(success)
                }
                
                # Print to log
                success_str = "SUCCESSFUL" if success else "UNSUCCESSFUL"
                print(f"[Prediction Tracker] Evaluated 1h Prediction from {get_local_time_str(pred['candle_timestamp']/1000)}: Direction: {direction} | Ref Price: {ref_price:.2f} | Exit Price: {exit_price:.2f} | Change: {change:+.2f} ({change_pct:+.3f}%) | Result: {success_str} | Status: {pred['status']}")

# =========================
# NEWS & SOCIAL SENTIMENT ANALYSIS
# =========================
sentiment_pipeline = None

def get_reddit_posts():
    """
    Fetches the top crypto/bitcoin post titles from Reddit RSS feeds.
    Does not require API keys, but does require a descriptive User-Agent.
    """
    subreddits = ["CryptoCurrency", "Bitcoin"]
    posts = []
    # Using the recommended Reddit API user agent format to prevent blocks
    headers = {"User-Agent": "btc-trading-bot:v1.0.0 (by /u/btc-trading-bot-user)"}
    
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot/.rss"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                xml_content = res.content.decode("utf-8")
                # Parse the Atom XML feed
                root = ET.fromstring(xml_content)
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                sub_posts = []
                for entry in root.findall("atom:entry", namespace):
                    title_elem = entry.find("atom:title", namespace)
                    if title_elem is not None and title_elem.text:
                        sub_posts.append(title_elem.text.strip())
                # Limit to top 5 posts per subreddit to avoid skewing sentiment
                posts.extend(sub_posts[:5])
                print(f"[News/Sentiment] Fetched {len(sub_posts[:5])} posts from r/{sub} RSS.")
            else:
                print(f"[News/Sentiment] Reddit r/{sub} feed returned status code {res.status_code}")
        except Exception as e:
            print(f"[News/Sentiment] Exception fetching Reddit r/{sub} feed: {e}")
    return posts

def get_cryptopanic_posts():
    """
    Fetches aggregated cryptocurrency news and social posts from CryptoPanic.
    Requires a free CryptoPanic developer API token in .env.
    """
    token = os.environ.get("CRYPTOPANIC_API_TOKEN")
    if not token:
        return []
    
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": token,
        "public": "true",
        "filter": "hot",
        "regions": "en"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            posts = []
            for item in data.get("results", []):
                title = item.get("title")
                if title:
                    posts.append(title.strip())
            print(f"[News/Sentiment] Fetched {len(posts[:10])} posts from CryptoPanic API.")
            return posts[:10]
        else:
            print(f"[News/Sentiment] CryptoPanic API returned status code {res.status_code}")
    except Exception as e:
        print(f"[News/Sentiment] Exception fetching CryptoPanic API: {e}")
    return []

def get_x_tweets():
    """
    Fetches recent tweets matching crypto/bitcoin search query.
    Requires an X Developer Bearer Token in .env (Basic or Pro subscription).
    """
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        return []
    
    query = os.environ.get("X_SEARCH_QUERY", "Bitcoin OR BTC lang:en -is:retweet")
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "v2RecentSearchPython"
    }
    params = {
        "query": query,
        "max_results": 10,
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            tweets = []
            for t in data.get("data", []):
                text = t.get("text")
                if text:
                    tweets.append(text.strip())
            print(f"[News/Sentiment] Fetched {len(tweets)} tweets from X API.")
            return tweets
        else:
            print(f"[News/Sentiment] X API returned status code {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[News/Sentiment] Exception fetching X tweets: {e}")
    return []

def get_news_sentiment():
    global sentiment_pipeline
    titles = []

    # 1. Fetch from Cointelegraph RSS (standard news)
    url = "https://cointelegraph.com/rss"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            xml_content = res.content.decode("utf-8")
            root = ET.fromstring(xml_content)
            rss_titles = []
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                if title_elem is not None and title_elem.text:
                    rss_titles.append(title_elem.text.strip())
            titles.extend(rss_titles[:10])
            print(f"[News/Sentiment] Fetched {len(rss_titles[:10])} articles from Cointelegraph RSS.")
    except Exception as e:
        print(f"[News/Sentiment] Error fetching Cointelegraph RSS: {e}")

    # 2. Fetch from Reddit RSS (free social sentiment fallback)
    reddit_posts = get_reddit_posts()
    titles.extend(reddit_posts)

    # 3. Fetch from CryptoPanic (optional aggregated news/social)
    cryptopanic_posts = get_cryptopanic_posts()
    titles.extend(cryptopanic_posts)

    # 4. Fetch from X / Twitter (optional premium social sentiment)
    x_tweets = get_x_tweets()
    titles.extend(x_tweets)

    # Clean up empty or duplicate titles
    seen = set()
    cleaned_titles = []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            cleaned_titles.append(t)
            
    # Limit combined sources to top 25 to avoid overloaded HuggingFace pipeline times
    cleaned_titles = cleaned_titles[:25]

    if not cleaned_titles:
        print("[News/Sentiment] No content found across any source. Sentiment defaults to Neutral.")
        return "Neutral", []

    try:
        # Lazy load FinBERT pipeline (approx. 400MB download on first run)
        if sentiment_pipeline is None:
            from transformers import pipeline
            sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

        print(f"[News/Sentiment] Running FinBERT sentiment analysis on {len(cleaned_titles)} text inputs...")
        results = sentiment_pipeline(cleaned_titles)
        
        total_score = 0.0
        for r in results:
            label = r["label"].lower()
            score = float(r["score"])
            if label == "positive":
                total_score += score
            elif label == "negative":
                total_score -= score
                
        avg_score = total_score / len(cleaned_titles)
        
        sentiment = "Neutral"
        if avg_score > 0.15:
            sentiment = "Bullish"
        elif avg_score < -0.15:
            sentiment = "Bearish"
            
        print(f"[News/Sentiment] Analysis complete. Avg Score: {avg_score:.4f} | Aggregated Sentiment: {sentiment}")
        return sentiment, cleaned_titles
    except Exception as e:
        print(f"[News/Sentiment] Error executing FinBERT pipeline: {e}")
    return "Neutral", []

# =========================
# ORDER BOOK PRESSURE
# =========================
def get_orderbook_imbalance():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "spot", "symbol": SYMBOL, "limit": 25}, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"[Orderbook] Error: received HTTP {response.status_code}")
            return {"imbalance": 0.0, "spread": 0.0}
        res = response.json()
        if "result" in res and "b" in res["result"] and "a" in res["result"]:
            bids = res["result"]["b"]  # list of [price, size]
            asks = res["result"]["a"]
            
            if not bids or not asks:
                return {"imbalance": 0.0, "spread": 0.0}
                
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2.0
            
            # Normalized bid-ask spread
            spread = (best_ask - best_bid) / mid_price
            
            # Weighted sizes by proximity to mid price (1 / distance)
            weighted_bid_size = sum(float(b[1]) / (abs(float(b[0]) - mid_price) + 1e-8) for b in bids)
            weighted_ask_size = sum(float(a[1]) / (abs(float(a[0]) - mid_price) + 1e-8) for a in asks)
            
            weighted_imbalance = (weighted_bid_size - weighted_ask_size) / (weighted_bid_size + weighted_ask_size + 1e-8)
            
            return {
                "imbalance": weighted_imbalance,
                "spread": spread
            }
    except Exception as e:
        print(f"[Orderbook] Error fetching/calculating orderbook metrics: {e}")
    return {"imbalance": 0.0, "spread": 0.0}

# ==========================================
# CONFIDENCE CALIBRATION & HISTORICAL STATS
# ==========================================
def calculate_historical_thresholds(model_trend):
    print(f"Fetching historical data to calibrate confidence percentiles (last 5,000 candles for {SYMBOL} + BTCUSDT)...")
    try:
        df_target = get_history(symbol=SYMBOL, interval=INTERVAL, limit=1000, pages=5)
        df_btc = get_history(symbol="BTCUSDT", interval=INTERVAL, limit=1000, pages=5)
        
        if df_target is not None and len(df_target) > 0 and df_btc is not None and len(df_btc) > 0:
            df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
            df = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
            if len(df) > 0:
                df = add_features(df)
                
                X_hist = df[features].values
                probs = model_trend.predict_proba(X_hist)
                confidences = np.max(probs, axis=1)
                
                p95 = float(np.percentile(confidences, 95))
                max_conf = float(np.max(confidences))
                mean_conf = float(np.mean(confidences))
                
                print("Confidence Calibration Done:")
                print(f"  - Historical Mean: {mean_conf*100:.2f}%")
                print(f"  - 95th Percentile Threshold (Maps to 80%): {p95*100:.2f}%")
                print(f"  - Maximum Confidence (Maps to 100%): {max_conf*100:.2f}%")
                return p95, max_conf
    except Exception as e:
        print(f"Error calculating calibration: {e}. Using defaults.")
    
    return 0.55, 0.75

def calibrate_confidence(raw_conf, p95, max_conf):
    if max_conf <= p95:
        max_conf = p95 + 0.01
    if p95 <= 0.50:
        p95 = 0.51
        
    if raw_conf < p95:
        # Piecewise linear mapping [0.50, p95] -> [50%, 80%]
        calibrated = 50.0 + (raw_conf - 0.50) / (p95 - 0.50) * 30.0
    else:
        # Piecewise linear mapping [p95, max_conf] -> [80%, 100%]
        calibrated = 80.0 + (raw_conf - p95) / (max_conf - p95) * 20.0
        
    return min(100.0, max(50.0, calibrated)) / 100.0

def get_funding_rate(symbol=SYMBOL):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "linear", "symbol": symbol}, headers=headers, timeout=5)
        if response.status_code != 200:
            print(f"Error fetching funding rate: HTTP status {response.status_code}")
            return 0.0
        res = response.json()
        if "result" in res and "list" in res["result"] and len(res["result"]["list"]) > 0:
            rate_str = res["result"]["list"][0].get("fundingRate")
            if rate_str:
                return float(rate_str)
    except Exception as e:
        print(f"Error fetching funding rate: {e}")
    return 0.0

# ==========================================
# PRE-TRADE CONFLUENCE ANALYSIS
# ==========================================
def check_pre_trade_confluence(current_price, df_1h, ml_trend, news_sentiment, expected_pct_change):
    """
    Runs 12 pre-trade confluence checks before a signal is authorized.
    Returns: (bool_all_pass, dict_results_details)
    """
    results = {}
    all_pass = True

    # 1. 1-Day Structural Trend Check (Macro Bias)
    try:
        # Fetch Daily historical candles
        df_1d = get_history(symbol=SYMBOL, interval="D", limit=100)
    except Exception as e:
        print(f"Error fetching 1d candle history for confluence: {e}")
        df_1d = None

    if df_1d is None or len(df_1d) < 21:
        results["1d_Trend"] = {"pass": False, "detail": "Could not fetch 1d data"}
        all_pass = False
    else:
        # Calculate 1d EMA 9 and EMA 21 using completed candles
        df_1d_completed = df_1d.iloc[:-1].copy()
        ema9_1d = EMAIndicator(df_1d_completed["close"], window=9).ema_indicator().iloc[-1]
        ema21_1d = EMAIndicator(df_1d_completed["close"], window=21).ema_indicator().iloc[-1]
        
        # Trend check
        trend_1d = "Bullish" if ema9_1d > ema21_1d else "Bearish"
        if ml_trend == "Bullish":
            trend_1d_pass = (trend_1d == "Bullish")
        else:
            trend_1d_pass = (trend_1d == "Bearish")
            
        results["1d_Trend"] = {
            "pass": trend_1d_pass,
            "detail": f"1d Trend is {trend_1d} (EMA9: {ema9_1d:.2f}, EMA21: {ema21_1d:.2f})"
        }
        if not trend_1d_pass:
            all_pass = False

    # 2. 4-Hour Tactical Trend Check & 4-Hour RSI Check
    try:
        # Fetch 4h historical candles (interval "240")
        df_4h = get_history(symbol=SYMBOL, interval="240", limit=100)
    except Exception as e:
        print(f"Error fetching 4h candle history for confluence: {e}")
        df_4h = None

    if df_4h is None or len(df_4h) < 21:
        results["4h_Trend"] = {"pass": False, "detail": "Could not fetch 4h data"}
        results["4h_RSI"] = {"pass": False, "detail": "Could not fetch 4h data"}
        all_pass = False
    else:
        # Calculate 4h EMA 9, EMA 21, and RSI using completed candles
        df_4h_completed = df_4h.iloc[:-1].copy()
        ema9_4h = EMAIndicator(df_4h_completed["close"], window=9).ema_indicator().iloc[-1]
        ema21_4h = EMAIndicator(df_4h_completed["close"], window=21).ema_indicator().iloc[-1]
        
        # Calculate 4h RSI
        rsi_4h = RSIIndicator(df_4h_completed["close"], window=14).rsi().iloc[-1]
        
        # Trend check
        trend_4h = "Bullish" if ema9_4h > ema21_4h else "Bearish"
        if ml_trend == "Bullish":
            trend_pass = (trend_4h == "Bullish")
        else:
            trend_pass = (trend_4h == "Bearish")
            
        results["4h_Trend"] = {
            "pass": trend_pass,
            "detail": f"4h Trend is {trend_4h} (EMA9: {ema9_4h:.2f}, EMA21: {ema21_4h:.2f})"
        }
        if not trend_pass:
            all_pass = False
            
        # RSI check
        if ml_trend == "Bullish":
            rsi_4h_pass = (rsi_4h < 70.0)
            detail_msg = f"4h RSI is {rsi_4h:.2f} (< 70, Safe)" if rsi_4h_pass else f"4h RSI is {rsi_4h:.2f} (>= 70, Overbought)"
        else:
            rsi_4h_pass = (rsi_4h > 30.0)
            detail_msg = f"4h RSI is {rsi_4h:.2f} (> 30, Safe)" if rsi_4h_pass else f"4h RSI is {rsi_4h:.2f} (<= 30, Oversold)"
            
        results["4h_RSI"] = {
            "pass": rsi_4h_pass,
            "detail": detail_msg
        }
        if not rsi_4h_pass:
            all_pass = False

    # 3. 1h RSI Check
    rsi_1h = df_1h["RSI"].iloc[-1]
    if ml_trend == "Bullish":
        rsi_1h_pass = (rsi_1h < 70.0)
        detail_msg = f"1h RSI is {rsi_1h:.2f} (< 70, Safe)" if rsi_1h_pass else f"1h RSI is {rsi_1h:.2f} (>= 70, Overbought)"
    else:
        rsi_1h_pass = (rsi_1h > 30.0)
        detail_msg = f"1h RSI is {rsi_1h:.2f} (> 30, Safe)" if rsi_1h_pass else f"1h RSI is {rsi_1h:.2f} (<= 30, Oversold)"
    
    results["1h_RSI"] = {
        "pass": rsi_1h_pass,
        "detail": detail_msg
    }
    if not rsi_1h_pass:
        all_pass = False

    # 4. Volume Check (Completed 1h volume >= 80% of 20-period average volume of completed candles)
    try:
        vol_series = df_1h["volume"]
        avg_vol_20 = vol_series.iloc[:-1].rolling(20).mean().iloc[-1]
        latest_vol = vol_series.iloc[-2]
        
        volume_pass = (latest_vol >= 0.8 * avg_vol_20)
        results["Volume_Participation"] = {
            "pass": volume_pass,
            "detail": f"Vol: {latest_vol:.1f} vs Avg20: {avg_vol_20:.1f} ({latest_vol/avg_vol_20*100:.1f}%, Req >= 80%)"
        }
    except Exception as e:
        volume_pass = True
        results["Volume_Participation"] = {
            "pass": True,
            "detail": f"Skipped volume check (Error: {e})"
        }
    if not volume_pass:
        all_pass = False

    # 5. Bollinger Band Edge Guard Check (BB_pct)
    bb_pct_val = df_1h["BB_pct"].iloc[-1]
    if ml_trend == "Bullish":
        bb_pass = (bb_pct_val < 0.95)
        detail_msg = f"BB Pct is {bb_pct_val:.3f} (< 0.95, Room to run)" if bb_pass else f"BB Pct is {bb_pct_val:.3f} (>= 0.95, Overextended long)"
    else:
        bb_pass = (bb_pct_val > 0.05)
        detail_msg = f"BB Pct is {bb_pct_val:.3f} (> 0.05, Room to run)" if bb_pass else f"BB Pct is {bb_pct_val:.3f} (<= 0.05, Overextended short)"
        
    results["BB_Edge_Guard"] = {
        "pass": bb_pass,
        "detail": detail_msg
    }
    if not bb_pass:
        all_pass = False

    # 6. Counter-Momentum Candle Guard (Check if last 3 completed 1h candles oppose the direction)
    try:
        c1 = df_1h.iloc[-2]
        c2 = df_1h.iloc[-3]
        c3 = df_1h.iloc[-4]
        
        is_red = [c1["close"] < c1["open"], c2["close"] < c2["open"], c3["close"] < c3["open"]]
        is_green = [c1["close"] > c1["open"], c2["close"] > c2["open"], c3["close"] > c3["open"]]
        
        if ml_trend == "Bullish":
            candle_pass = not all(is_red)
            detail_msg = "Safe (No consecutive 3 red candles)" if candle_pass else "Blocked (Knife Falling: 3 consecutive red candles)"
        else:
            candle_pass = not all(is_green)
            detail_msg = "Safe (No consecutive 3 green candles)" if candle_pass else "Blocked (Rocket Rising: 3 consecutive green candles)"
    except Exception as e:
        candle_pass = True
        detail_msg = f"Skipped (Not enough candles: {e})"
        
    results["Counter_Momentum"] = {
        "pass": candle_pass,
        "detail": detail_msg
    }
    if not candle_pass:
        all_pass = False

    # 7. Volatility (ATR) Safety Guard
    try:
        atr_series = df_1h["ATR_norm"]
        recent_atr = atr_series.iloc[-100:]
        p10 = float(np.percentile(recent_atr, 10))
        p90 = float(np.percentile(recent_atr, 90))
        current_atr = atr_series.iloc[-1]
        
        atr_pass = (p10 <= current_atr <= p90)
        if atr_pass:
            detail_msg = f"ATR Norm: {current_atr:.6f} (P10: {p10:.6f}, P90: {p90:.6f}, Safe)"
        elif current_atr < p10:
            detail_msg = f"ATR Norm: {current_atr:.6f} (< P10 {p10:.6f}, Market too flat)"
        else:
            detail_msg = f"ATR Norm: {current_atr:.6f} (> P90 {p90:.6f}, Volatility too extreme)"
    except Exception as e:
        atr_pass = True
        detail_msg = f"Skipped volatility guard (Error: {e})"
        
    results["Volatility_Guard"] = {
        "pass": atr_pass,
        "detail": detail_msg
    }
    if not atr_pass:
        all_pass = False

    # 8. ADX Trend Regime Check (Informational - routes dynamically to trending or ranging models)
    adx_val = df_1h["ADX"].iloc[-1]
    adx_pass = True
    results["ADX_Regime"] = {
        "pass": adx_pass,
        "detail": f"ADX is {adx_val:.2f} ({'Trending Regime' if adx_val >= 20.0 else 'Ranging Regime'})"
    }

    # 9. Fee Coverage Check (using volatility-based ATR norm to avoid model point-estimate shrinkage)
    atr_norm_val = df_1h["ATR_norm"].iloc[-1]
    fee_pass = (atr_norm_val >= 0.0025)
    results["Fee_Coverage"] = {
        "pass": fee_pass,
        "detail": f"ATR Volatility: {atr_norm_val*100:.3f}% (Req >= 0.25% to cover roundtrip Spot fees)"
    }
    if not fee_pass:
        all_pass = False

    # 10. Order Book Imbalance Check
    ob_metrics = get_orderbook_imbalance()
    ob_imbalance = ob_metrics["imbalance"]
    spread = ob_metrics["spread"]
    
    # 10a. Spread Guard
    spread_pass = (spread <= 0.001)  # Spread must be <= 0.1% to prevent entry on low liquidity / spikes
    
    # 10b. Weighted Imbalance
    if ml_trend == "Bullish":
        ob_pass = (ob_imbalance >= -0.20)
        imbalance_detail = f"Weighted Imbalance: {ob_imbalance:+.2f} (>= -0.20, Safe)" if ob_pass else f"Weighted Imbalance: {ob_imbalance:+.2f} (< -0.20, Heavy Sell)"
    else:
        ob_pass = (ob_imbalance <= 0.20)
        imbalance_detail = f"Weighted Imbalance: {ob_imbalance:+.2f} (<= +0.20, Safe)" if ob_pass else f"Weighted Imbalance: {ob_imbalance:+.2f} (> +0.20, Heavy Buy)"
        
    combined_ob_pass = ob_pass and spread_pass
    spread_detail = f"Spread: {spread*100:.3f}% (Req <= 0.10%, Safe)" if spread_pass else f"Spread: {spread*100:.3f}% (> 0.10%, High Spread)"
    
    results["Orderbook_Imbalance"] = {
        "pass": combined_ob_pass,
        "detail": f"{imbalance_detail} | {spread_detail}"
    }
    if not combined_ob_pass:
        all_pass = False

    # 11. News Sentiment Check
    is_opposed = (ml_trend == "Bullish" and news_sentiment == "Bearish") or (ml_trend == "Bearish" and news_sentiment == "Bullish")
    news_pass = not is_opposed
    results["News_Sentiment"] = {
        "pass": news_pass,
        "detail": f"Model: {ml_trend} vs News: {news_sentiment}"
    }
    if not news_pass:
        all_pass = False

    # 12. Funding Rate Guard Check (using linear perp funding rate)
    funding_rate = get_funding_rate(SYMBOL)
    if ml_trend == "Bullish":
        funding_pass = (funding_rate < 0.0005)
        detail_msg = f"Funding rate is {funding_rate*100:+.4f}% (< +0.05%, Safe)" if funding_pass else f"Funding rate is {funding_rate*100:+.4f}% (>= +0.05%, Congested Longs)"
    else:
        funding_pass = (funding_rate > -0.0005)
        detail_msg = f"Funding rate is {funding_rate*100:+.4f}% (> -0.05%, Safe)" if funding_pass else f"Funding rate is {funding_rate*100:+.4f}% (<= -0.05%, Congested Shorts)"
        
    results["Funding_Rate_Guard"] = {
        "pass": funding_pass,
        "detail": detail_msg
    }
    # Convert all results pass values and all_pass to standard python bool/str
    std_results = {}
    for key, val in results.items():
        std_results[str(key)] = {
            "pass": bool(val["pass"]),
            "detail": str(val["detail"])
        }

    return bool(all_pass), std_results

# =========================
# LIVE LOOP
# =========================
def get_fallback_price():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "spot", "symbol": SYMBOL}, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching ticker price fallback: HTTP status {response.status_code}")
            return None
        res = response.json()
        return float(res["result"]["list"][0]["lastPrice"])
    except Exception as e:
        print(f"Error fetching ticker price fallback: {e}")
        return None

def main():
    global live_price, last_ws_update_time
    load_history()
    print(f"{SYMBOL} LIVE BOT RUNNING...")
    print("Connecting to WebSocket and waiting for initial price...")

    startup_timeout = 5
    start_wait = time.time()
    while live_price is None:
        if time.time() - start_wait > startup_timeout:
            print("WebSocket connecting... Fetching ticker price from API fallback...")
            fallback = get_fallback_price()
            if fallback is not None:
                live_price = fallback
                last_ws_update_time = time.time()
            else:
                time.sleep(2)
        time.sleep(0.5)

    print(f"Initial price loaded: {live_price:.2f}")
    print(f"\n==================================================")
    print(f"👉 Local Web Dashboard is running at http://localhost:5001")
    print(f"==================================================\n")
    bot_state["live_price"] = live_price

    # Calculate calibration boundaries at startup
    p95, max_conf = calculate_historical_thresholds(models_trending["trend"])
    bot_state["calibration"] = {
        "p95": p95,
        "max_conf": max_conf,
        "mean": 54.81
    }

    print(f"Starting loop... Checking for new completed hourly candles and exit monitoring...")
    bot_state["status"] = "Running"

    last_processed_candle_timestamp = None
    active_trade = None

    while True:
        current_time = time.time()
        
        # 1. Health check & current price update
        if live_price is None or (current_time - last_ws_update_time > 15.0):
            fallback_price = get_fallback_price()
            if fallback_price is not None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket price is stale or disconnected. Fallback price: {fallback_price:.2f}")
                live_price = fallback_price
                last_ws_update_time = current_time
            
        current_price = live_price
        if current_price is None:
            print("Could not obtain price. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # 2. Check Exits if a trade is active
        if active_trade is not None:
            stop_loss = active_trade["stop_loss"]
            take_profit = active_trade["take_profit"]
            direction = active_trade["direction"]
            end_time = active_trade["end_time"]
            entry_price = active_trade["entry_price"]
            predicted_price = active_trade["predicted_price"]

            remaining_seconds = max(0, int(end_time - current_time))
            mins, secs = divmod(remaining_seconds, 60)
            countdown_str = f"{mins:02d}m {secs:02d}s"

            print(f"Active Trade: {direction} | Price: {current_price:.2f} (Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}) | Countdown: {countdown_str}")

            exit_reason = None
            if direction == "Bullish":
                if current_price <= stop_loss:
                    exit_reason = "STOP LOSS HIT [FAIL]"
                elif current_price >= take_profit:
                    exit_reason = "TAKE PROFIT HIT [SUCCESS]"
            else:
                if current_price >= stop_loss:
                    exit_reason = "STOP LOSS HIT [FAIL]"
                elif current_price <= take_profit:
                    exit_reason = "TAKE PROFIT HIT [SUCCESS]"

            if current_time >= end_time:
                exit_reason = "1-HOUR TIMER ELAPSED"

            if exit_reason is not None:
                actual_price = current_price
                price_diff = actual_price - predicted_price
                price_diff_pct = (price_diff / predicted_price) * 100
                price_accuracy = max(0.0, 100.0 - abs((actual_price - predicted_price) / actual_price * 100))
                actual_change = actual_price - entry_price
                actual_trend = "Bullish" if actual_change > 0 else "Bearish"
                signal_correct = (actual_trend == direction)
                trend_status = f"{direction} was CORRECT [OK]" if signal_correct else f"{direction} was INCORRECT [FAIL]"
                
                print("\n==================================================")
                print(f"TRADE EXITED: {exit_reason}")
                print(f"Start Price: {entry_price:.2f}")
                print(f"Predicted Price: {predicted_price:.2f}")
                print(f"Exit Price: {actual_price:.2f}")
                print(f"Actual Change: {actual_change:+.2f} ({actual_change/entry_price*100:+.4f}%)")
                print(f"Prediction Error: {price_diff:+.2f} ({price_diff_pct:+.4f}%)")
                print(f"Price Accuracy: {price_accuracy:.4f}%")
                print(f"Predicted Signal: {direction}")
                print(f"Actual Trend: {actual_trend} ({trend_status})")
                print("==================================================\n")
                
                # Update Completed Trade History in global state
                bot_state["trade_history"].append({
                    "exit_time": float(time.time()),
                    "direction": str(direction),
                    "entry_price": float(entry_price),
                    "exit_price": float(actual_price),
                    "change_pct": float((actual_change / entry_price) * 100),
                    "success": bool(signal_correct),
                    "reason": str(exit_reason)
                })
                save_history()
                active_trade = None
                bot_state["active_trade"] = None

        # 3. Check for completed hourly candle closes to search for a new signal
        try:
            df_raw = get_history(symbol=SYMBOL, interval=INTERVAL, limit=300)
            if df_raw is not None and len(df_raw) > 1:
                df_completed = df_raw.iloc[:-1].copy()
                latest_completed_ts = int(df_completed.iloc[-1]["timestamp"])

                if last_processed_candle_timestamp is None:
                    # Force immediate evaluation of the current candle on startup
                    last_processed_candle_timestamp = 0
                    print(f"Initialized completed candle timestamp tracking: {get_local_time_str(latest_completed_ts/1000)}")

                if latest_completed_ts != last_processed_candle_timestamp:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New completed 1-hour candle detected (TS: {latest_completed_ts})")
                    
                    # 1. Always evaluate ML prediction
                    df_target = df_completed.copy()
                    df_target["close_btc"] = df_target["close"]
                    df = add_features(df_target)
                    
                    latest_candle = df.iloc[-1]
                    X_live = latest_candle[features].values.reshape(1, -1)
                    
                    # Dynamic Regime Routing based on ADX
                    adx_regime = latest_candle["ADX"]
                    if adx_regime >= 20.0:
                        active_model_price = models_trending["price"]
                        active_model_trend = models_trending["trend"]
                        regime_name = "Trending (ADX >= 20)"
                    else:
                        active_model_price = models_ranging["price"]
                        active_model_trend = models_ranging["trend"]
                        regime_name = "Ranging (ADX < 20)"

                    pred_pct = float(active_model_price.predict(X_live)[0])
                    pred_change = pred_pct * float(latest_candle["close"])
                    predicted_price = float(latest_candle["close"]) + pred_change
                    prob_bullish = float(active_model_trend.predict_proba(X_live)[0][1])

                    if prob_bullish >= 0.50:
                        ml_trend = "Bullish"
                        ml_confidence = prob_bullish
                    else:
                        ml_trend = "Bearish"
                        ml_confidence = 1.0 - prob_bullish

                    calibrated_confidence = calibrate_confidence(ml_confidence, p95, max_conf)
                    expected_pct_change = (abs(pred_change) / latest_candle["close"]) * 100

                    # Update global state prediction metrics
                    bot_state["regime"] = regime_name
                    bot_state["adx"] = adx_regime
                    bot_state["latest_prediction"] = {
                        "predicted_change": pred_change,
                        "predicted_price": predicted_price,
                        "direction": ml_trend,
                        "raw_confidence": ml_confidence,
                        "calibrated_confidence": calibrated_confidence
                    }

                    print(f"Regime Selected: {regime_name} | ML Output: {ml_trend} | Raw Conf: {ml_confidence*100:.2f}% | Calibrated Conf: {calibrated_confidence*100:.2f}% | Expected Change: {pred_change:+.2f}")

                    # Determine dynamic confidence threshold based on regime and volatility
                    atr_norm_val = latest_candle["ATR_norm"]
                    dynamic_conf_threshold = 0.70
                    
                    # 1. Regime Adjustment (ADX)
                    if adx_regime >= 25.0:
                        dynamic_conf_threshold = 0.65  # lower threshold to ride strong trend early
                    elif adx_regime < 15.0:
                        dynamic_conf_threshold = 0.75  # raise threshold to filter ranging noise
                        
                    # 2. Volatility Adjustment (ATR)
                    if atr_norm_val > 0.008:
                        dynamic_conf_threshold = max(0.60, dynamic_conf_threshold - 0.05)
                    elif atr_norm_val < 0.003:
                        dynamic_conf_threshold = min(0.80, dynamic_conf_threshold + 0.05)
                        
                    print(f"Dynamic Confidence Threshold: {dynamic_conf_threshold * 100:.2f}% (Regime: {regime_name}, Volatility: {atr_norm_val * 100:.3f}%)")

                    # Determine tracking status
                    direction_conflict = (ml_trend == "Bullish" and pred_change < 0) or (ml_trend == "Bearish" and pred_change > 0)
                    
                    status_msg = "Pending"
                    
                    if active_trade is not None:
                        status_msg = "Skipped (Trade Active)"
                        print("New completed candle detected, but trade entry skipped because a trade is already active.")
                    elif direction_conflict:
                        status_msg = "Skipped (Contradiction)"
                        print(f"Prediction skipped: Directional contradiction (Trend: {ml_trend}, Regressor Price Change: {pred_change:+.2f}).")
                    elif calibrated_confidence < dynamic_conf_threshold:
                        status_msg = "Skipped (Low Confidence)"
                        print(f"Prediction skipped (calibrated confidence {calibrated_confidence*100:.2f}% < {dynamic_conf_threshold*100:.2f}%).")
                    else:
                        # Confluence checks
                        print("Triggering pre-trade confluence analysis...")
                        news_sentiment, latest_titles = get_news_sentiment()
                        all_pass, confluence_results = check_pre_trade_confluence(
                            latest_candle["close"], df, ml_trend, news_sentiment, expected_pct_change
                        )

                        # Update global confluence status
                        bot_state["confluence_results"] = {
                            "approved": all_pass,
                            "checks": confluence_results
                        }

                        print("\n==================================================")
                        print("PRE-TRADE CONFLUENCE ANALYSIS REPORT")
                        print("--------------------------------------------------")
                        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbol: {SYMBOL}")
                        print(f"Signal: {ml_trend} | Calibrated Confidence: {calibrated_confidence * 100:.2f}%")
                        print(f"Current Price: {latest_candle['close']:.2f} | Predicted Price: {predicted_price:.2f} (Expected: {pred_change:+.2f} [{expected_pct_change:.3f}%])")
                        print("--------------------------------------------------")
                        print("Checks Status:")
                        for idx, (check_name, res_val) in enumerate(confluence_results.items(), 1):
                            status_str = "[PASS]" if res_val["pass"] else "[FAIL]"
                            print(f"  {status_str} {idx}. {check_name.replace('_', ' '):<22}: {res_val['detail']}")
                        
                        if all_pass:
                            status_msg = "Traded"
                            print("--------------------------------------------------")
                            print("CONFLUENCE RESULT: APPROVED (All checks passed)")
                            print("==================================================\n")
                            
                            atr_norm_val = latest_candle["ATR_norm"]
                            atr_dollars = atr_norm_val * latest_candle["close"]
                            
                            # Regime-Adaptive Take-Profit Multiplier
                            if latest_candle["ADX"] >= 20.0:
                                tp_multiplier = 1.50
                            else:
                                tp_multiplier = 1.00
                                
                            if ml_trend == "Bullish":
                                stop_loss_price = latest_candle["close"] - 0.75 * atr_dollars
                                take_profit_price = latest_candle["close"] + tp_multiplier * atr_dollars
                            else:
                                stop_loss_price = latest_candle["close"] + 0.75 * atr_dollars
                                take_profit_price = latest_candle["close"] - tp_multiplier * atr_dollars

                            active_trade = {
                                "entry_price": float(latest_candle["close"]),
                                "predicted_price": float(predicted_price),
                                "stop_loss": float(stop_loss_price),
                                "take_profit": float(take_profit_price),
                                "direction": str(ml_trend),
                                "end_time": float(time.time() + 3600.0)
                            }
                            bot_state["active_trade"] = active_trade
                            
                            print(f"Trade Opened: {ml_trend} at price {latest_candle['close']:.2f} (SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f})\n")
                        else:
                            status_msg = "Skipped (Confluence Failed)"
                            failed_list = [name.replace('_', ' ') for name, res_val in confluence_results.items() if not res_val["pass"]]
                            print("--------------------------------------------------")
                            print(f"CONFLUENCE RESULT: REJECTED (Failed: {', '.join(failed_list)})")
                            print("==================================================\n")
                    # Prevent duplicate predictions for the same candle timestamp (e.g. on restarts)
                    exists = any(p.get("candle_timestamp") == int(latest_completed_ts) for p in bot_state["prediction_history"])
                    if not exists:
                        # Add prediction to history list in bot_state
                        bot_state["prediction_history"].append({
                            "timestamp": float(time.time()),
                            "candle_timestamp": int(latest_completed_ts),
                            "direction": str(ml_trend),
                            "ref_price": float(latest_candle["close"]),
                            "predicted_change": float(pred_change),
                            "predicted_price": float(predicted_price),
                            "status": str(status_msg),
                            "evaluation": {
                                "evaluated": False,
                                "exit_price": None,
                                "change": None,
                                "change_pct": None,
                                "success": None
                            }
                        })
                        
                        # Limit predictions history length to 200 to prevent memory leak
                        if len(bot_state["prediction_history"]) > 200:
                            bot_state["prediction_history"] = bot_state["prediction_history"][-200:]
                    else:
                        print(f"[System] Prediction for candle timestamp {get_local_time_str(latest_completed_ts/1000)} already exists in history. Skipping duplicate append.")
                    # Evaluate previous predictions
                    evaluate_predictions(df_completed)
                    save_history()
                    
                    last_processed_candle_timestamp = latest_completed_ts
        except Exception as e:
            print(f"Error checking candle close signals: {e}")

        time.sleep(10)

if __name__ == "__main__":
    import threading
    # Start Bybit WebSocket feed in a background thread
    threading.Thread(target=start_ws, daemon=True).start()
    # Start local web dashboard server in a background thread
    threading.Thread(target=run_flask, daemon=True).start()
    main()