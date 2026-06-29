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
from data import get_history, merge_derivatives_sentiment_features
import xml.etree.ElementTree as ET
from flask import Flask, jsonify, render_template

# ==========================================
# WEB DASHBOARD CONFIGURATION & STATE
# ==========================================
app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True
bot_logs = []
logs_lock = threading.Lock()
retraining_lock = threading.Lock()

bot_state = {
    "live_price": None,
    "last_update": 0.0,
    
    "active_trade_1h": None,
    "active_trade_2h": None,
    "active_trade_4h": None,
    "active_trade_6h": None,
    
    "latest_prediction_1h": None,
    "latest_prediction_2h": None,
    "latest_prediction_4h": None,
    "latest_prediction_6h": None,
    
    "confluence_results_1h": None,
    "confluence_results_2h": None,
    "confluence_results_4h": None,
    "confluence_results_6h": None,
    
    "regime_1h": "Unknown",
    "regime_2h": "Unknown",
    "regime_4h": "Unknown",
    "regime_6h": "Unknown",
    
    "adx_1h": 0.0,
    "adx_2h": 0.0,
    "adx_4h": 0.0,
    "adx_6h": 0.0,
    
    "status": "Initializing",
    "retraining_status": "Idle",
    
    "calibration_1h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
    "calibration_2h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
    "calibration_4h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
    "calibration_6h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
    
    "simulated_balance": 10000.0,
    "daily_drawdown_start_balance": 10000.0,
    "daily_drawdown_reset_day": -1,
    "circuit_breaker_active": False,
    "trade_history": [],
    "prediction_history": [],
    "win_rate_by_tf": {"60": None, "120": None, "240": None, "360": None}
}

def save_history():
    # Cap prediction history at 500 entries
    if len(bot_state["prediction_history"]) > 500:
        bot_state["prediction_history"] = bot_state["prediction_history"][-500:]
    # Recompute win rate by TF from trade history
    for tf_key in ["60", "120", "240", "360"]:
        tf_trades = [t for t in bot_state["trade_history"] if str(t.get("interval")) == tf_key]
        if tf_trades:
            wins = sum(1 for t in tf_trades if t.get("success"))
            bot_state["win_rate_by_tf"][tf_key] = round(wins / len(tf_trades) * 100, 1)
        else:
            bot_state["win_rate_by_tf"][tf_key] = None
    data = {
        "simulated_balance": bot_state["simulated_balance"],
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
                bot_state["simulated_balance"] = data.get("simulated_balance", 10000.0)
                bot_state["trade_history"] = data.get("trade_history", [])
                for t in bot_state["trade_history"]:
                    if "interval" not in t:
                        t["interval"] = "60"
                bot_state["prediction_history"] = data.get("prediction_history", [])
                for p in bot_state["prediction_history"]:
                    if "interval" not in p:
                        p["interval"] = "60"
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

@app.route("/api/retrain", methods=["POST"])
def trigger_retrain():
    started = retrain_models_thread(is_manual=True)
    if started:
        return jsonify({"status": "started", "message": "Model optimization started in background."})
    else:
        return jsonify({"status": "ignored", "message": "Optimization already in progress."}), 409

@app.route("/")
def index():
    return render_template("index.html")

def retrain_models_thread(is_manual=False):
    """
    Worker function to retrain all models (5m, 15m, 60m) inside a background thread.
    Uses retraining_lock to prevent concurrent retraining.
    """
    if not retraining_lock.acquire(blocking=False):
        print("[Retraining] Retraining is already in progress. Skipping request.")
        return False
    
    def run_training():
        global bot_state
        try:
            bot_state["retraining_status"] = "Optimizing..."
            print(f"[Retraining] Starting {'manual ' if is_manual else 'scheduled '}rolling retraining of models for 1h, 2h, 4h, and 6h intervals...")
            
            # Import train_models dynamically to avoid circular import issues
            from train import train_models
            
            # Retrain for all intervals
            for iv in ["60", "120", "240", "360"]:
                print(f"[Retraining] Retraining models for interval {iv}m...")
                train_models(interval=iv, pages=20)
                
            print("[Retraining] Rolling retraining completed successfully. Model files updated on disk.")
        except Exception as e:
            print(f"[Retraining] Error during retraining process: {e}")
        finally:
            bot_state["retraining_status"] = "Idle"
            retraining_lock.release()

    threading.Thread(target=run_training, daemon=True).start()
    return True

def run_rolling_retrain_scheduler():
    """
    Background scheduler that runs indefinitely.
    Every 1 hour, it checks if the models on disk are older than 7 days (604,800 seconds).
    If they are, or if any model file is missing, it triggers rolling retraining.
    """
    print("[Scheduler] Automated weekly rolling retraining scheduler started.")
    # Give the bot some time to initialize before running the first check
    time.sleep(30)
    
    retrain_interval_seconds = 7 * 24 * 60 * 60  # 7 days
    
    while True:
        try:
            now = time.time()
            needs_retrain = False
            
            # Check if any model file is missing or older than 7 days
            for iv in ["60", "120", "240", "360"]:
                filenames = [
                    f"xgb_trending_trend_{iv}.json",
                    f"xgb_trending_price_{iv}.json",
                    f"xgb_ranging_trend_{iv}.json",
                    f"xgb_ranging_price_{iv}.json"
                ]
                for filename in filenames:
                    if not os.path.exists(filename):
                        print(f"[Scheduler] Model file {filename} is missing. Triggering retraining.")
                        needs_retrain = True
                        break
                    else:
                        mtime = os.path.getmtime(filename)
                        age = now - mtime
                        if age > retrain_interval_seconds:
                            print(f"[Scheduler] Model file {filename} is {age/(24*3600):.1f} days old (exceeds 7 days). Triggering retraining.")
                            needs_retrain = True
                            break
                if needs_retrain:
                    break
            
            if needs_retrain:
                retrain_models_thread(is_manual=False)
                
        except Exception as e:
            print(f"[Scheduler] Error in rolling retraining scheduler: {e}")
            
        # Sleep for 1 hour before checking again
        time.sleep(3600)

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

# =========================
from xgboost import XGBClassifier, XGBRegressor
import joblib
from ensemble import load_ensemble_classifier, load_ensemble_regressor

models_by_interval = {}
model_files_mtime = {}

for iv in ["60", "120", "240", "360"]:
    models_by_interval[iv] = {
        "trending": {
            "trend": None,
            "price": None,
            "meta": None
        },
        "ranging": {
            "trend": None,
            "price": None,
            "meta": None
        }
    }

def load_model_weights(iv):
    prefixes = {
        "trending_trend": f"ensemble_trending_trend_{iv}",
        "trending_price": f"ensemble_trending_price_{iv}",
        "ranging_trend": f"ensemble_ranging_trend_{iv}",
        "ranging_price": f"ensemble_ranging_price_{iv}",
        "trending_meta": f"meta_trending_trend_{iv}.json",
        "ranging_meta": f"meta_ranging_trend_{iv}.json"
    }
    
    # Update modification times
    for key, filename in prefixes.items():
        if os.path.exists(filename):
            model_files_mtime[f"{iv}_{key}"] = os.path.getmtime(filename)
        elif os.path.exists(f"{filename}_xgb.json"):
            model_files_mtime[f"{iv}_{key}"] = os.path.getmtime(f"{filename}_xgb.json")
            
    # Load
    try:
        n_features = len(features)
        
        if os.path.exists(f"{prefixes['trending_trend']}_xgb.json"):
            models_by_interval[iv]["trending"]["trend"] = load_ensemble_classifier(prefixes["trending_trend"], n_features)
        if os.path.exists(f"{prefixes['trending_price']}_xgb.json"):
            models_by_interval[iv]["trending"]["price"] = load_ensemble_regressor(prefixes["trending_price"], n_features)
        if os.path.exists(prefixes["trending_meta"]):
            meta_clf = XGBClassifier()
            meta_clf.load_model(prefixes["trending_meta"])
            models_by_interval[iv]["trending"]["meta"] = meta_clf
            
        if os.path.exists(f"{prefixes['ranging_trend']}_xgb.json"):
            models_by_interval[iv]["ranging"]["trend"] = load_ensemble_classifier(prefixes["ranging_trend"], n_features)
        if os.path.exists(f"{prefixes['ranging_price']}_xgb.json"):
            models_by_interval[iv]["ranging"]["price"] = load_ensemble_regressor(prefixes["ranging_price"], n_features)
        if os.path.exists(prefixes["ranging_meta"]):
            meta_clf = XGBClassifier()
            meta_clf.load_model(prefixes["ranging_meta"])
            models_by_interval[iv]["ranging"]["meta"] = meta_clf
            
        print(f"Successfully loaded ensemble and meta models for interval {iv}")
    except Exception as e:
        print(f"Warning: Could not load ensemble models for interval {iv}: {e}")



def check_and_hot_reload_models():
    for iv in ["60", "120", "240", "360"]:
        filenames = {
            "trending_trend": f"ensemble_trending_trend_{iv}_xgb.json",
            "trending_price": f"ensemble_trending_price_{iv}_xgb.json",
            "trending_meta": f"meta_trending_trend_{iv}.json",
            "ranging_trend": f"ensemble_ranging_trend_{iv}_xgb.json",
            "ranging_price": f"ensemble_ranging_price_{iv}_xgb.json",
            "ranging_meta": f"meta_ranging_trend_{iv}.json"
        }
        
        changed = False
        for key, filename in filenames.items():
            if os.path.exists(filename):
                current_mtime = os.path.getmtime(filename)
                mtime_key = f"{iv}_{key}"
                if mtime_key not in model_files_mtime or current_mtime > model_files_mtime[mtime_key]:
                    changed = True
                    break
                    
        if changed:
            print(f"[Hot-Reload] Model update detected for {iv} on disk. Reloading in memory...")
            load_model_weights(iv)
            try:
                p95, max_conf = calculate_historical_thresholds(models_by_interval[iv]["trending"]["trend"], iv)
                tf_map_startup = {"60": "1h", "120": "2h", "240": "4h", "360": "6h"}
                tf_key = tf_map_startup[iv]
                bot_state[f"calibration_{tf_key}"] = {
                    "p95": p95,
                    "max_conf": max_conf,
                    "mean": 54.81
                }
                print(f"[Hot-Reload] Recalculated calibration thresholds for {iv} (p95: {p95:.2f}, max_conf: {max_conf:.2f})")
            except Exception as e:
                print(f"[Hot-Reload] Warning: Could not recalculate thresholds for {iv}m: {e}")

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
    "RSI_24", "ROC_24", "volatility_24h",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "RSI_z", "ADX_z"
]
for lag in [1, 2, 3, 4, 5]:
    features.append(f"return_5m_lag{lag}")
for lag in [1, 2, 3]:
    features.append(f"volume_ratio_lag{lag}")
for lag in [1, 2]:
    features.append(f"RSI_lag{lag}")
    features.append(f"MACD_diff_lag{lag}")
    features.append(f"BB_pct_lag{lag}")

# New features from Option 2
features.extend(["open_interest", "funding_rate", "fear_greed"])
for lag in [1, 2]:
    features.append(f"open_interest_lag{lag}")
    features.append(f"funding_rate_lag{lag}")
    features.append(f"fear_greed_lag{lag}")

# New microstructure and derivatives momentum features
features.extend([
    "open_interest_pct_change", "funding_rate_diff", 
    "CVD_rolling_1h", "CVD_rolling_4h"
])
for lag in [1, 2]:
    features.append(f"open_interest_pct_change_lag{lag}")
    features.append(f"funding_rate_diff_lag{lag}")
    features.append(f"CVD_rolling_1h_lag{lag}")
    features.append(f"CVD_rolling_4h_lag{lag}")

# Initial load
for iv in ["60", "120", "240", "360"]:
    load_model_weights(iv)

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

    # Rolling z-score normalization for RSI and ADX (200-candle window)
    for col in ["RSI", "ADX"]:
        rolling_mean = df[col].rolling(200, min_periods=20).mean()
        rolling_std = df[col].rolling(200, min_periods=20).std().replace(0, 1)
        df[f"{col}_z"] = (df[col] - rolling_mean) / rolling_std

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
    
    # Derivatives & sentiment lags
    for lag in [1, 2]:
        df[f"open_interest_lag{lag}"] = df["open_interest"].shift(lag)
        df[f"funding_rate_lag{lag}"] = df["funding_rate"].shift(lag)
        df[f"fear_greed_lag{lag}"] = df["fear_greed"].shift(lag)
        
    # Derivatives momentum
    df["open_interest_pct_change"] = df["open_interest"].pct_change(1).fillna(0.0)
    df["funding_rate_diff"] = df["funding_rate"].diff(1).fillna(0.0)
    
    # Signed volume & Cumulative Volume Delta (CVD) proxies
    signed_vol = df["volume"] * np.sign(df["close"] - df["open"])
    df["CVD_rolling_1h"] = signed_vol.rolling(window=4, min_periods=1).sum()
    df["CVD_rolling_4h"] = signed_vol.rolling(window=16, min_periods=1).sum()
    
    # Lag new features
    for lag in [1, 2]:
        df[f"open_interest_pct_change_lag{lag}"] = df["open_interest_pct_change"].shift(lag)
        df[f"funding_rate_diff_lag{lag}"] = df["funding_rate_diff"].shift(lag)
        df[f"CVD_rolling_1h_lag{lag}"] = df["CVD_rolling_1h"].shift(lag)
        df[f"CVD_rolling_4h_lag{lag}"] = df["CVD_rolling_4h"].shift(lag)
        
    # Cyclical time features
    datetime_series = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["hour_sin"] = np.sin(2 * np.pi * datetime_series.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * datetime_series.dt.hour / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * datetime_series.dt.dayofweek / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * datetime_series.dt.dayofweek / 7.0)

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
                df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=INTERVAL)
                df = add_features(df)
                return df
    except Exception as e:
        print(f"Error fetching candle data: {e}")
    return None

def get_local_time_str(t):
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

def evaluate_predictions(df_completed, interval):
    if not bot_state["prediction_history"]:
        return

    # Create a map of timestamp to close price for quick lookup
    ts_map = {}
    for _, row in df_completed.iterrows():
        ts_map[int(row["timestamp"])] = float(row["close"])

    for pred in bot_state["prediction_history"]:
        if pred.get("interval") == interval and not pred["evaluation"]["evaluated"]:
            interval_mins = int(interval)
            lookahead = 10
            target_ts = int(pred["candle_timestamp"]) + (interval_mins * 60 * 1000 * lookahead)
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
                print(f"[Prediction Tracker] Evaluated {interval}m Prediction from {get_local_time_str(pred['candle_timestamp']/1000)}: Direction: {direction} | Ref Price: {ref_price:.2f} | Exit Price: {exit_price:.2f} | Change: {change:+.2f} ({change_pct:+.3f}%) | Result: {success_str} | Status: {pred['status']}")

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

    # 1b. Fetch from CoinDesk RSS
    url_coindesk = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    try:
        res = requests.get(url_coindesk, headers=headers, timeout=10)
        if res.status_code == 200:
            xml_content = res.content.decode("utf-8")
            root = ET.fromstring(xml_content)
            coindesk_titles = []
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                if title_elem is not None and title_elem.text:
                    coindesk_titles.append(title_elem.text.strip())
            titles.extend(coindesk_titles[:10])
            print(f"[News/Sentiment] Fetched {len(coindesk_titles[:10])} articles from CoinDesk RSS.")
    except Exception as e:
        print(f"[News/Sentiment] Error fetching CoinDesk RSS: {e}")

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
        url = "https://api.bybit.com/v5/market/orderbook"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "spot", "symbol": SYMBOL, "limit": 25}, headers=headers, timeout=10)
        res = None
        if response.status_code == 200:
            res = response.json()
        else:
            print(f"[Orderbook] Bybit returned HTTP {response.status_code}. Attempting Binance depth fallback...")
            # Try Binance depth fallback
            binance_url = "https://api.binance.com/api/v3/depth"
            resp = requests.get(binance_url, params={"symbol": SYMBOL.upper(), "limit": 25}, headers=headers, timeout=10)
            if resp.status_code == 200:
                binance_data = resp.json()
                res = {
                    "result": {
                        "b": binance_data.get("bids", []),
                        "a": binance_data.get("asks", [])
                    }
                }
            else:
                print(f"[Orderbook] Binance depth fallback failed: HTTP {resp.status_code}")
                return {"imbalance": 0.0, "spread": 0.0}

        if res and "result" in res and "b" in res["result"] and "a" in res["result"]:
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
def calculate_historical_thresholds(model_trend, interval):
    print(f"Fetching historical data to calibrate confidence percentiles (last 5,000 candles for {SYMBOL} + BTCUSDT on {interval}m interval)...")
    try:
        df_target = get_history(symbol=SYMBOL, interval=interval, limit=1000, pages=5)
        df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=1000, pages=5)
        
        if df_target is not None and len(df_target) > 0 and df_btc is not None and len(df_btc) > 0:
            df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
            df = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
            if len(df) > 0:
                df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=interval)
                df = add_features(df)
                
                X_hist = df[features].values
                probs = model_trend.predict_proba(X_hist)
                confidences = np.max(probs, axis=1)
                
                p95 = float(np.percentile(confidences, 95))
                max_conf = float(np.max(confidences))
                mean_conf = float(np.mean(confidences))
                
                print(f"Confidence Calibration Done for {interval}m:")
                print(f"  - Historical Mean: {mean_conf*100:.2f}%")
                print(f"  - 95th Percentile Threshold (Maps to 80%): {p95*100:.2f}%")
                print(f"  - Maximum Confidence (Maps to 100%): {max_conf*100:.2f}%")
                return p95, max_conf
    except Exception as e:
        print(f"Error calculating calibration for {interval}m: {e}. Using defaults.")
    
    return 0.55, 0.75

def calibrate_confidence(raw_conf, p95, max_conf):
    if max_conf <= p95:
        max_conf = p95 + 0.01
    if p95 <= 0.33:
        p95 = 0.34
        
    if raw_conf < p95:
        # Piecewise linear mapping [0.33, p95] -> [50%, 80%]
        calibrated = 50.0 + (raw_conf - 0.33) / (p95 - 0.33) * 30.0
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
def check_pre_trade_confluence(current_price, df_1h, ml_trend, news_sentiment, expected_pct_change, interval):
    """
    Runs pre-trade confluence checks using a WEIGHTED SCORING SYSTEM.
    Critical checks are hard gates (instant reject if failed).
    Other checks contribute weighted points to a total score.
    Trade is approved if score >= 75% of max possible points AND no hard gate fails.
    Returns: (bool_approved, dict_results_details)
    """
    results = {}
    hard_gate_failed = False
    total_score = 0
    max_score = 0

    # ======= CHECK 1: 1-Day Structural Trend (Weight: 1, Bypassed for 5m/15m) =======
    try:
        df_1d = get_history(symbol=SYMBOL, interval="D", limit=100)
    except Exception as e:
        print(f"Error fetching 1d candle history for confluence: {e}")
        df_1d = None

    weight_1d = 1
    if str(interval) in ["5", "15"]:
        results["1d_Trend"] = {"pass": True, "detail": "Bypassed for short TF", "weight": 0}
    elif df_1d is None or len(df_1d) < 21:
        results["1d_Trend"] = {"pass": False, "detail": "Could not fetch 1d data", "weight": weight_1d}
        max_score += weight_1d
    else:
        df_1d_completed = df_1d.iloc[:-1].copy()
        ema9_1d = EMAIndicator(df_1d_completed["close"], window=9).ema_indicator().iloc[-1]
        ema21_1d = EMAIndicator(df_1d_completed["close"], window=21).ema_indicator().iloc[-1]
        trend_1d = "Bullish" if ema9_1d > ema21_1d else "Bearish"
        trend_1d_pass = (ml_trend == "Bullish" and trend_1d == "Bullish") or (ml_trend == "Bearish" and trend_1d == "Bearish")
        results["1d_Trend"] = {
            "pass": trend_1d_pass,
            "detail": f"1d Trend is {trend_1d} (EMA9: {ema9_1d:.2f}, EMA21: {ema21_1d:.2f})",
            "weight": weight_1d
        }
        max_score += weight_1d
        if trend_1d_pass:
            total_score += weight_1d

    # ======= CHECK 2: 4-Hour Tactical Trend & RSI (Weight: 1 each, Bypassed for 5m/15m) =======
    try:
        df_4h = get_history(symbol=SYMBOL, interval="240", limit=100)
    except Exception as e:
        print(f"Error fetching 4h candle history for confluence: {e}")
        df_4h = None

    weight_4h = 1
    if str(interval) in ["5", "15"]:
        results["4h_Trend"] = {"pass": True, "detail": "Bypassed for short TF", "weight": 0}
        results["4h_RSI"] = {"pass": True, "detail": "Bypassed for short TF", "weight": 0}
    elif df_4h is None or len(df_4h) < 21:
        results["4h_Trend"] = {"pass": False, "detail": "Could not fetch 4h data", "weight": weight_4h}
        results["4h_RSI"] = {"pass": False, "detail": "Could not fetch 4h data", "weight": weight_4h}
        max_score += weight_4h * 2
    else:
        df_4h_completed = df_4h.iloc[:-1].copy()
        ema9_4h = EMAIndicator(df_4h_completed["close"], window=9).ema_indicator().iloc[-1]
        ema21_4h = EMAIndicator(df_4h_completed["close"], window=21).ema_indicator().iloc[-1]
        rsi_4h = RSIIndicator(df_4h_completed["close"], window=14).rsi().iloc[-1]

        trend_4h = "Bullish" if ema9_4h > ema21_4h else "Bearish"
        trend_pass = (ml_trend == "Bullish" and trend_4h == "Bullish") or (ml_trend == "Bearish" and trend_4h == "Bearish")
        results["4h_Trend"] = {
            "pass": trend_pass,
            "detail": f"4h Trend is {trend_4h} (EMA9: {ema9_4h:.2f}, EMA21: {ema21_4h:.2f})",
            "weight": weight_4h
        }
        max_score += weight_4h
        if trend_pass:
            total_score += weight_4h

        if ml_trend == "Bullish":
            rsi_4h_pass = (rsi_4h < 70.0)
            detail_msg = f"4h RSI is {rsi_4h:.2f} (< 70, Safe)" if rsi_4h_pass else f"4h RSI is {rsi_4h:.2f} (>= 70, Overbought)"
        else:
            rsi_4h_pass = (rsi_4h > 30.0)
            detail_msg = f"4h RSI is {rsi_4h:.2f} (> 30, Safe)" if rsi_4h_pass else f"4h RSI is {rsi_4h:.2f} (<= 30, Oversold)"
        results["4h_RSI"] = {"pass": rsi_4h_pass, "detail": detail_msg, "weight": weight_4h}
        max_score += weight_4h
        if rsi_4h_pass:
            total_score += weight_4h

    # ======= CHECK 3: 1h RSI — HARD GATE (extreme overbought/oversold) =======
    rsi_1h = df_1h["RSI"].iloc[-1]
    if ml_trend == "Bullish":
        rsi_1h_pass = (rsi_1h < 70.0)
        detail_msg = f"1h RSI is {rsi_1h:.2f} (< 70, Safe)" if rsi_1h_pass else f"1h RSI is {rsi_1h:.2f} (>= 70, Overbought)"
    else:
        rsi_1h_pass = (rsi_1h > 30.0)
        detail_msg = f"1h RSI is {rsi_1h:.2f} (> 30, Safe)" if rsi_1h_pass else f"1h RSI is {rsi_1h:.2f} (<= 30, Oversold)"
    results["1h_RSI"] = {"pass": rsi_1h_pass, "detail": detail_msg + " [HARD GATE]", "weight": "HARD"}
    if not rsi_1h_pass:
        hard_gate_failed = True

    # ======= CHECK 4: Volume Participation (Weight: 2) =======
    weight_vol = 2
    try:
        vol_series = df_1h["volume"]
        avg_vol_20 = vol_series.iloc[:-1].rolling(20).mean().iloc[-1]
        latest_vol = vol_series.iloc[-2]
        volume_pass = (latest_vol >= 0.8 * avg_vol_20)
        results["Volume_Participation"] = {
            "pass": volume_pass,
            "detail": f"Vol: {latest_vol:.1f} vs Avg20: {avg_vol_20:.1f} ({latest_vol/avg_vol_20*100:.1f}%, Req >= 80%)",
            "weight": weight_vol
        }
    except Exception as e:
        volume_pass = True
        results["Volume_Participation"] = {"pass": True, "detail": f"Skipped volume check (Error: {e})", "weight": weight_vol}
    max_score += weight_vol
    if volume_pass:
        total_score += weight_vol

    # ======= CHECK 5: Bollinger Band Edge Guard (Weight: 2) =======
    weight_bb = 2
    bb_pct_val = df_1h["BB_pct"].iloc[-1]
    if ml_trend == "Bullish":
        bb_pass = (bb_pct_val < 0.95)
        detail_msg = f"BB Pct is {bb_pct_val:.3f} (< 0.95, Room to run)" if bb_pass else f"BB Pct is {bb_pct_val:.3f} (>= 0.95, Overextended long)"
    else:
        bb_pass = (bb_pct_val > 0.05)
        detail_msg = f"BB Pct is {bb_pct_val:.3f} (> 0.05, Room to run)" if bb_pass else f"BB Pct is {bb_pct_val:.3f} (<= 0.05, Overextended short)"
    results["BB_Edge_Guard"] = {"pass": bb_pass, "detail": detail_msg, "weight": weight_bb}
    max_score += weight_bb
    if bb_pass:
        total_score += weight_bb

    # ======= CHECK 6: Counter-Momentum Guard (Weight: 2) =======
    weight_cm = 2
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
    results["Counter_Momentum"] = {"pass": candle_pass, "detail": detail_msg, "weight": weight_cm}
    max_score += weight_cm
    if candle_pass:
        total_score += weight_cm

    # ======= CHECK 7: Volatility (ATR) Safety Guard (Weight: 2) =======
    weight_atr = 2
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
    results["Volatility_Guard"] = {"pass": atr_pass, "detail": detail_msg, "weight": weight_atr}
    max_score += weight_atr
    if atr_pass:
        total_score += weight_atr

    # ======= CHECK 8: ADX Regime (Informational only — Weight: 0, always passes) =======
    adx_val = df_1h["ADX"].iloc[-1]
    results["ADX_Regime"] = {
        "pass": True,
        "detail": f"ADX is {adx_val:.2f} ({'Trending Regime' if adx_val >= 20.0 else 'Ranging Regime'}) [INFO]",
        "weight": 0
    }

    # ======= CHECK 9: Fee Coverage — HARD GATE =======
    atr_norm_val = df_1h["ATR_norm"].iloc[-1]
    if str(interval) in ["5", "15"]:
        fee_pass = (atr_norm_val >= 0.0010)
        req_str = ">= 0.10%"
    else:
        fee_pass = (atr_norm_val >= 0.0015)
        req_str = ">= 0.15%"
    results["Fee_Coverage"] = {
        "pass": fee_pass,
        "detail": f"ATR Volatility: {atr_norm_val*100:.3f}% (Req {req_str} to cover roundtrip Spot fees) [HARD GATE]",
        "weight": "HARD"
    }
    if not fee_pass:
        hard_gate_failed = True

    # ======= CHECK 10: Order Book Imbalance & Spread (Weight: 1) =======
    weight_ob = 1
    ob_metrics = get_orderbook_imbalance()
    ob_imbalance = ob_metrics["imbalance"]
    spread = ob_metrics["spread"]
    spread_pass = (spread <= 0.001)
    if str(interval) in ["5", "15"]:
        ob_pass = True
        imbalance_detail = f"Weighted Imbalance: {ob_imbalance:+.2f} (Bypassed for short TF)"
    elif ml_trend == "Bullish":
        ob_pass = (ob_imbalance >= -0.20)
        imbalance_detail = f"Weighted Imbalance: {ob_imbalance:+.2f} (>= -0.20, Safe)" if ob_pass else f"Weighted Imbalance: {ob_imbalance:+.2f} (< -0.20, Heavy Sell)"
    else:
        ob_pass = (ob_imbalance <= 0.20)
        imbalance_detail = f"Weighted Imbalance: {ob_imbalance:+.2f} (<= +0.20, Safe)" if ob_pass else f"Weighted Imbalance: {ob_imbalance:+.2f} (> +0.20, Heavy Buy)"
    combined_ob_pass = ob_pass and spread_pass
    spread_detail = f"Spread: {spread*100:.3f}% (Req <= 0.10%, Safe)" if spread_pass else f"Spread: {spread*100:.3f}% (> 0.10%, High Spread)"
    results["Orderbook_Imbalance"] = {
        "pass": combined_ob_pass,
        "detail": f"{imbalance_detail} | {spread_detail}",
        "weight": weight_ob
    }
    max_score += weight_ob
    if combined_ob_pass:
        total_score += weight_ob

    # ======= CHECK 11: News Sentiment (Weight: 1, Bypassed for 5m/15m) =======
    weight_news = 1
    is_opposed = (ml_trend == "Bullish" and news_sentiment == "Bearish") or (ml_trend == "Bearish" and news_sentiment == "Bullish")
    if str(interval) in ["5", "15"]:
        news_pass = True
        detail_msg = f"Model: {ml_trend} vs News: {news_sentiment} (Bypassed for short TF)"
        results["News_Sentiment"] = {"pass": news_pass, "detail": detail_msg, "weight": 0}
    else:
        news_pass = not is_opposed
        detail_msg = f"Model: {ml_trend} vs News: {news_sentiment}"
        results["News_Sentiment"] = {"pass": news_pass, "detail": detail_msg, "weight": weight_news}
        max_score += weight_news
        if news_pass:
            total_score += weight_news

    # ======= CHECK 12: Expected Price Change Threshold (Weight: 2) =======
    weight_exp = 2
    min_pct_map = {"5": 0.10, "15": 0.15, "60": 0.25}
    req_pct = min_pct_map.get(str(interval), 0.15)
    change_pass = (expected_pct_change >= req_pct)
    results["Expected_Change"] = {
        "pass": change_pass,
        "detail": f"Expected price change is {expected_pct_change:.3f}% (Req >= {req_pct:.2f}% to trade)",
        "weight": weight_exp
    }
    max_score += weight_exp
    if change_pass:
        total_score += weight_exp

    # ======= CHECK 13: Timeframe Trend Alignment (Weight: 2) =======
    weight_align = 2
    trend_align_pass = True
    align_detail = "Aligned with dominant trend"
    if str(interval) in ["5", "15"]:
        try:
            df_1h_align = get_history(symbol=SYMBOL, interval="60", limit=100)
            if df_1h_align is not None and len(df_1h_align) >= 21:
                df_1h_align_completed = df_1h_align.iloc[:-1].copy()
                ema9_1h = EMAIndicator(df_1h_align_completed["close"], window=9).ema_indicator().iloc[-1]
                ema21_1h = EMAIndicator(df_1h_align_completed["close"], window=21).ema_indicator().iloc[-1]
                trend_1h = "Bullish" if ema9_1h > ema21_1h else "Bearish"
                if ml_trend == "Bullish" and trend_1h != "Bullish":
                    trend_align_pass = False
                    align_detail = f"Blocked (5m/15m Bullish signal contradicts 1h Bearish trend)"
                elif ml_trend == "Bearish" and trend_1h != "Bearish":
                    trend_align_pass = False
                    align_detail = f"Blocked (5m/15m Bearish signal contradicts 1h Bullish trend)"
                else:
                    align_detail = f"Aligned with 1h {trend_1h} trend"
            else:
                align_detail = "Could not fetch 1h trend data (Bypassed)"
        except Exception as e:
            align_detail = f"Skipped trend alignment check (Error: {e})"
    else:
        align_detail = f"1h interval is already the dominant trend"
    results["Timeframe_Alignment"] = {"pass": trend_align_pass, "detail": align_detail, "weight": weight_align}
    max_score += weight_align
    if trend_align_pass:
        total_score += weight_align

    # ======= FINAL SCORING =======
    score_pct = (total_score / max_score * 100) if max_score > 0 else 100.0
    score_threshold = 75.0
    approved = (not hard_gate_failed) and (score_pct >= score_threshold)

    # Add score summary to results
    results["_Score_Summary"] = {
        "pass": approved,
        "detail": f"Score: {total_score}/{max_score} ({score_pct:.0f}%) | Threshold: {score_threshold:.0f}% | Hard Gates: {'FAILED' if hard_gate_failed else 'PASSED'}",
        "weight": "SUMMARY"
    }

    # Convert all results pass values to standard python bool/str
    std_results = {}
    for key, val in results.items():
        std_results[str(key)] = {
            "pass": bool(val["pass"]),
            "detail": str(val["detail"])
        }

    return bool(approved), std_results

# =========================
# LIVE LOOP
# =========================
def get_fallback_price():
    # 1. Try Bybit API
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "spot", "symbol": SYMBOL}, headers=headers, timeout=5)
        if response.status_code == 200:
            res = response.json()
            return float(res["result"]["list"][0]["lastPrice"])
        else:
            print(f"Bybit price ticker returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Bybit price fallback: {e}")

    # 2. Try Coinbase API (very permissive, no API key needed)
    try:
        response = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=5)
        if response.status_code == 200:
            res = response.json()
            return float(res["data"]["amount"])
        else:
            print(f"Coinbase price ticker returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Coinbase price fallback: {e}")

    # 3. Try Binance API
    try:
        response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
        if response.status_code == 200:
            res = response.json()
            return float(res["price"])
        else:
            print(f"Binance price ticker returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Binance price fallback: {e}")

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

    # Calculate calibration boundaries at startup for each interval
    tf_map_startup = {"60": "1h", "120": "2h", "240": "4h", "360": "6h"}
    for iv in ["60", "120", "240", "360"]:
        if iv in models_by_interval:
            p95, max_conf = calculate_historical_thresholds(models_by_interval[iv]["trending"]["trend"], iv)
            tf_key = tf_map_startup[iv]
            bot_state[f"calibration_{tf_key}"] = {
                "p95": p95,
                "max_conf": max_conf,
                "mean": 54.81
            }

    print(f"Starting loop... Checking for new completed candle signals and exit monitoring...")
    bot_state["status"] = "Running"

    last_processed_timestamps = {
        "last_processed_60_ts": None,
        "last_processed_120_ts": None,
        "last_processed_240_ts": None,
        "last_processed_360_ts": None
    }

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

        # 2. Check Exits for each timeframe if a trade is active
        tf_map = {"60": "1h", "120": "2h", "240": "4h", "360": "6h"}
        for iv in ["60", "120", "240", "360"]:
            tf = tf_map[iv]
            active_trade_key = f"active_trade_{tf}"
            active_trade = bot_state[active_trade_key]
            
            if active_trade is not None:
                stop_loss = active_trade["stop_loss"]
                take_profit = active_trade["take_profit"]
                direction = active_trade["direction"]
                end_time = active_trade["end_time"]
                entry_price = active_trade["entry_price"]
                predicted_price = active_trade["predicted_price"]

                # Trailing stop and break-even variables
                atr_dollars = active_trade.get("atr_dollars", 50.0)
                highest_price = active_trade.get("highest_price", entry_price)
                lowest_price = active_trade.get("lowest_price", entry_price)
                break_even_triggered = active_trade.get("break_even_triggered", False)
                position_size_usd = active_trade.get("position_size_usd", 100.0)

                # Update trailing stop peak prices
                if direction == "Bullish":
                    if current_price > highest_price:
                        highest_price = current_price
                        active_trade["highest_price"] = highest_price
                        # Trailing Stop: SL trails highest price by 1.25 * ATR
                        potential_sl = highest_price - 1.25 * atr_dollars
                        if potential_sl > stop_loss:
                            stop_loss = potential_sl
                            active_trade["stop_loss"] = stop_loss
                            print(f"[{iv}m Trailing Stop] Moved SL up to {stop_loss:.2f} (trailing highest: {highest_price:.2f})")
                    
                    # Break-Even Guard: if price goes up by 0.5 * ATR, move SL to entry
                    if not break_even_triggered and current_price >= entry_price + 0.5 * atr_dollars:
                        break_even_triggered = True
                        active_trade["break_even_triggered"] = True
                        stop_loss = max(stop_loss, entry_price)
                        active_trade["stop_loss"] = stop_loss
                        print(f"[{iv}m Break-Even Guard] Triggered! SL moved to entry price: {entry_price:.2f}")
                else:
                    if current_price < lowest_price:
                        lowest_price = current_price
                        active_trade["lowest_price"] = lowest_price
                        # Trailing Stop: SL trails lowest price by 1.25 * ATR
                        potential_sl = lowest_price + 1.25 * atr_dollars
                        if potential_sl < stop_loss:
                            stop_loss = potential_sl
                            active_trade["stop_loss"] = stop_loss
                            print(f"[{iv}m Trailing Stop] Moved SL down to {stop_loss:.2f} (trailing lowest: {lowest_price:.2f})")
                            
                    # Break-Even Guard: if price goes down by 0.5 * ATR, move SL to entry
                    if not break_even_triggered and current_price <= entry_price - 0.5 * atr_dollars:
                        break_even_triggered = True
                        active_trade["break_even_triggered"] = True
                        stop_loss = min(stop_loss, entry_price)
                        active_trade["stop_loss"] = stop_loss
                        print(f"[{iv}m Break-Even Guard] Triggered! SL moved to entry price: {entry_price:.2f}")

                remaining_seconds = max(0, int(end_time - current_time))
                mins, secs = divmod(remaining_seconds, 60)
                countdown_str = f"{mins:02d}m {secs:02d}s"

                print(f"[{iv}m Active Trade] {direction} | Price: {current_price:.2f} (Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}) | Countdown: {countdown_str}")

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
                    lookahead = 10
                    exit_reason = f"{int(iv)*lookahead}-MINUTE TIMER ELAPSED"

                if exit_reason is not None:
                    actual_price = current_price
                    price_diff = actual_price - predicted_price
                    price_diff_pct = (price_diff / predicted_price) * 100
                    price_accuracy = max(0.0, 100.0 - abs((actual_price - predicted_price) / actual_price * 100))
                    actual_change = actual_price - entry_price
                    actual_change_pct = (actual_change / entry_price) * 100
                    
                    # Calculate PnL (long vs short) and simulated fees
                    raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
                    net_return_pct = raw_return_pct - 0.2  # 0.2% roundtrip Spot fee
                    realized_pnl = position_size_usd * (net_return_pct / 100.0)
                    
                    # Update simulated balance
                    old_bal = bot_state.get("simulated_balance", 10000.0)
                    new_bal = old_bal + realized_pnl
                    bot_state["simulated_balance"] = new_bal
                    
                    actual_trend = "Bullish" if actual_change > 0 else "Bearish"
                    signal_correct = (actual_trend == direction)
                    trend_status = f"{direction} was CORRECT [OK]" if signal_correct else f"{direction} was INCORRECT [FAIL]"
                    
                    print("\n==================================================")
                    print(f"[{iv}m TRADE EXITED]: {exit_reason}")
                    print(f"Start Price: {entry_price:.2f} | Exit Price: {actual_price:.2f}")
                    print(f"Actual Change: {actual_change:+.2f} ({actual_change_pct:+.4f}%)")
                    print(f"Size: ${position_size_usd:.2f} | Net Return: {net_return_pct:+.4f}% (after 0.2% fees)")
                    print(f"Realized PnL: ${realized_pnl:+.2f} | New Balance: ${new_bal:.2f}")
                    print(f"Predicted Signal: {direction} ({trend_status})")
                    print("==================================================\n")
                    
                    # Update Completed Trade History in global state
                    bot_state["trade_history"].append({
                        "exit_time": float(time.time()),
                        "interval": str(iv),
                        "direction": str(direction),
                        "entry_price": float(entry_price),
                        "exit_price": float(actual_price),
                        "change_pct": float(net_return_pct),
                        "success": bool(signal_correct),
                        "reason": str(exit_reason),
                        "position_size_usd": float(position_size_usd),
                        "pnl_usd": float(realized_pnl),
                        "balance": float(new_bal)
                    })
                    save_history()
                    bot_state[active_trade_key] = None

        # 3. Check for completed candle closes to search for a new signal

        # --- Daily Drawdown Circuit Breaker ---
        today = datetime.now().day
        if bot_state["daily_drawdown_reset_day"] != today:
            bot_state["daily_drawdown_start_balance"] = bot_state.get("simulated_balance", 10000.0)
            bot_state["daily_drawdown_reset_day"] = today
            bot_state["circuit_breaker_active"] = False
            print(f"[Circuit Breaker] Daily reset. Start balance: ${bot_state['daily_drawdown_start_balance']:.2f}")
        else:
            start_bal = bot_state["daily_drawdown_start_balance"]
            curr_bal = bot_state.get("simulated_balance", start_bal)
            daily_dd_pct = (start_bal - curr_bal) / start_bal * 100 if start_bal > 0 else 0
            if daily_dd_pct >= 5.0 and not bot_state["circuit_breaker_active"]:
                bot_state["circuit_breaker_active"] = True
                print(f"[Circuit Breaker] ACTIVATED — daily drawdown {daily_dd_pct:.2f}% >= 5%. Trading paused for today.")
            elif daily_dd_pct < 5.0:
                bot_state["circuit_breaker_active"] = False

        # --- High-Impact News Window Guard ---
        def is_high_impact_news_window():
            """Returns True if within 15 minutes of a known high-impact event (CPI, FOMC, NFP)."""
            try:
                now_utc = datetime.utcnow()
                # Use investing.com economic calendar RSS or finnhub — use finnhub free tier
                resp = requests.get(
                    "https://finnhub.io/api/v1/calendar/economic",
                    params={"token": "free"},
                    timeout=5
                )
                if resp.status_code == 200:
                    events = resp.json().get("economicCalendar", [])
                    high_impact = ["CPI", "FOMC", "NFP", "Non-Farm", "Federal Reserve", "Interest Rate"]
                    for ev in events:
                        if any(kw.lower() in ev.get("event", "").lower() for kw in high_impact):
                            ev_time_str = ev.get("time", "")
                            try:
                                ev_time = datetime.strptime(ev_time_str, "%Y-%m-%d %H:%M:%S")
                                diff = abs((now_utc - ev_time).total_seconds())
                                if diff <= 900:  # 15 minute window
                                    return True, ev.get("event", "Unknown")
                            except Exception:
                                pass
            except Exception:
                pass
            return False, None

        check_and_hot_reload_models()
        for iv in ["60", "120", "240", "360"]:

            tf = tf_map[iv]
            try:
                df_raw = get_history(symbol=SYMBOL, interval=iv, limit=300)
                if df_raw is not None and len(df_raw) > 1:
                    df_completed = df_raw.iloc[:-1].copy()
                    latest_completed_ts = int(df_completed.iloc[-1]["timestamp"])

                    last_ts_key = f"last_processed_{iv}_ts"
                    if last_processed_timestamps[last_ts_key] is None:
                        last_processed_timestamps[last_ts_key] = 0
                        print(f"Initialized completed candle timestamp tracking for {iv}m: {get_local_time_str(latest_completed_ts/1000)}")

                    if latest_completed_ts != last_processed_timestamps[last_ts_key]:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New completed {iv}-minute candle detected (TS: {latest_completed_ts})")
                        
                        df_target = df_completed.copy()
                        df_target["close_btc"] = df_target["close"]
                        df_target = merge_derivatives_sentiment_features(df_target, symbol=SYMBOL, interval=iv)
                        df = add_features(df_target)
                        if len(df) == 0:
                            print(f"[{iv}m] Error: Dataframe became empty after feature engineering.")
                            continue
                        
                        latest_candle = df.iloc[-1]
                        X_live = latest_candle[features].values.reshape(1, -1)
                        
                        # Dynamic Regime Routing based on ADX
                        adx_regime = latest_candle["ADX"]
                        
                        if iv in models_by_interval:
                            models_tf = models_by_interval[iv]
                            if adx_regime >= 20.0:
                                active_model_price = models_tf["trending"]["price"]
                                active_model_trend = models_tf["trending"]["trend"]
                                regime_name = "Trending (ADX >= 20)"
                            else:
                                active_model_price = models_tf["ranging"]["price"]
                                active_model_trend = models_tf["ranging"]["trend"]
                                regime_name = "Ranging (ADX < 20)"

                            pred_pct = float(active_model_price.predict(X_live)[0])
                            pred_change = pred_pct * float(latest_candle["close"])
                            predicted_price = float(latest_candle["close"]) + pred_change
                            
                            # 3-class probabilities
                            probs = active_model_trend.predict_proba(X_live)[0]
                            prob_bearish = float(probs[0])
                            prob_neutral = float(probs[1])
                            prob_bullish = float(probs[2])
                            
                            winning_class = int(np.argmax(probs))
                            
                            if winning_class == 2:
                                ml_trend = "Bullish"
                                ml_confidence = prob_bullish
                            elif winning_class == 0:
                                ml_trend = "Bearish"
                                ml_confidence = prob_bearish
                            else:
                                ml_trend = "Neutral"
                                ml_confidence = prob_neutral

                            calibration = bot_state[f"calibration_{tf}"]
                            p95 = calibration["p95"]
                            max_conf = calibration["max_conf"]
                            calibrated_confidence = calibrate_confidence(ml_confidence, p95, max_conf)
                            expected_pct_change = (abs(pred_change) / latest_candle["close"]) * 100

                            # Update global state prediction metrics for this timeframe
                            bot_state[f"regime_{tf}"] = regime_name
                            bot_state[f"adx_{tf}"] = adx_regime
                            bot_state[f"latest_prediction_{tf}"] = {
                                "predicted_change": pred_change,
                                "predicted_price": predicted_price,
                                "direction": ml_trend,
                                "raw_confidence": ml_confidence,
                                "calibrated_confidence": calibrated_confidence
                            }

                            print(f"[{iv}m] Regime Selected: {regime_name} | ML Output: {ml_trend} (Bull: {prob_bullish*100:.1f}%, Bear: {prob_bearish*100:.1f}%, Neut: {prob_neutral*100:.1f}%) | Raw Conf: {ml_confidence*100:.2f}% | Calibrated Conf: {calibrated_confidence*100:.2f}% | Expected Change: {pred_change:+.2f}")

                            # Determine dynamic confidence threshold based on regime and volatility
                            atr_norm_val = latest_candle["ATR_norm"]
                            dynamic_conf_threshold = 0.55
                            
                            # 1. Regime Adjustment (ADX)
                            if adx_regime >= 25.0:
                                dynamic_conf_threshold = 0.50
                            elif adx_regime < 15.0:
                                dynamic_conf_threshold = 0.60
                                
                            # 2. Volatility Adjustment (ATR)
                            if atr_norm_val > 0.008:
                                dynamic_conf_threshold = max(0.45, dynamic_conf_threshold - 0.05)
                            elif atr_norm_val < 0.003:
                                dynamic_conf_threshold = min(0.65, dynamic_conf_threshold + 0.05)
                                
                            print(f"[{iv}m] Dynamic Confidence Threshold: {dynamic_conf_threshold * 100:.2f}% (Regime: {regime_name}, Volatility: {atr_norm_val * 100:.3f}%)")

                            # Meta-Classifier: Use as confidence MODIFIER instead of hard gate
                            meta_adjustment = 0.0
                            if ml_trend in ["Bullish", "Bearish"]:
                                active_meta_model = models_tf["trending"]["meta"] if adx_regime >= 20.0 else models_tf["ranging"]["meta"]
                                if active_meta_model is not None:
                                    meta_pred = int(active_meta_model.predict(X_live)[0])
                                    if meta_pred == 1:
                                        meta_adjustment = +0.05  # Meta predicts success: boost confidence
                                        print(f"[{iv}m] Meta-Classifier: PASS (confidence boosted +5%)")
                                    else:
                                        meta_adjustment = -0.10  # Meta predicts failure: reduce confidence
                                        print(f"[{iv}m] Meta-Classifier: FAIL (confidence reduced -10%)")
                                    calibrated_confidence = max(0.0, min(1.0, calibrated_confidence + meta_adjustment))
                                    print(f"[{iv}m] Adjusted Calibrated Confidence: {calibrated_confidence*100:.2f}%")

                            # Determine tracking status
                            # Softened contradiction: only block if regressor predicts > 0.05% in OPPOSITE direction
                            pred_pct = (abs(pred_change) / latest_candle["close"]) * 100
                            strong_conflict = (ml_trend == "Bullish" and pred_change < 0 and pred_pct > 0.05) or \
                                              (ml_trend == "Bearish" and pred_change > 0 and pred_pct > 0.05)
                            
                            status_msg = "Pending"
                            active_trade_key = f"active_trade_{tf}"
                            active_trade = bot_state[active_trade_key]

                            if active_trade is not None:
                                status_msg = "Skipped (Trade Active)"
                                print(f"[{iv}m] New completed candle detected, but trade entry skipped because a trade is already active.")
                            elif bot_state.get("circuit_breaker_active", False):
                                status_msg = "Skipped (Circuit Breaker)"
                                print(f"[{iv}m] Prediction skipped: Daily drawdown circuit breaker is active. Trading paused for today.")
                            elif ml_trend == "Neutral":
                                status_msg = "Skipped (Neutral)"
                                print(f"[{iv}m] Prediction skipped: Model output is Neutral/Hold.")
                            elif strong_conflict:
                                status_msg = "Skipped (Contradiction)"
                                print(f"[{iv}m] Prediction skipped: Strong directional contradiction (Trend: {ml_trend}, Regressor: {pred_change:+.2f} [{pred_pct:.3f}%]).")
                            elif calibrated_confidence < dynamic_conf_threshold:
                                status_msg = "Skipped (Low Confidence)"
                                print(f"[{iv}m] Prediction skipped (calibrated confidence {calibrated_confidence*100:.2f}% < {dynamic_conf_threshold*100:.2f}%).")
                            else:
                                # Check news window guard before running full confluence
                                in_news_window, news_event = is_high_impact_news_window()
                                if in_news_window:
                                    status_msg = "Skipped (News Window)"
                                    print(f"[{iv}m] Prediction skipped: High-impact event window ({news_event}). Trading paused.")
                                else:
                                    # Confluence checks
                                    print(f"[{iv}m] Triggering pre-trade confluence analysis...")
                                    news_sentiment, latest_titles = get_news_sentiment()
                                    all_pass, confluence_results = check_pre_trade_confluence(
                                        latest_candle["close"], df, ml_trend, news_sentiment, expected_pct_change, iv
                                    )

                                    # Update global confluence status
                                    bot_state[f"confluence_results_{tf}"] = {
                                        "approved": all_pass,
                                        "checks": confluence_results
                                    }

                                    print(f"\n==================================================")
                                    print(f"[{iv}m] PRE-TRADE CONFLUENCE ANALYSIS REPORT")
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
                                        print(f"CONFLUENCE RESULT: APPROVED ({confluence_results.get('_Score_Summary', {}).get('detail', 'Score check passed')})")
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

                                        # Kelly Criterion position sizing calculation
                                        kelly_b = tp_multiplier / 0.75
                                        kelly_p = float(calibrated_confidence)
                                        f_star = (kelly_p * (kelly_b + 1) - 1) / kelly_b if kelly_b > 0 else 0
                                        kelly_fraction = max(0.01, min(0.20, 0.25 * f_star))
                                        current_bal = bot_state.get("simulated_balance", 10000.0)
                                        position_size_usd = current_bal * kelly_fraction

                                        lookahead = 10
                                        duration_seconds = int(iv) * 60.0 * lookahead
                                        active_trade = {
                                            "entry_price": float(latest_candle["close"]),
                                            "predicted_price": float(predicted_price),
                                            "stop_loss": float(stop_loss_price),
                                            "take_profit": float(take_profit_price),
                                            "direction": str(ml_trend),
                                            "end_time": float(time.time() + duration_seconds),
                                            "atr_dollars": float(atr_dollars),
                                            "highest_price": float(latest_candle["close"]),
                                            "lowest_price": float(latest_candle["close"]),
                                            "break_even_triggered": False,
                                            "position_size_usd": float(position_size_usd),
                                            "kelly_fraction": float(kelly_fraction)
                                        }
                                        bot_state[active_trade_key] = active_trade
                                        
                                        print(f"[{iv}m] Trade Opened: {ml_trend} at price {latest_candle['close']:.2f} (SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f})")
                                        print(f"[{iv}m Kelly Sizing] Confidence: {kelly_p*100:.2f}% | R:R ratio: {kelly_b:.2f} | Kelly allocation: {kelly_fraction*100:.2f}% | Size: ${position_size_usd:.2f}\n")
                                    else:
                                        status_msg = "Skipped (Confluence Failed)"
                                        failed_list = [name.replace('_', ' ') for name, res_val in confluence_results.items() if not res_val["pass"] and name != '_Score_Summary']
                                        print("--------------------------------------------------")
                                        print(f"CONFLUENCE RESULT: REJECTED ({confluence_results.get('_Score_Summary', {}).get('detail', 'Score too low')})")
                                        print(f"Failed checks: {', '.join(failed_list)}")
                                        print("==================================================\n")
                            
                            # Prevent duplicate predictions for the same candle timestamp
                            exists = any(p.get("candle_timestamp") == int(latest_completed_ts) and p.get("interval") == iv for p in bot_state["prediction_history"])
                            if not exists:
                                bot_state["prediction_history"].append({
                                    "timestamp": float(time.time()),
                                    "candle_timestamp": int(latest_completed_ts),
                                    "interval": str(iv),
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
                                
                                if len(bot_state["prediction_history"]) > 200:
                                    bot_state["prediction_history"] = bot_state["prediction_history"][-200:]
                            else:
                                print(f"[{iv}m] Prediction for candle timestamp {get_local_time_str(latest_completed_ts/1000)} already exists in history. Skipping duplicate append.")
                            
                            evaluate_predictions(df_completed, iv)
                            save_history()
                            
                            last_processed_timestamps[last_ts_key] = latest_completed_ts
            except Exception as e:
                print(f"Error checking {iv}m candle close signals: {e}")

        time.sleep(10)

if __name__ == "__main__":
    import threading
    # Start Bybit WebSocket feed in a background thread
    threading.Thread(target=start_ws, daemon=True).start()
    # Start local web dashboard server in a background thread
    threading.Thread(target=run_flask, daemon=True).start()
    # Start automated rolling retraining scheduler in a background thread
    threading.Thread(target=run_rolling_retrain_scheduler, daemon=True).start()
    main()