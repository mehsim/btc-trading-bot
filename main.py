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
from datetime import datetime, timedelta

def get_pkt_time():
    return datetime.utcnow() + timedelta(hours=5)

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
from data import get_history, merge_derivatives_sentiment_features
import xml.etree.ElementTree as ET
from flask import Flask, jsonify, render_template, request

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
    "live_price_BTCUSDT": None,
    "live_price_ETHUSDT": None,
    "live_price_SOLUSDT": None,
    "live_price_BNBUSDT": None,
    "live_price_ADAUSDT": None,
    "live_price_XRPUSDT": None,
    "live_price_AVAXUSDT": None,
    "live_price_NEARUSDT": None,
    "live_price_LINKUSDT": None,
    "live_price_LTCUSDT": None,
    "live_price_DOGEUSDT": None,
    "last_update": 0.0,
    
    "active_trade_1h": [],
    "active_trade_2h": [],
    "active_trade_4h": [],
    "active_trade_6h": [],
    
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
    
    "simulated_balance": 80.0,
    "daily_drawdown_start_balance": 80.0,
    "daily_drawdown_reset_day": -1,
    "circuit_breaker_active": False,
    "bot_running": True,
    "trade_history": [],
    "prediction_history": [],
    "win_rate_by_tf": {"60": None, "120": None, "240": None, "360": None}
}

cached_news_sentiment = "Neutral"
cached_news_titles = []
news_sentiment_lock = threading.Lock()

economic_calendar_cache = None
economic_calendar_lock = threading.Lock()

HISTORY_FILE = "/data/dashboard_history.json" if os.path.exists("/data") and os.access("/data", os.W_OK) else "dashboard_history.json"

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
        "prediction_history": bot_state["prediction_history"],
        "active_trade_1h": bot_state.get("active_trade_1h", []),
        "active_trade_2h": bot_state.get("active_trade_2h", []),
        "active_trade_4h": bot_state.get("active_trade_4h", []),
        "active_trade_6h": bot_state.get("active_trade_6h", []),
        "bot_running": bot_state.get("bot_running", True),
        "fresh_reset_v3": bot_state.get("fresh_reset_v3", False)
    }
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f)
            
        # If running on Hugging Face and write token is available, backup to HF Dataset
        token = os.environ.get("HF_TOKEN") or os.environ.get("token")
        space_id = os.environ.get("SPACE_ID")
        if token and space_id:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                dataset_id = f"{space_id}-history"
                api.create_repo(repo_id=dataset_id, repo_type="dataset", exist_ok=True, token=token)
                api.upload_file(
                    path_or_fileobj=HISTORY_FILE,
                    path_in_repo="dashboard_history.json",
                    repo_id=dataset_id,
                    repo_type="dataset",
                    token=token
                )
            except Exception as hf_err:
                print(f"HF Space Sync: Failed to backup history to Dataset: {hf_err}")
    except Exception as e:
        print(f"Error saving history to disk: {e}")

def load_history():
    token = os.environ.get("HF_TOKEN") or os.environ.get("token")
    space_id = os.environ.get("SPACE_ID")
    
    # 1. If running on HF and token is available, restore history from HF Dataset
    if space_id and token:
        try:
            from huggingface_hub import hf_hub_download
            dataset_id = f"{space_id}-history"
            print(f"[Sync] Attempting to download history from Dataset {dataset_id}...")
            downloaded_path = hf_hub_download(
                repo_id=dataset_id,
                filename="dashboard_history.json",
                repo_type="dataset",
                token=token
            )
            import shutil
            shutil.copy(downloaded_path, HISTORY_FILE)
            print("[Sync] Successfully restored history from Hugging Face Dataset.")
        except Exception as hf_err:
            print(f"[Sync] Could not restore history from HF Dataset (normal if first run): {hf_err}")

    # 2. Sync from Hugging Face Space if running locally (not in Space environment itself)
    elif not space_id:
        try:
            print("Syncing: Attempting to pull latest history from Hugging Face Space API...")
            resp = requests.get("https://mehsimleo-btc-trading-bot.hf.space/api/status", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                hf_trades = data.get("trade_history", [])
                hf_predictions = data.get("prediction_history", [])
                hf_balance = data.get("simulated_balance", 80.0)
                
                # Filter out old 5m and 15m intervals
                hf_trades = [t for t in hf_trades if str(t.get("interval", "60")) not in ["5", "15"]]
                hf_predictions = [p for p in hf_predictions if str(p.get("interval", "60")) not in ["5", "15"]]
                
                if len(hf_trades) > 0 or len(hf_predictions) > 0:
                    bot_state["simulated_balance"] = hf_balance
                    bot_state["trade_history"] = hf_trades
                    bot_state["prediction_history"] = hf_predictions
                    bot_state["active_trade_1h"] = data.get("active_trade_1h", [])
                    bot_state["active_trade_2h"] = data.get("active_trade_2h", [])
                    bot_state["active_trade_4h"] = data.get("active_trade_4h", [])
                    bot_state["active_trade_6h"] = data.get("active_trade_6h", [])
                    bot_state["bot_running"] = data.get("bot_running", True)
                    bot_state["fresh_reset_v3"] = data.get("fresh_reset_v3", False)
                    print(f"Sync Success: Loaded {len(hf_trades)} trades and {len(hf_predictions)} predictions from Hugging Face Space.")
                    save_history()
                    return
        except Exception as e:
            print(f"HF Space Sync: Could not fetch from Hugging Face Space: {e}")

    # 2. Local/Persistent history fallback load
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                bot_state["simulated_balance"] = data.get("simulated_balance", 80.0)
                bot_state["trade_history"] = [t for t in data.get("trade_history", []) if str(t.get("interval", "60")) not in ["5", "15"]]
                for t in bot_state["trade_history"]:
                    if "interval" not in t:
                        t["interval"] = "60"
                bot_state["prediction_history"] = [p for p in data.get("prediction_history", []) if str(p.get("interval", "60")) not in ["5", "15"]]
                for p in bot_state["prediction_history"]:
                    if "interval" not in p:
                        p["interval"] = "60"
                
                # Load active trades
                bot_state["active_trade_1h"] = data.get("active_trade_1h", [])
                bot_state["active_trade_2h"] = data.get("active_trade_2h", [])
                bot_state["active_trade_4h"] = data.get("active_trade_4h", [])
                bot_state["active_trade_6h"] = data.get("active_trade_6h", [])
                bot_state["bot_running"] = data.get("bot_running", True)
                bot_state["fresh_reset_v3"] = data.get("fresh_reset_v3", False)
                print(f"Loaded {len(bot_state['trade_history'])} trades and {len(bot_state['prediction_history'])} predictions from {HISTORY_FILE}")
        except Exception as e:
            print(f"Error loading history from disk: {e}")

    # Force auto-reset if it's the first time running this updated version
    if not bot_state.get("fresh_reset_v3", False):
        print("[System Reset] Migrating history to fresh reset v3. Setting balance to 80.0 and clearing all old trades.")
        bot_state["simulated_balance"] = 80.0
        bot_state["daily_drawdown_start_balance"] = 80.0
        bot_state["trade_history"] = []
        bot_state["prediction_history"] = []
        bot_state["active_trade_1h"] = []
        bot_state["active_trade_2h"] = []
        bot_state["active_trade_4h"] = []
        bot_state["active_trade_6h"] = []
        bot_state["fresh_reset_v3"] = True
        save_history()
        
    bot_state["retraining_status"] = "Idle"

# Thread-safe print wrapper to redirect logs to dashboard log panel
_print = print
def print(*args, **kwargs):
    _print(*args, **kwargs)
    if "file" not in kwargs or kwargs["file"] is None:
        msg = " ".join(map(str, args))
        timestamp = get_pkt_time().strftime("%H:%M:%S")
        lines = msg.split('\n')
        with logs_lock:
            for line in lines:
                if line.strip(): # ignore empty lines in console
                    bot_logs.append(f"[{timestamp}] {line}")
            # Keep history to 200 lines
            if len(bot_logs) > 200:
                bot_logs[:] = bot_logs[-200:]

_last_balance_fetch = 0.0
_cached_balance = None
_balance_lock = threading.Lock()

def get_bybit_proxies():
    import os
    proxy = os.environ.get("BYBIT_PROXY")
    if proxy:
        return {
            "http": proxy,
            "https": proxy
        }
    return None

def get_bybit_time_offset():
    import requests
    import time
    try:
        resp = requests.get("https://api.bybit.com/v5/market/time", proxies=get_bybit_proxies(), timeout=3)
        if resp.status_code == 200:
            server_time = int(resp.json()["result"]["timeNano"]) // 1000000
            local_time = int(time.time() * 1000)
            return server_time - local_time
    except Exception:
        pass
    return 0

def get_real_bybit_balance():
    import os
    import hmac
    import hashlib
    import time
    import requests
    
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    
    if not api_key or not api_secret:
        return "API_KEYS_MISSING"
        
    offset = get_bybit_time_offset()
    timestamp = str(int(time.time() * 1000) + offset)
    recv_window = "5000"
    
    max_balance = 0.0
    geo_blocked_encountered = False
    
    for account_type in ["UNIFIED", "CONTRACT", "SPOT", "FUND"]:
        if account_type == "FUND":
            query_string = "accountType=FUND"
            url = f"https://api.bybit.com/v5/asset/transfer/query-account-coins-balance?{query_string}"
        else:
            query_string = f"accountType={account_type}"
            url = f"https://api.bybit.com/v5/account/wallet-balance?{query_string}"
            
        val_str = timestamp + api_key + recv_window + query_string
        sign = hmac.new(
            api_secret.encode("utf-8"),
            val_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-SIGN": sign,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }
        
        try:
            resp = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                ret_code = data.get("retCode")
                if ret_code == 0:
                    if account_type == "FUND":
                        balances = data.get("result", {}).get("balance", [])
                        fund_sum = 0.0
                        for b_item in balances:
                            coin_name = b_item.get("coin", "")
                            coin_bal = float(b_item.get("walletBalance", "0"))
                            if coin_name in ["USDT", "USDC"]:
                                fund_sum += coin_bal
                            elif coin_name == "BTC":
                                fund_sum += coin_bal * float(get_fallback_price("BTCUSDT") or 60000.0)
                            elif coin_name == "ETH":
                                fund_sum += coin_bal * float(get_fallback_price("ETHUSDT") or 33000.0)
                            elif coin_name == "SOL":
                                fund_sum += coin_bal * float(get_fallback_price("SOLUSDT") or 140.0)
                        max_balance = max(max_balance, fund_sum)
                    else:
                        list_data = data.get("result", {}).get("list", [])
                        if list_data:
                            total_equity = list_data[0].get("totalEquity") or list_data[0].get("totalWalletBalance") or "0"
                            max_balance = max(max_balance, float(total_equity))
                else:
                    # Ignore the expected "accountType only support UNIFIED" error (Code 10001) for SPOT/CONTRACT to avoid log clutter
                    if not (ret_code == 10001 and account_type in ["SPOT", "CONTRACT"]):
                        print(f"[Bybit Balance] Query error for {account_type}: Code {ret_code} - {data.get('retMsg')}")
            else:
                print(f"[Bybit Balance] HTTP {resp.status_code} for {account_type}: {resp.text}")
                if resp.status_code == 403 and ("cloudfront" in resp.text.lower() or "block" in resp.text.lower()):
                    geo_blocked_encountered = True
        except Exception as e:
            print(f"[Bybit Balance] Exception during {account_type} fetch: {e}")
            
    if max_balance > 0.0:
        return max_balance
    if geo_blocked_encountered:
        return "GEO_BLOCKED"
    return 0.0

def get_real_bybit_balance_cached():
    with _balance_lock:
        return _cached_balance

def run_bybit_balance_updater():
    global _cached_balance, _last_balance_fetch
    print("[Bybit Balance] Background updater thread started.")
    # Fetch immediately at startup
    try:
        val = get_real_bybit_balance()
        with _balance_lock:
            _cached_balance = val
            _last_balance_fetch = time.time()
        print(f"[Bybit Balance] Startup background update success: {val}")
    except Exception as e:
        print(f"[Bybit Balance] Startup background update error: {e}")
        
    while True:
        time.sleep(20)
        try:
            val = get_real_bybit_balance()
            with _balance_lock:
                _cached_balance = val
                _last_balance_fetch = time.time()
        except Exception as e:
            print(f"[Bybit Balance] Error in background balance update: {e}")

@app.route("/api/status")
def get_status():
    # Thread-safe dictionary copy with fallback
    for _ in range(5):
        try:
            state_copy = bot_state.copy()
            break
        except RuntimeError:
            time.sleep(0.01)
    else:
        state_copy = {k: v for k, v in list(bot_state.items())}

    with logs_lock:
        state_copy["logs"] = list(bot_logs)
    
    # Inject cached real Bybit balance
    state_copy["real_bybit_balance"] = get_real_bybit_balance_cached()
    
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

@app.route("/api/close_trade", methods=["POST"])
def force_close_trade():
    data = request.json or {}
    interval = str(data.get("interval", ""))
    symbol = str(data.get("symbol", "")).upper()
    
    tf_map_local = {"60": "1h", "120": "2h", "240": "4h", "360": "6h"}
    tf = tf_map_local.get(interval)
    if not tf:
        return jsonify({"status": "error", "message": "Invalid interval specified."}), 400
        
    active_trade_key = f"active_trade_{tf}"
    active_trades_list = bot_state.get(active_trade_key, [])
    if not isinstance(active_trades_list, list):
        active_trades_list = [] if active_trades_list is None else [active_trades_list]
        bot_state[active_trade_key] = active_trades_list

    # Find the trade for the specified symbol or trade_id
    trade_id = data.get("trade_id")
    trade_to_close = None
    if trade_id:
        for t in active_trades_list:
            if t.get("trade_id") == trade_id:
                trade_to_close = t
                break
    if not trade_to_close:
        for t in active_trades_list:
            if t.get("symbol", "").upper() == symbol:
                trade_to_close = t
                break
                
    if not trade_to_close:
        # Fallback if no symbol specified, close the first one
        if len(active_trades_list) > 0:
            trade_to_close = active_trades_list[0]
            symbol = trade_to_close.get("symbol", "BTCUSDT")
        else:
            return jsonify({"status": "error", "message": f"No active trade found for {tf}."}), 400
            
    # Exiting trade manually
    entry_price = trade_to_close["entry_price"]
    direction = trade_to_close["direction"]
    position_size_usd = trade_to_close.get("position_size_usd", 100.0)
    
    live_symbol_price = get_fallback_price(symbol)
    if live_symbol_price is None:
        live_symbol_price = bot_state.get(f"live_price_{symbol}")
    actual_exit_price = live_symbol_price if live_symbol_price is not None else entry_price
    
    # Maker execution: zero slippage on limit close
    slippage_pct = 0.0
    actual_price = actual_exit_price
        
    actual_change = actual_price - entry_price
    actual_change_pct = (actual_change / entry_price) * 100
    
    raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
    leverage = trade_to_close.get("leverage", 1.0)
    net_return_pct = (raw_return_pct * leverage) - 0.04  # 0.04% maker roundtrip fee
    realized_pnl = position_size_usd * (net_return_pct / 100.0)
    
    old_bal = bot_state.get("simulated_balance", 80.0)
    new_bal = round(old_bal + position_size_usd + realized_pnl, 2)
    bot_state["simulated_balance"] = new_bal
    
    actual_trend = "Bullish" if actual_change > 0 else "Bearish"
    signal_correct = (actual_trend == direction)
    trend_status = f"{direction} was CORRECT [OK]" if signal_correct else f"{direction} was INCORRECT [FAIL]"
    
    exit_reason = "Manual Exit (Force Closed)"
    
    print("\n==================================================")
    print(f"[{symbol} {tf.upper()} MANUAL EXIT]: {exit_reason} (Slippage: {slippage_pct:.3f}%)")
    print(f"Start Price: {entry_price:.2f} | Exit Price: {actual_price:.2f}")
    print(f"Actual Change: {actual_change:+.2f} ({actual_change_pct:+.4f}%)")
    print(f"Size: ${position_size_usd:.2f} | Net Return: {net_return_pct:+.4f}% (after 0.2% fees with {leverage}x leverage)")
    print(f"Realized PnL: ${realized_pnl:+.2f} | New Balance: ${new_bal:.2f}")
    print(f"Predicted Signal: {direction} ({trend_status})")
    print("==================================================\n")
    
    bot_state["trade_history"].append({
        "symbol": symbol,
        "exit_time": float(time.time()),
        "interval": interval,
        "direction": direction,
        "entry_price": float(entry_price),
        "exit_price": float(actual_price),
        "change_pct": float(net_return_pct),
        "success": bool(signal_correct),
        "reason": exit_reason,
        "position_size_usd": float(position_size_usd),
        "pnl_usd": float(realized_pnl),
        "balance": float(new_bal),
        "leverage": float(leverage)
    })
    
    for p in bot_state["prediction_history"]:
        if p.get("interval") == interval and p.get("symbol") == symbol and p.get("status") == "Traded" and (not p.get("evaluation") or not p["evaluation"].get("evaluated")):
            p["evaluation"] = {
                "evaluated": True,
                "exit_price": float(actual_price),
                "change": float(actual_change if direction == "Bullish" else -actual_change),
                "change_pct": float(raw_return_pct),
                "success": bool(signal_correct)
            }
            break
            
    save_history()
    
    # Remove from active trades
    if trade_id:
        active_trades_list = [t for t in active_trades_list if t.get("trade_id") != trade_id]
    else:
        active_trades_list = [t for t in active_trades_list if not (t.get("symbol", "").upper() == symbol)]
    bot_state[active_trade_key] = active_trades_list
    
    return jsonify({"status": "success", "message": f"Successfully force-closed {symbol} {tf.upper()} trade at ${actual_price:.2f}"})

def close_all_trades_internal(exit_reason):
    closed_count = 0
    tf_map_local = {"60": "1h", "120": "2h", "240": "4h", "360": "6h"}
    
    # Iterate over all active trade timeframes
    for tf_key in ["1h", "2h", "4h", "6h"]:
        active_trade_key = f"active_trade_{tf_key}"
        active_trades = bot_state.get(active_trade_key, [])
        if not isinstance(active_trades, list):
            active_trades = [] if active_trades is None else [active_trades]
            bot_state[active_trade_key] = active_trades

        # Close each trade in this list
        for t in list(active_trades):
            symbol = t.get("symbol", "BTCUSDT").upper()
            direction = t.get("direction", "Bullish")
            entry_price = t.get("entry_price", 0.0)
            position_size_usd = t.get("position_size_usd", 100.0)
            trade_id = t.get("trade_id")
            
            # Fetch interval number (e.g. "60") corresponding to tf_key
            interval = "60"
            for k, v in tf_map_local.items():
                if v == tf_key:
                    interval = k
                    break

            live_symbol_price = get_fallback_price(symbol)
            if live_symbol_price is None:
                live_symbol_price = bot_state.get(f"live_price_{symbol}")
            actual_exit_price = live_symbol_price if live_symbol_price is not None else entry_price
            
            # Maker execution: zero slippage on limit close
            slippage_pct = 0.0
            actual_price = actual_exit_price
                
            actual_change = actual_price - entry_price
            actual_change_pct = (actual_change / entry_price) * 100 if entry_price > 0 else 0.0
            
            raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
            leverage = t.get("leverage", 1.0)
            net_return_pct = (raw_return_pct * leverage) - 0.04  # 0.04% maker roundtrip fee
            realized_pnl = position_size_usd * (net_return_pct / 100.0)
            
            old_bal = bot_state.get("simulated_balance", 80.0)
            new_bal = round(old_bal + position_size_usd + realized_pnl, 2)
            bot_state["simulated_balance"] = new_bal
            
            actual_trend = "Bullish" if actual_change > 0 else "Bearish"
            signal_correct = (actual_trend == direction)
            trend_status = f"{direction} was CORRECT [OK]" if signal_correct else f"{direction} was INCORRECT [FAIL]"
            
            print("\n==================================================")
            print(f"[{symbol} {tf_key.upper()} MANUAL EXIT ALL]: {exit_reason} (Slippage: {slippage_pct:.3f}%)")
            print(f"Start Price: {entry_price:.2f} | Exit Price: {actual_price:.2f}")
            print(f"Actual Change: {actual_change:+.2f} ({actual_change_pct:+.4f}%)")
            print(f"Size: ${position_size_usd:.2f} | Net Return: {net_return_pct:+.4f}% (after 0.2% fees with {leverage}x leverage)")
            print(f"Realized PnL: ${realized_pnl:+.2f} | New Balance: ${new_bal:.2f}")
            print(f"Predicted Signal: {direction} ({trend_status})")
            print("==================================================\n")
            
            bot_state["trade_history"].append({
                "symbol": symbol,
                "exit_time": float(time.time()),
                "interval": interval,
                "direction": direction,
                "entry_price": float(entry_price),
                "exit_price": float(actual_price),
                "change_pct": float(net_return_pct),
                "success": bool(signal_correct),
                "reason": exit_reason,
                "position_size_usd": float(position_size_usd),
                "pnl_usd": float(realized_pnl),
                "balance": float(new_bal),
                "leverage": float(leverage)
            })
            
            for p in bot_state["prediction_history"]:
                if p.get("interval") == interval and p.get("symbol") == symbol and p.get("status") == "Traded" and (not p.get("evaluation") or not p["evaluation"].get("evaluated")):
                    p["evaluation"] = {
                        "evaluated": True,
                        "exit_price": float(actual_price),
                        "change": float(actual_change if direction == "Bullish" else -actual_change),
                        "change_pct": float(raw_return_pct),
                        "success": bool(signal_correct)
                    }
                    break
            
            closed_count += 1
            
        # Clear this active trades list
        bot_state[active_trade_key] = []
        
    if closed_count > 0:
        save_history()
    return closed_count

@app.route("/api/close_all_trades", methods=["POST"])
def force_close_all_trades():
    closed_count = close_all_trades_internal("Manual Exit (Force Closed All)")
    if closed_count > 0:
        return jsonify({"status": "success", "message": f"Successfully force-closed all {closed_count} open trades."})
    else:
        return jsonify({"status": "success", "message": "No active open trades found to close."})

@app.route("/api/toggle_bot", methods=["POST"])
def toggle_bot():
    current_status = bot_state.get("bot_running", True)
    new_status = not current_status
    bot_state["bot_running"] = new_status
    
    message = ""
    if not new_status:
        closed_count = close_all_trades_internal("Manual Exit (Bot Stopped)")
        message = f"Bot stopped successfully. Closed {closed_count} open trades."
    else:
        message = "Bot is now running."
        
    save_history()
    return jsonify({"status": "success", "bot_running": new_status, "message": message})

@app.route("/api/reset_circuit_breaker", methods=["POST"])
def reset_circuit_breaker():
    bot_state["circuit_breaker_active"] = False
    bot_state["daily_drawdown_start_balance"] = bot_state.get("simulated_balance", 80.0)
    save_history()
    return jsonify({"status": "success", "message": "Daily drawdown circuit breaker successfully reset. Trading resumed!"})

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
            # Sync latest predictions and trade history from Hugging Face Space first
            load_history()
            bot_state["retraining_status"] = "Optimizing..."
            print(f"[Retraining] Starting {'manual ' if is_manual else 'scheduled '}rolling retraining of models for 1h, 2h, 4h, and 6h intervals...")
            
            # Import train_models dynamically to avoid circular import issues
            from train import train_models
            
            # Retrain for all intervals
            for iv in ["60", "120", "240", "360"]:
                print(f"[Retraining] Retraining models for interval {iv}m...")
                train_models(interval=iv, pages=5)
                
            print("[Retraining] Rolling retraining completed successfully. Model files updated on disk.")
        except Exception as e:
            print(f"[Retraining] Error during retraining process: {e}")
        finally:
            bot_state["retraining_status"] = "Idle"
            save_history()
            retraining_lock.release()

    threading.Thread(target=run_training, daemon=True).start()
    return True

def run_rolling_retrain_scheduler():
    """
    Background scheduler that runs indefinitely.
    Every 1 hour, it checks if the models on disk are older than 3 days (259,200 seconds).
    If they are, or if any model file is missing, it triggers rolling retraining.
    """
    print("[Scheduler] Automated 72-hour rolling retraining scheduler started.")
    # Give the bot some time to initialize before running the first check
    time.sleep(30)
    
    retrain_interval_seconds = 3 * 24 * 60 * 60  # 3 days
    
    while True:
        try:
            now = time.time()
            needs_retrain = False
            
            # Check if any model file is missing or older than 3 days
            for iv in ["60", "120", "240", "360"]:
                filenames = [
                    f"ensemble_trending_trend_{iv}_xgb.json",
                    f"ensemble_trending_price_{iv}_xgb.json",
                    f"ensemble_ranging_trend_{iv}_xgb.json",
                    f"ensemble_ranging_price_{iv}_xgb.json",
                    f"meta_trending_trend_{iv}.json",
                    f"meta_ranging_trend_{iv}.json"
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
                            print(f"[Scheduler] Model file {filename} is {age/(24*3600):.1f} days old (exceeds 3 days). Triggering retraining.")
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
INTERVAL = "60"
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "NEARUSDT", "LINKUSDT", "LTCUSDT", "DOGEUSDT"]

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
            "meta": None,
            "calibrator": None
        },
        "ranging": {
            "trend": None,
            "price": None,
            "meta": None,
            "calibrator": None
        },
        "selected_features": None
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
        # Load selected features if they exist
        selected_features_filename = f"selected_features_{iv}.json"
        if os.path.exists(selected_features_filename):
            with open(selected_features_filename, "r") as f:
                selected_features_list = json.load(f)
            models_by_interval[iv]["selected_features"] = selected_features_list
            n_features = len(selected_features_list)
            print(f"Loaded {n_features} selected features for interval {iv}")
        else:
            models_by_interval[iv]["selected_features"] = None
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
            
        # Load calibrators if they exist
        trending_cal_file = f"calibrator_trending_{iv}.json"
        if os.path.exists(trending_cal_file):
            with open(trending_cal_file, "r") as f:
                models_by_interval[iv]["trending"]["calibrator"] = json.load(f)
            print(f"Loaded Isotonic Regression calibrator: {trending_cal_file}")
        else:
            models_by_interval[iv]["trending"]["calibrator"] = None

        ranging_cal_file = f"calibrator_ranging_{iv}.json"
        if os.path.exists(ranging_cal_file):
            with open(ranging_cal_file, "r") as f:
                models_by_interval[iv]["ranging"]["calibrator"] = json.load(f)
            print(f"Loaded Isotonic Regression calibrator: {ranging_cal_file}")
        else:
            models_by_interval[iv]["ranging"]["calibrator"] = None
            
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

def run_fallback_price_updater():
    """
    Periodic thread that queries Bybit REST API for spot prices.
    Acts as a failover if WebSocket is geoblocked or disconnected.
    """
    global live_price, last_ws_update_time
    print("[Price Fallback] Background updater thread started.")
    while True:
        try:
            # If WebSocket hasn't updated in the last 12 seconds, query REST ticker
            if (time.time() - last_ws_update_time) > 12.0:
                url = "https://api.bybit.com/v5/market/tickers"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                resp = requests.get(url, params={"category": "spot"}, headers=headers, proxies=get_bybit_proxies(), timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    ticker_list = data.get("result", {}).get("list", [])
                    for ticker in ticker_list:
                        sym = ticker.get("symbol")
                        if sym in SUPPORTED_SYMBOLS:
                            val_str = ticker.get("lastPrice")
                            if val_str:
                                val = float(val_str)
                                bot_state[f"live_price_{sym}"] = val
                                if sym == "BTCUSDT":
                                    live_price = val
                                    bot_state["live_price"] = val
                                    bot_state["last_update"] = time.time()
                else:
                    pass
        except Exception:
            pass
        time.sleep(6)

def send_email_notification(subject, body):
    """
    Sends an email alert using SMTP.
    Reads SMTP configuration from environment variables.
    """
    import smtplib
    from email.mime.text import MIMEText
    
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        smtp_port = 587
        
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    email_to = os.getenv("EMAIL_TO", "mehsimleo@gmail.com")
    
    if not smtp_user or not smtp_password:
        print("[Email Notification] Skipped: SMTP_USER or SMTP_PASSWORD environment variables not set.")
        return False
        
    try:
        msg = MIMEText(body, "html")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = email_to
        
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            print(f"[Email Notification] Successfully sent email to {email_to}")
            return True
    except Exception as e:
        print(f"[Email Notification] Error sending email: {e}")
        return False

def on_message(ws, message):
    global live_price, last_ws_update_time
    try:
        data = json.loads(message)
        if "data" in data and isinstance(data["data"], dict):
            price_str = data["data"].get("lastPrice")
            sym = data["data"].get("symbol")
            if price_str and sym:
                val = float(price_str)
                bot_state[f"live_price_{sym}"] = val
                if sym == "BTCUSDT":
                    live_price = val
                    bot_state["live_price"] = val
                    last_ws_update_time = time.time()
                    bot_state["last_update"] = last_ws_update_time
    except Exception:
        pass

def on_open(ws):
    print("Connected to Bybit WebSocket for multi-asset prices")
    ws.send(json.dumps({
        "op": "subscribe",
        "args": [f"tickers.{s}" for s in SUPPORTED_SYMBOLS]
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
    "RSI_z", "ADX_z", "close_to_Kalman"
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

# New Wick Volume features (absorption/liquidation proxies)
features.extend([
    "upper_wick_volume_ratio", "lower_wick_volume_ratio"
])
for lag in [1, 2]:
    features.append(f"upper_wick_volume_ratio_lag{lag}")
    features.append(f"lower_wick_volume_ratio_lag{lag}")

features.append("hours_to_news")

# Initial load
for iv in ["60", "120", "240", "360"]:
    load_model_weights(iv)

try:
    from numba import jit
    @jit(nopython=True, cache=True)
    def _kalman_loop(prices, state_estimate, error_covariance, process_variance, measurement_variance):
        n = len(prices)
        for i in range(1, n):
            pred_state = state_estimate[i-1]
            pred_error = error_covariance[i-1] + process_variance
            kalman_gain = pred_error / (pred_error + measurement_variance)
            state_estimate[i] = pred_state + kalman_gain * (prices[i] - pred_state)
            error_covariance[i] = (1.0 - kalman_gain) * pred_error
        return state_estimate
except ImportError:
    def _kalman_loop(prices, state_estimate, error_covariance, process_variance, measurement_variance):
        n = len(prices)
        for i in range(1, n):
            pred_state = state_estimate[i-1]
            pred_error = error_covariance[i-1] + process_variance
            kalman_gain = pred_error / (pred_error + measurement_variance)
            state_estimate[i] = pred_state + kalman_gain * (prices[i] - pred_state)
            error_covariance[i] = (1.0 - kalman_gain) * pred_error
        return state_estimate

def calculate_kalman_feature(prices):
    n = len(prices)
    if n == 0:
        return np.zeros(0)
    state_estimate = np.zeros(n)
    error_covariance = np.zeros(n)
    state_estimate[0] = prices[0]
    error_covariance[0] = 1.0
    
    process_variance = 1e-4
    measurement_variance = 1e-2
    
    state_estimate = _kalman_loop(prices, state_estimate, error_covariance, process_variance, measurement_variance)
    return (prices / (state_estimate + 1e-8)) - 1.0

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
    
    # High-fidelity Delta Volume / CVD Proxy
    high_low_range = df["high"] - df["low"] + 1e-8
    delta_volume = df["volume"] * (2 * (df["close"] - df["low"]) / high_low_range - 1.0)
    df["CVD_rolling_1h"] = delta_volume.rolling(window=4, min_periods=1).sum()
    df["CVD_rolling_4h"] = delta_volume.rolling(window=16, min_periods=1).sum()
    
    # Wick Volume (Liquidation & Stop-Loss Sweep Proxies)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    
    upper_wick_vol = df["volume"] * (upper_wick / high_low_range)
    lower_wick_vol = df["volume"] * (lower_wick / high_low_range)
    
    df["upper_wick_volume_ratio"] = upper_wick_vol / (upper_wick_vol.rolling(20).mean() + 1e-8)
    df["lower_wick_volume_ratio"] = lower_wick_vol / (lower_wick_vol.rolling(20).mean() + 1e-8)
    
    # Lag new features
    for lag in [1, 2]:
        df[f"open_interest_pct_change_lag{lag}"] = df["open_interest_pct_change"].shift(lag)
        df[f"funding_rate_diff_lag{lag}"] = df["funding_rate_diff"].shift(lag)
        df[f"CVD_rolling_1h_lag{lag}"] = df["CVD_rolling_1h"].shift(lag)
        df[f"CVD_rolling_4h_lag{lag}"] = df["CVD_rolling_4h"].shift(lag)
        df[f"upper_wick_volume_ratio_lag{lag}"] = df["upper_wick_volume_ratio"].shift(lag)
        df[f"lower_wick_volume_ratio_lag{lag}"] = df["lower_wick_volume_ratio"].shift(lag)
        
    # Cyclical time features
    datetime_series = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["hour_sin"] = np.sin(2 * np.pi * datetime_series.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * datetime_series.dt.hour / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * datetime_series.dt.dayofweek / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * datetime_series.dt.dayofweek / 7.0)

    # 1D Kalman Filter trend feature
    df["close_to_Kalman"] = calculate_kalman_feature(df["close"].values)

    df = add_news_proximity_feature(df)
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
    # Pakistan timezone is UTC + 5 hours (18000 seconds)
    return datetime.utcfromtimestamp(t + 18000).strftime('%Y-%m-%d %H:%M:%S')

def evaluate_predictions(df_completed, interval, symbol):
    if not bot_state["prediction_history"]:
        return

    # Create a map of timestamp to close price for quick lookup
    ts_map = {}
    for _, row in df_completed.iterrows():
        ts_map[int(row["timestamp"])] = float(row["close"])

    for pred in bot_state["prediction_history"]:
        eval_dict = pred.get("evaluation")
        if eval_dict is None:
            eval_dict = {
                "evaluated": False,
                "exit_price": None,
                "change": None,
                "change_pct": None,
                "success": None
            }
            pred["evaluation"] = eval_dict
            
        if pred.get("interval") == interval and pred.get("symbol", "BTCUSDT") == symbol and not eval_dict.get("evaluated"):
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

def run_news_sentiment_updater():
    global cached_news_sentiment, cached_news_titles
    print("[News/Sentiment] Background updater thread started.")
    try:
        sentiment, titles = get_news_sentiment()
        with news_sentiment_lock:
            cached_news_sentiment = sentiment
            cached_news_titles = titles
        print(f"[News/Sentiment] Startup background update success: {sentiment} (based on {len(titles)} inputs).")
    except Exception as e:
        print(f"[News/Sentiment] Startup background update error: {e}")
        
    while True:
        time.sleep(15 * 60)
        try:
            print("[News/Sentiment] Triggering periodic background news sentiment update...")
            sentiment, titles = get_news_sentiment()
            with news_sentiment_lock:
                cached_news_sentiment = sentiment
                cached_news_titles = titles
            print(f"[News/Sentiment] Background update success: {sentiment} (based on {len(titles)} inputs).")
        except Exception as e:
            print(f"[News/Sentiment] Error in background news sentiment update: {e}")

def fetch_economic_calendar_cached(start_ts_ms=None, end_ts_ms=None):
    global economic_calendar_cache
    with economic_calendar_lock:
        if economic_calendar_cache is not None:
            return economic_calendar_cache
            
        try:
            finnhub_token = os.environ.get("FINNHUB_TOKEN", "free")
            from datetime import datetime, timedelta
            now = datetime.utcnow()
            if start_ts_ms:
                from_dt = datetime.utcfromtimestamp(start_ts_ms / 1000.0)
            else:
                from_dt = now - timedelta(days=60)
                
            if end_ts_ms:
                to_dt = datetime.utcfromtimestamp(end_ts_ms / 1000.0) + timedelta(days=2)
            else:
                to_dt = now + timedelta(days=7)
                
            from_str = from_dt.strftime("%Y-%m-%d")
            to_str = to_dt.strftime("%Y-%m-%d")
            
            print(f"[News/Sentiment] Fetching economic calendar from {from_str} to {to_str}...")
            resp = requests.get(
                "https://finnhub.io/api/v1/calendar/economic",
                params={"token": finnhub_token, "from": from_str, "to": to_str},
                timeout=10
            )
            if resp.status_code == 200:
                events = resp.json().get("economicCalendar", [])
                high_impact = ["CPI", "FOMC", "NFP", "Non-Farm", "Federal Reserve", "Interest Rate"]
                filtered_events = []
                for ev in events:
                    if any(kw.lower() in ev.get("event", "").lower() for kw in high_impact):
                        ev_time_str = ev.get("time", "")
                        try:
                            ev_time = datetime.strptime(ev_time_str, "%Y-%m-%d %H:%M:%S")
                            filtered_events.append(ev_time)
                        except Exception:
                            pass
                economic_calendar_cache = sorted(filtered_events)
                print(f"[News/Sentiment] Cached {len(economic_calendar_cache)} high-impact calendar events.")
                return economic_calendar_cache
        except Exception as e:
            print(f"[News/Sentiment] Error caching economic calendar: {e}")
        
        economic_calendar_cache = []
        return economic_calendar_cache

def add_news_proximity_feature(df):
    if df.empty:
        df["hours_to_news"] = 72.0
        return df
        
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    events = fetch_economic_calendar_cached(start_ts, end_ts)
    
    if not events:
        df["hours_to_news"] = 72.0
        return df
        
    import bisect
    df_dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    hours_to_news_list = []
    events_utc = [pd.Timestamp(ev).tz_localize("UTC") for ev in events]
    
    for current_time in df_dt:
        idx = bisect.bisect_right(events_utc, current_time)
        if idx < len(events_utc):
            next_event = events_utc[idx]
            diff_hours = (next_event - current_time).total_seconds() / 3600.0
            hours_to_news_list.append(min(72.0, max(0.0, diff_hours)))
        else:
            hours_to_news_list.append(72.0)
            
    df["hours_to_news"] = hours_to_news_list
    return df

# =========================
# ORDER BOOK PRESSURE
# =========================
def get_orderbook_imbalance(symbol=None):
    if symbol is None:
        symbol = SYMBOL
    try:
        url = "https://api.bybit.com/v5/market/orderbook"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "spot", "symbol": symbol, "limit": 25}, headers=headers, proxies=get_bybit_proxies(), timeout=10)
        res = None
        if response.status_code == 200:
            res = response.json()
        else:
            print(f"[Orderbook] Bybit returned HTTP {response.status_code}. Attempting Binance depth fallback...")
            # Try Binance depth fallback
            binance_url = "https://api.binance.com/api/v3/depth"
            resp = requests.get(binance_url, params={"symbol": symbol.upper(), "limit": 25}, headers=headers, timeout=10)
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
                
                selected_features_list = models_by_interval[interval].get("selected_features")
                if selected_features_list is not None:
                    X_hist = df[selected_features_list].values
                else:
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
# PORTFOLIO COVARIANCE CONSTRAINTS
# ==========================================
def calculate_covariance_multiplier(new_symbol, new_direction):
    """
    Calculates a position sizing multiplier based on portfolio covariance.
    Penalizes highly correlated assets in the same direction.
    Allows offsetting/hedging for assets in opposite directions.
    """
    CORRELATION_MAP = {
        ("BTCUSDT", "BTCUSDT"): 1.0,
        ("ETHUSDT", "ETHUSDT"): 1.0,
        ("SOLUSDT", "SOLUSDT"): 1.0,
        ("BNBUSDT", "BNBUSDT"): 1.0,
        ("ADAUSDT", "ADAUSDT"): 1.0,
        ("XRPUSDT", "XRPUSDT"): 1.0,
        
        ("BTCUSDT", "ETHUSDT"): 0.85,
        ("BTCUSDT", "SOLUSDT"): 0.75,
        ("BTCUSDT", "BNBUSDT"): 0.70,
        ("BTCUSDT", "ADAUSDT"): 0.70,
        ("BTCUSDT", "XRPUSDT"): 0.65,
        
        ("ETHUSDT", "SOLUSDT"): 0.80,
        ("ETHUSDT", "BNBUSDT"): 0.75,
        ("ETHUSDT", "ADAUSDT"): 0.75,
        ("ETHUSDT", "XRPUSDT"): 0.65,
        
        ("SOLUSDT", "BNBUSDT"): 0.70,
        ("SOLUSDT", "ADAUSDT"): 0.70,
        ("SOLUSDT", "XRPUSDT"): 0.60,
        
        ("BNBUSDT", "ADAUSDT"): 0.70,
        ("BNBUSDT", "XRPUSDT"): 0.60,
        
        ("ADAUSDT", "XRPUSDT"): 0.65
    }

    def get_correlation(s1, s2):
        if s1 == s2:
            return 1.0
        return CORRELATION_MAP.get((s1, s2)) or CORRELATION_MAP.get((s2, s1)) or 0.70

    # Collect active trades from all timeframes
    open_trades = []
    for tf_key in ["1h", "2h", "4h", "6h"]:
        open_trades.extend(bot_state.get(f"active_trade_{tf_key}", []))

    if not open_trades:
        return 1.0, 0.0

    total_risk = 0.0
    breakdown = []
    
    for t in open_trades:
        open_sym = t.get("symbol")
        open_dir = t.get("direction")
        if not open_sym or not open_dir:
            continue
        r = get_correlation(new_symbol, open_sym)
        
        if new_direction == open_dir:
            impact = r
            risk_type = "CONCENTRATION"
        else:
            impact = -r
            risk_type = "HEDGE"
            
        total_risk += impact
        breakdown.append(f"  - Active: {open_sym} {open_dir} | Correlation: {r:.2f} | Risk impact: {impact:+.2f} ({risk_type})")

    if total_risk <= 0:
        multiplier = 1.0
    else:
        multiplier = 1.0 / (1.0 + total_risk)
        multiplier = max(0.20, min(1.0, multiplier))

    print(f"\n[Portfolio Covariance Analysis] New Entry: {new_symbol} {new_direction}")
    for item in breakdown:
        print(item)
    print(f"  - Total Net Correlation Risk: {total_risk:+.2f} -> Covariance Multiplier: {multiplier:.2f}x\n")

    return float(multiplier), float(total_risk)

# ==========================================
# PRE-TRADE CONFLUENCE ANALYSIS
# ==========================================
def check_pre_trade_confluence(current_price, df_1h, ml_trend, news_sentiment, expected_pct_change, interval="60", symbol=None, htf_cache=None):
    """
    Runs pre-trade confluence checks using a WEIGHTED SCORING SYSTEM.
    Critical checks are hard gates (instant reject if failed).
    Other checks contribute weighted points to a total score.
    Trade is approved if score >= 75% of max possible points AND no hard gate fails.
    Returns: (bool_approved, dict_results_details)
    """
    if symbol is None:
        symbol = SYMBOL
    results = {}
    hard_gate_failed = False
    total_score = 0
    max_score = 0

    # ======= CHECK 1: 1-Day Structural Trend (Weight: 1, Bypassed for 5m/15m) =======
    df_1d = None
    if htf_cache is not None and (symbol, "D") in htf_cache:
        df_1d = htf_cache[(symbol, "D")]
    if df_1d is None:
        try:
            df_1d = get_history(symbol=symbol, interval="D", limit=100)
            if htf_cache is not None and df_1d is not None:
                htf_cache[(symbol, "D")] = df_1d
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
    df_4h = None
    if htf_cache is not None and (symbol, "240") in htf_cache:
        df_4h = htf_cache[(symbol, "240")]
    if df_4h is None:
        try:
            df_4h = get_history(symbol=symbol, interval="240", limit=100)
            if htf_cache is not None and df_4h is not None:
                htf_cache[(symbol, "240")] = df_4h
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

    # ======= CHECK 3: 1h RSI (Weight: 2) =======
    weight_rsi = 2
    rsi_1h = df_1h["RSI"].iloc[-1]
    if ml_trend == "Bullish":
        rsi_1h_pass = (rsi_1h < 62.0)
        detail_msg = f"1h RSI is {rsi_1h:.2f} (< 62, Safe)" if rsi_1h_pass else f"1h RSI is {rsi_1h:.2f} (>= 62, Overbought)"
    else:
        rsi_1h_pass = (rsi_1h > 38.0)
        detail_msg = f"1h RSI is {rsi_1h:.2f} (> 38, Safe)" if rsi_1h_pass else f"1h RSI is {rsi_1h:.2f} (<= 38, Oversold)"
    results["1h_RSI"] = {"pass": rsi_1h_pass, "detail": detail_msg, "weight": weight_rsi}
    max_score += weight_rsi
    if rsi_1h_pass:
        total_score += weight_rsi

    # ======= CHECK 4: Volume Confirmation (RVOL) Hard Gate =======
    # Filter out low-liquidity fakeouts (RVOL must be >= 1.0x)
    try:
        vol_series = df_1h["volume"]
        avg_vol_20 = vol_series.iloc[:-1].rolling(20).mean().iloc[-1]
        latest_vol = vol_series.iloc[-2]
        rvol = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0
        volume_pass = (rvol >= 1.0)
        if not volume_pass:
            hard_gate_failed = True
        results["Volume_Confirmation"] = {
            "pass": volume_pass,
            "detail": f"RVOL: {rvol:.2f}x (Vol: {latest_vol:.1f} / Avg20: {avg_vol_20:.1f}), required >= 1.0x",
            "weight": 0
        }
    except Exception as e:
        volume_pass = True
        results["Volume_Confirmation"] = {"pass": True, "detail": f"Skipped volume check (Error: {e})", "weight": 0}

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

    # ======= CHECK 9: Fee Coverage (Weight: 2) =======
    weight_fee = 2
    atr_norm_val = df_1h["ATR_norm"].iloc[-1]
    if str(interval) in ["5", "15"]:
        fee_pass = (atr_norm_val >= 0.0010)
        req_str = ">= 0.10%"
    else:
        fee_pass = (atr_norm_val >= 0.0015)
        req_str = ">= 0.15%"
    results["Fee_Coverage"] = {
        "pass": fee_pass,
        "detail": f"ATR Volatility: {atr_norm_val*100:.3f}% (Req {req_str} to cover roundtrip Spot fees)",
        "weight": weight_fee
    }
    max_score += weight_fee
    if fee_pass:
        total_score += weight_fee

    # ======= CHECK 10: Order Book Imbalance & Spread (Weight: 1) =======
    weight_ob = 1
    ob_metrics = get_orderbook_imbalance(symbol=symbol)
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

    # ======= CHECK 11: News Sentiment (Hard Gate & Direction Lock) =======
    # Dynamic Sentiment Lock: Hard block trades contradicting general news sentiment (e.g. Bullish models blocked on Bearish news)
    is_opposed = (ml_trend == "Bullish" and news_sentiment == "Bearish") or (ml_trend == "Bearish" and news_sentiment == "Bullish")
    news_pass = not is_opposed
    if is_opposed:
        hard_gate_failed = True
        detail_msg = f"Blocked (Hard Direction Lock: Model is {ml_trend} but News sentiment is {news_sentiment})"
    else:
        detail_msg = f"Passed (Direction Lock: Model is {ml_trend}, News sentiment is {news_sentiment})"
        
    results["News_Sentiment"] = {"pass": news_pass, "detail": detail_msg, "weight": 0}

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

    # ======= CHECK 13: Multi-Timeframe Trend Enforcer (HTF Gate - Weight: 2, Hard Gate for 5m/15m) =======
    weight_align = 2
    trend_align_pass = True
    align_detail = "Aligned with dominant trend"
    if str(interval) in ["5", "15"]:
        try:
            # 1. Fetch 1h trend
            df_1h_align = get_history(symbol=symbol, interval="60", limit=100)
            ema9_1h, ema21_1h = None, None
            if df_1h_align is not None and len(df_1h_align) >= 21:
                df_1h_align_completed = df_1h_align.iloc[:-1].copy()
                ema9_1h = EMAIndicator(df_1h_align_completed["close"], window=9).ema_indicator().iloc[-1]
                ema21_1h = EMAIndicator(df_1h_align_completed["close"], window=21).ema_indicator().iloc[-1]

            # 2. Fetch/Retrieve 4h trend
            if df_4h is None:
                try:
                    df_4h = get_history(symbol=symbol, interval="240", limit=100)
                except Exception as e:
                    print(f"Error fetching 4h history for HTF Gate: {e}")
            
            ema9_4h, ema21_4h = None, None
            if df_4h is not None and len(df_4h) >= 21:
                df_4h_completed = df_4h.iloc[:-1].copy()
                ema9_4h = EMAIndicator(df_4h_completed["close"], window=9).ema_indicator().iloc[-1]
                ema21_4h = EMAIndicator(df_4h_completed["close"], window=21).ema_indicator().iloc[-1]

            # 3. Check alignment
            if ema9_1h is not None and ema21_1h is not None and ema9_4h is not None and ema21_4h is not None:
                trend_1h = "Bullish" if ema9_1h > ema21_1h else "Bearish"
                trend_4h = "Bullish" if ema9_4h > ema21_4h else "Bearish"
                
                if ml_trend == "Bullish":
                    if trend_1h == "Bullish" and trend_4h == "Bullish":
                        align_detail = "Aligned with 1h Bullish and 4h Bullish trends"
                    else:
                        trend_align_pass = False
                        hard_gate_failed = True
                        align_detail = f"Blocked (HTF Gate: Bullish signal contradicts 1h {trend_1h} or 4h {trend_4h} trend)"
                elif ml_trend == "Bearish":
                    if trend_1h == "Bearish" and trend_4h == "Bearish":
                        align_detail = "Aligned with 1h Bearish and 4h Bearish trends"
                    else:
                        trend_align_pass = False
                        hard_gate_failed = True
                        align_detail = f"Blocked (HTF Gate: Bearish signal contradicts 1h {trend_1h} or 4h {trend_4h} trend)"
                else:
                    align_detail = f"Neutral ML trend, HTF check not applicable"
            else:
                trend_align_pass = False
                hard_gate_failed = True
                align_detail = "Could not calculate 1h/4h EMA trends (HTF Gate Blocked)"
        except Exception as e:
            trend_align_pass = False
            hard_gate_failed = True
            align_detail = f"Error in HTF Gate alignment check: {e} (HTF Gate Blocked)"
    else:
        align_detail = f"1h/4h intervals are already dominant or equal"
    
    results["Timeframe_Alignment"] = {"pass": trend_align_pass, "detail": align_detail, "weight": weight_align}
    max_score += weight_align
    if trend_align_pass:
        total_score += weight_align

    # ======= CHECK 14: Open Interest Delta Confirmation (Weight: 2) =======
    weight_oi = 2
    try:
        oi_delta = df_1h["open_interest_pct_change"].iloc[-1] * 100.0  # as percentage
        oi_pass = (oi_delta >= -2.0)
        detail_msg = f"OI Delta: {oi_delta:+.2f}% (Req >= -2.00% to confirm momentum, Safe)" if oi_pass else f"OI Delta: {oi_delta:+.2f}% (< -2.00%, Position unwinding / exhaustion)"
    except Exception as e:
        oi_pass = True
        detail_msg = f"Skipped (Error: {e})"
    results["Open_Interest_Delta"] = {"pass": oi_pass, "detail": detail_msg, "weight": weight_oi}
    max_score += weight_oi
    if oi_pass:
        total_score += weight_oi

    # ======= FINAL SCORING =======
    score_pct = (total_score / max_score * 100) if max_score > 0 else 100.0
    
    # Standardize confluence threshold to flat 80% for consistent accuracy across all market regimes
    score_threshold = 80.0
        
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

    return bool(approved), std_results, float(score_pct)

# =========================
# LIVE LOOP
# =========================
def get_fallback_price(symbol=SYMBOL):
    # 1. Try Bybit API
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "spot", "symbol": symbol}, headers=headers, proxies=get_bybit_proxies(), timeout=5)
        if response.status_code == 200:
            res = response.json()
            return float(res["result"]["list"][0]["lastPrice"])
        else:
            print(f"Bybit price ticker for {symbol} returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Bybit price fallback for {symbol}: {e}")

    # 2. Try Coinbase API (only for BTCUSDT)
    if symbol == "BTCUSDT":
        try:
            response = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=5)
            if response.status_code == 200:
                res = response.json()
                return float(res["data"]["amount"])
        except Exception:
            pass

    # 3. Try Binance API
    try:
        response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5)
        if response.status_code == 200:
            res = response.json()
            return float(res["price"])
        else:
            print(f"Binance price ticker for {symbol} returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Binance price fallback for {symbol}: {e}")

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
    startup_check_done = False
    last_check_hour = -1

    while True:
        current_time = time.time()
        
        # 1. Health check & current price update
        if live_price is None or (current_time - last_ws_update_time > 15.0):
            fallback_price = get_fallback_price()
            if fallback_price is not None:
                print(f"[{get_pkt_time().strftime('%H:%M:%S')}] WebSocket price is stale or disconnected. Fallback price: {fallback_price:.2f}")
                live_price = fallback_price
                last_ws_update_time = current_time
            
        current_price = live_price
        if current_price is None:
            print("Could not obtain price. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # 2. Check Exits for each timeframe if a trade is active
        tf_map = {"60": "1h", "120": "2h", "240": "4h", "360": "6h"}
        active_trades_updated = False
        for iv in ["60", "120", "240", "360"]:
            tf = tf_map[iv]
            active_trade_key = f"active_trade_{tf}"
            active_trades_list = bot_state.get(active_trade_key, [])
            if not isinstance(active_trades_list, list):
                active_trades_list = [] if active_trades_list is None else [active_trades_list]
                bot_state[active_trade_key] = active_trades_list
            
            updated_trades = []
            for active_trade in active_trades_list:
                active_symbol = active_trade.get("symbol", "BTCUSDT")
                symbol_price = bot_state.get(f"live_price_{active_symbol}")
                if symbol_price is None:
                    symbol_price = get_fallback_price(active_symbol)
                if symbol_price is None:
                    updated_trades.append(active_trade)
                    continue
                current_price = symbol_price
                
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

                # Volatility-Scaled Trailing Stops: multiplier is dynamic based on current ADX
                # If the position has scaled out (half closed), we use a tighter trailing stop of 1.0 * ATR
                if active_trade.get("half_closed", False):
                    trailing_multiplier = 1.0
                else:
                    current_adx = bot_state.get(f"adx_{tf}", 20.0)
                    if current_adx >= 25.0:
                        trailing_multiplier = 1.50
                    elif current_adx < 18.0:
                        trailing_multiplier = 0.90
                    else:
                        trailing_multiplier = 1.25

                # Update trailing stop peak prices
                if direction == "Bullish":
                    if current_price > highest_price:
                        highest_price = current_price
                        active_trade["highest_price"] = highest_price
                        # Trailing Stop: SL trails highest price by dynamic multiplier
                        potential_sl = highest_price - trailing_multiplier * atr_dollars
                        if potential_sl > stop_loss:
                            stop_loss = potential_sl
                            active_trade["stop_loss"] = stop_loss
                            active_trades_updated = True
                            print(f"[{iv}m Trailing Stop] Moved SL up to {stop_loss:.2f} (trailing highest: {highest_price:.2f}, multiplier: {trailing_multiplier}x)")
                    
                    # Break-Even Guard: if price goes up by 0.5 * ATR, move SL to entry
                    if not break_even_triggered and current_price >= entry_price + 0.5 * atr_dollars:
                        break_even_triggered = True
                        active_trade["break_even_triggered"] = True
                        stop_loss = max(stop_loss, entry_price)
                        active_trade["stop_loss"] = stop_loss
                        active_trades_updated = True
                        print(f"[{iv}m Break-Even Guard] Triggered! SL moved to entry price: {entry_price:.2f}")
                else:
                    if current_price < lowest_price:
                        lowest_price = current_price
                        active_trade["lowest_price"] = lowest_price
                        # Trailing Stop: SL trails lowest price by dynamic multiplier
                        potential_sl = lowest_price + trailing_multiplier * atr_dollars
                        if potential_sl < stop_loss:
                            stop_loss = potential_sl
                            active_trade["stop_loss"] = stop_loss
                            active_trades_updated = True
                            print(f"[{iv}m Trailing Stop] Moved SL down to {stop_loss:.2f} (trailing lowest: {lowest_price:.2f}, multiplier: {trailing_multiplier}x)")
                            
                    # Break-Even Guard: if price goes down by 0.5 * ATR, move SL to entry
                    if not break_even_triggered and current_price <= entry_price - 0.5 * atr_dollars:
                        break_even_triggered = True
                        active_trade["break_even_triggered"] = True
                        stop_loss = min(stop_loss, entry_price)
                        active_trade["stop_loss"] = stop_loss
                        active_trades_updated = True
                        print(f"[{iv}m Break-Even Guard] Triggered! SL moved to entry price: {entry_price:.2f}")

                # Scale-Out (50% partial profit taking at 0.6 * ATR)
                half_closed = active_trade.get("half_closed", False)
                if not half_closed:
                    if direction == "Bullish" and current_price >= entry_price + 0.6 * atr_dollars:
                        # Scale-Out Triggered for Long
                        half_closed = True
                        active_trade["half_closed"] = True
                        
                        # Close 50% of the position
                        closed_size = round(position_size_usd * 0.5, 2)
                        remaining_size = round(position_size_usd - closed_size, 2)
                        
                        # Calculate profit on closed half (use standard Taker fee on exit)
                        raw_return_pct = ((current_price - entry_price) / entry_price) * 100.0
                        net_return_pct = (raw_return_pct * active_trade.get("leverage", 1.0)) - 0.08
                        pnl_usd = round(closed_size * (net_return_pct / 100.0), 2)
                        
                        # Save scaled out pnl
                        active_trade["scaled_out_pnl"] = pnl_usd
                        
                        # Refund closed size + PnL to wallet balance
                        bot_state["simulated_balance"] = round(bot_state["simulated_balance"] + closed_size + pnl_usd, 2)
                        
                        # Update position details
                        position_size_usd = remaining_size
                        active_trade["position_size_usd"] = remaining_size
                        
                        # Move stop loss to entry price (break-even)
                        stop_loss = entry_price
                        active_trade["stop_loss"] = entry_price
                        active_trade["break_even_triggered"] = True
                        active_trades_updated = True
                        
                        print(f"[{active_symbol} {iv}m Scale-Out] 50% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to entry: {entry_price:.2f}")
                        
                    elif direction == "Bearish" and current_price <= entry_price - 0.6 * atr_dollars:
                        # Scale-Out Triggered for Short
                        half_closed = True
                        active_trade["half_closed"] = True
                        
                        # Close 50% of the position
                        closed_size = round(position_size_usd * 0.5, 2)
                        remaining_size = round(position_size_usd - closed_size, 2)
                        
                        # Calculate profit on closed half (use standard Taker fee on exit)
                        raw_return_pct = ((entry_price - current_price) / entry_price) * 100.0
                        net_return_pct = (raw_return_pct * active_trade.get("leverage", 1.0)) - 0.08
                        pnl_usd = round(closed_size * (net_return_pct / 100.0), 2)
                        
                        # Save scaled out pnl
                        active_trade["scaled_out_pnl"] = pnl_usd
                        
                        # Refund closed size + PnL to wallet balance
                        bot_state["simulated_balance"] = round(bot_state["simulated_balance"] + closed_size + pnl_usd, 2)
                        
                        # Update position details
                        position_size_usd = remaining_size
                        active_trade["position_size_usd"] = remaining_size
                        
                        # Move stop loss to entry price (break-even)
                        stop_loss = entry_price
                        active_trade["stop_loss"] = entry_price
                        active_trade["break_even_triggered"] = True
                        active_trades_updated = True
                        
                        print(f"[{active_symbol} {iv}m Scale-Out] 50% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to entry: {entry_price:.2f}")

                remaining_seconds = max(0, int(end_time - current_time))
                mins, secs = divmod(remaining_seconds, 60)
                countdown_str = f"{mins:02d}m {secs:02d}s"

                print(f"[{active_symbol} {iv}m Active Trade] {direction} | Price: {current_price:.2f} (Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}) | Countdown: {countdown_str}")

                exit_reason = None
                half_closed = active_trade.get("half_closed", False)
                if direction == "Bullish":
                    if current_price <= stop_loss:
                        exit_reason = "TRAILING STOP HIT [SUCCESS]" if half_closed else "STOP LOSS HIT [FAIL]"
                    elif current_price >= take_profit and not half_closed:
                        exit_reason = "TAKE PROFIT HIT [SUCCESS]"
                else:
                    if current_price >= stop_loss:
                        exit_reason = "TRAILING STOP HIT [SUCCESS]" if half_closed else "STOP LOSS HIT [FAIL]"
                    elif current_price <= take_profit and not half_closed:
                        exit_reason = "TAKE PROFIT HIT [SUCCESS]"

                if current_time >= end_time and not half_closed:
                    lookahead = 10
                    exit_reason = f"{int(iv)*lookahead}-MINUTE TIMER ELAPSED"

                if exit_reason is not None:
                    # Maker vs Taker execution logic
                    is_stop_loss = "STOP LOSS" in str(exit_reason).upper()
                    
                    if is_stop_loss:
                        # Maker limit execution for Stop Loss exit (Post-Only model)
                        slippage_pct = 0.0
                        actual_price = current_price
                        fee_rate_roundtrip = 0.04  # Maker Entry + Maker Exit roundtrip
                        exit_reason = str(exit_reason) + " [Limit order Maker close]"
                    else:
                        # Maker execution for Take Profit, Timer, etc.
                        slippage_pct = 0.0
                        actual_price = current_price
                        fee_rate_roundtrip = 0.04  # Maker Entry + Maker Exit roundtrip

                    price_diff = actual_price - predicted_price
                    price_diff_pct = (price_diff / predicted_price) * 100
                    price_accuracy = max(0.0, 100.0 - abs((actual_price - predicted_price) / actual_price * 100))
                    actual_change = actual_price - entry_price
                    actual_change_pct = (actual_change / entry_price) * 100
                    
                    # Calculate PnL (long vs short) and simulated fees with leverage
                    raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
                    leverage = active_trade.get("leverage", 1.0)
                    net_return_pct = (raw_return_pct * leverage) - fee_rate_roundtrip
                    realized_pnl = position_size_usd * (net_return_pct / 100.0)
                    
                    # Aggregate PnL and size for trade history logging if scaled out
                    original_size = float(active_trade.get("original_size", position_size_usd))
                    scaled_out_pnl = float(active_trade.get("scaled_out_pnl", 0.0))
                    total_pnl = round(realized_pnl + scaled_out_pnl, 2)
                    total_net_return_pct = round((total_pnl / original_size) * 100.0, 4)
                    
                    # Update simulated balance
                    old_bal = bot_state.get("simulated_balance", 80.0)
                    new_bal = round(old_bal + position_size_usd + realized_pnl, 2)
                    bot_state["simulated_balance"] = new_bal
                    
                    actual_trend = "Bullish" if actual_change > 0 else "Bearish"
                    signal_correct = (actual_trend == direction)
                    trend_status = f"{direction} was CORRECT [OK]" if signal_correct else f"{direction} was INCORRECT [FAIL]"
                    
                    print("\n==================================================")
                    print(f"[{active_symbol} {iv}m TRADE EXITED]: {exit_reason} (Slippage: {slippage_pct:.3f}%)")
                    print(f"Start Price: {entry_price:.2f} | Exit Price: {actual_price:.2f}")
                    print(f"Actual Change: {actual_change:+.2f} ({actual_change_pct:+.4f}%)")
                    if active_trade.get("half_closed", False):
                        print(f"Total Size: ${original_size:.2f} (Scaled-Out) | Net Return: {total_net_return_pct:+.4f}% (weighted)")
                        print(f"Scaled-Out PnL: ${scaled_out_pnl:+.2f} | Remaining PnL: ${realized_pnl:+.2f} | Total PnL: ${total_pnl:+.2f}")
                    else:
                        print(f"Size: ${position_size_usd:.2f} | Net Return: {net_return_pct:+.4f}% (after {fee_rate_roundtrip:.2f}% fees)")
                        print(f"Realized PnL: ${realized_pnl:+.2f}")
                    print(f"New Balance: ${new_bal:.2f} | Predicted Signal: {direction} ({trend_status})")
                    print("==================================================\n")
                    
                    # Update Completed Trade History in global state
                    bot_state["trade_history"].append({
                        "symbol": active_symbol,
                        "exit_time": float(time.time()),
                        "interval": str(iv),
                        "direction": str(direction),
                        "entry_price": float(entry_price),
                        "exit_price": float(actual_price),
                        "change_pct": float(total_net_return_pct if active_trade.get("half_closed", False) else net_return_pct),
                        "success": bool(signal_correct),
                        "reason": str(exit_reason) + (" (Scale-Out)" if active_trade.get("half_closed", False) else ""),
                        "position_size_usd": float(original_size),
                        "pnl_usd": float(total_pnl),
                        "balance": float(new_bal),
                        "leverage": float(leverage)
                    })
                    
                    # Send email alert on Take Profit hit
                    if "TAKE PROFIT" in str(exit_reason).upper():
                        subject = f"🚀 [UBOTE TP Hit] {active_symbol} {iv}m Take Profit Triggered!"
                        invested_margin_usd = original_size / leverage if leverage > 0 else original_size
                        body = f"""
                        <html>
                        <body style="font-family: Arial, sans-serif; background-color: #0b0e14; color: #f0f3fa; padding: 20px; border-radius: 8px;">
                            <h2 style="color: #00b0ff; margin-bottom: 20px;">🎉 Take Profit Target Hit!</h2>
                            <div style="background-color: #161a22; padding: 15px; border-radius: 6px; border-left: 4px solid #00c853;">
                                <table style="width: 100%; border-collapse: collapse;">
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3; width: 140px;"><b>Symbol:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace; font-size: 14px;">{active_symbol}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Timeframe:</b></td>
                                        <td style="padding: 6px 0;">{iv}m</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Direction:</b></td>
                                        <td style="padding: 6px 0; color: {'#00c853' if direction == 'Bullish' else '#ff3d00'}; font-weight: bold;">{direction}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Entry Price:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">${entry_price:.4f}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Exit Price:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">${actual_price:.4f}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Profit/Loss:</b></td>
                                        <td style="padding: 6px 0; color: #00c853; font-weight: bold; font-family: monospace;">+{total_pnl:+.2f} USD ({total_net_return_pct:+.4f}%)</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Position Size:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">${original_size:.2f} USD ({leverage}x Leverage)</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Invested Amount:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">${invested_margin_usd:.2f} USD</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Account Balance:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace; font-weight: bold;">${new_bal:.2f} USD</td>
                                    </tr>
                                </table>
                            </div>
                            <p style="font-size: 11px; color: #8f9bb3; margin-top: 20px;">Sent automatically by UBOTE Trading System.</p>
                        </body>
                        </html>
                        """
                        threading.Thread(target=send_email_notification, args=(subject, body), daemon=True).start()
                    
                    for p in bot_state["prediction_history"]:
                        if p.get("interval") == str(iv) and p.get("symbol") == active_symbol and p.get("status") == "Traded" and (not p.get("evaluation") or not p["evaluation"].get("evaluated")):
                            p["evaluation"] = {
                                "evaluated": True,
                                "exit_price": float(actual_price),
                                "change": float(actual_change if direction == "Bullish" else -actual_change),
                                "change_pct": float(raw_return_pct),
                                "success": bool(signal_correct)
                            }
                            break
                    save_history()
                else:
                    updated_trades.append(active_trade)
            bot_state[active_trade_key] = updated_trades
        
        if active_trades_updated:
            save_history()

        # 3. Check for completed candle closes to search for a new signal

        # --- Daily Drawdown Circuit Breaker & Profit Goal ---
        today = get_pkt_time().day
        if bot_state["daily_drawdown_reset_day"] != today:
            bot_state["daily_drawdown_start_balance"] = bot_state.get("simulated_balance", 80.0)
            bot_state["daily_drawdown_reset_day"] = today
            bot_state["circuit_breaker_active"] = False
            bot_state["daily_goal_reached"] = False
            print(f"[Circuit Breaker] Daily reset (PKT). Start balance: ${bot_state['daily_drawdown_start_balance']:.2f}")
        else:
            start_bal = bot_state["daily_drawdown_start_balance"]
            curr_bal = bot_state.get("simulated_balance", start_bal)
            daily_dd_pct = (start_bal - curr_bal) / start_bal * 100 if start_bal > 0 else 0
            daily_profit = curr_bal - start_bal
            # Circuit breaker is deactivated for now
            bot_state["circuit_breaker_active"] = False
            if daily_profit >= 1000.0 and not bot_state.get("daily_goal_reached", False):
                bot_state["daily_goal_reached"] = True
                print(f"[Daily Goal] REACHED — daily profit of ${daily_profit:.2f} >= $1000. Continuing trading to maximize gains (no maximum limit).")
            elif daily_profit < 1000.0:
                bot_state["daily_goal_reached"] = False

        # --- High-Impact News Window Guard ---
        def is_high_impact_news_window():
            """Returns True if within 15 minutes of a known high-impact event (CPI, FOMC, NFP)."""
            try:
                now_utc = datetime.utcnow()
                finnhub_token = os.environ.get("FINNHUB_TOKEN", "free")
                resp = requests.get(
                    "https://finnhub.io/api/v1/calendar/economic",
                    params={"token": finnhub_token},
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

        # --- Consecutive Losses Cooldown Circuit Breaker ---
        def is_symbol_interval_cooling_off(symbol, interval):
            """
            Checks if a symbol and interval combination is in a 6-hour cool-off period
            after suffering 2 consecutive loss trades.
            """
            trades = [t for t in bot_state.get("trade_history", []) if t.get("symbol") == symbol and str(t.get("interval")) == str(interval)]
            if len(trades) < 2:
                return False, 0
                
            # Sort by exit_time descending to get latest trades
            sorted_trades = sorted(trades, key=lambda x: x.get("exit_time", 0.0), reverse=True)
            
            latest_trade = sorted_trades[0]
            second_latest = sorted_trades[1]
            
            is_latest_loss = (latest_trade.get("success") is False) or (latest_trade.get("pnl_usd", 0.0) < 0.0)
            is_second_loss = (second_latest.get("success") is False) or (second_latest.get("pnl_usd", 0.0) < 0.0)
            
            if is_latest_loss and is_second_loss:
                exit_time = latest_trade.get("exit_time", 0.0)
                cooldown_duration = 6 * 3600  # 6 hours
                time_elapsed = time.time() - exit_time
                if time_elapsed < cooldown_duration:
                    remaining_minutes = int((cooldown_duration - time_elapsed) / 60)
                    return True, remaining_minutes
                    
            return False, 0

        check_and_hot_reload_models()
        current_time_pkt = get_pkt_time()
        # Trigger check once at the boundary (minute 0, second between 5 and 55) or on startup
        is_boundary_time = (current_time_pkt.minute == 0) and (5 <= current_time_pkt.second <= 55)
        
        if (is_boundary_time and current_time_pkt.hour != last_check_hour) or (not startup_check_done):
            if not startup_check_done:
                print("[Startup] Executing fast initial candle check for BTCUSDT to update cards instantly...")
                check_queue = [("BTCUSDT", iv) for iv in ["60", "120", "240", "360"]]
                startup_check_done = True
            else:
                last_check_hour = current_time_pkt.hour
                check_queue = []
                for iv_q in ["60", "120", "240", "360"]:
                    iv_hours = int(iv_q) // 60
                    if current_time_pkt.hour % iv_hours == 0:
                        for symbol_q in SUPPORTED_SYMBOLS:
                            check_queue.append((symbol_q, iv_q))
        else:
            check_queue = []
            
        htf_cache = {}
        fetched_data = {}
        if check_queue:
            from concurrent.futures import ThreadPoolExecutor
            
            # Fetch BTCUSDT history once per interval to cache and share among workers (avoids rate limits and duplicate REST calls)
            btc_hist_cache = {}
            unique_intervals = set(iv for sym, iv in check_queue)
            for iv_val in unique_intervals:
                df_btc = get_history(symbol="BTCUSDT", interval=iv_val, limit=300)
                btc_hist_cache[iv_val] = df_btc
            
            def fetch_single_history(sym, interval_val):
                if sym == "BTCUSDT" and interval_val in btc_hist_cache:
                    df_raw_val = btc_hist_cache[interval_val]
                else:
                    df_raw_val = get_history(symbol=sym, interval=interval_val, limit=300)
                if df_raw_val is None or len(df_raw_val) < 2:
                    return sym, interval_val, None, None
                
                df_completed_val = df_raw_val.iloc[:-1].copy()
                latest_completed_ts_val = int(df_completed_val.iloc[-1]["timestamp"])
                
                last_ts_key_val = f"last_processed_{sym}_{interval_val}_ts"
                if last_processed_timestamps.get(last_ts_key_val) is not None:
                    if latest_completed_ts_val == last_processed_timestamps[last_ts_key_val]:
                        return sym, interval_val, df_raw_val, None
                
                df_target_val = df_completed_val.copy()
                if sym != "BTCUSDT":
                    df_btc_val = btc_hist_cache.get(interval_val)
                    if df_btc_val is not None and len(df_btc_val) > 0:
                        df_btc_sub_val = df_btc_val[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                        df_target_val = pd.merge(df_target_val, df_btc_sub_val, on="timestamp", how="inner")
                    else:
                        df_target_val["close_btc"] = df_target_val["close"]
                else:
                    df_target_val["close_btc"] = df_target_val["close"]
                
                df_target_val = merge_derivatives_sentiment_features(df_target_val, symbol=sym, interval=interval_val)
                df_feat_val = add_features(df_target_val)
                
                return sym, interval_val, df_raw_val, df_feat_val

            print(f"[Parallel Fetch] Querying {len(check_queue)} candle combinations in parallel...")
            t_start = time.time()
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(fetch_single_history, sym, iv) for sym, iv in check_queue]
                for fut in futures:
                    try:
                        sym, iv, df_raw_val, df_feat_val = fut.result(timeout=25)
                        if df_raw_val is not None:
                            fetched_data[(sym, iv)] = (df_raw_val, df_feat_val)
                    except Exception as e:
                        print(f"[Parallel Fetch] Error fetching {sym} {iv}: {e}")
            print(f"[Parallel Fetch] Completed in {time.time() - t_start:.2f} seconds.")

        for symbol, iv in check_queue:
            tf = tf_map[iv]
            active_trade_key = f"active_trade_{tf}"
            active_trades_list = bot_state.get(active_trade_key, [])
            if not isinstance(active_trades_list, list):
                active_trades_list = [] if active_trades_list is None else [active_trades_list]
                bot_state[active_trade_key] = active_trades_list
                
            if (symbol, iv) not in fetched_data:
                continue
            df_raw, df = fetched_data[(symbol, iv)]
            if df is None or len(df) == 0:
                continue
                
            try:
                df_completed = df_raw.iloc[:-1].copy()
                latest_completed_ts = int(df_completed.iloc[-1]["timestamp"])

                last_ts_key = f"last_processed_{symbol}_{iv}_ts"
                if last_processed_timestamps.get(last_ts_key) is None:
                    last_processed_timestamps[last_ts_key] = 0
                    print(f"Initialized completed candle timestamp tracking for {symbol} on {iv}m: {get_local_time_str(latest_completed_ts/1000)}")

                if latest_completed_ts != last_processed_timestamps[last_ts_key]:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New completed {symbol} {iv}-minute candle detected (TS: {latest_completed_ts})")
                    
                    latest_candle = df.iloc[-1]
                    
                    # Slicing features based on selected_features if loaded
                    if iv in models_by_interval and models_by_interval[iv].get("selected_features") is not None:
                        X_live = latest_candle[models_by_interval[iv]["selected_features"]].values.reshape(1, -1)
                    else:
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

                        # Refined regime switching voting weights (Trending: CatBoost dominant; Ranging: LightGBM dominant)
                        ensemble_weights = [0.3, 0.2, 0.5] if adx_regime >= 20.0 else [0.3, 0.5, 0.2]
                        
                        pred_pct = float(active_model_price.predict(X_live, weights=ensemble_weights)[0])
                        pred_change = pred_pct * float(latest_candle["close"])
                        predicted_price = float(latest_candle["close"]) + pred_change
                        
                        # 3-class probabilities
                        probs = active_model_trend.predict_proba(X_live, weights=ensemble_weights)[0]
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

                        # Apply Isotonic Regression probability calibration if available
                        calibrator = models_tf["trending"]["calibrator"] if adx_regime >= 20.0 else models_tf["ranging"]["calibrator"]
                        if calibrator is not None and "X" in calibrator and "y" in calibrator and ml_trend in ["Bullish", "Bearish"]:
                            calibrated_confidence = float(np.interp(ml_confidence, calibrator["X"], calibrator["y"]))
                            print(f"[{symbol} {iv}m Isotonic Calibration] Raw: {ml_confidence*100:.2f}% -> Calibrated: {calibrated_confidence*100:.2f}%")
                        else:
                            # Fallback to piecewise linear calibration
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

                        print(f"[{iv}m] Regime Selected: {regime_name} | ML Output: {ml_trend} (Bull: {prob_bullish*100:.1f}%, Bear: {prob_bearish*100:.1f}%, Neut: {prob_neutral*100:.1f}%) | Raw Conf: {ml_confidence*100:.2f}% | Calibrated Conf: {calibrated_confidence*100:.2f}% | Expected Change: {pred_change:+.3f}")

                        # Determine dynamic confidence threshold based on regime and volatility
                        atr_norm_val = latest_candle["ATR_norm"]
                        dynamic_conf_threshold = 0.63
                        
                        # 1. Regime Adjustment (ADX)
                        if adx_regime >= 25.0:
                            dynamic_conf_threshold = 0.58
                        elif adx_regime < 15.0:
                            dynamic_conf_threshold = 0.65
                            
                        # 2. Volatility Adjustment (ATR)
                        if atr_norm_val > 0.008:
                            # High volatility has wider stops, increase threshold to enter only high-conviction trades
                            dynamic_conf_threshold = min(0.70, dynamic_conf_threshold + 0.03)
                        elif atr_norm_val < 0.003:
                            dynamic_conf_threshold = min(0.70, dynamic_conf_threshold + 0.02)
                            
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
                        
                        is_cooling, remaining_mins = is_symbol_interval_cooling_off(symbol, iv)
                        
                        status_msg = "Pending"
                        active_trade_key = f"active_trade_{tf}"
                        active_trades_list = bot_state.get(active_trade_key, [])
                        
                        # Prevent duplicate parallel trades of the same symbol on the same interval/timeframe
                        already_active = any(t.get("symbol") == symbol for t in active_trades_list)
                        
                        if not bot_state.get("bot_running", True):
                            status_msg = "Skipped (Bot Stopped)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Bot is currently stopped by the user.")
                        elif already_active:
                            status_msg = "Skipped (Already Active)"
                            print(f"[{symbol} {iv}m] Prediction skipped: A trade is already active for this symbol on this timeframe.")
                        elif is_cooling:
                            status_msg = "Skipped (Cool-Off)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Interval is in a 6-hour cool-off period after consecutive losses ({remaining_mins} mins remaining).")
                        elif ml_trend == "Neutral":
                            status_msg = "Skipped (Neutral)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Model output is Neutral/Hold.")
                        elif strong_conflict:
                            status_msg = "Skipped (Contradiction)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Strong directional contradiction (Trend: {ml_trend}, Regressor: {pred_change:+.3f} [{pred_pct:.3f}%]).")
                        elif calibrated_confidence < dynamic_conf_threshold:
                            status_msg = "Skipped (Low Confidence)"
                            print(f"[{symbol} {iv}m] Prediction skipped (calibrated confidence {calibrated_confidence*100:.2f}% < {dynamic_conf_threshold*100:.2f}%).")

                        if status_msg == "Pending":
                            # Check news window proximity status for logging purposes
                            in_news_window, news_event = is_high_impact_news_window()
                            if in_news_window:
                                print(f"[{iv}m WARNING] High-impact event window active ({news_event}). Proceeding since ML models incorporate news proximity features.")
                                
                            with news_sentiment_lock:
                                news_sentiment = cached_news_sentiment
                                latest_titles = cached_news_titles
                                all_pass, confluence_results, confluence_score_pct = check_pre_trade_confluence(
                                    latest_candle["close"], df, ml_trend, news_sentiment, expected_pct_change, iv, symbol=symbol, htf_cache=htf_cache
                                )

                                # Update global confluence status
                                bot_state[f"confluence_results_{tf}"] = {
                                    "approved": all_pass,
                                    "checks": confluence_results
                                }

                                print(f"\n==================================================")
                                print(f"[{iv}m] PRE-TRADE CONFLUENCE ANALYSIS REPORT")
                                print("--------------------------------------------------")
                                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbol: {symbol}")
                                print(f"Signal: {ml_trend} | Calibrated Confidence: {calibrated_confidence * 100:.2f}%")
                                print(f"Current Price: {latest_candle['close']:.2f} | Predicted Price: {predicted_price:.2f} (Expected: {pred_change:+.3f} [{expected_pct_change:.3f}%])")
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
                                    
                                    # Volatility (ATR)-Adaptive Take-Profit Multiplier
                                    # High Volatility (ATR_norm >= 0.008) -> Smaller targets to lock profits
                                    # Low Volatility (ATR_norm <= 0.003) -> Larger targets to capture extensions
                                    base_tp = 2.0 if latest_candle["ADX"] >= 20.0 else 1.2
                                    vol_factor = 1.0
                                    if atr_norm_val > 0:
                                        vol_factor = 1.5 - ((atr_norm_val - 0.003) / 0.005) * 0.75
                                        vol_factor = max(0.75, min(1.5, vol_factor))
                                    tp_multiplier = round(base_tp * vol_factor, 2)
                                    print(f"[{iv}m Volatility Sizing] ADX: {latest_candle['ADX']:.1f} (Base TP: {base_tp:.1f}) | ATR Norm: {atr_norm_val*100:.3f}% (Vol Factor: {vol_factor:.2f}x) -> Dynamic TP Multiplier: {tp_multiplier:.2f}x")
                                    
                                    # Align stop loss and take profit multipliers strictly with model training labels
                                    sl_multiplier = 0.80
                                    tp_multiplier_adjusted = 1.20
                                    print(f"[{iv}m Target Alignment] Aligned multipliers with ML training: SL = {sl_multiplier}x, TP = {tp_multiplier_adjusted}x")
                                    
                                    # Maker execution: zero entry slippage for limit orders
                                    slippage_pct = 0.0
                                    raw_entry_price = float(latest_candle["close"])
                                    entry_price = raw_entry_price

                                    if ml_trend == "Bullish":
                                        stop_loss_price = entry_price - sl_multiplier * atr_dollars
                                        take_profit_price = entry_price + tp_multiplier_adjusted * atr_dollars
                                    else:
                                        stop_loss_price = entry_price + sl_multiplier * atr_dollars
                                        take_profit_price = entry_price - tp_multiplier_adjusted * atr_dollars

                                    # Calibrated Position Sizing based on Isotonic Probability (Kelly scaling)
                                    c_prob = float(calibrated_confidence)
                                    if c_prob < 0.60:
                                        position_size_usd = 10.0
                                    elif c_prob <= 0.75:
                                        position_size_usd = 11.5
                                    else:
                                        position_size_usd = 13.0
                                        
                                    # Apply covariance multiplier and clamp final size between $10 and $13
                                    cov_multiplier, net_risk = calculate_covariance_multiplier(symbol, ml_trend)
                                    position_size_usd = max(10.0, min(13.0, position_size_usd * cov_multiplier))
                                    print(f"[{iv}m Calibrated Sizing] Calibrated Conf: {calibrated_confidence*100:.1f}% -> Final Position Size: ${position_size_usd:.2f} (Covariance: {cov_multiplier:.2f}x)")

                                    # Ensure total size of active trades does not exceed the wallet balance
                                    total_active_size = sum(t.get("position_size_usd", 0.0) for tf_key in ["1h", "2h", "4h", "6h"] for t in bot_state.get(f"active_trade_{tf_key}", []))
                                    current_bal = bot_state.get("simulated_balance", 80.0)
                                    
                                    wallet_exceeded = False
                                    if total_active_size + position_size_usd > current_bal:
                                        remaining_bal = current_bal - total_active_size
                                        if remaining_bal >= 10.0:
                                            print(f"[{symbol} {iv}m] Sizing scaled down from ${position_size_usd:.2f} to ${remaining_bal:.2f} to fit remaining wallet balance (Total Active: ${total_active_size:.2f}, Wallet: ${current_bal:.2f}).")
                                            position_size_usd = remaining_bal
                                        else:
                                            print(f"[{symbol} {iv}m] Trade skipped: Insufficient wallet balance to maintain minimum $10 trade size (Total Active: ${total_active_size:.2f}, Wallet: ${current_bal:.2f}, Proposed: ${position_size_usd:.2f}).")
                                            status_msg = "Skipped (Exceeds Wallet)"
                                            wallet_exceeded = True

                                    if not wallet_exceeded:
                                        # Dynamic Leverage Scaling: scale between 10x-15x (at 70% confidence) and 30x-50x (at 85%+)
                                        c = float(calibrated_confidence)
                                        if c >= 0.85:
                                            leverage_val = 35.0 + (c - 0.85) / 0.15 * 15.0
                                        else:
                                            leverage_val = 10.0 + (c - 0.70) / 0.15 * 25.0
                                            if leverage_val < 1.0:
                                                 leverage_val = 1.0

                                        # Risk check: cap leverage so stop loss doesn't exceed 90% of capital, with absolute limit at 50.0x
                                        stop_loss_pct = (sl_multiplier * atr_dollars / entry_price) * 100
                                        max_safe_lev = 90.0 / stop_loss_pct if stop_loss_pct > 0 else 100.0
                                        leverage_val = round(max(1.0, min(50.0, min(leverage_val, max_safe_lev))), 1)

                                        lookahead = 10
                                        duration_seconds = int(iv) * 60.0 * lookahead
                                        import uuid
                                        trade_uuid = str(uuid.uuid4())[:8]
                                        active_trade = {
                                            "trade_id": f"{symbol}_{trade_uuid}",
                                            "symbol": symbol,
                                            "entry_price": float(entry_price),
                                            "predicted_price": float(predicted_price),
                                            "stop_loss": float(stop_loss_price),
                                            "take_profit": float(take_profit_price),
                                            "direction": str(ml_trend),
                                            "end_time": float(time.time() + duration_seconds),
                                            "atr_dollars": float(atr_dollars),
                                            "highest_price": float(entry_price),
                                            "lowest_price": float(entry_price),
                                            "break_even_triggered": False,
                                            "half_closed": False,
                                            "original_size": float(position_size_usd),
                                            "position_size_usd": float(position_size_usd),
                                            "scaled_out_pnl": 0.0,
                                            "kelly_fraction": float(kelly_fraction),
                                            "leverage": float(leverage_val)
                                        }
                                        active_trades_list.append(active_trade)
                                        bot_state[active_trade_key] = active_trades_list
                                        
                                        # Deduct size from wallet balance immediately
                                        bot_state["simulated_balance"] = round(bot_state["simulated_balance"] - position_size_usd, 2)
                                        
                                        print(f"[{symbol} {iv}m] Trade Opened: {ml_trend} at price {entry_price:.2f} (SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}, Slippage: {slippage_pct:.3f}%)")
                                        print(f"[{iv}m Kelly Sizing] Confidence: {kelly_p*100:.2f}% | R:R ratio: {kelly_b:.2f} | Size: ${position_size_usd:.2f} | Leverage: {leverage_val}x (New Balance: ${bot_state['simulated_balance']:.2f})\n")
                                else:
                                    status_msg = "Skipped (Confluence Failed)"
                                    failed_list = [name.replace('_', ' ') for name, res_val in confluence_results.items() if not res_val["pass"] and name != '_Score_Summary']
                                    print("--------------------------------------------------")
                                    print(f"CONFLUENCE RESULT: REJECTED ({confluence_results.get('_Score_Summary', {}).get('detail', 'Score too low')})")
                                    print(f"Failed checks: {', '.join(failed_list)}")
                                    print("==================================================\n")
                        
                        # Prevent duplicate predictions for the same candle timestamp
                        exists = any(p.get("candle_timestamp") == int(latest_completed_ts) and p.get("interval") == iv and p.get("symbol") == symbol for p in bot_state["prediction_history"])
                        if not exists:
                            bot_state["prediction_history"].append({
                                "symbol": symbol,
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
                            print(f"[{symbol} {iv}m] Prediction for candle timestamp {get_local_time_str(latest_completed_ts/1000)} already exists in history. Skipping duplicate append.")
                        
                        evaluate_predictions(df_completed, iv, symbol)
                        save_history()
                        
                        last_processed_timestamps[last_ts_key] = latest_completed_ts
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error checking {iv}m candle close signals: {e}")

        time.sleep(10)

if __name__ == "__main__":
    import threading
    # Start background news sentiment updater thread
    threading.Thread(target=run_news_sentiment_updater, daemon=True).start()
    # Start background Bybit balance updater thread
    threading.Thread(target=run_bybit_balance_updater, daemon=True).start()
    # Start Bybit WebSocket feed in a background thread
    threading.Thread(target=start_ws, daemon=True).start()
    # Start Bybit REST API fallback price updater thread
    threading.Thread(target=run_fallback_price_updater, daemon=True).start()
    # Start local web dashboard server in a background thread
    threading.Thread(target=run_flask, daemon=True).start()
    # Start automated rolling retraining scheduler in a background thread
    threading.Thread(target=run_rolling_retrain_scheduler, daemon=True).start()
    main()