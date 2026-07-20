print("[System Debug] main.py global execution started.")
import sys
import re
from datetime import datetime

class CircularLogBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.logs_list = []
        self.original_stdout = sys.stdout
        
    def write(self, message):
        self.original_stdout.write(message)
        if message.strip():
            # Strip ANSI escape codes to keep logs clean in Telegram
            clean_msg = re.sub(r'\x1b\[[0-9;]*[mK]', '', message.strip())
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.logs_list.append(f"[{timestamp}] {clean_msg}")
            if len(self.logs_list) > self.capacity:
                self.logs_list.pop(0)
                
    def flush(self):
        self.original_stdout.flush()

    def __getattr__(self, name):
        return getattr(self.original_stdout, name)

log_buffer = CircularLogBuffer(capacity=100)
sys.stdout = log_buffer

import os
print("[System Debug] os imported.")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
print("[System Debug] Thread env vars set.")
import time
startup_time = time.time()

from dotenv import load_dotenv
load_dotenv()
import features as features_module
import state_manager

TRADE_MODE = os.environ.get("TRADE_MODE", "simulation").lower()
BYBIT_BASE_URL = "https://api-testnet.bybit.com" if TRADE_MODE == "testnet" else "https://api.bybit.com"
BYBIT_WS_URL = "wss://stream-testnet.bybit.com/v5/public/linear" if TRADE_MODE == "testnet" else "wss://stream.bybit.com/v5/public/linear"
BYBIT_PRIVATE_WS_URL = "wss://stream-testnet.bybit.com/v5/private" if TRADE_MODE == "testnet" else "wss://stream.bybit.com/v5/private"
private_ws_connected = False
private_ws_retry_delay = 5

active_public_ws = None
active_private_ws = None
last_private_ws_update_time = 0.0

# ==========================================
# TIMING & API/PROXY HIT INTERVALS
# ==========================================
CANDLE_CHECK_WINDOW_MINS = int(os.environ.get("CANDLE_CHECK_WINDOW_MINS", "5"))
CANDLE_CHECK_INTERVAL_SECS = int(os.environ.get("CANDLE_CHECK_INTERVAL_SECS", "20"))
BALANCE_UPDATE_INTERVAL_SECS = int(os.environ.get("BALANCE_UPDATE_INTERVAL_SECS", "120"))
POSITION_SYNC_INTERVAL_SECS = float(os.environ.get("POSITION_SYNC_INTERVAL_SECS", "30.0"))
POSITION_SYNC_IDLE_INTERVAL_SECS = float(os.environ.get("POSITION_SYNC_IDLE_INTERVAL_SECS", "120.0"))

# Centralized timeframe parameters for training labels and live execution alignment
TIMEFRAME_CONFIG = {
    "60": {   # 1H Timeframe
        "lookahead": 10,
        "sl_mult": 1.5,
        "tp_mult_ranging": 1.5,
        "tp_mult_trending": 2.5
    },
    "120": {  # 2H Timeframe
        "lookahead": 12,
        "sl_mult": 1.5,
        "tp_mult_ranging": 1.4,
        "tp_mult_trending": 2.2
    },
    "240": {  # 4H Timeframe
        "lookahead": 12,
        "sl_mult": 1.8,
        "tp_mult_ranging": 1.3,
        "tp_mult_trending": 2.0
    },
    "360": {  # 6H Timeframe
        "lookahead": 16,
        "sl_mult": 2.0,
        "tp_mult_ranging": 1.2,
        "tp_mult_trending": 1.8
    }
}


print("[System Debug] Importing websocket...")
import websocket
print("[System Debug] websocket imported.")
import json
import requests
import asyncio
import aiohttp
import threading
import time

# Dedicated background thread and loop for async HTTP calls
_async_loop = None
_aiohttp_session = None

def _ensure_async_loop():
    global _async_loop, _aiohttp_session
    if _async_loop is None or not _async_loop.is_running():
        _async_loop = asyncio.new_event_loop()
        def _run():
            asyncio.set_event_loop(_async_loop)
            _async_loop.run_forever()
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        while not _async_loop.is_running():
            time.sleep(0.01)
        async def _init_session():
            global _aiohttp_session
            connector = aiohttp.TCPConnector(ssl=False, limit=100, keepalive_timeout=30)
            _aiohttp_session = aiohttp.ClientSession(connector=connector)
        asyncio.run_coroutine_threadsafe(_init_session(), _async_loop).result()

print("[System Debug] Importing pandas/numpy/joblib...")
import pandas as pd
import numpy as np
import joblib
print("[System Debug] pandas/numpy/joblib imported.")
import threading
import time
from datetime import datetime, timedelta

def get_pkt_time():
    return datetime.utcnow() + timedelta(hours=5)

def execute_telegram_api_call(method: str, payload: dict) -> dict:
    """Helper to send POST requests to Telegram API using proxy routing."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return {}
        
    custom_url = os.environ.get("TELEGRAM_API_URL")
    if custom_url:
        if not custom_url.endswith("/"):
            custom_url += "/"
        url = f"{custom_url}bot{token}/{method}"
    else:
        url = f"https://api.telegram.org/bot{token}/{method}"
        
    tg_proxy = os.environ.get("TELEGRAM_PROXY") or os.environ.get("BYBIT_PROXY")
    proxies_dict = None
    if tg_proxy:
        if custom_url:
            # If using a custom Cloudflare worker URL, route using standard HTTP proxy CONNECT
            proxies_dict = {"http": tg_proxy, "https": tg_proxy}
        else:
            # If connecting directly to api.telegram.org, use socks5h to delegate DNS resolution
            if "://" in tg_proxy:
                tg_proxy_clean = tg_proxy.split("://", 1)[1]
            else:
                tg_proxy_clean = tg_proxy
            socks_proxy = f"socks5h://{tg_proxy_clean}"
            proxies_dict = {"http": socks_proxy, "https": socks_proxy}
            
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Connection": "close"
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=20, proxies=proxies_dict, verify=True)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception as e:
            err_str = str(e).lower()
            is_eof = "eof occurred" in err_str or "unexpected eof" in err_str or "connection aborted" in err_str
            
            if custom_url:
                try:
                    fallback_url = f"https://api.telegram.org/bot{token}/{method}"
                    fallback_proxies = None
                    if tg_proxy:
                        tg_clean = tg_proxy.split("://", 1)[-1]
                        fallback_proxies = {"http": f"socks5h://{tg_clean}", "https": f"socks5h://{tg_clean}"}
                    resp2 = requests.post(fallback_url, json=payload, headers=headers, timeout=20, proxies=fallback_proxies)
                    if resp2.status_code == 200:
                        return resp2.json()
                except Exception:
                    pass
                    
            if is_eof and attempt < 2:
                time.sleep(0.5)
                continue
                
            is_silent = method == "getUpdates" and any(x in err_str for x in ["timed out", "eof", "ssl", "connection"])
            if not is_silent:
                print(f"[Telegram API Exception] method={method}: {e}")
            return {}

def send_telegram_alert(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
        
    def _post():
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        execute_telegram_api_call("sendMessage", payload)
        
    threading.Thread(target=_post, daemon=True).start()
        
def estimate_liquidation_pool(df_history, direction, entry_price):
    """
    Estimates the location of the nearest high-leverage liquidation pool
    based on historical swing highs/lows (support and resistance levels).
    """
    import numpy as np
    
    # Look at the last 60 candles to find recent swing high/low
    lookback = min(len(df_history), 60)
    df_recent = df_history.iloc[-lookback:]
    
    if direction == "Bullish":
        # We are Long. Take Profit is above entry.
        # Short sellers entered near the recent swing high. Their liquidations (buy stops)
        # are clustered 1% to 2% above the swing high (representing 100x and 50x leverage liquidations).
        swing_high = float(df_recent["high"].max())
        # Target just inside the 50x/100x liquidation pool (1.2% above swing high)
        liq_pool_target = swing_high * 1.012
        # Ensure it's higher than entry price
        return max(liq_pool_target, entry_price * 1.005)
    elif direction == "Bearish":
        # We are Short. Take Profit is below entry.
        # Long buyers entered near the recent swing low. Their liquidations (sell stops)
        # are clustered 1% to 2% below the swing low.
        swing_low = float(df_recent["low"].min())
        # Target just inside the 50x/100x liquidation pool (1.2% below swing low)
        liq_pool_target = swing_low * 0.988
        # Ensure it's lower than entry price
        return min(liq_pool_target, entry_price * 0.995)
    else:
        return entry_price

def run_manual_confluence_report(symbol, interval):
    try:
        from data import get_history, merge_derivatives_sentiment_features, classify_market_regime
        import numpy as np
        df_raw = get_history(symbol=symbol, interval=interval, limit=300)
        if df_raw is None or len(df_raw) < 2:
            return f"❌ Failed to fetch price history from Bybit/Binance/Kraken for *{symbol}*."
            
        # Ensure close_btc is populated (required for features calculations)
        if symbol == "BTCUSDT":
            df_raw["close_btc"] = df_raw["close"]
        else:
            df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=300)
            if df_btc is not None and len(df_btc) > 0:
                df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                df_raw = pd.merge(df_raw, df_btc_sub, on="timestamp", how="inner")
            else:
                df_raw["close_btc"] = df_raw["close"]
                
        df_raw = merge_derivatives_sentiment_features(df_raw, symbol=symbol, interval=interval)
        df_features = add_features(df_raw)
        latest_candle = df_features.iloc[-1]
        
        iv = str(interval)
        models_tf = models_by_interval.get(iv)
        if not models_tf or not models_tf["trending"]["price"]:
            return "❌ Models are currently not fully loaded or active."
            
        # Unsupervised GMM Market Regime Classification
        regime = classify_market_regime(df_features, interval=interval)
        if regime == "Trending":
            active_model_price = models_tf["trending"]["price"]
            active_model_trend = models_tf["trending"]["trend"]
            calibrator = models_tf["trending"]["calibrator"]
            feat_list = models_tf.get("selected_features_trending")
        else:
            active_model_price = models_tf["ranging"]["price"]
            active_model_trend = models_tf["ranging"]["trend"]
            calibrator = models_tf["ranging"]["calibrator"]
            feat_list = models_tf.get("selected_features_ranging")
            
        if feat_list is not None:
            X_live = latest_candle[feat_list].values.reshape(1, -1)
        else:
            X_live = latest_candle[features].values.reshape(1, -1)
            
        ensemble_weights = [0.3, 0.2, 0.5] if regime == "Trending" else [0.3, 0.5, 0.2]
        pred_pct = float(active_model_price.predict(X_live, weights=ensemble_weights)[0])
        pred_change = pred_pct * float(latest_candle["close"])
        expected_pct_change = (abs(pred_change) / latest_candle["close"]) * 100
        
        probs = active_model_trend.predict_proba(X_live, weights=ensemble_weights)[0]
        winning_class = int(np.argmax(probs))
        if winning_class == 2:
            ml_trend = "Bullish"
            ml_confidence = float(probs[2])
        elif winning_class == 0:
            ml_trend = "Bearish"
            ml_confidence = float(probs[0])
        else:
            ml_trend = "Neutral"
            ml_confidence = float(probs[1])
            
        if calibrator is not None and "X" in calibrator and "y" in calibrator and ml_trend in ["Bullish", "Bearish"]:
            calibrated_confidence = float(np.interp(ml_confidence, calibrator["X"], calibrator["y"]))
        else:
            calibration = bot_state.get(f"calibration_{iv.replace('60','1h').replace('120','2h').replace('240','4h').replace('360','6h')}", {"p95": 0.8, "max_conf": 0.95})
            calibrated_confidence = calibrate_confidence(ml_confidence, calibration["p95"], calibration["max_conf"])
            
        with news_sentiment_lock:
            news_sentiment = cached_news_sentiment
            
        all_pass, confluence_results, confluence_score_pct = check_pre_trade_confluence(
            latest_candle["close"], df_features, ml_trend, news_sentiment, expected_pct_change, iv, symbol=symbol,
            calibrated_confidence=calibrated_confidence, dynamic_conf_threshold=0.58
        )
        
        adx_regime = latest_candle["ADX"]
        est_tp_val = estimate_liquidation_pool(df_features, ml_trend, latest_candle["close"])
        
        report = (
            f"🔍 *CONFLUENCE REPORT: {symbol} ({iv.replace('60','1H').replace('120','2H').replace('240','4H').replace('360','6H')})*\n"
            f"• *Signal*: {ml_trend} ({calibrated_confidence*100:.1f}% confidence)\n"
            f"• *Regime*: {regime} (ADX: {adx_regime:.1f})\n"
            f"• *Expected Move*: {pred_change:+.4f} ({expected_pct_change:.2f}%)\n"
            f"• *Liquidation TP Target*: ${est_tp_val:.2f}\n"
            f"• *Decision*: *{'APPROVED' if all_pass else 'REJECTED'}*\n\n"
            f"*Check Details:*\n"
        )
        for idx, (check_name, res_val) in enumerate(confluence_results.items(), 1):
            if check_name == "_Score_Summary":
                continue
            circle = "🟢" if res_val["pass"] else "🔴"
            report += f"{circle} *{check_name.replace('_', ' ')}*: {res_val['detail']}\n"
            
        report += f"\n📊 *{confluence_results.get('_Score_Summary', {}).get('detail', '')}*"
        return report
    except Exception as e:
        return f"❌ *Error running manual check:* {str(e)}"

def start_telegram_command_listener():
    """Starts the background thread to poll and handle incoming Telegram commands."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    raw_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not raw_chat_id:
        print("[Telegram Command Listener] Unconfigured credentials. Listener skipped.")
        return
        
    # Build list of allowed chat IDs (support comma-separated string)
    allowed_chat_ids = [cid.strip() for cid in raw_chat_id.split(",") if cid.strip()]
    if "8827929671" not in allowed_chat_ids:
        allowed_chat_ids.append("8827929671")
        
    # Load dynamically authorized chat IDs
    with bot_state_lock:
        dyn_list = bot_state.get("telegram_allowed_ids", [])
        for dyn_id in dyn_list:
            if dyn_id not in allowed_chat_ids:
                allowed_chat_ids.append(dyn_id)

    pending_auth = {} # {sender_chat_id: {"code": str, "step": str, "timestamp": float}}
    pending_confluence = {} # {sender_chat_id: {"step": str, "symbol": str, "timestamp": float}}
    pending_manual_trade = {} # {sender_chat_id: {"step": str, "timestamp": float}}
 
    def listener_loop():
        offset = 0
        init_res = execute_telegram_api_call("getUpdates", {"limit": 1})
        if init_res.get("ok") and init_res.get("result"):
            offset = init_res["result"][-1]["update_id"] + 1
 
        # Automatically configure the Telegram bot commands menu
        commands_payload = {
            "commands": [
                {"command": "active", "description": "View all active open trades"},
                {"command": "balance", "description": "View account/wallet balance"},
                {"command": "profit", "description": "View profit/loss stats of all days"},
                {"command": "skipped", "description": "View recently skipped/filtered trades"},
                {"command": "confluence", "description": "Get live confluence report for coin"},
                {"command": "create_manual_trade", "description": "Open a manual trade with bot management"},
                {"command": "clean_duplicates", "description": "Prune duplicate active trade records from memory"},
                {"command": "retrain_status", "description": "View model retraining status"},
                {"command": "latency", "description": "Check Bybit API round-trip latency"},
                {"command": "logs", "description": "View latest bot running logs"},
                {"command": "add_user", "description": "Authorize a new user via email verification"},
                {"command": "retrain", "description": "Retrain models with recent live trade feedback"},
                {"command": "stop_all", "description": "Emergency stop bot and close all trades"},
                {"command": "start_bot", "description": "Resume bot and enable new trade entries"}
            ]
        }
        execute_telegram_api_call("setMyCommands", commands_payload)
 
        print(f"[Telegram Command Listener] Started polling background loop (initial offset={offset}).")
        while True:
            try:
                updates_res = execute_telegram_api_call("getUpdates", {"offset": offset, "timeout": 5})
                if updates_res.get("ok") and updates_res.get("result"):
                    for update in updates_res["result"]:
                        offset = update["update_id"] + 1
                        message_obj = update.get("message")
                        if not message_obj:
                            continue
                        
                        sender_chat_id = str(message_obj.get("chat", {}).get("id"))
                        text = message_obj.get("text", "").strip()
                        if text.startswith("/"):
                            parts = text.split(" ")
                            cmd = parts[0]
                            if "@" in cmd:
                                cmd = cmd.split("@")[0]
                            parts[0] = cmd
                            text = " ".join(parts)
                        print(f"[Telegram Command Listener] Received message: '{text}' from chat_id '{sender_chat_id}'")
                        
                        # Handle verification flow logic resets
                        if text in ["/cancel", "/add_user"] and sender_chat_id in pending_auth:
                            pending_auth.pop(sender_chat_id, None)
                            
                        if sender_chat_id in pending_auth:
                            user_flow = pending_auth[sender_chat_id]
                            if time.time() - user_flow["timestamp"] > 300:
                                pending_auth.pop(sender_chat_id, None)
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": "❌ *Session expired.* Please start over by sending /add_user.",
                                    "parse_mode": "Markdown"
                                })
                                continue
                                
                            if user_flow["step"] == "awaiting_code":
                                if text == user_flow["code"]:
                                    user_flow["step"] = "awaiting_chat_id"
                                    user_flow["timestamp"] = time.time()
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "✅ *Authentication successful!*\n\nPlease reply with the new Telegram Chat ID you want to authorize.",
                                        "parse_mode": "Markdown"
                                    })
                                else:
                                    pending_auth.pop(sender_chat_id, None)
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Invalid verification code.* Request cancelled.",
                                        "parse_mode": "Markdown"
                                    })
                                continue
                                
                            elif user_flow["step"] == "awaiting_chat_id":
                                if text.isdigit():
                                    new_id = text
                                    with bot_state_lock:
                                        dyn_list = bot_state.get("telegram_allowed_ids", [])
                                        if new_id not in allowed_chat_ids:
                                            allowed_chat_ids.append(new_id)
                                        if new_id not in dyn_list:
                                            dyn_list.append(new_id)
                                            bot_state["telegram_allowed_ids"] = dyn_list
                                            save_history()
                                            
                                    pending_auth.pop(sender_chat_id, None)
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": f"🎉 *Success! Chat ID {new_id} is now authorized to use the bot.*",
                                        "parse_mode": "Markdown"
                                    })
                                else:
                                    pending_auth.pop(sender_chat_id, None)
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Invalid Chat ID format.* Request cancelled.",
                                        "parse_mode": "Markdown"
                                    })
                                continue

                        if sender_chat_id not in allowed_chat_ids:
                            print(f"[Telegram Command Listener] Mismatched chat ID: expected one of {allowed_chat_ids}, got '{sender_chat_id}'")
                            continue
                        
                        # Handle confluence interactive flow logic resets
                        if text in ["/cancel", "/add_user", "/confluence"] and sender_chat_id in pending_confluence:
                            pending_confluence.pop(sender_chat_id, None)
                            
                        # Handle manual trade interactive flow resets
                        if text in ["/cancel", "/create_manual_trade"] and sender_chat_id in pending_manual_trade:
                            pending_manual_trade.pop(sender_chat_id, None)
                            
                        if sender_chat_id in pending_manual_trade:
                            flow = pending_manual_trade[sender_chat_id]
                            if time.time() - flow["timestamp"] > 300:
                                pending_manual_trade.pop(sender_chat_id, None)
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": "❌ *Session expired.* Please start over by sending /create_manual_trade.",
                                    "parse_mode": "Markdown",
                                    "reply_markup": {"remove_keyboard": True}
                                })
                                continue
                                
                            if text == "/cancel":
                                pending_manual_trade.pop(sender_chat_id, None)
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": "❌ *Operation cancelled.*",
                                    "parse_mode": "Markdown",
                                    "reply_markup": {"remove_keyboard": True}
                                })
                                continue
                                
                            step = flow["step"]
                            
                            # 1. Symbol Step
                            if step == "awaiting_symbol":
                                symbol = text.upper().strip()
                                if not symbol.endswith("USDT"):
                                    symbol += "USDT"
                                    
                                if not (3 <= len(symbol) <= 12):
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Invalid symbol format.* Please enter a valid coin name (e.g., BTC, ETHUSDT):",
                                        "parse_mode": "Markdown"
                                    })
                                    continue
                                    
                                flow["symbol"] = symbol
                                flow["step"] = "awaiting_direction"
                                flow["timestamp"] = time.time()
                                
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": f"Select the *Direction* for {symbol}:",
                                    "parse_mode": "Markdown",
                                    "reply_markup": {
                                        "keyboard": [[{"text": "Long (Buy)"}, {"text": "Short (Sell)"}], [{"text": "/cancel"}]],
                                        "resize_keyboard": True,
                                        "one_time_keyboard": True
                                    }
                                })
                                continue
                                
                            # 2. Direction Step
                            elif step == "awaiting_direction":
                                val = text.lower()
                                if "long" in val or "buy" in val:
                                    direction = "Bullish"
                                elif "short" in val or "sell" in val:
                                    direction = "Bearish"
                                else:
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Invalid direction.* Please select 'Long (Buy)' or 'Short (Sell)':",
                                        "parse_mode": "Markdown"
                                    })
                                    continue
                                    
                                flow["direction"] = direction
                                flow["step"] = "awaiting_entry"
                                flow["timestamp"] = time.time()
                                
                                symbol = flow["symbol"]
                                cur_p = bot_state.get(f"live_price_{symbol}")
                                if cur_p is None:
                                    cur_p = get_fallback_price(symbol)
                                    
                                keyboard = []
                                text_msg = "Enter your desired *Entry Price*:"
                                if cur_p:
                                    flow["live_price"] = cur_p
                                    text_msg = f"💡 *Current Price:* `{cur_p:.4f}`\n\nEnter or select your *Entry Price*:"
                                    keyboard = [[{"text": f"Current Price ({cur_p:.4f})"}], [{"text": "/cancel"}]]
                                else:
                                    keyboard = [[{"text": "/cancel"}]]
                                    
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": text_msg,
                                    "parse_mode": "Markdown",
                                    "reply_markup": {
                                        "keyboard": keyboard,
                                        "resize_keyboard": True,
                                        "one_time_keyboard": True
                                    }
                                })
                                continue
                                
                            # 3. Entry Price Step
                            elif step == "awaiting_entry":
                                symbol = flow["symbol"]
                                raw_text = text.lower()
                                if "current price" in raw_text:
                                    entry_price = flow.get("live_price")
                                    if not entry_price:
                                        entry_price = get_fallback_price(symbol)
                                    flow["entry_type"] = "market"
                                else:
                                    try:
                                        entry_price = float(text)
                                        if entry_price <= 0:
                                            raise ValueError
                                        flow["entry_type"] = "limit"
                                    except ValueError:
                                        execute_telegram_api_call("sendMessage", {
                                            "chat_id": sender_chat_id,
                                            "text": "❌ *Invalid price.* Please enter a positive number for entry price:",
                                            "parse_mode": "Markdown"
                                        })
                                        continue
                                        
                                if not entry_price or entry_price <= 0:
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Could not resolve entry price.* Please type a manual price:",
                                        "parse_mode": "Markdown"
                                    })
                                    continue
                                    
                                flow["entry_price"] = entry_price
                                flow["step"] = "awaiting_investment"
                                flow["timestamp"] = time.time()
                                
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": "Enter the *Investment margin (USD)* to allocate to this trade (e.g., 20):",
                                    "parse_mode": "Markdown",
                                    "reply_markup": {"remove_keyboard": True}
                                })
                                continue
                                
                            # 4. Investment Step
                            elif step == "awaiting_investment":
                                try:
                                    investment = float(text)
                                    if investment <= 0:
                                        raise ValueError
                                except ValueError:
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Invalid amount.* Please enter a positive number for investment margin:",
                                        "parse_mode": "Markdown"
                                    })
                                    continue
                                    
                                flow["investment"] = investment
                                flow["step"] = "awaiting_leverage"
                                flow["timestamp"] = time.time()
                                
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": "Enter or select the *Leverage* (e.g., 5 or 10):",
                                    "parse_mode": "Markdown",
                                    "reply_markup": {
                                        "keyboard": [[{"text": "1x"}, {"text": "2x"}, {"text": "5x"}, {"text": "10x"}], [{"text": "/cancel"}]],
                                        "resize_keyboard": True,
                                        "one_time_keyboard": True
                                    }
                                })
                                continue
                                
                            # 5. Leverage Step
                            elif step == "awaiting_leverage":
                                try:
                                    lev_str = text.lower().replace("x", "")
                                    leverage = int(lev_str)
                                    if leverage <= 0 or leverage > 100:
                                        raise ValueError
                                except ValueError:
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Invalid leverage.* Please enter a valid integer between 1 and 100:",
                                        "parse_mode": "Markdown"
                                    })
                                    continue
                                    
                                flow["leverage"] = leverage
                                flow["step"] = "awaiting_tp"
                                flow["timestamp"] = time.time()
                                
                                suggested_tp = None
                                suggested_sl = None
                                symbol = flow["symbol"]
                                direction = flow["direction"]
                                entry_p = flow["entry_price"]
                                
                                try:
                                    calc_atr = 0.015 * entry_p
                                    if symbol in SUPPORTED_SYMBOLS:
                                        df_temp = get_history(symbol, "60", limit=100)
                                        if not df_temp.empty:
                                            high_low = df_temp["high"] - df_temp["low"]
                                            high_close = (df_temp["high"] - df_temp["close"].shift()).abs()
                                            low_close = (df_temp["low"] - df_temp["close"].shift()).abs()
                                            ranges = pd.concat([high_low, high_close, low_close], axis=1)
                                            true_range = ranges.max(axis=1)
                                            calc_atr = float(true_range.rolling(14).mean().iloc[-1])
                                            
                                    flow["atr_dollars"] = calc_atr
                                    
                                    if direction == "Bullish":
                                        suggested_tp = entry_p + 1.5 * calc_atr
                                        suggested_sl = entry_p - 1.0 * calc_atr
                                    else:
                                        suggested_tp = entry_p - 1.5 * calc_atr
                                        suggested_sl = entry_p + 1.0 * calc_atr
                                except Exception:
                                    pass
                                    
                                reply_markup = {"remove_keyboard": True}
                                msg_text = "Enter your *Take Profit (TP) price*:"
                                if suggested_tp:
                                    flow["suggested_tp"] = round(suggested_tp, 4)
                                    flow["suggested_sl"] = round(suggested_sl, 4)
                                    msg_text = f"💡 *Suggested TP (1.5x ATR):* `{suggested_tp:.4f}`\n\nEnter or select your *Take Profit (TP) price*:"
                                    reply_markup = {
                                        "keyboard": [[{"text": f"{suggested_tp:.4f}"}], [{"text": "/cancel"}]],
                                        "resize_keyboard": True,
                                        "one_time_keyboard": True
                                    }
                                    
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": msg_text,
                                    "parse_mode": "Markdown",
                                    "reply_markup": reply_markup
                                })
                                continue
                                
                            # 6. TP Step
                            elif step == "awaiting_tp":
                                try:
                                    tp = float(text)
                                    if tp <= 0:
                                        raise ValueError
                                except ValueError:
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Invalid TP.* Please enter a positive number for TP price:",
                                        "parse_mode": "Markdown"
                                    })
                                    continue
                                    
                                flow["tp"] = tp
                                flow["step"] = "awaiting_sl"
                                flow["timestamp"] = time.time()
                                
                                reply_markup = {"remove_keyboard": True}
                                msg_text = "Enter your *Stop Loss (SL) price*:"
                                suggested_sl = flow.get("suggested_sl")
                                if suggested_sl:
                                    msg_text = f"💡 *Suggested SL (1.0x ATR):* `{suggested_sl:.4f}`\n\nEnter or select your *Stop Loss (SL) price*:"
                                    reply_markup = {
                                        "keyboard": [[{"text": f"{suggested_sl:.4f}"}], [{"text": "/cancel"}]],
                                        "resize_keyboard": True,
                                        "one_time_keyboard": True
                                    }
                                    
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": msg_text,
                                    "parse_mode": "Markdown",
                                    "reply_markup": reply_markup
                                })
                                continue
                                
                            # 7. SL Step
                            elif step == "awaiting_sl":
                                try:
                                    sl = float(text)
                                    if sl <= 0:
                                        raise ValueError
                                except ValueError:
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Invalid SL.* Please enter a positive number for SL price:",
                                        "parse_mode": "Markdown"
                                    })
                                    continue
                                    
                                flow["sl"] = sl
                                flow["step"] = "awaiting_confirm"
                                flow["timestamp"] = time.time()
                                
                                summary = (
                                    f"📊 *MANUAL TRADE SUMMARY* 📊\n\n"
                                    f"• *Symbol*: {flow['symbol']}\n"
                                    f"• *Direction*: {flow['direction']}\n"
                                    f"• *Entry Type*: {flow['entry_type'].upper()}\n"
                                    f"• *Entry Price*: ${flow['entry_price']:.4f}\n"
                                    f"• *Investment*: ${flow['investment']:.2f}\n"
                                    f"• *Leverage*: {flow['leverage']}x\n"
                                    f"• *TP Price*: ${flow['tp']:.4f}\n"
                                    f"• *SL Price*: ${flow['sl']:.4f}\n\n"
                                    f"Please confirm if you want to open this trade on Bybit:"
                                )
                                
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": summary,
                                    "parse_mode": "Markdown",
                                    "reply_markup": {
                                        "keyboard": [[{"text": "Confirm Open"}, {"text": "Cancel"}], [{"text": "/cancel"}]],
                                        "resize_keyboard": True,
                                        "one_time_keyboard": True
                                    }
                                })
                                continue
                                
                            # 8. Confirmation Step
                            elif step == "awaiting_confirm":
                                if "confirm" in text.lower():
                                    symbol = flow["symbol"]
                                    direction = flow["direction"]
                                    margin = flow["investment"]
                                    leverage = flow["leverage"]
                                    tp = flow["tp"]
                                    sl = flow["sl"]
                                    entry_type = flow["entry_type"]
                                    entry_price = flow["entry_price"]
                                    atr_val = flow.get("atr_dollars", 0.015 * entry_price)
                                    
                                    pending_manual_trade.pop(sender_chat_id, None)
                                    
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": f"⏳ Sending {entry_type} order to Bybit for *{symbol}*...",
                                        "parse_mode": "Markdown",
                                        "reply_markup": {"remove_keyboard": True}
                                    })
                                    
                                    def _execute_manual_trade_bg(cid, sym, d, m, l, t_p, s_l, atr, e_type, e_price):
                                        try:
                                            set_bybit_leverage(sym, l)
                                            
                                            notional = m * l
                                            qty = notional / e_price
                                            qty_str = format_bybit_qty(sym, qty)
                                            actual_qty = float(qty_str)
                                            
                                            side = "Buy" if d == "Bullish" else "Sell"
                                            
                                            if e_type == "limit":
                                                # Place limit order directly with TP/SL attached
                                                order_res = place_bybit_limit_order(sym, side, qty_str, e_price, sl=s_l, tp=t_p)
                                                if order_res and order_res.get("retCode") == 0:
                                                    execute_telegram_api_call("sendMessage", {
                                                        "chat_id": cid,
                                                        "text": (
                                                            f"🎯 *MANUAL LIMIT ORDER PLACED* 🎯\n\n"
                                                            f"• *Asset*: {sym}\n"
                                                            f"• *Side*: {side}\n"
                                                            f"• *Limit Price*: ${e_price:.4f}\n"
                                                            f"• *Size*: ${m:.2f} ({l}x leverage)\n"
                                                            f"• *TP Price*: ${t_p:.4f}\n"
                                                            f"• *SL Price*: ${s_l:.4f}\n\n"
                                                            f"_Once the limit order gets filled on Bybit, the bot will automatically pick up the position and start tracking it._"
                                                        ),
                                                        "parse_mode": "Markdown"
                                                    })
                                                else:
                                                    err_msg = order_res.get("retMsg", "Unknown error")
                                                    execute_telegram_api_call("sendMessage", {
                                                        "chat_id": cid,
                                                        "text": f"❌ *Failed to place Limit order on Bybit:* {err_msg}",
                                                        "parse_mode": "Markdown"
                                                    })
                                            else:
                                                # Market order execution
                                                order_res = place_bybit_order(sym, side, qty_str, reduce_only=False)
                                                if order_res and order_res.get("retCode") == 0:
                                                    res_data = order_res.get("result", {})
                                                    order_id = res_data.get("orderId", "MANUAL")
                                                    fill_price = e_price
                                                    
                                                    try:
                                                        time.sleep(0.5)
                                                        pos_info = get_bybit_position(sym)
                                                        if pos_info:
                                                            fill_price = float(pos_info.get("avgPrice", fill_price))
                                                    except Exception:
                                                        pass
                                                        
                                                    temp_trade = {"qty": str(actual_qty), "direction": d}
                                                    update_bybit_stop_loss(sym, s_l, temp_trade)
                                                    update_bybit_take_profit(sym, t_p, temp_trade)
                                                    
                                                    limit_side = "Sell" if d == "Bullish" else "Buy"
                                                    limit_price = fill_price + 1.0 * atr if d == "Bullish" else fill_price - 1.0 * atr
                                                    limit_qty_str = format_bybit_qty(sym, actual_qty * 0.5)
                                                    scale_out_order_id = None
                                                    if (float(limit_qty_str) * limit_price) >= 5.0:
                                                        limit_res = place_bybit_limit_order(sym, limit_side, limit_qty_str, limit_price, reduce_only=True)
                                                        if limit_res.get("retCode") == 0:
                                                            scale_out_order_id = limit_res.get("result", {}).get("orderId")
                                                            
                                                    import uuid
                                                    trade_uuid = str(uuid.uuid4())[:8]
                                                    new_trade = {
                                                        "trade_id": f"{sym}_{trade_uuid}",
                                                        "bybit_order_id": order_id,
                                                        "bybit_scale_out_order_id": scale_out_order_id,
                                                        "symbol": sym,
                                                        "entry_price": fill_price,
                                                        "predicted_price": fill_price,
                                                        "stop_loss": s_l,
                                                        "take_profit": t_p,
                                                        "direction": d,
                                                        "end_time": float(time.time() + 3600 * 48),
                                                        "entry_time": int(time.time() * 1000),
                                                        "atr_dollars": atr,
                                                        "highest_price": fill_price,
                                                        "lowest_price": fill_price,
                                                        "break_even_triggered": False,
                                                        "half_closed": False,
                                                        "original_size": m,
                                                        "position_size_usd": m,
                                                        "fill_pct": 100.0,
                                                        "original_qty": actual_qty,
                                                        "qty": actual_qty,
                                                        "leverage": float(l),
                                                        "confidence": "MT",
                                                        "recovered": False
                                                    }
                                                    
                                                    with active_trades_lock:
                                                        active_list = bot_state.get("active_trade_1h", [])
                                                        exists = any(t.get("symbol") == sym for t in active_list)
                                                        if not exists:
                                                            active_list.append(new_trade)
                                                            bot_state["active_trade_1h"] = active_list
                                                            save_history()
                                                        
                                                    execute_telegram_api_call("sendMessage", {
                                                        "chat_id": cid,
                                                        "text": (
                                                            f"🚀 *MANUAL TRADE OPENED ON BYBIT* 🚀\n\n"
                                                            f"• *Asset*: {sym}\n"
                                                            f"• *Direction*: {d}\n"
                                                            f"• *Entry Price*: ${fill_price:.4f}\n"
                                                            f"• *Size*: ${m:.2f} ({l}x leverage)\n"
                                                            f"• *TP Price*: ${t_p:.4f}\n"
                                                            f"• *SL Price*: ${s_l:.4f}\n\n"
                                                            f"_The bot will now manage this position automatically._"
                                                        ),
                                                        "parse_mode": "Markdown"
                                                    })
                                                else:
                                                    err_msg = order_res.get("retMsg", "Unknown error")
                                                    execute_telegram_api_call("sendMessage", {
                                                        "chat_id": cid,
                                                        "text": f"❌ *Failed to open trade on Bybit:* {err_msg}",
                                                        "parse_mode": "Markdown"
                                                    })
                                        except Exception as e_run:
                                            execute_telegram_api_call("sendMessage", {
                                                "chat_id": cid,
                                                "text": f"❌ *Runtime error during order placement:* {str(e_run)}",
                                                "parse_mode": "Markdown"
                                            })
                                            
                                    threading.Thread(target=_execute_manual_trade_bg, args=(sender_chat_id, symbol, direction, margin, leverage, tp, sl, atr_val, entry_type, entry_price), daemon=True).start()
                                    continue
                                else:
                                    pending_manual_trade.pop(sender_chat_id, None)
                                    execute_telegram_api_call("sendMessage", {
                                        "chat_id": sender_chat_id,
                                        "text": "❌ *Operation cancelled.*",
                                        "parse_mode": "Markdown",
                                        "reply_markup": {"remove_keyboard": True}
                                    })
                                    continue
                                    
                        # Handle confluence interactive flow logic resets
                        if text in ["/cancel", "/add_user", "/confluence"] and sender_chat_id in pending_confluence:
                            pending_confluence.pop(sender_chat_id, None)
                            
                        if sender_chat_id in pending_confluence:
                            flow = pending_confluence[sender_chat_id]
                            if time.time() - flow["timestamp"] > 300:
                                pending_confluence.pop(sender_chat_id, None)
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": "❌ *Session expired.* Please start over by sending /confluence.",
                                    "parse_mode": "Markdown",
                                    "reply_markup": {"remove_keyboard": True}
                                })
                                continue
                                
                            if text == "/cancel":
                                pending_confluence.pop(sender_chat_id, None)
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": sender_chat_id,
                                    "text": "❌ *Operation cancelled.*",
                                    "parse_mode": "Markdown",
                                    "reply_markup": {"remove_keyboard": True}
                                })
                                continue
                                
                            if text.startswith("/"):
                                pending_confluence.pop(sender_chat_id, None)
                            else:
                                if flow["step"] == "awaiting_symbol":
                                    raw_sym = text.upper()
                                    target_sym = raw_sym if raw_sym.endswith("USDT") else f"{raw_sym}USDT"
                                    if target_sym not in SUPPORTED_SYMBOLS:
                                        reply_text = f"❌ *Symbol {target_sym} is not supported.*\n\nActive symbols: {', '.join(SUPPORTED_SYMBOLS)}\n\nPlease select one of the supported symbols from the keyboard or send /cancel to abort."
                                        
                                        syms_keyboard = []
                                        row = []
                                        for idx, s in enumerate(SUPPORTED_SYMBOLS):
                                            row.append({"text": s})
                                            if (idx + 1) % 3 == 0:
                                                syms_keyboard.append(row)
                                                row = []
                                        if row:
                                            syms_keyboard.append(row)
                                        syms_keyboard.append([{"text": "/cancel"}])
                                        
                                        execute_telegram_api_call("sendMessage", {
                                            "chat_id": sender_chat_id,
                                            "text": reply_text,
                                            "parse_mode": "Markdown",
                                            "reply_markup": {
                                                "keyboard": syms_keyboard,
                                                "resize_keyboard": True,
                                                "one_time_keyboard": True
                                            }
                                        })
                                    else:
                                        flow["symbol"] = target_sym
                                        flow["step"] = "awaiting_timeframe"
                                        flow["timestamp"] = time.time()
                                        
                                        tf_keyboard = [
                                            [{"text": "1h"}, {"text": "2h"}],
                                            [{"text": "4h"}, {"text": "6h"}],
                                            [{"text": "/cancel"}]
                                        ]
                                        execute_telegram_api_call("sendMessage", {
                                            "chat_id": sender_chat_id,
                                            "text": f"✅ *Selected Symbol:* {target_sym}\n\nPlease select a *Timeframe* below:",
                                            "parse_mode": "Markdown",
                                            "reply_markup": {
                                                "keyboard": tf_keyboard,
                                                "resize_keyboard": True,
                                                "one_time_keyboard": True
                                            }
                                        })
                                    continue
                                    
                                elif flow["step"] == "awaiting_timeframe":
                                    tf_str = text.lower()
                                    tf_mapping = {"1h": "60", "2h": "120", "4h": "240", "6h": "360"}
                                    if tf_str not in tf_mapping:
                                        reply_text = f"❌ *Timeframe {tf_str} is not supported.*\n\nSupported: `1h`, `2h`, `4h`, `6h`\n\nPlease select a supported timeframe from the keyboard or send /cancel to abort."
                                        tf_keyboard = [
                                            [{"text": "1h"}, {"text": "2h"}],
                                            [{"text": "4h"}, {"text": "6h"}],
                                            [{"text": "/cancel"}]
                                        ]
                                        execute_telegram_api_call("sendMessage", {
                                            "chat_id": sender_chat_id,
                                            "text": reply_text,
                                            "parse_mode": "Markdown",
                                            "reply_markup": {
                                                "keyboard": tf_keyboard,
                                                "resize_keyboard": True,
                                                "one_time_keyboard": True
                                            }
                                        })
                                    else:
                                        target_sym = flow["symbol"]
                                        pending_confluence.pop(sender_chat_id, None)
                                        
                                        execute_telegram_api_call("sendMessage", {
                                            "chat_id": sender_chat_id,
                                            "text": f"⏳ Fetching live market metrics for *{target_sym}* ({tf_str})...",
                                            "parse_mode": "Markdown",
                                            "reply_markup": {"remove_keyboard": True}
                                        })
                                        
                                        def _run_confluence_bg(cid, sym, interval_val):
                                            rep = run_manual_confluence_report(sym, interval_val)
                                            execute_telegram_api_call("sendMessage", {
                                                "chat_id": cid,
                                                "text": rep,
                                                "parse_mode": "Markdown"
                                            })
                                        threading.Thread(target=_run_confluence_bg, args=(sender_chat_id, target_sym, tf_mapping[tf_str]), daemon=True).start()
                                    continue

                        if text == "/active":
                            active_trades_summary = []
                            with active_trades_lock:
                                for tf_key in ["1h", "2h", "4h", "6h"]:
                                    trades = bot_state.get(f"active_trade_{tf_key}", [])
                                    for t in trades:
                                        symbol = t.get("symbol")
                                        direction = t.get("direction")
                                        entry_p = t.get("entry_price")
                                        current_p = bot_state.get(f"live_price_{symbol}") or entry_p
                                        stop_loss = t.get("stop_loss")
                                        take_profit = t.get("take_profit")
                                        confidence = t.get("confidence", 0.63)
                                        conf_str = "MT" if confidence == "MT" else f"{float(confidence) * 100:.1f}%"
                                        leverage = t.get("leverage", 1.0)
                                        size_usd = t.get("position_size_usd", 0.0)
                                        
                                        change_pct = ((current_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0
                                        raw_pnl = change_pct if direction == "Bullish" else -change_pct
                                        pnl_pct = raw_pnl * leverage
                                        pnl_usd = size_usd * (pnl_pct / 100.0)
                                        
                                        active_trades_summary.append(
                                            f"💼 *{symbol} ({tf_key.upper()})*\n"
                                            f"• *Signal*: {direction}\n"
                                            f"• *Confidence*: {conf_str}\n"
                                            f"• *Leverage*: {leverage:.1f}x\n"
                                            f"• *Investment*: ${size_usd:.2f} (Value: ${size_usd * leverage:.2f})\n"
                                            f"• *Price*: ${current_p:.2f} (Entry: ${entry_p:.2f})\n"
                                            f"• *TP / SL*: ${take_profit:.2f} / ${stop_loss:.2f}\n"
                                            f"• *PnL*: {pnl_usd:+.2f} ({pnl_pct:+.2f}%)\n"
                                        )
                            
                            if active_trades_summary:
                                reply_text = "📊 *ACTIVE OPEN TRADES* 📊\n\n" + "\n".join(active_trades_summary)
                            else:
                                reply_text = "ℹ️ *No active trades currently open.*"
                                
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": reply_text,
                                "parse_mode": "Markdown"
                            })
                            
                        elif text == "/balance":
                            if TRADE_MODE == "simulation":
                                sim_bal = bot_state.get("simulated_balance", 80.0)
                                reply_text = f"💵 *SIMULATION ACCOUNT BALANCE*\n\n• *Cash Balance*: ${sim_bal:.2f}"
                            else:
                                try:
                                    live_bal = get_real_bybit_balance_cached(force=True)
                                    reply_text = f"💰 *LIVE BYBIT ACCOUNT BALANCE* 💰\n\n• *Wallet Balance*: ${live_bal:.2f} USDT"
                                except Exception as bal_err:
                                    reply_text = f"❌ *Failed to fetch balance:* {bal_err}"
                                    
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": reply_text,
                                "parse_mode": "Markdown"
                            })
                            
                        elif text == "/latency":
                            try:
                                # Warm up connection first to ensure we measure reused connection latency
                                get_shared_session().get(f"{BYBIT_BASE_URL}/v5/market/time", timeout=5)
                                t_start = time.time()
                                resp = get_shared_session().get(f"{BYBIT_BASE_URL}/v5/market/time", timeout=5)
                                elapsed = (time.time() - t_start) * 1000.0
                                if resp.status_code == 200:
                                    reply_text = f"⚡ *Bybit API Latency:* `{elapsed:.1f} ms`"
                                else:
                                    reply_text = f"⚠️ *Bybit API Latency:* `{elapsed:.1f} ms` (Status: {resp.status_code})"
                            except Exception as e:
                                reply_text = f"❌ *Latency Check Failed:* {type(e).__name__}"
                                
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": reply_text,
                                "parse_mode": "Markdown"
                            })
                            
                        elif text == "/profit":
                            total_pnl = 0.0
                            today_pnl = 0.0
                            week_pnl = 0.0
                            month_pnl = 0.0
                            wins = 0
                            losses = 0
                            
                            now_ts = time.time()
                            one_day_secs = 24 * 3600
                            
                            with bot_state_lock:
                                for t in bot_state.get("trade_history", []):
                                    pnl = float(t.get("pnl_usd", 0.0))
                                    total_pnl += pnl
                                    
                                    exit_time = float(t.get("exit_time", 0.0))
                                    age = now_ts - exit_time
                                    
                                    if age <= one_day_secs:
                                        today_pnl += pnl
                                    if age <= 7 * one_day_secs:
                                        week_pnl += pnl
                                    if age <= 30 * one_day_secs:
                                        month_pnl += pnl
                                        
                                    if pnl > 0:
                                        wins += 1
                                    elif pnl < 0:
                                        losses += 1
                                        
                            total_trades = wins + losses
                            win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
                            
                            reply_text = (
                                f"📈 *PROFIT / LOSS SUMMARY* 📈\n\n"
                                f"• *Total PnL*: {total_pnl:+.2f} USD\n"
                                f"• *Today (24h)*: {today_pnl:+.2f} USD\n"
                                f"• *7 Days*: {week_pnl:+.2f} USD\n"
                                f"• *30 Days*: {month_pnl:+.2f} USD\n\n"
                                f"📊 *Statistics*:\n"
                                f"• *Win Rate*: {win_rate:.1f}% ({wins} W / {losses} L)"
                            )
                            
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": reply_text,
                                "parse_mode": "Markdown"
                            })
                            
                        elif text == "/skipped":
                            skipped_summary = []
                            with bot_state_lock:
                                preds = [p for p in bot_state.get("prediction_history", []) 
                                         if p.get("status", "").startswith("Skipped (") 
                                         and p.get("status") not in ["Skipped (Neutral)", "Skipped (Bot Stopped)"]]
                                
                                # Sort by timestamp descending
                                preds = sorted(preds, key=lambda x: x.get("timestamp", 0.0), reverse=True)
                                
                                # Take top 5
                                for p in preds[:5]:
                                    symbol = p.get("symbol")
                                    iv = p.get("interval")
                                    direction = p.get("direction")
                                    status = p.get("status", "").replace("Skipped (", "").replace(")", "")
                                    conf = p.get("calibrated_confidence", 0.0)
                                    ts = p.get("timestamp", 0.0)
                                    time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                                    
                                    skipped_summary.append(
                                        f"🚫 *{symbol} ({iv}m)* | signal: {direction}\n"
                                        f"• *Filter*: {status.upper()}\n"
                                        f"• *Confidence*: {conf*100:.1f}%\n"
                                        f"• *Time*: {time_str}"
                                    )
                                    
                            if skipped_summary:
                                reply_text = "🚫 *LATEST FILTERED/SKIPPED SIGNALS* 🚫\n\n" + "\n\n".join(skipped_summary)
                            else:
                                reply_text = "ℹ️ *No skipped trades logged recently.*"
                                
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": reply_text,
                                "parse_mode": "Markdown"
                            })
                            
                        elif text == "/add_user":
                            import random
                            code = str(random.randint(100000, 999999))
                            pending_auth[sender_chat_id] = {
                                "code": code,
                                "step": "awaiting_code",
                                "timestamp": time.time()
                            }
                            
                            subject = "🔑 Bot Authorization Verification Code"
                            body = (
                                f"Hello,\n\n"
                                f"A request was made to authorize a new Telegram Chat ID for your trading bot.\n\n"
                                f"Your verification code is: {code}\n\n"
                                f"Please enter this code in Telegram to authenticate the request."
                            )
                            
                            def _send_email():
                                try:
                                    send_email_notification(subject, body)
                                except Exception as mail_err:
                                    print(f"[Telegram Command Listener] Email auth send error: {mail_err}")
                                    
                            threading.Thread(target=_send_email, daemon=True).start()
                            
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": "🔑 *Verification code sent to mehsimleo@gmail.com.*\n\nPlease reply with the 6-digit code to verify your request.",
                                "parse_mode": "Markdown"
                            })

                        elif text.startswith("/confluence"):
                            pending_confluence[sender_chat_id] = {
                                "step": "awaiting_symbol",
                                "timestamp": time.time()
                            }
                            syms_keyboard = []
                            row = []
                            for idx, s in enumerate(SUPPORTED_SYMBOLS):
                                row.append({"text": s})
                                if (idx + 1) % 3 == 0:
                                    syms_keyboard.append(row)
                                    row = []
                            if row:
                                syms_keyboard.append(row)
                            syms_keyboard.append([{"text": "/cancel"}])
                            
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": "📝 *Confluence Report*\n\nPlease select or reply with the *Symbol* you want a report for:",
                                "parse_mode": "Markdown",
                                "reply_markup": {
                                    "keyboard": syms_keyboard,
                                    "resize_keyboard": True,
                                    "one_time_keyboard": True
                                }
                            })

                        elif text == "/create_manual_trade":
                            pending_manual_trade[sender_chat_id] = {
                                "step": "awaiting_symbol",
                                "timestamp": time.time()
                            }
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": "📝 *Create Manual Trade*\n\nPlease reply with the *Symbol* you want to trade (e.g., BTC or ETHUSDT):",
                                "parse_mode": "Markdown",
                                "reply_markup": {"remove_keyboard": True}
                            })

                        elif text == "/clean_duplicates":
                            cleaned_count = 0
                            with active_trades_lock:
                                for tf_key in ["1h", "2h", "4h", "6h"]:
                                    trades = bot_state.get(f"active_trade_{tf_key}", [])
                                    seen_symbols = set()
                                    unique_trades = []
                                    for t in trades:
                                        sym = t.get("symbol")
                                        if sym in seen_symbols:
                                            cleaned_count += 1
                                            print(f"[Cleanup] Removed duplicate active trade record for {sym} in {tf_key}")
                                        else:
                                            seen_symbols.add(sym)
                                            unique_trades.append(t)
                                    bot_state[f"active_trade_{tf_key}"] = unique_trades
                                
                                if cleaned_count > 0:
                                    save_history()
                                    
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": f"🧹 *Cleaned active trades:*\n\nRemoved `{cleaned_count}` duplicate trade records from the bot's state.",
                                "parse_mode": "Markdown"
                            })

                        elif text == "/retrain_status":
                            status = bot_state.get("retraining_status", "Idle")
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": f"🔄 *Model Retraining Status:* `{status}`",
                                "parse_mode": "Markdown"
                            })

                        elif text == "/logs":
                            logs = "\n".join(log_buffer.logs_list[-15:])
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": f"📋 *Latest Bot Logs:*\n\n```\n{logs}\n```",
                                "parse_mode": "Markdown"
                            })

                        elif text == "/retrain":
                            started = retrain_models_thread(is_manual=True)
                            if started:
                                reply_text = "🔄 *Model retraining started with live trade feedback.*\n\nThis runs in the background (~20-40 min). You'll receive a Telegram alert when complete."
                            else:
                                reply_text = "⚠️ *Retraining already in progress.* Please wait for it to complete."
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": reply_text,
                                "parse_mode": "Markdown"
                            })

                        elif text == "/stop_all":
                            try:
                                with bot_state_lock:
                                    bot_state["bot_running"] = False
                                    save_history()
                                closed_count = close_all_trades_internal("Emergency Telegram Stop")
                                reply_text = (
                                    f"🚨 *EMERGENCY PANIC EXIT TRIGGERED* 🚨\n\n"
                                    f"• Status: *Bot Stopped/Paused*\n"
                                    f"• Closed *{closed_count}* active open trades on Bybit.\n"
                                    f"• Cancelled all pending limit and conditional orders.\n"
                                    f"• New trade entry has been paused."
                                )
                            except Exception as stop_err:
                                reply_text = f"❌ *Failed to stop trades:* {stop_err}"
                                
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": reply_text,
                                "parse_mode": "Markdown"
                            })
                            
                        elif text == "/start_bot":
                            with bot_state_lock:
                                bot_state["bot_running"] = True
                                save_history()
                            reply_text = "▶️ *BTC Trading Bot resumed successfully. New trade entries are enabled.*"
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": sender_chat_id,
                                "text": reply_text,
                                "parse_mode": "Markdown"
                            })
                            
                time.sleep(3)
            except Exception as e:
                print(f"[Telegram Command Listener Error] {e}")
                time.sleep(10)
 
    threading.Thread(target=listener_loop, daemon=True).start()

print("[System Debug] Importing ta...")
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
print("[System Debug] ta imported.")
print("[System Debug] Importing data.py...")
from data import get_history, merge_derivatives_sentiment_features, classify_market_regime
import xml.etree.ElementTree as ET
print("[System Debug] Importing Flask...")
from flask import Flask, jsonify, render_template, request

# ==========================================
# WEB DASHBOARD CONFIGURATION & STATE
# ==========================================
app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True
bot_logs = []
logs_lock = threading.Lock()
retraining_lock = threading.Lock()
active_trades_lock = threading.Lock()

bot_state = state_manager.StateManager()

cached_news_sentiment = "Neutral"
cached_news_titles = []
news_sentiment_lock = threading.Lock()

# Thread-safe real-time Order Flow (CVD & OFI)
order_flow_lock = threading.Lock()
order_flow_data = {} # {symbol: {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, ...}}

# Thread-safe active background order execution guard
active_execution_lock = threading.Lock()
active_execution_symbols = set()

economic_calendar_cache = None
last_calendar_fetch = 0.0
economic_calendar_lock = threading.Lock()

# Re-entrant lock for thread-safe access to bot_state and file IO
bot_state_lock = threading.RLock()

HISTORY_FILE = "/data/dashboard_history.json" if os.path.exists("/data") and os.access("/data", os.W_OK) else "dashboard_history.json"

def save_history():
    with bot_state_lock:
        # Deduplicate completed trades
        trades = bot_state.get("trade_history", [])
        if trades:
            sorted_trades = sorted(trades, key=lambda x: float(x.get("exit_time", 0.0)))
            deduped = []
            for t in sorted_trades:
                duplicate = False
                t_exit = float(t.get("exit_time", 0.0))
                t_entry_p = round(float(t.get("entry_price", 0.0)), 4)
                t_exit_p = round(float(t.get("exit_price", 0.0)), 4)
                t_sym = t.get("symbol")
                t_iv = str(t.get("interval"))
                t_dir = t.get("direction")
                for existing in deduped:
                    if (t_sym == existing.get("symbol") and str(t_iv) == str(existing.get("interval")) and 
                        t_dir == existing.get("direction") and 
                        abs(t_entry_p - round(float(existing.get("entry_price", 0.0)), 4)) < 1e-4 and 
                        abs(t_exit_p - round(float(existing.get("exit_price", 0.0)), 4)) < 1e-4 and 
                        abs(t_exit - float(existing.get("exit_time", 0.0))) < 43200):
                        duplicate = True
                        break
                if not duplicate:
                    deduped.append(t)
            bot_state["trade_history"] = deduped

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
            dir_name = os.path.dirname(HISTORY_FILE)
            temp_file = os.path.join(dir_name, "dashboard_history_temp.json") if dir_name else "dashboard_history_temp.json"
            with open(temp_file, "w") as f:
                json.dump(data, f)
            os.replace(temp_file, HISTORY_FILE)
                
            # If running on Hugging Face and write token is available, backup to HF Dataset
            token = os.environ.get("HF_TOKEN") or os.environ.get("token")
            space_id = os.environ.get("SPACE_ID")
            if token and space_id:
                def _hf_backup():
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
                threading.Thread(target=_hf_backup, daemon=True).start()
        except Exception as e:
            print(f"Error saving history to disk: {e}")

def migrate_active_trades(active_trades_list):
    if not isinstance(active_trades_list, list):
        return
    for t in active_trades_list:
        if "confidence" not in t:
            # Estimate confidence based on original sizing thresholds
            orig_size = t.get("original_size", t.get("position_size_usd", 9.5))
            if orig_size >= 11.0:
                t["confidence"] = 0.785
            elif orig_size >= 9.5:
                t["confidence"] = 0.685
            else:
                t["confidence"] = 0.585

def heal_completed_trades_bybit_order_ids():
    predictions_by_key = {}
    for p in bot_state.get("prediction_history", []):
        if p.get("status") == "Traded" and p.get("bybit_order_id"):
            key = (p.get("symbol"), str(p.get("interval", "60")), p.get("direction"))
            predictions_by_key.setdefault(key, []).append(p)

    healed_count = 0
    for t in bot_state.get("trade_history", []):
        if not t.get("bybit_order_id") or t.get("bybit_order_id") == "N/A":
            key = (t.get("symbol"), str(t.get("interval", "60")), t.get("direction"))
            candidates = predictions_by_key.get(key, [])
            if candidates:
                best_p = None
                min_diff = float("inf")
                exit_ts = t.get("exit_time", 0.0)
                for p in candidates:
                    pred_ts = p.get("timestamp", 0.0)
                    if pred_ts < exit_ts:
                        diff = exit_ts - pred_ts
                        if diff < min_diff:
                            min_diff = diff
                            best_p = p
                if best_p and min_diff < 86400 * 5:
                    t["bybit_order_id"] = best_p["bybit_order_id"]
                    if best_p.get("bybit_scale_out_order_id"):
                        t["bybit_scale_out_order_id"] = best_p["bybit_scale_out_order_id"]
                    healed_count += 1

    if healed_count > 0:
        print(f"[Heal] Successfully recovered missing bybit_order_id for {healed_count} completed trades.")
        save_history()

def deduplicate_completed_trades():
    trades = bot_state.get("trade_history", [])
    if not trades:
        return
        
    sorted_trades = sorted(trades, key=lambda x: float(x.get("exit_time", 0.0)))
    deduped = []
    for t in sorted_trades:
        duplicate = False
        t_exit = float(t.get("exit_time", 0.0))
        t_entry_p = round(float(t.get("entry_price", 0.0)), 4)
        t_exit_p = round(float(t.get("exit_price", 0.0)), 4)
        t_sym = t.get("symbol")
        t_iv = str(t.get("interval"))
        t_dir = t.get("direction")
        
        for existing in deduped:
            if (t_sym == existing.get("symbol") and str(t_iv) == str(existing.get("interval")) and 
                t_dir == existing.get("direction") and 
                abs(t_entry_p - round(float(existing.get("entry_price", 0.0)), 4)) < 1e-4 and 
                abs(t_exit_p - round(float(existing.get("exit_price", 0.0)), 4)) < 1e-4 and 
                abs(t_exit - float(existing.get("exit_time", 0.0))) < 43200):
                duplicate = True
                break
                
        if not duplicate:
            deduped.append(t)
            
    if len(deduped) != len(trades):
        print(f"[Heal] Deduplicated trade history: removed {len(trades) - len(deduped)} duplicate records within 12h window.")
        bot_state["trade_history"] = deduped
        save_history()

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
            target_space = os.environ.get("TARGET_HF_SPACE") or "mehsimleo/btc-trading-bot"
            if "/" in target_space:
                username, space_name = target_space.split("/")
                space_url = f"https://{username}-{space_name.replace('_', '-')}.hf.space/api/status"
            else:
                space_url = f"https://{target_space.replace('_', '-')}.hf.space/api/status"
            print(f"Syncing: Attempting to pull latest history from Hugging Face Space API ({space_url})...")
            resp = requests.get(space_url, timeout=5)
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
                    
                    # Migrate legacy active trades
                    for tf_key in ["1h", "2h", "4h", "6h"]:
                        migrate_active_trades(bot_state[f"active_trade_{tf_key}"])
                        
                    bot_state["bot_running"] = data.get("bot_running", True)
                    bot_state["fresh_reset_v3"] = data.get("fresh_reset_v3", False)
                    print(f"Sync Success: Loaded {len(hf_trades)} trades and {len(hf_predictions)} predictions from Hugging Face Space.")
                    
                    # Startup Balance Audit
                    active_margin = sum(t.get("position_size_usd", 0.0) for tf_key in ["1h", "2h", "4h", "6h"] for t in bot_state.get(f"active_trade_{tf_key}", []))
                    print(f"[Startup Sync Balance Audit] Cash Balance: ${bot_state['simulated_balance']:.2f} | Active Position Margin: ${active_margin:.2f} | Total Account Value: ${bot_state['simulated_balance'] + active_margin:.2f}")
                    
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
                
                # Migrate legacy active trades
                for tf_key in ["1h", "2h", "4h", "6h"]:
                    migrate_active_trades(bot_state[f"active_trade_{tf_key}"])
                    
                bot_state["bot_running"] = data.get("bot_running", True)
                bot_state["fresh_reset_v3"] = data.get("fresh_reset_v3", False)
                print(f"Loaded {len(bot_state['trade_history'])} trades and {len(bot_state['prediction_history'])} predictions from {HISTORY_FILE}")
                
                # Startup Balance Audit
                active_margin = sum(t.get("position_size_usd", 0.0) for tf_key in ["1h", "2h", "4h", "6h"] for t in bot_state.get(f"active_trade_{tf_key}", []))
                print(f"[Startup Balance Audit] Cash Balance: ${bot_state['simulated_balance']:.2f} | Active Position Margin: ${active_margin:.2f} | Total Account Value: ${bot_state['simulated_balance'] + active_margin:.2f}")
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
        
    deduplicate_completed_trades()
    heal_completed_trades_bybit_order_ids()
        
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
    # If running on Hugging Face and no explicit BYBIT_PROXY is set, bypass internal HF proxy
    if os.environ.get("SPACE_ID") and not os.environ.get("BYBIT_PROXY"):
        return None

    proxy = (
        os.environ.get("BYBIT_PROXY") or
        os.environ.get("HTTPS_PROXY") or
        os.environ.get("HTTP_PROXY") or
        os.environ.get("https_proxy") or
        os.environ.get("http_proxy")
    )
    if proxy:
        if "://" not in proxy:
            proxy = "http://" + proxy
        return {
            "http": proxy,
            "https": proxy
        }
    return None

_session = None
_session_lock = threading.Lock()

def get_shared_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                proxies = get_bybit_proxies()
                if proxies:
                    _session.proxies.update(proxies)
    return _session

def parse_proxy_url(proxy_url):
    """Parse proxy URL into components for websocket-client."""
    from urllib.parse import urlparse
    if "://" not in proxy_url:
        proxy_url = "http://" + proxy_url
    parsed = urlparse(proxy_url)
    host = parsed.hostname
    port = parsed.port
    proxy_type = parsed.scheme or "http"
    if proxy_type == "socks":
        proxy_type = "socks5"
    auth = None
    if parsed.username and parsed.password:
        auth = (parsed.username, parsed.password)
    return host, port, auth, proxy_type

_cached_time_offset = None
_last_time_sync = 0.0
_time_offset_lock = threading.Lock()

def get_bybit_time_offset():
    global _cached_time_offset, _last_time_sync
    with _time_offset_lock:
        if _cached_time_offset is not None and (time.time() - _last_time_sync) < 7200:
            return _cached_time_offset
    
    async def do_time_sync():
        proxy_dict = get_bybit_proxies()
        proxy_url = proxy_dict.get("https") or proxy_dict.get("http") if proxy_dict else None
        timeout = aiohttp.ClientTimeout(total=5)
        async with _aiohttp_session.get(f"{BYBIT_BASE_URL}/v5/market/time", proxy=proxy_url, timeout=timeout) as resp:
            status = resp.status
            data = await resp.json()
            return status, data

    for attempt in range(3):
        try:
            _ensure_async_loop()
            future = asyncio.run_coroutine_threadsafe(do_time_sync(), _async_loop)
            status, res = future.result(timeout=7)
            if status == 200:
                server_time = int(res["result"]["timeNano"]) // 1000000
                local_time = int(time.time() * 1000)
                offset = server_time - local_time
                print(f"[Bybit API] Successfully synced time offset: {offset}ms")
                with _time_offset_lock:
                    _cached_time_offset = offset
                    _last_time_sync = time.time()
                return offset
        except Exception as e:
            if attempt == 2:
                print(f"[Bybit API Error] Failed to sync time after 3 attempts: {e}")
            time.sleep(1)
    return 0

def bybit_post_request(endpoint, payload):
    import time
    import hmac
    import hashlib
    
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    
    if not api_key or not api_secret:
        return {"retCode": -1, "retMsg": "API keys missing"}
        
    offset = get_bybit_time_offset()
    timestamp = str(int(time.time() * 1000) + offset)
    recv_window = "30000"
    
    payload_str = json.dumps(payload)
    val_str = timestamp + api_key + recv_window + payload_str
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
    
    url = f"{BYBIT_BASE_URL}{endpoint}"
    
    async def do_post(url, headers, json_data):
        proxy_dict = get_bybit_proxies()
        proxy_url = proxy_dict.get("https") or proxy_dict.get("http") if proxy_dict else None
        timeout = aiohttp.ClientTimeout(total=8)
        async with _aiohttp_session.post(url, headers=headers, json=json_data, proxy=proxy_url, timeout=timeout) as resp:
            status = resp.status
            try:
                data = await resp.json()
            except Exception:
                text = await resp.text()
                data = {"retCode": status, "retMsg": f"HTTP Error: {text}"}
            
            # Adaptive rate limiting dynamic sleep
            try:
                remaining = int(resp.headers.get("X-Bapi-Limit-Remaining", 100))
                if remaining <= 5:
                    print(f"[Rate Limiter Warning] Post remaining limit is low: {remaining}. Sleeping 1s...")
                    await asyncio.sleep(1.0)
            except Exception:
                pass
                
            return status, data

    max_retries = 3
    for attempt in range(max_retries):
        try:
            _ensure_async_loop()
            future = asyncio.run_coroutine_threadsafe(do_post(url, headers, payload), _async_loop)
            status, res = future.result(timeout=10)
            if status == 200:
                return res
            else:
                if status in [402, 403, 429, 502, 503, 504] and attempt < max_retries - 1:
                    time.sleep(1 + attempt * 1.5)
                    continue
                return res if isinstance(res, dict) else {"retCode": status, "retMsg": str(res)}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 1.5)
                continue
            return {"retCode": -1, "retMsg": f"Connection Error: {e}"}

def set_bybit_leverage(symbol, leverage):
    payload = {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    }
    res = bybit_post_request("/v5/position/set-leverage", payload)
    # 110043 means leverage is already set to this value, which is safe to ignore
    if res.get("retCode") in [0, 110043]:
        print(f"[Bybit API] Leverage set to {leverage}x for {symbol} successfully.")
        return True
    else:
        print(f"[Bybit API Error] Failed to set leverage for {symbol}: {res.get('retMsg')} (code: {res.get('retCode')})")
        return False

def format_bybit_price(symbol, price):
    price_precisions = {
        "BTCUSDT": 2,
        "ETHUSDT": 2,
        "SOLUSDT": 3,
        "BNBUSDT": 2,
        "AVAXUSDT": 3,
        "NEARUSDT": 3,
        "LINKUSDT": 3,
        "LTCUSDT": 2,
        "ADAUSDT": 4,
        "XRPUSDT": 4,
        "DOGEUSDT": 5,
        "DOTUSDT": 3,
        "SUIUSDT": 4,
        "APTUSDT": 3
    }
    p = price_precisions.get(symbol, 2)
    return str(round(price, p))

def format_bybit_qty(symbol, qty):
    precisions = {
        "BTCUSDT": 3,
        "ETHUSDT": 2,
        "SOLUSDT": 1,
        "BNBUSDT": 1,
        "AVAXUSDT": 1,
        "NEARUSDT": 1,
        "LINKUSDT": 1,
        "LTCUSDT": 1,
        "ADAUSDT": 0,
        "XRPUSDT": 0,
        "DOGEUSDT": 0,
        "DOTUSDT": 0,
        "SUIUSDT": 0,
        "APTUSDT": 1
    }
    min_limits = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.01,
        "SOLUSDT": 0.1,
        "BNBUSDT": 0.1,
        "AVAXUSDT": 0.1,
        "NEARUSDT": 0.1,
        "LINKUSDT": 0.1,
        "LTCUSDT": 0.1,
        "ADAUSDT": 1.0,
        "XRPUSDT": 1.0,
        "DOGEUSDT": 1.0,
        "DOTUSDT": 1.0,
        "SUIUSDT": 1.0,
        "APTUSDT": 0.1
    }
    p = precisions.get(symbol, 1)
    min_val = min_limits.get(symbol, 0.1)
    
    # Enforce minimum order quantity limits
    if qty < min_val:
        qty = min_val
        
    if p == 0:
        return str(int(round(qty)))
    return str(round(qty, p))

def get_bybit_min_qty_step(symbol):
    min_limits = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.01,
        "SOLUSDT": 0.1,
        "BNBUSDT": 0.1,
        "AVAXUSDT": 0.1,
        "NEARUSDT": 0.1,
        "LINKUSDT": 0.1,
        "LTCUSDT": 0.1,
        "ADAUSDT": 1.0,
        "XRPUSDT": 1.0,
        "DOGEUSDT": 1.0,
        "DOTUSDT": 1.0,
        "SUIUSDT": 1.0,
        "APTUSDT": 0.1
    }
    return min_limits.get(symbol, 0.1)

_ws_responses = {}
_ws_responses_lock = threading.Lock()

def execute_bybit_order_ws_or_rest(endpoint, payload):
    global private_ws_connected, active_private_ws
    # Add clientOrderId (orderLinkId) for request deduplication on order creation
    if endpoint == "/v5/order/create" and "orderLinkId" not in payload:
        payload["orderLinkId"] = f"cl_{payload.get('symbol', 'generic')}_{int(time.time() * 1000)}"
        
    op_map = {
        "/v5/order/create": "order.create",
        "/v5/order/cancel": "order.cancel"
    }
    op = op_map.get(endpoint)
    if op and private_ws_connected and active_private_ws:
        req_id = f"req_{payload.get('symbol', 'generic')}_{int(time.time() * 1000)}"
        ws_payload = {
            "op": op,
            "reqId": req_id,
            "args": [payload]
        }
        try:
            print(f"[WebSocket Private Execution] Sending {op} reqId={req_id}")
            active_private_ws.send(json.dumps(ws_payload))
            
            # Wait for response (increased timeout to 4.0s for high volatility/latency resilience)
            timeout = 4.0 # 4,000ms
            start_t = time.time()
            while time.time() - start_t < timeout:
                with _ws_responses_lock:
                    if len(_ws_responses) > 500:
                        _ws_responses.clear()
                    if req_id in _ws_responses:
                        resp = _ws_responses.pop(req_id)
                        print(f"[WebSocket Private Execution] Received response for reqId={req_id} retCode={resp.get('retCode')}")
                        return resp
                time.sleep(0.01) # 10ms
            print(f"[WebSocket Private Execution Warning] Timeout waiting for reqId={req_id}. Falling back to REST...")
        except Exception as e:
            print(f"[WebSocket Private Execution Error] {e}. Falling back to REST...")
            
    # Fallback to standard REST API request
    return bybit_post_request(endpoint, payload)

def place_bybit_order(symbol, side, qty, price=None, sl=None, tp=None, reduce_only=False):
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Market", # Market order ensures instant execution
        "qty": str(qty),
        "timeInForce": "IOC",
        "positionIdx": 0
    }
    if reduce_only:
        payload["reduceOnly"] = True
    if sl:
        payload["stopLoss"] = format_bybit_price(symbol, sl)
    if tp:
        payload["takeProfit"] = format_bybit_price(symbol, tp)
        
    res = execute_bybit_order_ws_or_rest("/v5/order/create", payload)
    return res

def get_bybit_order_details(symbol, order_id):
    params = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id
    }
    # 1. Try active/realtime orders first
    res = bybit_get_request("/v5/order/realtime", params)
    if res.get("retCode") == 0:
        orders = res.get("result", {}).get("list", [])
        if orders:
            return orders[0]
            
    # 2. Fall back to order history
    res = bybit_get_request("/v5/order/history", params)
    if res.get("retCode") == 0:
        orders = res.get("result", {}).get("list", [])
        if orders:
            return orders[0]
    return None

def cancel_bybit_order(symbol, order_id):
    cancel_payload = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id
    }
    return execute_bybit_order_ws_or_rest("/v5/order/cancel", cancel_payload)

def bybit_get_request(endpoint, query_params):
    import time
    import hmac
    import hashlib
    import urllib.parse
    
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    
    if not api_key or not api_secret:
        return {"retCode": -1, "retMsg": "API keys missing"}
        
    offset = get_bybit_time_offset()
    timestamp = str(int(time.time() * 1000) + offset)
    recv_window = "30000"
    
    query_string = urllib.parse.urlencode(query_params)
    
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
    
    url = f"{BYBIT_BASE_URL}{endpoint}?{query_string}"
    
    async def do_get(url, headers):
        proxy_dict = get_bybit_proxies()
        proxy_url = proxy_dict.get("https") or proxy_dict.get("http") if proxy_dict else None
        timeout = aiohttp.ClientTimeout(total=8)
        async with _aiohttp_session.get(url, headers=headers, proxy=proxy_url, timeout=timeout) as resp:
            status = resp.status
            try:
                data = await resp.json()
            except Exception:
                text = await resp.text()
                data = {"retCode": status, "retMsg": f"HTTP Error: {text}"}
            
            # Adaptive rate limiting dynamic sleep
            try:
                remaining = int(resp.headers.get("X-Bapi-Limit-Remaining", 100))
                if remaining <= 5:
                    print(f"[Rate Limiter Warning] Get remaining limit is low: {remaining}. Sleeping 1s...")
                    await asyncio.sleep(1.0)
            except Exception:
                pass
                
            return status, data

    max_retries = 3
    for attempt in range(max_retries):
        try:
            _ensure_async_loop()
            future = asyncio.run_coroutine_threadsafe(do_get(url, headers), _async_loop)
            status, res = future.result(timeout=10)
            if status == 200:
                return res
            else:
                if status in [402, 403, 429, 502, 503, 504] and attempt < max_retries - 1:
                    time.sleep(1 + attempt * 1.5)
                    continue
                return res if isinstance(res, dict) else {"retCode": status, "retMsg": str(res)}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 1.5)
                continue
            return {"retCode": -1, "retMsg": f"Connection Error: {e}"}

def get_bybit_position(symbol):
    res = bybit_get_request("/v5/position/list", {"category": "linear", "symbol": symbol})
    if res.get("retCode") == 0:
        pos_list = res.get("result", {}).get("list", [])
        for pos in pos_list:
            if pos.get("symbol") == symbol:
                return pos
    return None

def get_bybit_closed_pnl(symbol, limit=1):
    """Fetch the most recent closed PnL record from Bybit for exact settled realized PnL."""
    res = bybit_get_request("/v5/position/closed-pnl", {"category": "linear", "symbol": symbol, "limit": str(limit)})
    if res.get("retCode") == 0:
        pnl_list = res.get("result", {}).get("list", [])
        if pnl_list:
            return pnl_list[0]
    return None

def get_bybit_accumulated_closed_pnl(symbol, entry_time_ms):
    """Retrieve all closed PnL records for a symbol from entry_time_ms to now and sum them up."""
    res = bybit_get_request("/v5/position/closed-pnl", {"category": "linear", "symbol": symbol, "limit": "50"})
    if res.get("retCode") == 0:
        pnl_list = res.get("result", {}).get("list", [])
        total_pnl = 0.0
        exit_prices = []
        exit_quantities = []
        entry_values = []
        
        for item in pnl_list:
            updated_time = int(item.get("updatedTime", 0))
            if updated_time >= entry_time_ms:
                pnl_val = float(item.get("closedPnl", 0.0))
                total_pnl += pnl_val
                
                ep = float(item.get("avgExitPrice", 0.0))
                eq = float(item.get("closedSize", 0.0))
                ev = float(item.get("cumEntryValue", 0.0))
                if ep > 0 and eq > 0:
                    exit_prices.append(ep)
                    exit_quantities.append(eq)
                if ev > 0:
                    entry_values.append(ev)
                    
        avg_exit_price = None
        if exit_prices and exit_quantities:
            total_qty = sum(exit_quantities)
            if total_qty > 0:
                avg_exit_price = sum(p * q for p, q in zip(exit_prices, exit_quantities)) / total_qty
                
        total_entry_val = sum(entry_values)
        
        return {
            "total_pnl": total_pnl,
            "avg_exit_price": avg_exit_price,
            "total_entry_value": total_entry_val if total_entry_val > 0 else None
        }
    return None

def update_bybit_stop_loss(symbol, sl_price, active_trade=None):
    if active_trade:
        qty_val = float(active_trade.get("qty", 0.0))
        side = "Buy" if active_trade.get("direction") == "Bullish" else "Sell"
    else:
        pos = get_bybit_position(symbol)
        if not pos:
            print(f"[Bybit API] Stop Loss update skipped for {symbol}: No active position found.")
            return False
        qty_val = float(pos.get("size", "0"))
        side = pos.get("side", "Buy")  # "Buy" for Long, "Sell" for Short
        
    if qty_val == 0:
        print(f"[Bybit API] Stop Loss update skipped for {symbol}: Position size is 0.")
        return False
        
    live_price = bot_state.get(f"live_price_{symbol}")
    if live_price is None:
        live_price = get_fallback_price(symbol)
        
    if live_price is not None:
        if side == "Buy" or side == "Long":  # Long position: Stop Loss must be < current price
            if sl_price >= live_price:
                print(f"[Bybit API] Stop Loss update skipped for Long {symbol}: Proposed SL {sl_price:.4f} is >= current price {live_price:.4f}.")
                return False
        else:  # Short position: Stop Loss must be > current price
            if sl_price <= live_price:
                print(f"[Bybit API] Stop Loss update skipped for Short {symbol}: Proposed SL {sl_price:.4f} is <= current price {live_price:.4f}.")
                return False

    payload = {
        "category": "linear",
        "symbol": symbol,
        "stopLoss": format_bybit_price(symbol, sl_price),
        "positionIdx": 0
    }
    res = bybit_post_request("/v5/position/trading-stop", payload)
    if res.get("retCode") == 0:
        print(f"[Bybit API] Successfully updated Stop Loss on Bybit to {sl_price:.4f} for {symbol}.")
        return True
    elif "not modified" in str(res.get("retMsg", "")).lower() or res.get("retCode") == 130089:
        print(f"[Bybit API] Stop Loss for {symbol} is already set to {sl_price:.4f} (not modified).")
        return True
    else:
        print(f"[Bybit API Error] Failed to update Stop Loss for {symbol}: {res.get('retMsg')}")
        return False

def update_bybit_take_profit(symbol, tp_price, active_trade=None):
    """Sync the Take Profit on the Bybit server."""
    if active_trade:
        qty_val = float(active_trade.get("qty", 0.0))
        side = "Buy" if active_trade.get("direction") == "Bullish" else "Sell"
    else:
        pos = get_bybit_position(symbol)
        if not pos:
            print(f"[Bybit API] Take Profit update skipped for {symbol}: No active position found.")
            return False
        qty_val = float(pos.get("size", "0"))
        side = pos.get("side", "Buy")  # "Buy" for Long, "Sell" for Short
        
    if qty_val == 0:
        print(f"[Bybit API] Take Profit update skipped for {symbol}: Position size is 0.")
        return False
        
    live_price = bot_state.get(f"live_price_{symbol}")
    if live_price is None:
        live_price = get_fallback_price(symbol)
        
    if live_price is not None:
        if side == "Buy" or side == "Long":  # Long position: Take Profit must be > current price
            if tp_price <= live_price:
                print(f"[Bybit API] Take Profit update skipped for Long {symbol}: Proposed TP {tp_price:.4f} is <= current price {live_price:.4f}.")
                return False
        else:  # Short position: Take Profit must be < current price
            if tp_price >= live_price:
                print(f"[Bybit API] Take Profit update skipped for Short {symbol}: Proposed TP {tp_price:.4f} is >= current price {live_price:.4f}.")
                return False

    payload = {
        "category": "linear",
        "symbol": symbol,
        "takeProfit": format_bybit_price(symbol, tp_price),
        "positionIdx": 0
    }
    res = bybit_post_request("/v5/position/trading-stop", payload)
    if res.get("retCode") == 0:
        print(f"[Bybit API] Successfully updated Take Profit on Bybit to {tp_price:.4f} for {symbol}.")
        return True
    elif "not modified" in str(res.get("retMsg", "")).lower() or res.get("retCode") == 130089:
        print(f"[Bybit API] Take Profit for {symbol} is already set to {tp_price:.4f} (not modified).")
        return True
    else:
        print(f"[Bybit API Error] Failed to update Take Profit for {symbol}: {res.get('retMsg')}")
        return False

def place_bybit_limit_order(symbol, side, qty, price, sl=None, tp=None, reduce_only=False):
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Limit",
        "qty": str(qty),
        "price": format_bybit_price(symbol, price),
        "timeInForce": "GTC",
        "positionIdx": 0
    }
    if reduce_only:
        payload["reduceOnly"] = True
    if sl:
        payload["stopLoss"] = format_bybit_price(symbol, sl)
    if tp:
        payload["takeProfit"] = format_bybit_price(symbol, tp)
    res = execute_bybit_order_ws_or_rest("/v5/order/create", payload)
    return res

def place_bybit_taker_ioc_order(symbol, side, qty, sl=None, tp=None, reduce_only=False):
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "IOC",
        "positionIdx": 0
    }
    if reduce_only:
        payload["reduceOnly"] = True
    if sl:
        payload["stopLoss"] = format_bybit_price(symbol, sl)
    if tp:
        payload["takeProfit"] = format_bybit_price(symbol, tp)
    res = execute_bybit_order_ws_or_rest("/v5/order/create", payload)
    return res

def get_bybit_bid_ask(symbol):
    res = bybit_get_request("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if res.get("retCode") == 0:
        lst = res.get("result", {}).get("list", [])
        if lst:
            tick = lst[0]
            bid = float(tick.get("bid1Price", 0.0))
            ask = float(tick.get("ask1Price", 0.0))
            last = float(tick.get("lastPrice", 0.0))
            return bid, ask, last
    return None, None, None

def get_bybit_last_execution(symbol):
    res = bybit_get_request("/v5/execution/list", {"category": "linear", "symbol": symbol, "limit": 1})
    if res.get("retCode") == 0:
        exec_list = res.get("result", {}).get("list", [])
        if exec_list:
            return exec_list[0]
    return None

def get_real_bybit_balance():
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    
    if not api_key or not api_secret:
        return "API_KEYS_MISSING"
        
    max_balance = 0.0
    geo_blocked_encountered = False
    
    for account_type in ["UNIFIED", "CONTRACT", "SPOT", "FUND"]:
        if account_type == "FUND":
            res = bybit_get_request("/v5/asset/transfer/query-account-coins-balance", {"accountType": "FUND"})
        else:
            res = bybit_get_request("/v5/account/wallet-balance", {"accountType": account_type})
            
        ret_code = res.get("retCode")
        if ret_code == 0:
            if account_type == "FUND":
                balances = res.get("result", {}).get("balance", [])
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
                list_data = res.get("result", {}).get("list", [])
                if list_data:
                    total_equity = list_data[0].get("totalEquity") or list_data[0].get("totalWalletBalance") or "0"
                    max_balance = max(max_balance, float(total_equity))
        else:
            ret_msg = res.get("retMsg", "")
            # If the response is HTTP error (retCode is HTTP status code)
            if isinstance(ret_code, int) and (400 <= ret_code <= 599):
                print(f"[Bybit Balance] HTTP {ret_code} for {account_type}: {ret_msg}")
                if ret_code == 403 and ("cloudfront" in ret_msg.lower() or "block" in ret_msg.lower()):
                    geo_blocked_encountered = True
            else:
                # Suppress legacy warnings (10001, 10003) for Unified accounts
                if not (ret_code in [10001, 10003] and account_type in ["SPOT", "CONTRACT", "FUND"]):
                    print(f"[Bybit Balance] Query error for {account_type}: Code {ret_code} - {ret_msg}")
                    
    if max_balance > 0.0:
        return max_balance
    if geo_blocked_encountered:
        return "GEO_BLOCKED"
    return 0.0

def get_real_bybit_balance_cached(force=False):
    global _cached_balance, _last_balance_fetch
    now = time.time()
    if force or (now - _last_balance_fetch > BALANCE_UPDATE_INTERVAL_SECS):
        try:
            val = get_real_bybit_balance()
            with _balance_lock:
                _cached_balance = val
                _last_balance_fetch = now
            if TRADE_MODE != "simulation" and isinstance(val, (int, float)) and val > 0:
                bot_state["simulated_balance"] = val
        except Exception as e:
            print(f"[Bybit Balance] Error in balance update (forced={force}): {e}")
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
        if TRADE_MODE != "simulation" and isinstance(val, (int, float)) and val > 0:
            bot_state["simulated_balance"] = val
        print(f"[Bybit Balance] Startup background update success: {val}")
    except Exception as e:
        print(f"[Bybit Balance] Startup background update error: {e}")
        
    while True:
        time.sleep(BALANCE_UPDATE_INTERVAL_SECS)  # Query Bybit balance periodically based on configuration
        try:
            val = get_real_bybit_balance()
            with _balance_lock:
                _cached_balance = val
                _last_balance_fetch = time.time()
            if TRADE_MODE != "simulation" and isinstance(val, (int, float)) and val > 0:
                bot_state["simulated_balance"] = val
        except Exception as e:
            print(f"[Bybit Balance] Error in background balance update: {e}")

@app.route("/api/health")
def get_health():
    ws_status = "live" if ws_connected else "disconnected"
    private_ws_status = "live" if private_ws_connected else "disconnected"
    
    max_age_hours = 0.0
    if model_files_mtime:
        newest_mtime = max(model_files_mtime.values())
        max_age_hours = (time.time() - newest_mtime) / 3600.0
        
    active_count = 0
    with bot_state_lock:
        for tf in ["1h", "2h", "4h", "6h"]:
            active_count += len(bot_state.get(f"active_trade_{tf}", []))
            
    return jsonify({
        "status": "ok",
        "uptime_seconds": int(time.time() - startup_time),
        "bybit_api": "connected" if get_real_bybit_balance_cached() != "API_KEYS_MISSING" else "API_KEYS_MISSING",
        "websocket": ws_status,
        "private_websocket": private_ws_status,
        "model_weights_age_hours": round(max_age_hours, 2),
        "active_trades": active_count,
        "last_candle_close": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(last_ws_update_time)) if last_ws_update_time > 0 else "N/A"
    })

@app.route("/api/status")
def get_status():
    # Thread-safe dictionary copy using the global bot_state_lock
    with bot_state_lock:
        state_copy = bot_state.copy()
        for tf in ["1h", "2h", "4h", "6h"]:
            tf_key = f"active_trade_{tf}"
            if tf_key in state_copy and isinstance(state_copy[tf_key], list):
                state_copy[tf_key] = list(state_copy[tf_key])

    # Filter out simulated skipped trades from the frontend view
    if "trade_history" in state_copy:
        state_copy["trade_history"] = [
            t for t in state_copy["trade_history"]
            if "SIMULATED" not in str(t.get("reason", "")).upper()
        ]

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
    with active_trades_lock:
        active_trades_list = bot_state.get(active_trade_key, [])
        if not isinstance(active_trades_list, list):
            active_trades_list = [] if active_trades_list is None else [active_trades_list]
            bot_state[active_trade_key] = active_trades_list
        active_trades_list = list(active_trades_list)

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
    original_size = trade_to_close.get("original_size", position_size_usd)
    
    live_symbol_price = get_fallback_price(symbol)
    if live_symbol_price is None:
        live_symbol_price = bot_state.get(f"live_price_{symbol}")
    actual_exit_price = live_symbol_price if live_symbol_price is not None else entry_price
    
    # 1. Close position on Bybit if in live/testnet mode
    bybit_exit_price = None
    bybit_realized_pnl = None
    
    if TRADE_MODE != "simulation":
        # Cancel scale-out limit order if it exists
        scale_out_id = trade_to_close.get("bybit_scale_out_order_id")
        if scale_out_id:
            cancel_payload = {
                "category": "linear",
                "symbol": symbol,
                "orderId": scale_out_id
            }
            bybit_post_request("/v5/order/cancel", cancel_payload)
            print(f"[Bybit API] Canceled scale-out limit order {scale_out_id} for {symbol}.")
            
        # Close position
        pos = get_bybit_position(symbol)
        if pos:
            qty_str = pos.get("size", "0")
            qty_val = float(qty_str)
            if qty_val > 0:
                side = "Sell" if direction == "Bullish" else "Buy"
                print(f"[Bybit API] Placing Market close order for {qty_str} {symbol}...")
                close_res = place_bybit_order(
                    symbol=symbol,
                    side=side,
                    qty=qty_str,
                    reduce_only=True
                )
                if close_res.get("retCode") == 0:
                    print(f"[Bybit API] Successfully closed position for {symbol} on Bybit.")
                    time.sleep(0.5) # Brief sleep for order registration
                    closed_pnl_record = get_bybit_closed_pnl(symbol)
                    if closed_pnl_record:
                        record_time_ms = int(closed_pnl_record.get("updatedTime", 0))
                        current_time_ms = int(time.time() * 1000)
                        if abs(current_time_ms - record_time_ms) <= 300000:
                            bybit_realized_pnl = float(closed_pnl_record.get("closedPnl", 0.0))
                            bybit_exit_price = float(closed_pnl_record.get("avgExitPrice", live_symbol_price))
                            # Correct database position size to match actual filled size on Bybit
                            entry_val = float(closed_pnl_record.get("cumEntryValue", 0.0))
                            if entry_val > 0:
                                lev = float(trade_to_close.get("leverage", 1.0))
                                actual_margin = round(entry_val / lev, 2)
                                trade_to_close["position_size_usd"] = actual_margin
                                # Keep original_size as the unscaled full size to calculate correct assumed mainnet PnL
                                trade_to_close["original_size"] = float(trade_to_close.get("original_size", actual_margin))
                                position_size_usd = actual_margin
                        else:
                            print(f"[Bybit API] Stale closed PnL record ignored (Age: {int((current_time_ms - record_time_ms)/1000)}s).")
                            closed_pnl_record = None
                            
                    if not closed_pnl_record:
                        exec_log = get_bybit_last_execution(symbol)
                        if exec_log:
                            exec_time_ms = int(exec_log.get("execTime", 0))
                            current_time_ms = int(time.time() * 1000)
                            if abs(current_time_ms - exec_time_ms) <= 300000:
                                bybit_exit_price = float(exec_log.get("execPrice", live_symbol_price))
                            else:
                                print(f"[Bybit API] Stale execution log ignored (Age: {int((current_time_ms - exec_time_ms)/1000)}s).")

    # Maker execution: zero slippage on limit close
    slippage_pct = 0.0
    actual_price = bybit_exit_price if bybit_exit_price is not None else actual_exit_price
        
    actual_change = actual_price - entry_price
    actual_change_pct = (actual_change / entry_price) * 100
    
    raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
    leverage = trade_to_close.get("leverage", 1.0)
    gross_pnl = position_size_usd * (raw_return_pct * leverage / 100.0)
    taker_fee_cost = position_size_usd * leverage * 0.00055 * 2  # 0.055% taker fee per side on leveraged size
    realized_pnl = gross_pnl - taker_fee_cost
    net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
    
    if TRADE_MODE != "simulation" and bybit_realized_pnl is not None:
        realized_pnl = bybit_realized_pnl
        net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
        
    if realized_pnl < -position_size_usd:
        realized_pnl = -position_size_usd
        net_return_pct = -100.0
    
    # Update simulated balance (only in simulation)
    if TRADE_MODE == "simulation":
        old_bal = bot_state.get("simulated_balance", 80.0)
        new_bal = round(old_bal + position_size_usd + realized_pnl, 2)
        bot_state["simulated_balance"] = new_bal
    else:
        new_bal = bot_state.get("simulated_balance", 0.0)
    
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
        "original_size": float(original_size),
        "pnl_usd": float(realized_pnl),
        "balance": float(new_bal),
        "leverage": float(leverage),
        "fill_pct": float(trade_to_close.get("fill_pct", 100.0)),
        "bybit_order_id": trade_to_close.get("bybit_order_id"),
        "bybit_scale_out_order_id": trade_to_close.get("bybit_scale_out_order_id")
    })
    
    send_telegram_alert(
        f"🔴 *POSITION CLOSED (MANUAL)* 🔴\n"
        f"• *Asset*: {symbol}\n"
        f"• *Interval*: {interval}m\n"
        f"• *Direction*: {direction}\n"
        f"• *Exit Price*: ${actual_price:.2f}\n"
        f"• *Realized PnL*: ${realized_pnl:+.2f} ({net_return_pct:+.2f}%)\n"
        f"• *New Balance*: ${new_bal:.2f}"
    )
    
    for p in bot_state["prediction_history"]:
        if p.get("interval") == interval and p.get("symbol") == symbol and p.get("status") == "Traded" and (not p.get("evaluation") or not p["evaluation"].get("evaluated")):
            p["evaluation"] = {
                "evaluated": True,
                "exit_price": float(actual_price),
                "change": float(actual_change if direction == "Bullish" else -actual_change),
                "change_pct": float(raw_return_pct),
                "success": bool(signal_correct)
            }
            bot_state.save_prediction(p)
            break
            
    save_history()
    
    # Remove from active trades
    with active_trades_lock:
        current_list = bot_state.get(active_trade_key, [])
        if not isinstance(current_list, list):
            current_list = []
        if trade_id:
            current_list = [t for t in current_list if t.get("trade_id") != trade_id]
        else:
            current_list = [t for t in current_list if not (t.get("symbol", "").upper() == symbol)]
        bot_state[active_trade_key] = current_list
    
    # Sync positions immediately to make UI responsive
    sync_active_positions_from_bybit()
    
    return jsonify({"status": "success", "message": f"Successfully force-closed {symbol} {tf.upper()} trade at ${actual_price:.2f}"})

def close_all_trades_internal(exit_reason):
    closed_count = 0
    tf_map_local = {"60": "1h", "120": "2h", "240": "4h", "360": "6h"}
    
    # 1. Direct Bybit Fail-safe Panic Close
    if TRADE_MODE != "simulation":
        try:
            print("[Panic Close All] Cancelling ALL pending limit and conditional orders on Bybit...")
            # Cancel all USDT-settled orders globally
            cancel_usdt_res = bybit_post_request("/v5/order/cancel-all", {
                "category": "linear",
                "settleCoin": "USDT"
            })
            print(f"[Panic Close All] Cancel USDT orders response: {cancel_usdt_res.get('retMsg')}")
            
            # Cancel all USDC-settled orders globally
            cancel_usdc_res = bybit_post_request("/v5/order/cancel-all", {
                "category": "linear",
                "settleCoin": "USDC"
            })
            print(f"[Panic Close All] Cancel USDC orders response: {cancel_usdc_res.get('retMsg')}")
            
            print("[Panic Close All] Querying all active positions on Bybit to close them...")
            bybit_positions = get_all_bybit_positions()
            for pos in (bybit_positions or []):
                symbol = pos.get("symbol")
                qty_str = pos.get("size", "0")
                qty_val = float(qty_str)
                if qty_val > 0:
                    side_pos = pos.get("side")
                    close_side = "Sell" if side_pos == "Buy" else "Buy"
                    
                    print(f"[Panic Close All] Closing position: {qty_str} {symbol} ({side_pos}) via Market close...")
                    close_res = place_bybit_order(
                        symbol=symbol,
                        side=close_side,
                        qty=qty_str,
                        reduce_only=True
                    )
                    if close_res.get("retCode") == 0:
                        print(f"[Panic Close All] Successfully closed {symbol} on Bybit.")
                        closed_count += 1
                    else:
                        print(f"[Panic Close All Error] Failed to close {symbol}: {close_res.get('retMsg')}")
        except Exception as e:
            print(f"[Panic Close All Exception] Error: {e}")
            
    # 2. Iterate and clear all local active trades from bot_state
    with active_trades_lock:
        for tf_key in ["1h", "2h", "4h", "6h"]:
            active_trade_key = f"active_trade_{tf_key}"
            active_trades = bot_state.get(active_trade_key, [])
            if not isinstance(active_trades, list):
                active_trades = [] if active_trades is None else [active_trades]
                
            for t in list(active_trades):
                symbol = t.get("symbol", "BTCUSDT").upper()
                direction = t.get("direction", "Bullish")
                entry_price = t.get("entry_price", 0.0)
                position_size_usd = t.get("position_size_usd", 100.0)
                original_size = t.get("original_size", position_size_usd)
                
                interval = "60"
                for k, v in tf_map_local.items():
                    if v == tf_key:
                        interval = k
                        break
                        
                live_symbol_price = get_fallback_price(symbol)
                if live_symbol_price is None:
                    live_symbol_price = bot_state.get(f"live_price_{symbol}")
                actual_price = live_symbol_price if live_symbol_price is not None else entry_price
                
                actual_change = actual_price - entry_price
                actual_change_pct = (actual_change / entry_price) * 100 if entry_price > 0 else 0.0
                raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
                leverage = t.get("leverage", 1.0)
                gross_pnl = position_size_usd * (raw_return_pct * leverage / 100.0)
                taker_fee_cost = position_size_usd * leverage * 0.00055 * 2
                realized_pnl = gross_pnl - taker_fee_cost
                net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
                
                if realized_pnl < -position_size_usd:
                    realized_pnl = -position_size_usd
                    net_return_pct = -100.0
                    
                if TRADE_MODE == "simulation":
                    old_bal = bot_state.get("simulated_balance", 80.0)
                    new_bal = round(old_bal + position_size_usd + realized_pnl, 2)
                    bot_state["simulated_balance"] = new_bal
                    closed_count += 1
                else:
                    new_bal = bot_state.get("simulated_balance", 0.0)
                    
                actual_trend = "Bullish" if actual_change > 0 else "Bearish"
                signal_correct = (actual_trend == direction)
                trend_status = f"{direction} was CORRECT [OK]" if signal_correct else f"{direction} was INCORRECT [FAIL]"
                
                print("\n==================================================")
                print(f"[{symbol} {tf_key.upper()} MANUAL EXIT ALL]: {exit_reason}")
                print(f"Start Price: {entry_price:.2f} | Exit Price: {actual_price:.2f}")
                print(f"Actual Change: {actual_change:+.2f} ({actual_change_pct:+.4f}%)")
                print(f"PnL: ${realized_pnl:+.2f} | New Balance: ${new_bal:.2f}")
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
                    "original_size": float(original_size),
                    "pnl_usd": float(realized_pnl),
                    "balance": float(new_bal),
                    "leverage": float(leverage),
                    "bybit_order_id": t.get("bybit_order_id"),
                    "bybit_scale_out_order_id": t.get("bybit_scale_out_order_id")
                })
                
                send_telegram_alert(
                    f"🔴 *POSITION CLOSED (FORCE ALL)* 🔴\n"
                    f"• *Asset*: {symbol}\n"
                    f"• *Interval*: {interval}m\n"
                    f"• *Direction*: {direction}\n"
                    f"• *Exit Price*: ${actual_price:.2f}\n"
                    f"• *Realized PnL*: ${realized_pnl:+.2f} ({net_return_pct:+.2f}%)\n"
                    f"• *New Balance*: ${new_bal:.2f}"
                )
                
                for p in bot_state["prediction_history"]:
                    if p.get("interval") == interval and p.get("symbol") == symbol and p.get("status") == "Traded" and (not p.get("evaluation") or not p["evaluation"].get("evaluated")):
                        p["evaluation"] = {
                            "evaluated": True,
                            "exit_price": float(actual_price),
                            "change": float(actual_change if direction == "Bullish" else -actual_change),
                            "change_pct": float(raw_return_pct),
                            "success": bool(signal_correct)
                        }
                        bot_state.save_prediction(p)
                        break
                        
            bot_state[active_trade_key] = []
        
    if closed_count > 0:
        save_history()
        sync_active_positions_from_bybit()
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

@app.route("/api/clear_history", methods=["POST"])
def clear_history_endpoint():
    bot_state["trade_history"] = []
    bot_state["prediction_history"] = []
    if TRADE_MODE == "simulation":
        bot_state["simulated_balance"] = 80.0
        bot_state["daily_drawdown_start_balance"] = 80.0
    else:
        real_bal = get_real_bybit_balance_cached() or 0.0
        bot_state["simulated_balance"] = real_bal
        bot_state["daily_drawdown_start_balance"] = real_bal
    bot_state["circuit_breaker_active"] = False
    for tf_key in ["60", "120", "240", "360"]:
        bot_state["win_rate_by_tf"][tf_key] = None
    save_history()
    return jsonify({"status": "success", "message": "All completed trades and prediction history have been successfully cleared from the backend and Hugging Face Dataset space. Simulated balance has been reset to $80.00."})

@app.route("/api/test_email", methods=["POST"])
def test_email_endpoint():
    resend_key = os.getenv("RESEND_API_KEY", "")
    if resend_key:
        subject = "🚀 [UBOTE Test Alert] Resend API Verification Test"
        body = """
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #0b0e14; color: #f0f3fa; padding: 20px; border-radius: 8px;">
            <h2 style="color: #00b0ff; margin-bottom: 20px;">✅ Resend HTTP API Test Successful!</h2>
            <p>If you are reading this email, your UBOTE trading bot email notification setup via Resend API is correctly configured and working over HTTPS.</p>
            <p style="font-size: 11px; color: #8f9bb3; margin-top: 20px;">Sent automatically by UBOTE Trading System.</p>
        </body>
        </html>
        """
    else:
        subject = "🚀 [UBOTE Test Alert] SMTP Verification Test"
        body = """
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #0b0e14; color: #f0f3fa; padding: 20px; border-radius: 8px;">
            <h2 style="color: #00b0ff; margin-bottom: 20px;">✅ SMTP Test Successful!</h2>
            <p>If you are reading this email, your UBOTE trading bot email notification setup is correctly configured and working.</p>
            <p style="font-size: 11px; color: #8f9bb3; margin-top: 20px;">Sent automatically by UBOTE Trading System.</p>
        </body>
        </html>
        """
    success = send_email_notification(subject, body)
    if success:
        return jsonify({"status": "success", "message": f"Test email sent successfully via {'Resend HTTPS API' if resend_key else 'SMTP'}! Please check your inbox (including Spam folder)."})
    else:
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        if resend_key:
            return jsonify({"status": "error", "message": "Resend HTTP API request failed. Please check your Hugging Face Space logs for the exact HTTP error response."}), 500
        elif not smtp_user or not smtp_password:
            return jsonify({"status": "error", "message": "SMTP credentials are not configured. To bypass Hugging Face firewall SMTP blocks, please set a RESEND_API_KEY secret to use HTTPS emails instead."}), 400
        else:
            return jsonify({"status": "error", "message": "SMTP connection failed. Hugging Face blocks outgoing SMTP ports (587/465). To fix this, register for a free account at Resend.com and add a RESEND_API_KEY secret to use firewall-resistant HTTPS emails instead."}), 500

@app.route("/api/test_telegram", methods=["POST", "GET"])
def test_telegram_endpoint():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    custom_url = os.environ.get("TELEGRAM_API_URL")
    if not token or not chat_id:
        return jsonify({"status": "error", "message": "Telegram credentials not configured."}), 400

    results = {}
    
    # 0. Environment Check
    results["Environment"] = {
        "TELEGRAM_API_URL": custom_url,
        "TELEGRAM_PROXY": os.environ.get("TELEGRAM_PROXY"),
        "BYBIT_PROXY": os.environ.get("BYBIT_PROXY"),
        "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
        "http_proxy": os.environ.get("http_proxy"),
        "https_proxy": os.environ.get("https_proxy"),
    }

    # DNS and Connection test for custom URL host
    if custom_url:
        import socket
        import urllib.parse
        parsed = urllib.parse.urlparse(custom_url)
        host = parsed.hostname
        dns_res = {}
        try:
            dns_res["ip"] = socket.gethostbyname(host)
            dns_res["status"] = "resolved"
        except Exception as e:
            dns_res["status"] = "failed"
            dns_res["error"] = str(e)
        results["DNS: Custom URL Host"] = dns_res

        socket_res = {}
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((dns_res.get("ip", host), 443))
            s.close()
            socket_res["status"] = "connected"
        except Exception as e:
            socket_res["status"] = "failed"
            socket_res["error"] = str(e)
        results["Socket: Custom URL Host:443"] = socket_res

    # Method 1: Direct requests.post
    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        resp = requests.post(url, timeout=10)
        results["Method 1: Direct requests.post (getMe)"] = {
            "status": "success",
            "code": resp.status_code,
            "body": resp.json()
        }
    except Exception as e:
        results["Method 1: Direct requests.post (getMe)"] = {
            "status": "failed",
            "error": str(e)
        }

    # Method 2: Custom Socket Proxy Tunnel
    try:
        tg_proxy = os.environ.get("TELEGRAM_PROXY") or os.environ.get("BYBIT_PROXY")
        if tg_proxy:
            import socket
            import ssl
            import urllib.parse
            import json
            import base64

            parsed_proxy = urllib.parse.urlparse(tg_proxy)
            proxy_host = parsed_proxy.hostname
            proxy_port = parsed_proxy.port or 80
            proxy_user = parsed_proxy.username
            proxy_pass = parsed_proxy.password

            auth_header = ""
            if proxy_user and proxy_pass:
                cred = f"{proxy_user}:{proxy_pass}".encode('utf-8')
                auth_header = f"Proxy-Authorization: Basic {base64.b64encode(cred).decode('utf-8')}\r\n"

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((proxy_host, proxy_port))

            connect_req = (
                f"CONNECT api.telegram.org:443 HTTP/1.1\r\n"
                f"Host: api.telegram.org:443\r\n"
                f"{auth_header}"
                f"Proxy-Connection: Keep-Alive\r\n\r\n"
            ).encode('utf-8')
            s.sendall(connect_req)

            resp_tunnel = s.recv(4096).decode('utf-8', errors='ignore')
            if "200" not in resp_tunnel:
                s.close()
                raise Exception(f"Proxy tunnel CONNECT response non-200: {resp_tunnel}")

            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_REQUIRED
            ssl_sock = context.wrap_socket(s, server_hostname="api.telegram.org")

            body = json.dumps({}).encode('utf-8')
            path = f"/bot{token}/getMe"
            http_req = (
                f"POST {path} HTTP/1.1\r\n"
                f"Host: api.telegram.org\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode('utf-8') + body

            ssl_sock.sendall(http_req)

            response_data = b""
            while True:
                chunk = ssl_sock.read(4096)
                if not chunk:
                    break
                response_data += chunk
            ssl_sock.close()
            resp_str = response_data.decode('utf-8', errors='ignore')
            results["Method 2: Custom Socket Proxy Tunnel (getMe)"] = {
                "status": "success",
                "raw_response": resp_str
            }
        else:
            results["Method 2: Custom Socket Proxy Tunnel (getMe)"] = {
                "status": "skipped",
                "reason": "No proxy configured in environment."
            }
    except Exception as e:
        results["Method 2: Custom Socket Proxy Tunnel (getMe)"] = {
            "status": "failed",
            "error": str(e)
        }

    # Method 3: Requests post through proxy (proxies dict)
    try:
        tg_proxy = os.environ.get("TELEGRAM_PROXY") or os.environ.get("BYBIT_PROXY")
        if tg_proxy:
            url = f"https://api.telegram.org/bot{token}/getMe"
            resp = requests.post(url, proxies={"https": tg_proxy, "http": tg_proxy}, timeout=10)
            results["Method 3: requests.post with proxies dict (getMe)"] = {
                "status": "success",
                "code": resp.status_code,
                "body": resp.json()
            }
        else:
            results["Method 3: requests.post with proxies dict (getMe)"] = {
                "status": "skipped",
                "reason": "No proxy configured in environment."
            }
    except Exception as e:
        results["Method 3: requests.post with proxies dict (getMe)"] = {
            "status": "failed",
            "error": str(e)
        }

    # Custom URL Direct test
    if custom_url:
        # Test 1: Direct requests without proxy
        try:
            url = f"{custom_url}bot{token}/getMe"
            resp = requests.post(url, timeout=10, proxies={"http": "", "https": ""})
            results["Custom URL: Direct requests.post without proxy"] = {
                "status": "success",
                "code": resp.status_code,
                "body": resp.json()
            }
        except Exception as e:
            results["Custom URL: Direct requests.post without proxy"] = {
                "status": "failed",
                "error": str(e)
            }

        # Test 2: Requests through proxy (POST request with timeout=30)
        try:
            url = f"{custom_url}bot{token}/getMe"
            tg_proxy = os.environ.get("TELEGRAM_PROXY") or os.environ.get("BYBIT_PROXY")
            proxies_dict = None
            if tg_proxy:
                if "://" not in tg_proxy:
                    tg_proxy = "http://" + tg_proxy
                proxies_dict = {"http": tg_proxy, "https": tg_proxy}
            resp = requests.post(url, timeout=30, proxies=proxies_dict)
            results["Custom URL: POST requests routed through proxy (timeout=30)"] = {
                "status": "success",
                "code": resp.status_code,
                "body": resp.json()
            }
        except Exception as e:
            results["Custom URL: POST requests routed through proxy (timeout=30)"] = {
                "status": "failed",
                "error": str(e)
            }

        # Test 3: Requests through proxy (GET request with timeout=30)
        try:
            url = f"{custom_url}bot{token}/getMe"
            tg_proxy = os.environ.get("TELEGRAM_PROXY") or os.environ.get("BYBIT_PROXY")
            proxies_dict = None
            if tg_proxy:
                if "://" not in tg_proxy:
                    tg_proxy = "http://" + tg_proxy
                proxies_dict = {"http": tg_proxy, "https": tg_proxy}
            resp = requests.get(url, timeout=30, proxies=proxies_dict)
            results["Custom URL: GET requests routed through proxy (timeout=30)"] = {
                "status": "success",
                "code": resp.status_code,
                "body": resp.json()
            }
        except Exception as e:
            results["Custom URL: GET requests routed through proxy (timeout=30)"] = {
                "status": "failed",
                "error": str(e)
            }

    return jsonify(results)

@app.route("/api/backtest")
def api_backtest():
    history = bot_state.get("prediction_history", [])[-30:]
    total = len(history)
    wins = sum(1 for p in history if p.get("evaluation", {}).get("success") is True)
    losses = sum(1 for p in history if p.get("evaluation", {}).get("success") is False)
    pending = total - wins - losses
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else None
    return jsonify({
        "summary": {
            "total": total,
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate_pct": round(win_rate, 1) if win_rate is not None else None
        },
        "signals": [
            {
                "trade_id": p.get("trade_id"),
                "symbol": p.get("symbol"),
                "interval": p.get("interval"),
                "direction": p.get("direction"),
                "confidence_pct": round(p.get("calibrated_confidence", 0) * 100, 1),
                "threshold_pct": round(p.get("dynamic_threshold", 0) * 100, 1),
                "status": p.get("status"),
                "outcome": p.get("evaluation", {}).get("success"),
                "change_pct": p.get("evaluation", {}).get("change_pct"),
                "timestamp": p.get("timestamp")
            }
            for p in reversed(history)
        ]
    })

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
            
            import sys
            import subprocess
            
            # Configure environment variables to restrict multi-threading inside the child process
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["VECLIB_MAXIMUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            
            # Retrain for all intervals sequentially using nice -n 19 to yield CPU to active trading bot
            for iv in ["60", "120", "240", "360"]:
                print(f"[Retraining] Spawning throttled subprocess for interval {iv}m...")
                cmd = ["nice", "-n", "19", sys.executable, "train.py", "--interval", iv, "--pages", "10", "--live-feedback"]
                p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = p.communicate()
                if p.returncode == 0:
                    print(f"[Retraining] Retraining for interval {iv}m finished successfully.")
                else:
                    print(f"[Retraining Error] Retraining for interval {iv}m failed: {stderr}")
                    
            print("[Retraining] Rolling retraining completed successfully. Model files updated on disk.")
            send_telegram_alert("🔄 *MODEL RETRAINING COMPLETE* 🔄\n• Rolling model retraining finished successfully.\n• Ensemble and meta-classifiers updated and re-loaded on disk.")
        except Exception as e:
            print(f"[Retraining] Error during retraining process: {e}")
        finally:
            bot_state["retraining_status"] = "Idle"
            save_history()
            retraining_lock.release()

    threading.Thread(target=run_training, daemon=True).start()
    return True

JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_journal.csv")
JOURNAL_HEADER = ["timestamp", "symbol", "interval", "direction", "entry_price", "exit_price", "pnl_usd", "change_pct", "success", "reason", "balance", "leverage", "confidence"]

def log_trade_journal(trade: dict):
    """Append a closed trade to trade_journal.csv."""
    import csv
    write_header = not os.path.exists(JOURNAL_PATH)
    try:
        with open(JOURNAL_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_HEADER, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(trade.get("exit_time", time.time()))),
                "symbol": trade.get("symbol"),
                "interval": trade.get("interval"),
                "direction": trade.get("direction"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "pnl_usd": trade.get("pnl_usd"),
                "change_pct": trade.get("change_pct"),
                "success": trade.get("success"),
                "reason": trade.get("reason"),
                "balance": trade.get("balance"),
                "leverage": trade.get("leverage"),
                "confidence": trade.get("confidence"),
            })
    except Exception as e:
        print(f"[Journal] Failed to write journal: {e}")

def send_daily_journal_digest():
    """Send a Telegram daily summary of yesterday's closed trades."""
    import csv
    target_day = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    if not os.path.exists(JOURNAL_PATH):
        send_telegram_alert(f"📓 *Daily Trade Journal — {target_day}*\nNo trades recorded today.")
        return
    try:
        wins, losses, total_pnl, rows = 0, 0, 0.0, []
        with open(JOURNAL_PATH, "r") as f:
            for row in csv.DictReader(f):
                if row["timestamp"].startswith(target_day):
                    rows.append(row)
                    pnl = float(row.get("pnl_usd", 0))
                    total_pnl += pnl
                    if row.get("success") == "True":
                        wins += 1
                    else:
                        losses += 1

        total = wins + losses
        wr = f"{wins/total*100:.1f}%" if total > 0 else "N/A"
        lines = [f"📓 *Daily Trade Journal — {target_day}*",
                 f"• Trades: {total} | ✅ {wins} / ❌ {losses} | WR: {wr}",
                 f"• Total PnL: *${total_pnl:+.2f}*", ""]
        for r in rows[-10:]:
            emoji = "✅" if r.get("success") == "True" else "❌"
            lines.append(f"{emoji} {r['symbol']} {r['direction']} {r['interval']}m | ${float(r['pnl_usd']):+.2f} | {r['reason']}")
        send_telegram_alert("\n".join(lines))
    except Exception as e:
        print(f"[Journal Digest] Error: {e}")

def run_daily_journal_scheduler():
    """Send daily Telegram digest at 00:00 UTC every day."""
    print("[Journal Scheduler] Daily digest scheduler started.")
    while True:
        now = time.gmtime()
        # Sleep until next midnight UTC
        seconds_to_midnight = 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
        time.sleep(seconds_to_midnight)
        try:
            send_daily_journal_digest()
        except Exception as e:
            print(f"[Journal Scheduler] Error: {e}")

FUNDING_ARB_THRESHOLD = 0.001  # 0.1% — above this, shorts earn funding income
FUNDING_ARB_SIZE_USD = 20.0    # Fixed notional size per arbitrage position

def run_funding_rate_arbitrage_monitor():
    """Monitor funding rates. When rate > 0.1%, open a small short to collect funding income."""
    print("[Funding Arb] Monitor started.")
    time.sleep(60)  # Wait for bot to initialize
    while True:
        try:
            for sym in SUPPORTED_SYMBOLS:
                rate = get_funding_rate(sym)
                arb_key = f"funding_arb_{sym}"
                existing = bot_state.get(arb_key)

                if rate > FUNDING_ARB_THRESHOLD and not existing:
                    # High positive funding — shorts earn. Open small short.
                    print(f"[Funding Arb] {sym} funding rate {rate*100:.4f}% > 0.1%. Opening arb short.")
                    if TRADE_MODE == "live":
                        qty_str = format_bybit_qty(sym, FUNDING_ARB_SIZE_USD / (bot_state.get(f"live_price_{sym}") or live_price))
                        res = place_bybit_market_order(sym, "Sell", qty_str, reduce_only=False)
                        if res and res.get("retCode") == 0:
                            bot_state[arb_key] = {"qty": qty_str, "open_rate": rate}
                            send_telegram_alert(
                                f"💰 *FUNDING ARB OPENED*\n"
                                f"• Asset: {sym}\n"
                                f"• Funding Rate: {rate*100:.4f}%\n"
                                f"• Side: Short (collecting funding)\n"
                                f"• Size: ${FUNDING_ARB_SIZE_USD}"
                            )
                elif existing and rate < FUNDING_ARB_THRESHOLD * 0.3:
                    # Funding rate has normalized — close the arb short
                    print(f"[Funding Arb] {sym} funding rate normalized ({rate*100:.4f}%). Closing arb short.")
                    if TRADE_MODE == "live":
                        qty_str = existing["qty"]
                        res = place_bybit_market_order(sym, "Buy", qty_str, reduce_only=True)
                        if res and res.get("retCode") == 0:
                            bot_state.pop(arb_key, None)
                            send_telegram_alert(
                                f"✅ *FUNDING ARB CLOSED*\n"
                                f"• Asset: {sym}\n"
                                f"• Current Rate: {rate*100:.4f}%"
                            )
        except Exception as e:
            print(f"[Funding Arb] Error: {e}")
        time.sleep(300)  # Check every 5 minutes

def run_daily_backup_scheduler():
    """
    Background scheduler that runs daily at 00:00 UTC.
    Calculates time to UTC midnight, sleeps, then creates a compressed zip file
    of trading_bot.db and trade_journal.csv, uploading to AWS S3 if credentials are set.
    """
    print("[Backup Scheduler] Daily database backup scheduler started.")
    import zipfile
    import shutil
    from database import DB_FILE
    
    while True:
        now = time.gmtime()
        # Sleep until next midnight UTC
        seconds_to_midnight = 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
        time.sleep(max(1, seconds_to_midnight))
        
        try:
            print("[Backup Scheduler] Triggering daily backup...")
            backup_dir = "backups"
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            zip_filename = os.path.join(backup_dir, f"backup_{timestamp_str}.zip")
            
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(DB_FILE):
                    zipf.write(DB_FILE, os.path.basename(DB_FILE))
                if os.path.exists(JOURNAL_PATH):
                    zipf.write(JOURNAL_PATH, os.path.basename(JOURNAL_PATH))
                    
            print(f"[Backup Scheduler] Created local compressed backup: {zip_filename}")
            
            # Optional S3 upload
            s3_bucket = os.environ.get("AWS_S3_BUCKET")
            if s3_bucket:
                try:
                    import boto3
                    s3_client = boto3.client('s3')
                    s3_key = f"backups/{os.path.basename(zip_filename)}"
                    s3_client.upload_file(zip_filename, s3_bucket, s3_key)
                    print(f"[Backup Scheduler] Successfully uploaded backup to S3: s3://{s3_bucket}/{s3_key}")
                except Exception as s3_err:
                    print(f"[Backup Scheduler Warning] S3 upload failed (boto3 or credentials missing): {s3_err}")
        except Exception as e:
            print(f"[Backup Scheduler Error] Daily backup failed: {e}")

def run_rolling_retrain_scheduler():
    """
    Background scheduler that runs weekly on Sundays at 00:00 UTC.
    Checks time every 15 minutes.
    """
    print("[Scheduler] Automated weekly Sunday retraining scheduler started.")
    while True:
        try:
            now_utc = datetime.utcnow()
            # Sunday is weekday 6. Check if it is Sunday between 00:00 and 00:20 UTC.
            if now_utc.weekday() == 6 and now_utc.hour == 0 and now_utc.minute < 20:
                print(f"[Scheduler] Sunday 00:00 UTC detected. Triggering weekly model retraining...")
                retrain_models_thread(is_manual=False)
                # Sleep 45 minutes to prevent double trigger within the same hour
                time.sleep(2700)
        except Exception as e:
            print(f"[Scheduler] Error in weekly retraining scheduler: {e}")
        
        # Sleep for 15 minutes before checking time again
        time.sleep(900)

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
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

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
        # Load selected features for both regimes
        selected_features_filename = f"selected_features_{iv}.json"
        trending_features_filename = f"selected_features_{iv}_trending.json"
        ranging_features_filename = f"selected_features_{iv}_ranging.json"
        
        feat_trending = None
        if os.path.exists(trending_features_filename):
            with open(trending_features_filename, "r") as f:
                feat_trending = json.load(f)
        elif os.path.exists(selected_features_filename):
            with open(selected_features_filename, "r") as f:
                feat_trending = json.load(f)
                
        feat_ranging = None
        if os.path.exists(ranging_features_filename):
            with open(ranging_features_filename, "r") as f:
                feat_ranging = json.load(f)
        elif os.path.exists(selected_features_filename):
            with open(selected_features_filename, "r") as f:
                feat_ranging = json.load(f)
                
        if feat_trending is None or feat_ranging is None:
            print(f"[Model Load Warning] selected_features for interval {iv} missing! Disabling model loading.")
            models_by_interval[iv]["selected_features"] = None
            return
            
        models_by_interval[iv]["selected_features_trending"] = feat_trending
        models_by_interval[iv]["selected_features_ranging"] = feat_ranging
        models_by_interval[iv]["selected_features"] = feat_trending
        
        n_features_trending = len(feat_trending)
        n_features_ranging = len(feat_ranging)
        print(f"Loaded feature counts - Trending: {n_features_trending}, Ranging: {n_features_ranging} for interval {iv}")
        
        if os.path.exists(f"{prefixes['trending_trend']}_xgb.json"):
            models_by_interval[iv]["trending"]["trend"] = load_ensemble_classifier(prefixes["trending_trend"], n_features_trending)
        if os.path.exists(f"{prefixes['trending_price']}_xgb.json"):
            models_by_interval[iv]["trending"]["price"] = load_ensemble_regressor(prefixes["trending_price"], n_features_trending)
        if os.path.exists(prefixes["trending_meta"]):
            meta_clf = XGBClassifier()
            meta_clf.load_model(prefixes["trending_meta"])
            models_by_interval[iv]["trending"]["meta"] = meta_clf
            
        if os.path.exists(f"{prefixes['ranging_trend']}_xgb.json"):
            models_by_interval[iv]["ranging"]["trend"] = load_ensemble_classifier(prefixes["ranging_trend"], n_features_ranging)
        if os.path.exists(f"{prefixes['ranging_price']}_xgb.json"):
            models_by_interval[iv]["ranging"]["price"] = load_ensemble_regressor(prefixes["ranging_price"], n_features_ranging)
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
    reloaded = []
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
            reloaded.append(iv)
    return reloaded

# =========================
# WEB SOCKET FOR LIVE PRICE
# =========================
live_price = None
last_ws_update_time = 0.0
ws_connected = False  # Track if WebSocket is currently connected
ws_retry_delay = 3  # Reconnection backoff delay (reset on successful connect)

def run_fallback_price_updater():
    """
    Periodic thread that queries Bybit REST API for spot prices.
    Acts as a failover if WebSocket is geoblocked or disconnected.
    Polls adaptively: every 10s when WS is down, every 5min when WS is active.
    """
    global live_price, last_ws_update_time
    print("[Price Fallback] Background updater thread started.")
    last_fallback_run = 0.0
    last_binance_run = 0.0
    while True:
        try:
            now = time.time()
            ws_active = ws_connected and (now - last_ws_update_time < 30)
            has_active_trades = any(len(bot_state.get(f"active_trade_{tf}", [])) > 0 for tf in ["1h", "2h", "4h", "6h"])

            # Adaptive interval: 20s if WS down with trades, 60s if WS down idle, 900s if WS up
            if not ws_active:
                poll_interval = 20 if has_active_trades else 60
                binance_interval = 60 if has_active_trades else 180
            else:
                poll_interval = 900  # Throttle to 15 minutes to minimize proxy bandwidth
                binance_interval = 900

            if now - last_fallback_run >= poll_interval:
                last_fallback_run = now
                url = f"{BYBIT_BASE_URL}/v5/market/tickers"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                resp = requests.get(url, params={"category": "linear"}, headers=headers, proxies=get_bybit_proxies(), timeout=8)
                found_symbols = set()
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
                                found_symbols.add(sym)
                                if sym == "BTCUSDT":
                                    live_price = val
                                    bot_state["live_price"] = val
                                    bot_state["last_update"] = time.time()
                                    if not ws_active:
                                        last_ws_update_time = time.time()

                # Try Bulk Bybit Live API fallback for missing symbols first (such as AVAX and LTC on testnet)
                missing = [s for s in SUPPORTED_SYMBOLS if s not in found_symbols]
                if missing and TRADE_MODE == "testnet":
                    try:
                        url = "https://api.bybit.com/v5/market/tickers"
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        }
                        bresp = requests.get(url, params={"category": "linear"}, headers=headers, proxies=get_bybit_proxies(), timeout=8)
                        if bresp.status_code == 200:
                            ticker_list = bresp.json().get("result", {}).get("list", [])
                            for ticker in ticker_list:
                                sym = ticker.get("symbol")
                                if sym in missing:
                                    val_str = ticker.get("lastPrice")
                                    if val_str:
                                        bot_state[f"live_price_{sym}"] = float(val_str)
                                        found_symbols.add(sym)
                    except Exception as ble:
                        print(f"[Price Fallback] Bybit Live bulk fetch error: {ble}")

                # Bulk Binance fallback for missing symbols (throttled)
                missing = [s for s in SUPPORTED_SYMBOLS if s not in found_symbols]
                if missing and (now - last_binance_run >= binance_interval):
                    last_binance_run = now
                    try:
                        bresp = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=8)
                        if bresp.status_code == 200:
                            binance_prices = {t["symbol"]: float(t["price"]) for t in bresp.json()}
                            for sym in missing:
                                if sym in binance_prices:
                                    bot_state[f"live_price_{sym}"] = binance_prices[sym]
                                    if sym == "BTCUSDT":
                                        live_price = binance_prices[sym]
                                        bot_state["live_price"] = binance_prices[sym]
                                        bot_state["last_update"] = time.time()
                                        if not ws_active:
                                            last_ws_update_time = time.time()
                    except Exception as be:
                        print(f"[Price Fallback] Binance bulk fetch error: {be}")
        except Exception as e:
            print(f"[Price Fallback Exception] {e}")

        time.sleep(5)  # Short sleep, actual polling gated by interval checks above

def send_email_notification(subject, body):
    """
    Sends an email alert.
    Uses Resend HTTP API (port 443) if RESEND_API_KEY is configured.
    Otherwise, falls back to standard SMTP (port 587/465).
    """
    import requests
    import smtplib
    from email.mime.text import MIMEText
    
    email_to = os.getenv("EMAIL_TO", "mehsimleo@gmail.com")
    resend_api_key = os.getenv("RESEND_API_KEY", "")
    
    # 1. Attempt to send via Resend HTTPS API (firewall resistant)
    if resend_api_key:
        try:
            print("[Email Notification] Sending email via Resend HTTP API...")
            url = "https://api.resend.com/emails"
            headers = {
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json"
            }
            # Resend onboarding@resend.dev allows sending to your own email address (email_to)
            payload = {
                "from": "UBOTE Alerts <onboarding@resend.dev>",
                "to": email_to,
                "subject": subject,
                "html": body
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code in [200, 201, 202]:
                print(f"[Email Notification] Successfully sent email via Resend to {email_to}")
                return True
            else:
                print(f"[Email Notification] Resend API error ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"[Email Notification] Resend HTTP API failed: {e}")
            
    # 2. Fallback to SMTP
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        smtp_port = 587
        
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    
    if not smtp_user or not smtp_password:
        print("[Email Notification] Skipped SMTP fallback: SMTP_USER or SMTP_PASSWORD environment variables not set.")
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
            print(f"[Email Notification] Successfully sent email via SMTP to {email_to}")
            return True
    except Exception as e:
        print(f"[Email Notification] SMTP connection failed: {e}")
        return False

def on_message(ws, message):
    global live_price, last_ws_update_time
    last_ws_update_time = time.time()
    try:
        data = json.loads(message)
        topic = data.get("topic", "")
        
        # 1. Price Tickers Handler
        if topic.startswith("tickers."):
            ticker_data = data.get("data", {})
            sym = ticker_data.get("symbol")
            price_str = ticker_data.get("lastPrice")
            if price_str and sym:
                val = float(price_str)
                bot_state[f"live_price_{sym}"] = val
                if sym == "BTCUSDT":
                    live_price = val
                    bot_state["live_price"] = val
                    bot_state["last_update"] = last_ws_update_time
                    
        # 2. Public Trade (CVD) Handler
        elif topic.startswith("publicTrade."):
            trade_list = data.get("data", [])
            for t in trade_list:
                sym = t.get("s")
                side = t.get("S")
                qty = float(t.get("v", 0.0))
                if sym:
                    with order_flow_lock:
                        if sym not in order_flow_data:
                            order_flow_data[sym] = {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0}
                        order_flow_data[sym]["cvd"] += qty if side == "Buy" else -qty
                        
        # 3. Order Book L2 (OFI & Depth Cache) Handler
        elif topic.startswith("orderbook.50."):
            ob_data = data.get("data", {})
            sym = ob_data.get("s")
            bids = ob_data.get("b", [])
            asks = ob_data.get("a", [])
            
            if sym:
                with order_flow_lock:
                    if sym not in order_flow_data:
                        order_flow_data[sym] = {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0}
                    
                    state = order_flow_data[sym]
                    is_snapshot = (data.get("type") == "snapshot")
                    if is_snapshot:
                        state["latest_bids"] = bids[:25]
                        state["latest_asks"] = asks[:25]
                    else:
                        if bids:
                            cached_b = {float(p): float(s) for p, s in state["latest_bids"]}
                            for p, s in bids:
                                price, size = float(p), float(s)
                                if size == 0.0:
                                    cached_b.pop(price, None)
                                else:
                                    cached_b[price] = size
                            state["latest_bids"] = sorted([[str(p), str(s)] for p, s in cached_b.items()], key=lambda x: float(x[0]), reverse=True)[:25]
                        if asks:
                            cached_a = {float(p): float(s) for p, s in state["latest_asks"]}
                            for p, s in asks:
                                price, size = float(p), float(s)
                                if size == 0.0:
                                    cached_a.pop(price, None)
                                else:
                                    cached_a[price] = size
                            state["latest_asks"] = sorted([[str(p), str(s)] for p, s in cached_a.items()], key=lambda x: float(x[0]))[:25]

                    bids_cache = state["latest_bids"]
                    asks_cache = state["latest_asks"]
                    
                    # Compute actual L2 imbalance and spread
                    if bids_cache and asks_cache:
                        bid_L1 = float(bids_cache[0][0])
                        ask_L1 = float(asks_cache[0][0])
                        state["ob_spread_L2"] = (ask_L1 - bid_L1) / bid_L1 if bid_L1 > 0 else 0.0
                        
                        top_bids_size = sum(float(b[1]) for b in bids_cache[:10])
                        top_asks_size = sum(float(a[1]) for a in asks_cache[:10])
                        tot_size = top_bids_size + top_asks_size + 1e-8
                        state["ob_imbalance_L2"] = (top_bids_size - top_asks_size) / tot_size

                    # Compute L1 OFI
                    if bids_cache:
                        bid_p = float(bids_cache[0][0])
                        bid_q = float(bids_cache[0][1])
                        if bid_p > state["prev_bid_price"]:
                            db = bid_q
                        elif bid_p == state["prev_bid_price"]:
                            db = bid_q - state["prev_bid_size"]
                        else:
                            db = 0.0
                        state["prev_bid_price"] = bid_p
                        state["prev_bid_size"] = bid_q
                    else:
                        db = 0.0
                        
                    if asks_cache:
                        ask_p = float(asks_cache[0][0])
                        ask_q = float(asks_cache[0][1])
                        if ask_p > state["prev_ask_price"]:
                            da = 0.0
                        elif ask_p == state["prev_ask_price"]:
                            da = ask_q - state["prev_ask_size"]
                        else:
                            da = ask_q
                        state["prev_ask_price"] = ask_p
                        state["prev_ask_size"] = ask_q
                    else:
                        da = 0.0
                        
                    state["ofi"] += (db - da)

        # 4. Public Liquidation Feed Handler
        elif topic.startswith("liquidation."):
            liq_data = data.get("data", {})
            sym = liq_data.get("symbol")
            side = liq_data.get("side") # Buy/Sell
            qty = float(liq_data.get("size", 0.0))
            price = float(liq_data.get("price", 0.0))
            usd_val = qty * price
            if sym and usd_val > 0.0:
                with order_flow_lock:
                    if sym not in order_flow_data:
                        order_flow_data[sym] = {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0}
                    state = order_flow_data[sym]
                    if side == "Buy": # Short liquidation (market buy filled)
                        state["liq_short_1h"] += usd_val
                    else: # Long liquidation (market sell filled)
                        state["liq_long_1h"] += usd_val
    except Exception as e:
        print(f"[WebSocket msg exception] {e}")

def on_open(ws):
    global ws_connected, ws_retry_delay, active_public_ws, last_ws_update_time
    ws_connected = True
    active_public_ws = ws
    last_ws_update_time = time.time()
    ws_retry_delay = 3  # Reset backoff on successful connection
    print("Connected to Bybit WebSocket for multi-asset prices and order flow")
    
    # Subscribe to tickers, publicTrade, orderbook, and liquidation topics for all supported symbols
    args = []
    for s in SUPPORTED_SYMBOLS:
        args.append(f"tickers.{s}")
        args.append(f"publicTrade.{s}")
        args.append(f"orderbook.50.{s}")
        args.append(f"liquidation.{s}")
        
    chunk_size = 10
    for i in range(0, len(args), chunk_size):
        chunk = args[i:i + chunk_size]
        ws.send(json.dumps({
            "op": "subscribe",
            "args": chunk
        }))
        
    # Heartbeat Daemon Thread to send custom text pings every 20 seconds
    def send_heartbeat():
        while ws_connected:
            try:
                ws.send(json.dumps({"op": "ping"}))
            except Exception:
                break
            time.sleep(20)
    threading.Thread(target=send_heartbeat, daemon=True).start()

def on_close(ws, close_status_code, close_msg):
    global ws_connected, active_public_ws
    ws_connected = False
    active_public_ws = None
    print(f"[WebSocket Closed] code={close_status_code}, msg={close_msg}")

def on_error(ws, error):
    print(f"[WebSocket Error] {error}")

def start_ws():
    global ws_connected, ws_retry_delay, active_public_ws
    url = BYBYIT_WS_URL if 'BYBYIT_WS_URL' in globals() else BYBIT_WS_URL
    print(f"[WebSocket Connecting] url={url}")
    # Parse proxy settings from BYBIT_PROXY env var
    proxy_host, proxy_port, proxy_auth, proxy_type_str = None, None, None, None
    proxy_url = os.environ.get("BYBIT_PROXY")
    if proxy_url:
        proxy_host, proxy_port, proxy_auth, proxy_type_str = parse_proxy_url(proxy_url)
        print(f"[WebSocket] Using proxy: {proxy_host}:{proxy_port} (type={proxy_type_str})")
    while True:
        try:
            ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            active_public_ws = ws
            import ssl
            ws.run_forever(
                ping_interval=20, ping_timeout=10,
                http_proxy_host=proxy_host,
                http_proxy_port=proxy_port,
                http_proxy_auth=proxy_auth,
                proxy_type=proxy_type_str,
                sslopt={"cert_reqs": ssl.CERT_NONE}
            )
        except Exception as e:
            print(f"[WebSocket run_forever exception] {e}")
        ws_connected = False
        active_public_ws = None
        print(f"[WebSocket] Reconnecting in {ws_retry_delay}s...")
        time.sleep(ws_retry_delay)
        ws_retry_delay = min(ws_retry_delay * 2, 60)  # Backoff up to 60s

# WebSocket thread is started inside if __name__ == "__main__" block at the bottom

def on_private_open(ws):
    global private_ws_connected, active_private_ws, last_private_ws_update_time
    private_ws_connected = True
    active_private_ws = ws
    last_private_ws_update_time = time.time()
    print("[WebSocket Private] Connected. Authenticating...")
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print("[WebSocket Private] API Key or Secret missing. Cannot authenticate.")
        ws.close()
        return
    import time as t_module
    import hmac
    import hashlib
    import json
    expires = int((t_module.time() + 10) * 1000)
    signature_raw = f"GET/realtime{expires}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        signature_raw.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    auth_payload = {
        "op": "auth",
        "args": [api_key, expires, signature]
    }
    ws.send(json.dumps(auth_payload))
    
    # Heartbeat Daemon Thread for private WebSocket to send custom text pings every 20 seconds
    def send_private_heartbeat():
        import json as j_module
        import time as t_module
        while private_ws_connected:
            try:
                ws.send(j_module.dumps({"op": "ping"}))
            except Exception:
                break
            t_module.sleep(20)
    threading.Thread(target=send_private_heartbeat, daemon=True).start()

_ws_filled_orders = {}
_ws_filled_orders_lock = threading.Lock()

def on_private_message(ws, message):
    import json
    import time
    global last_private_ws_update_time
    last_private_ws_update_time = time.time()
    try:
        data = json.loads(message)
        op = data.get("op")
        topic = data.get("topic")
        
        # Capture operations responses (e.g. order.create or order.cancel callbacks)
        req_id = data.get("reqId")
        if req_id:
            with _ws_responses_lock:
                _ws_responses[req_id] = data
                
        if op == "auth":
            if data.get("success") is True:
                print("[WebSocket Private] Authentication successful. Subscribing to topics...")
                sub_payload = {
                    "op": "subscribe",
                    "args": ["position", "wallet", "order"]
                }
                ws.send(json.dumps(sub_payload))
            else:
                print(f"[WebSocket Private] Authentication failed: {data.get('ret_msg')}")
                ws.close()
        elif topic == "wallet":
            wallet_data = data.get("data", [])
            if wallet_data:
                total_equity = wallet_data[0].get("totalEquity") or wallet_data[0].get("totalWalletBalance")
                if total_equity:
                    val = float(total_equity)
                    with _balance_lock:
                        global _cached_balance, _last_balance_fetch
                        _cached_balance = val
                        _last_balance_fetch = time.time()
                    if TRADE_MODE != "simulation":
                        bot_state["simulated_balance"] = val
                    print(f"[WebSocket Private] Balance updated dynamically from wallet stream: {val}")
        elif topic == "order":
            order_list = data.get("data", [])
            for ord in order_list:
                status = ord.get("orderStatus")
                ord_id = ord.get("orderId")
                if ord_id and status == "Filled":
                    with _ws_filled_orders_lock:
                        _ws_filled_orders[ord_id] = ord
            import threading
            threading.Thread(target=sync_active_positions_from_bybit, daemon=True).start()
        elif topic == "position":
            import threading
            threading.Thread(target=sync_active_positions_from_bybit, daemon=True).start()
    except Exception as e:
        print(f"[WebSocket Private Message Error] {e}")

def on_private_error(ws, error):
    print(f"[WebSocket Private Error] {error}")

def on_private_close(ws, close_status_code, close_msg):
    global private_ws_connected, active_private_ws
    private_ws_connected = False
    active_private_ws = None
    print(f"[WebSocket Private Closed] code={close_status_code}, msg={close_msg}")

def start_private_ws():
    global private_ws_connected, private_ws_retry_delay, active_private_ws
    url = BYBIT_PRIVATE_WS_URL
    print(f"[WebSocket Private Connecting] url={url}")
    proxy_host, proxy_port, proxy_auth, proxy_type_str = None, None, None, None
    proxy_url = os.environ.get("BYBIT_PROXY")
    if proxy_url:
        proxy_host, proxy_port, proxy_auth, proxy_type_str = parse_proxy_url(proxy_url)
    while True:
        try:
            ws = websocket.WebSocketApp(
                url,
                on_open=on_private_open,
                on_message=on_private_message,
                on_error=on_private_error,
                on_close=on_private_close
            )
            active_private_ws = ws
            import ssl
            ws.run_forever(
                ping_interval=20, ping_timeout=10,
                http_proxy_host=proxy_host,
                http_proxy_port=proxy_port,
                http_proxy_auth=proxy_auth,
                proxy_type=proxy_type_str,
                sslopt={"cert_reqs": ssl.CERT_NONE}
            )
        except Exception as e:
            print(f"[WebSocket Private run_forever exception] {e}")
        private_ws_connected = False
        active_private_ws = None
        print(f"[WebSocket Private] Reconnecting in {private_ws_retry_delay}s...")
        time.sleep(private_ws_retry_delay)
        private_ws_retry_delay = min(private_ws_retry_delay * 2, 60)

def run_websocket_watchdog():
    global last_ws_update_time, last_private_ws_update_time
    global active_public_ws, active_private_ws
    global ws_connected, private_ws_connected
    
    print("[WebSocket Watchdog] Active keep-alive thread started.")
    last_ws_update_time = time.time()
    last_private_ws_update_time = time.time()
    
    while True:
        time.sleep(15)
        now = time.time()
        
        # Check Public WebSocket (Ticker/Prices)
        if ws_connected and active_public_ws:
            silent_duration = now - last_ws_update_time
            if silent_duration > 60:
                print(f"[WebSocket Watchdog] Public WebSocket silent for {silent_duration:.1f}s (>60s). Force closing to trigger reconnect...")
                try:
                    active_public_ws.close()
                except Exception as e:
                    print(f"[WebSocket Watchdog] Error closing public ws: {e}")
                    
        # Check Private WebSocket (Position/Orders/Wallet)
        if private_ws_connected and active_private_ws:
            silent_duration = now - last_private_ws_update_time
            if silent_duration > 60:
                print(f"[WebSocket Watchdog] Private WebSocket silent for {silent_duration:.1f}s (>60s). Force closing to trigger reconnect...")
                try:
                    active_private_ws.close()
                except Exception as e:
                    print(f"[WebSocket Watchdog] Error closing private ws: {e}")


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
    "CVD_true", "OFI_true"
])
for lag in [1, 2]:
    features.append(f"open_interest_pct_change_lag{lag}")
    features.append(f"funding_rate_diff_lag{lag}")
    features.append(f"CVD_true_lag{lag}")
    features.append(f"OFI_true_lag{lag}")

# New Wick Volume features (absorption/liquidation proxies)
features.extend([
    "upper_wick_volume_ratio", "lower_wick_volume_ratio"
])
for lag in [1, 2]:
    features.append(f"upper_wick_volume_ratio_lag{lag}")
    features.append(f"lower_wick_volume_ratio_lag{lag}")

features.append("hours_to_news")

# New Correlation and OI momentum features
features.extend(["oi_change_1h", "oi_change_4h", "btc_close", "btc_volume", "btc_rsi"])
for lag in [1, 2]:
    features.append(f"oi_change_1h_lag{lag}")
    features.append(f"oi_change_4h_lag{lag}")
    features.append(f"btc_close_lag{lag}")
    features.append(f"btc_volume_lag{lag}")
    features.append(f"btc_rsi_lag{lag}")

# Advanced Microstructure features
features.extend([
    "roll_spread", "leverage_divergence", "oi_velocity", "funding_acceleration", "bid_ask_imbalance_ohlc",
    "ob_imbalance_L2", "ob_spread_L2", "liq_long_1h", "liq_short_1h"
])
for lag in [1, 2]:
    features.append(f"roll_spread_lag{lag}")
    features.append(f"leverage_divergence_lag{lag}")
    features.append(f"oi_velocity_lag{lag}")
    features.append(f"funding_acceleration_lag{lag}")
    features.append(f"bid_ask_imbalance_ohlc_lag{lag}")
    features.append(f"close_to_Kalman_lag{lag}")
    features.append(f"ob_imbalance_L2_lag{lag}")
    features.append(f"ob_spread_L2_lag{lag}")
    features.append(f"liq_long_1h_lag{lag}")
    features.append(f"liq_short_1h_lag{lag}")

# Garman-Klass Volatility features
features.extend(["volatility_gk", "volatility_gk_lag1", "volatility_gk_lag2"])

# Cross-Asset Lead-Lag Correlation features
features.extend(["lead_lag_diff_5m", "lead_lag_diff_1h", "lead_lag_diff_4h", "volume_ratio_to_btc"])
for lag in [1, 2]:
    features.append(f"lead_lag_diff_5m_lag{lag}")
    features.append(f"lead_lag_diff_1h_lag{lag}")
    features.append(f"lead_lag_diff_4h_lag{lag}")
    features.append(f"volume_ratio_to_btc_lag{lag}")

# Initial model loading is deferred to main() to ensure Flask starts immediately on HF Spaces.


def add_features(df):
    return features_module.add_features(df, fetch_calendar_callback=fetch_economic_calendar_cached)

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
            bot_state.save_prediction(pred)
            
        if pred.get("interval") == interval and pred.get("symbol", "BTCUSDT") == symbol and not eval_dict.get("evaluated"):
            interval_mins = int(interval)
            cfg = TIMEFRAME_CONFIG.get(str(interval), {"lookahead": 10})
            lookahead = cfg.get("lookahead", 10)
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
                bot_state.save_prediction(pred)
                
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
            res = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=10)
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
                if res.status_code != 429:
                    print(f"[News/Sentiment] Reddit r/{sub} feed returned status code {res.status_code}")
        except Exception as e:
            print(f"[News/Sentiment] Exception fetching Reddit r/{sub} feed: {e}")
    return posts

def get_cryptopanic_posts():
    """
    Fetches the top live crypto news from RSS feeds of Cointelegraph, CoinDesk, and Decrypt.
    Replaces the discontinued CryptoPanic developer API. No API key is needed.
    """
    import xml.etree.ElementTree as ET
    feeds = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        "https://decrypt.co/feed"
    ]
    posts = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    for url in feeds:
        try:
            res = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=8)
            if res.status_code == 200:
                xml_content = res.content
                root = ET.fromstring(xml_content)
                feed_posts = []
                for item in root.findall(".//item"):
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        feed_posts.append(title_elem.text.strip())
                posts.extend(feed_posts[:4])
                print(f"[News/Sentiment] Fetched {len(feed_posts[:4])} articles from RSS: {url}")
            else:
                print(f"[News/Sentiment] RSS feed {url} returned status code {res.status_code}")
        except Exception as e:
            print(f"[News/Sentiment] Exception fetching RSS feed {url}: {e}")
            
    return posts[:12]

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
        res = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=10)
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
    url_coindesk = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
    try:
        res = requests.get(url_coindesk, headers=headers, proxies=get_bybit_proxies(), timeout=10)
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
        token = os.environ.get("HF_TOKEN") or os.environ.get("token")
        if token:
            API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
            headers = {"Authorization": f"Bearer {token}"}
            
            total_score = 0.0
            processed = 0
            print(f"[News/Sentiment Serverless] Querying HF Inference API for {len(cleaned_titles)} inputs...")
            for text in cleaned_titles:
                try:
                    resp = requests.post(API_URL, headers=headers, json={"inputs": text[:300]}, timeout=5)
                    if resp.status_code == 200:
                        res = resp.json()
                        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                            scores = {item["label"].lower(): item["score"] for item in res[0]}
                            bullish = scores.get("positive", 0.0)
                            bearish = scores.get("negative", 0.0)
                            total_score += (bullish - bearish)
                            processed += 1
                except Exception as e:
                    pass
            
            if processed > 0:
                avg_score = total_score / processed
                sentiment = "Neutral"
                if avg_score > 0.15:
                    sentiment = "Bullish"
                elif avg_score < -0.15:
                    sentiment = "Bearish"
                print(f"[News/Sentiment Serverless] Analysis complete. Avg Score: {avg_score:.4f} | Aggregated: {sentiment}")
                return sentiment, cleaned_titles

        # Local pipeline fallback if HF_TOKEN is missing
        if sentiment_pipeline is None:
            try:
                from transformers import pipeline
                sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
            except ImportError:
                print("[News/Sentiment] transformers/torch not installed. Serverless HF_TOKEN must be configured. Sentiment defaults to Neutral.")
                return "Neutral", cleaned_titles

        print(f"[News/Sentiment Local] Running local FinBERT pipeline on {len(cleaned_titles)} inputs...")
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
            
        print(f"[News/Sentiment Local] Analysis complete. Avg Score: {avg_score:.4f} | Aggregated: {sentiment}")
        return sentiment, cleaned_titles
    except Exception as e:
        print(f"[News/Sentiment] Error executing FinBERT analysis: {e}")
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
    global economic_calendar_cache, last_calendar_fetch
    now_ts = time.time()
    with economic_calendar_lock:
        if economic_calendar_cache is not None and (now_ts - last_calendar_fetch < 86400):
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
            session = get_shared_session()
            resp = session.get(
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
                last_calendar_fetch = now_ts
                print(f"[News/Sentiment] Cached {len(economic_calendar_cache)} high-impact calendar events.")
                return economic_calendar_cache
        except Exception as e:
            print(f"[News/Sentiment] Error caching economic calendar: {e}")
        
        economic_calendar_cache = []
        last_calendar_fetch = now_ts
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
        
    # WebSocket Cache Check (Instantly bypass HTTP request if cache is warm)
    with order_flow_lock:
        cached = order_flow_data.get(symbol)
        if cached and cached.get("latest_bids") and cached.get("latest_asks"):
            bids = cached["latest_bids"]
            asks = cached["latest_asks"]
            try:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid_price = (best_bid + best_ask) / 2.0
                spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 0.0
                
                # Calculate weighted imbalance over top 10 levels
                bid_vol = 0.0
                ask_vol = 0.0
                for i in range(min(10, len(bids), len(asks))):
                    w = 1.0 / (i + 1.0)
                    bid_vol += float(bids[i][1]) * w
                    ask_vol += float(asks[i][1]) * w
                    
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                return {"imbalance": imbalance, "spread": spread}
            except Exception as cache_err:
                pass

    try:
        url = f"{BYBIT_BASE_URL}/v5/market/orderbook"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "linear", "symbol": symbol, "limit": 25}, headers=headers, proxies=get_bybit_proxies(), timeout=10)
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
                
                selected_features_list = None
                if interval in models_by_interval:
                    selected_features_list = models_by_interval[interval].get("selected_features")
                if selected_features_list is None:
                    import json
                    selected_features_filename = f"selected_features_{interval}.json"
                    if os.path.exists(selected_features_filename):
                        with open(selected_features_filename, "r") as f:
                            selected_features_list = json.load(f)
                            
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
        url = f"{BYBIT_BASE_URL}/v5/market/tickers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "linear", "symbol": symbol}, headers=headers, proxies=get_bybit_proxies(), timeout=5)
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

    is_stressed = False
    try:
        df_vol = get_history(symbol=new_symbol, interval="60", limit=30)
        if df_vol is not None and not df_vol.empty and "ATR_norm" in df_vol.columns:
            rolling_atr = df_vol["ATR_norm"].tail(30)
            atr_mean = rolling_atr.mean()
            atr_std = rolling_atr.std()
            vol_z_score = (df_vol["ATR_norm"].iloc[-1] - atr_mean) / (atr_std + 1e-8) if atr_std > 0 else 0.0
            is_stressed = vol_z_score > 2.0
            if is_stressed:
                print(f"[Stress Covariance] Volatility Z-score: {vol_z_score:.2f} > 2.0. Stressed correlation mode active.")
    except Exception as e:
        print(f"[Stress Covariance Warning] Could not calculate volatility z-score: {e}")

    def get_correlation(s1, s2):
        if s1 == s2:
            return 1.0
        if is_stressed:
            return 0.95
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

def calculate_recent_performance_leverage_multiplier(days=7):
    """
    Calculates a leverage multiplier based on the rolling Sharpe ratio of completed trades.
    Reduces max leverage during drawdowns to manage risk.
    """
    try:
        trades = bot_state.get("trade_history", [])
        if len(trades) < 5:
            return 1.0
            
        import time as _time
        cutoff = _time.time() - days * 86400
        recent_trades = [t for t in trades if float(t.get("exit_time", 0.0)) >= cutoff]
        
        if len(recent_trades) < 3:
            return 1.0
            
        pnls = [float(t.get("pnl_usd", 0.0)) for t in recent_trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        
        if std_pnl < 1e-4:
            return 1.0
            
        sharpe = mean_pnl / std_pnl
        if sharpe >= 0:
            return 1.0
        else:
            # Scale down leverage. Clamp multiplier between 0.30 and 1.0
            multiplier = max(0.30, min(1.0, 1.0 - 0.3 * abs(sharpe)))
            print(f"[Sharpe-Adaptive Leverage] Sharpe={sharpe:.2f} -> Sizing down leverage by {multiplier:.2f}x")
            return float(multiplier)
    except Exception as e:
        print(f"[Sharpe-Adaptive Leverage Error] {e}")
        return 1.0

# ==========================================
# PRE-TRADE CONFLUENCE ANALYSIS
# ==========================================
def check_pre_trade_confluence(current_price, df_1h, ml_trend, news_sentiment, expected_pct_change, interval="60", symbol=None, htf_cache=None, calibrated_confidence=0.5, dynamic_conf_threshold=0.58):
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
        if not trend_1d_pass:
            hard_gate_failed = True
        results["1d_Trend"] = {
            "pass": trend_1d_pass,
            "detail": f"1d Trend is {trend_1d} (EMA9: {ema9_1d:.2f}, EMA21: {ema21_1d:.2f}) [HARD GATE]",
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
        if not trend_pass:
            hard_gate_failed = True
        results["4h_Trend"] = {
            "pass": trend_pass,
            "detail": f"4h Trend is {trend_4h} (EMA9: {ema9_4h:.2f}, EMA21: {ema21_4h:.2f}) [HARD GATE]",
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
    tf_map_local = {"60": "1h", "120": "2h", "240": "4h", "360": "6h", "5": "5m", "15": "15m"}
    tf_label = tf_map_local.get(str(interval), "1h")
    if ml_trend == "Bullish":
        rsi_1h_pass = (rsi_1h < 62.0)
        detail_msg = f"{tf_label} RSI is {rsi_1h:.2f} (< 62, Safe)" if rsi_1h_pass else f"{tf_label} RSI is {rsi_1h:.2f} (>= 62, Overbought)"
    else:
        rsi_1h_pass = (rsi_1h > 38.0)
        detail_msg = f"{tf_label} RSI is {rsi_1h:.2f} (> 38, Safe)" if rsi_1h_pass else f"{tf_label} RSI is {rsi_1h:.2f} (<= 38, Oversold)"
    results["1h_RSI"] = {"pass": rsi_1h_pass, "detail": detail_msg, "weight": weight_rsi}
    max_score += weight_rsi
    if rsi_1h_pass:
        total_score += weight_rsi

    # ======= CHECK 4: Volume Confirmation (RVOL) Hard Gate =======
    # Filter out low-liquidity fakeouts using last completed candle volume
    try:
        vol_series_completed = df_1h["volume"].iloc[:-1]
        avg_vol_20 = vol_series_completed.rolling(20).mean().iloc[-1]
        latest_completed_vol = vol_series_completed.iloc[-1]
        rvol = latest_completed_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0
        
        regime = bot_state.get(f"regime_{interval}h", "Ranging")
        min_rvol_req = 0.8 if "Ranging" in regime else 1.2
        
        volume_pass = (rvol >= min_rvol_req)
        if not volume_pass:
            hard_gate_failed = True
        results["Volume_Participation"] = {
            "pass": volume_pass,
            "detail": f"RVOL: {rvol:.2f}x (Vol: {latest_completed_vol:.1f} / Avg20: {avg_vol_20:.1f}), required >= {min_rvol_req:.1f}x (Hard Gate)",
            "weight": 0
        }
    except Exception as e:
        volume_pass = True
        results["Volume_Participation"] = {"pass": True, "detail": f"Skipped volume check (Error: {e})", "weight": 0}

    # ======= CHECK 5: Bollinger Band Edge Guard (Weight: 2) =======
    weight_bb = 2
    bb_pct_val = df_1h["BB_pct"].iloc[-1]
    if ml_trend == "Bullish":
        bb_pass = (bb_pct_val < 0.85)
        detail_msg = f"BB Pct is {bb_pct_val:.3f} (< 0.85, Room to run)" if bb_pass else f"BB Pct is {bb_pct_val:.3f} (>= 0.85, Overextended long)"
    else:
        bb_pass = (bb_pct_val > 0.15)
        detail_msg = f"BB Pct is {bb_pct_val:.3f} (> 0.15, Room to run)" if bb_pass else f"BB Pct is {bb_pct_val:.3f} (<= 0.15, Overextended short)"
    results["BB_Edge_Guard"] = {"pass": bb_pass, "detail": detail_msg, "weight": weight_bb}
    max_score += weight_bb
    if bb_pass:
        total_score += weight_bb

    # ======= CHECK 6: Counter-Momentum Guard (Weight: 2) =======
    weight_cm = 2
    try:
        c1 = df_1h.iloc[-1]
        c2 = df_1h.iloc[-2]
        c3 = df_1h.iloc[-3]
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
        detail_msg = f"News Conflict: Model is {ml_trend} but News sentiment is {news_sentiment} [HARD GATE]"
    else:
        detail_msg = f"Passed (Direction Lock: Model is {ml_trend}, News sentiment is {news_sentiment}) [HARD GATE]"
        
    weight_news = 1
    results["News_Sentiment"] = {"pass": news_pass, "detail": detail_msg, "weight": weight_news}
    max_score += weight_news
    if news_pass:
        total_score += weight_news

    # ======= CHECK 12: Expected Price Change Threshold (Weight: 2) =======
    weight_exp = 2
    min_pct_map = {
        "5": 0.10,
        "15": 0.15,
        "60": 0.25,   # 1h requires >= 0.25% change
        "120": 0.35,  # 2h requires >= 0.35% change
        "240": 0.50,  # 4h requires >= 0.50% change
        "360": 0.65   # 6h requires >= 0.65% change
    }
    req_pct = min_pct_map.get(str(interval), 0.25)
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

    # ======= CHECK 15: Risk-Reward Ratio (R:R) Hard Gate =======
    try:
        cfg = TIMEFRAME_CONFIG.get(str(interval), {"sl_mult": 1.5, "tp_mult_ranging": 1.5, "tp_mult_trending": 2.5})
        sl_multiplier = cfg.get("sl_mult", 1.5)
        
        regime = bot_state.get(f"regime_{interval}h", "Ranging")
        if "Trending" in regime:
            tp_multiplier = cfg.get("tp_mult_trending", 2.5)
        else:
            tp_multiplier = cfg.get("tp_mult_ranging", 1.5)
            
        sl_multiplier_adjusted = sl_multiplier
        tp_multiplier_adjusted = tp_multiplier
        
        if calibrated_confidence > dynamic_conf_threshold and dynamic_conf_threshold < 1.0:
            confidence_ratio = (calibrated_confidence - dynamic_conf_threshold) / (1.0 - dynamic_conf_threshold)
            sl_multiplier_adjusted = sl_multiplier * (1.0 - 0.3 * confidence_ratio)
            tp_multiplier_adjusted = tp_multiplier * (1.0 + 0.2 * confidence_ratio)
            
        atr_norm = float(df_1h["ATR_norm"].iloc[-1])
        atr_dollars = atr_norm * current_price
        sl_distance = sl_multiplier_adjusted * atr_dollars
        tp_distance = tp_multiplier_adjusted * atr_dollars
        
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0
        
        min_rr_required = 0.9 if "Ranging" in regime else 1.2
        rr_pass = (rr_ratio >= min_rr_required)
        
        if not rr_pass:
            hard_gate_failed = True
            
        results["Risk_Reward_Ratio"] = {
            "pass": rr_pass,
            "detail": f"R:R Ratio: {rr_ratio:.2f}x (TP: {tp_distance:.4f} / SL: {sl_distance:.4f}), required >= {min_rr_required:.2f}x (Hard Gate)",
            "weight": 0
        }
    except Exception as e:
        results["Risk_Reward_Ratio"] = {"pass": True, "detail": f"Skipped R:R check (Error: {e})", "weight": 0}

    # ======= FINAL SCORING =======
    score_pct = (total_score / max_score * 100) if max_score > 0 else 100.0
    
    # Standardize confluence threshold to flat 75% for consistent accuracy across all market regimes
    score_threshold = 75.0
    traditional_approved = (not hard_gate_failed) and (score_pct >= score_threshold)
        
    # Pro-Level Meta-Labeling Gate: Approved if predicted probability of success >= 60% AND 1d/4h trend gates pass
    trend_gates_passed = results.get("1d_Trend", {}).get("pass", True) and results.get("4h_Trend", {}).get("pass", True)
    approved = (calibrated_confidence >= 0.60) and trend_gates_passed

    # Add score summary to results
    results["_Score_Summary"] = {
        "pass": approved,
        "detail": f"Meta-Gate: {'APPROVED' if approved else 'REJECTED'} (Calibrated Conf: {calibrated_confidence*100:.1f}% vs Req: 60%, Trend Pass: {trend_gates_passed}) | Traditional Score: {total_score}/{max_score} ({score_pct:.0f}%, Hard Gates: {'PASSED' if not hard_gate_failed else 'FAILED'})",
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
        url = f"{BYBIT_BASE_URL}/v5/market/tickers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "linear", "symbol": symbol}, headers=headers, proxies=get_bybit_proxies(), timeout=5)
        if response.status_code == 200:
            res = response.json()
            ticker_list = res.get("result", {}).get("list", [])
            if ticker_list:
                ticker = ticker_list[0]
                price_key = "lastPrice"
                return float(ticker.get("lastPrice"))
            else:
                print(f"Bybit price ticker list is empty for {symbol}")
        else:
            print(f"Bybit price ticker for {symbol} returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Bybit price fallback for {symbol}: {e}")

    # 1b. Try Bybit Live API as fallback for testnet missing symbols
    if TRADE_MODE == "testnet":
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, params={"category": "linear", "symbol": symbol}, headers=headers, proxies=get_bybit_proxies(), timeout=5)
            if response.status_code == 200:
                res = response.json()
                ticker_list = res.get("result", {}).get("list", [])
                if ticker_list:
                    ticker = ticker_list[0]
                    return float(ticker.get("lastPrice"))
        except Exception as e:
            print(f"Error fetching Bybit Live price fallback for {symbol}: {e}")

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

def load_initial_prices():
    global live_price, last_ws_update_time
    print("[Startup] Loading initial market prices for all assets...")
    try:
        url = f"{BYBIT_BASE_URL}/v5/market/tickers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, params={"category": "linear"}, headers=headers, proxies=get_bybit_proxies(), timeout=8)
        found_symbols = set()
        if resp.status_code == 200:
            ticker_list = resp.json().get("result", {}).get("list", [])
            for ticker in ticker_list:
                sym = ticker.get("symbol")
                if sym in SUPPORTED_SYMBOLS:
                    val_str = ticker.get("lastPrice")
                    if val_str:
                        val = float(val_str)
                        bot_state[f"live_price_{sym}"] = val
                        found_symbols.add(sym)
                        if sym == "BTCUSDT":
                            live_price = val
                            bot_state["live_price"] = val
                            last_ws_update_time = time.time()
                            bot_state["last_update"] = last_ws_update_time
        
        # Fall back to external sources (Binance/Coinbase) for any missing symbols (e.g. LINKUSDT on testnet)
        for sym in SUPPORTED_SYMBOLS:
            if sym not in found_symbols:
                val = get_fallback_price(sym)
                if val is not None:
                    bot_state[f"live_price_{sym}"] = val
                    if sym == "BTCUSDT" and live_price is None:
                        live_price = val
                        bot_state["live_price"] = val
                        last_ws_update_time = time.time()
                        bot_state["last_update"] = last_ws_update_time
    except Exception as e:
        print(f"[Initial Prices] Error loading prices at startup: {e}")

def get_all_bybit_positions():
    """Retrieve all open linear positions on Bybit in a single call."""
    res = bybit_get_request("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
    if res.get("retCode") == 0:
        return res.get("result", {}).get("list", [])
    return None

def get_bybit_entry_order_qty(symbol, side):
    """Query Bybit order history for the symbol to find the originally requested quantity of the entry order."""
    try:
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": 20
        }
        res = bybit_get_request("/v5/order/history", params)
        if res.get("retCode") == 0:
            orders = res.get("result", {}).get("list", [])
            for o in orders:
                o_side = o.get("side")
                o_cum = float(o.get("cumExecQty", 0.0))
                o_qty = float(o.get("qty", 0.0))
                if o_side == side and o_cum > 0.0:
                    return o_qty, o_cum, o.get("orderId")
    except Exception as e:
        print(f"[Entry Order Query Warning] Error for {symbol} {side}: {e}")
    return None, None, None

def get_bybit_active_limit_order_id(symbol, side):
    """Query Bybit active open orders to find if there is an active limit order on the specified side."""
    try:
        params = {
            "category": "linear",
            "symbol": symbol
        }
        res = bybit_get_request("/v5/order/realtime", params)
        if res.get("retCode") == 0:
            orders = res.get("result", {}).get("list", [])
            for o in orders:
                if o.get("orderType") == "Limit" and o.get("side") == side and o.get("orderStatus") in ["New", "PartiallyFilled"]:
                    return o.get("orderId")
    except Exception as e:
        print(f"[Active Limit Order Query Warning] Error for {symbol} {side}: {e}")
    return None

def sync_active_positions_from_bybit():
    """Real-time Sync: Sync all active trades from Bybit to keep bot_state completely aligned with testnet/live."""
    if TRADE_MODE == "simulation":
        return True
    
    try:
        pos_list = get_all_bybit_positions()
        if pos_list is None:
            print("[Position Sync] Failed to fetch positions from Bybit. Skipping sync to prevent false exits.")
            return False
        
        # Filter for positions with non-zero size
        open_positions = {}
        for pos in pos_list:
            qty_val = float(pos.get("size", "0"))
            if qty_val > 0:
                open_positions[pos.get("symbol")] = pos

        # Re-sync bot_state active trades
        with active_trades_lock:
            matched_symbols = set()
            
            for tf_key in ["1h", "2h", "4h", "6h"]:
                current_trades = bot_state.get(f"active_trade_{tf_key}", [])
                if not isinstance(current_trades, list):
                    current_trades = []
                
                seen_symbols_in_tf = set()
                updated_trades = []
                for t in current_trades:
                    symbol = t.get("symbol")
                    if not symbol:
                        continue
                    if symbol in seen_symbols_in_tf:
                        print(f"[De-duplication] Discarded duplicate active trade for {symbol} in timeframe {tf_key}.")
                        continue
                    seen_symbols_in_tf.add(symbol)
                    if symbol in open_positions:
                        pos = open_positions[symbol]
                        t["bybit_closed"] = False
                        
                        conf_val = t.get("confidence", 0.0)
                        is_zero_conf = False
                        if conf_val is not None and conf_val != "MT":
                            try:
                                is_zero_conf = (float(conf_val) == 0.0)
                            except (ValueError, TypeError):
                                pass
                        
                        if is_zero_conf:
                            trade_dir = t.get("direction", "Bullish")
                            for p in reversed(bot_state.get("prediction_history", [])):
                                if p.get("symbol") == symbol and p.get("direction") == trade_dir:
                                    if abs(p.get("timestamp", 0) - time.time()) < 86400 * 2:
                                        t["confidence"] = float(p.get("calibrated_confidence", p.get("confidence", 0.63)))
                                        print(f"[Sync Confidence Restore] Restored confidence for recovered {symbol} trade: {t['confidence']*100:.2f}%")
                                        save_history()
                                        break

                        # Side Mismatch Guard: verify Bybit position side aligns with trade direction
                        pos_side = pos.get("side") # "Buy" or "Sell"
                        trade_direction = t.get("direction") # "Bullish" or "Bearish"
                        mismatch = False
                        if trade_direction == "Bullish" and pos_side != "Buy":
                            mismatch = True
                        elif trade_direction == "Bearish" and pos_side != "Sell":
                            mismatch = True
                            
                        if mismatch:
                            print(f"[Side Mismatch Guard] WARNING: {symbol} in {tf_key} has direction {trade_direction} but Bybit position is {pos_side}! Force-closing to prevent inverted SL/TP.")
                            if TRADE_MODE != "simulation":
                                close_side = "Sell" if pos_side == "Buy" else "Buy"
                                place_bybit_order(symbol, close_side, str(pos.get("size", t["qty"])), reduce_only=True)
                                if t.get("bybit_scale_out_order_id"):
                                    cancel_bybit_order(symbol, t["bybit_scale_out_order_id"])
                            continue
    
                        t["entry_price"] = float(pos.get("avgPrice", t["entry_price"]))
                        t["liq_price"] = float(pos.get("liqPrice", 0.0)) if pos.get("liqPrice") else 0.0
                        t["mark_price"] = float(pos.get("markPrice", 0.0)) if pos.get("markPrice") else 0.0
                        t["qty"] = float(pos.get("size", t["qty"]))
                        t["leverage"] = float(pos.get("leverage", t["leverage"]))
                        
                        # Sync TP/SL from Bybit exchange if they exist and are non-zero
                        bybit_sl = float(pos.get("stopLoss", 0.0)) if pos.get("stopLoss") else 0.0
                        bybit_tp = float(pos.get("takeProfit", 0.0)) if pos.get("takeProfit") else 0.0
                        if bybit_sl > 0.0:
                            if t.get("break_even_triggered", False):
                                if t.get("direction") == "Bullish":
                                    t["stop_loss"] = max(bybit_sl, t["entry_price"])
                                else:
                                    t["stop_loss"] = min(bybit_sl, t["entry_price"])
                            else:
                                t["stop_loss"] = bybit_sl
                        if bybit_tp > 0.0:
                            t["take_profit"] = bybit_tp
                        
                        pos_val = float(pos.get("positionValue", 0.0))
                        t["position_size_usd"] = pos_val / t["leverage"] if t["leverage"] > 0 else pos_val
                        t["qty"] = abs(float(pos.get("size", 0.0)))
                        
                        # Proportional Unrealized PnL calculation
                        try:
                            same_symbol_trades = []
                            for tf_check in ["1h", "2h", "4h", "6h"]:
                                for t_item in bot_state.get(f"active_trade_{tf_check}", []):
                                    if t_item.get("symbol") == symbol:
                                        same_symbol_trades.append(t_item)
                            total_lev_size = sum(float(t_item.get("position_size_usd", 0.0)) * float(t_item.get("leverage", 1.0)) for t_item in same_symbol_trades)
                            position_pnl = float(pos.get("unrealisedPnl", 0.0))
                            if total_lev_size > 0:
                                this_lev_size = float(t.get("position_size_usd", 0.0)) * float(t.get("leverage", 1.0))
                                t["bybit_unrealized_pnl"] = round(position_pnl * (this_lev_size / total_lev_size), 2)
                            else:
                                t["bybit_unrealized_pnl"] = position_pnl
                        except Exception:
                            t["bybit_unrealized_pnl"] = float(pos.get("unrealisedPnl", 0.0))
                        
                        # Sanitize ATR, TP, and SL for active trades to prevent invalid/stuck parameters
                        avg_price = t["entry_price"]
                        mark_price = t["mark_price"]
                        liq_price = t["liq_price"]
                        direction = t.get("direction", "Bullish")
                        
                        # Fix ATR if it's unreasonably large or unset
                        current_atr = t.get("atr_dollars", 0.0)
                        if current_atr <= 0 or current_atr > 0.05 * avg_price:
                            current_atr = 0.015 * avg_price
                            t["atr_dollars"] = current_atr
                            
                        # Fix Take Profit if it is unset (0.0)
                        if t.get("take_profit", 0.0) == 0.0:
                            if direction == "Bullish":
                                t["take_profit"] = max(mark_price + 1.25 * current_atr, avg_price + 1.25 * current_atr)
                            else:
                                t["take_profit"] = min(mark_price - 1.25 * current_atr, avg_price - 1.25 * current_atr)
                            if TRADE_MODE != "simulation":
                                update_bybit_take_profit(symbol, t["take_profit"], t)
                                
                        # Fix Stop Loss if it is unset, below liquidation (for long), or above liquidation (for short)
                        sl_val = t.get("stop_loss", 0.0)
                        sl_updated = False
                        if direction == "Bullish":
                            if sl_val <= 0.0 or (liq_price > 0.0 and sl_val <= liq_price) or sl_val < avg_price - 3.0 * current_atr:
                                sl_val = avg_price - 0.75 * current_atr
                                if liq_price > 0.0 and sl_val <= liq_price:
                                    sl_val = liq_price + 0.2 * current_atr
                                sl_updated = True
                        else:
                            if sl_val <= 0.0 or (liq_price > 0.0 and sl_val >= liq_price) or sl_val > avg_price + 3.0 * current_atr:
                                sl_val = avg_price + 0.75 * current_atr
                                if liq_price > 0.0 and sl_val >= liq_price:
                                    sl_val = liq_price - 0.2 * current_atr
                                sl_updated = True
                                
                        if sl_updated:
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(symbol, sl_val, t)
                                if success:
                                    t["stop_loss"] = sl_val
                                else:
                                    t["stop_loss"] = bybit_sl
                            else:
                                t["stop_loss"] = sl_val
                        
                        # Dynamically reconstruct original size for recovered trades
                        if t.get("recovered", False) and t.get("original_size_reconstructed", False) is not True:
                            orig_qty, filled_qty, entry_ord_id = get_bybit_entry_order_qty(symbol, pos_side)
                            if orig_qty is not None and orig_qty > 0.0:
                                t["original_size"] = (orig_qty * avg_price) / t["leverage"] if t["leverage"] > 0 else (orig_qty * avg_price)
                                t["fill_pct"] = round((filled_qty / orig_qty) * 100.0, 2)
                                t["original_size_reconstructed"] = True
                                if entry_ord_id and not t.get("bybit_order_id"):
                                    t["bybit_order_id"] = entry_ord_id
                                print(f"[Crash Recovery Sync] Reconstructed original size for {symbol}: target={t['original_size']:.2f}, filled={t['position_size_usd']:.2f} ({t['fill_pct']}%)")
                                save_history()

                        # Recover missing/unset bybit_order_id
                        if not t.get("bybit_order_id") or t.get("bybit_order_id") == "N/A":
                            _, _, entry_ord_id = get_bybit_entry_order_qty(symbol, pos_side)
                            if entry_ord_id:
                                t["bybit_order_id"] = entry_ord_id
                                print(f"[Sync] Recovered missing bybit_order_id {entry_ord_id} for {symbol}")

                        # Recover missing/unset bybit_scale_out_order_id
                        if not t.get("bybit_scale_out_order_id") or t.get("bybit_scale_out_order_id") == "N/A":
                            limit_side = "Sell" if pos_side == "Buy" else "Buy"
                            limit_ord_id = get_bybit_active_limit_order_id(symbol, limit_side)
                            if limit_ord_id:
                                t["bybit_scale_out_order_id"] = limit_ord_id
                                print(f"[Sync] Recovered missing bybit_scale_out_order_id {limit_ord_id} for {symbol}")
                        
                        updated_trades.append(t)
                        matched_symbols.add(symbol)
                    else:
                        # Keep it so the exit checker can process its closure and fetch closed PnL
                        t["bybit_closed"] = True
                        updated_trades.append(t)
                
                bot_state[f"active_trade_{tf_key}"] = updated_trades
    
            # Reconstruct any open positions on Bybit that are NOT in bot_state (orphaned/manual positions)
            recovered = 0
            for symbol, pos in open_positions.items():
                if symbol not in matched_symbols:
                    avg_price = float(pos.get("avgPrice", "0"))
                    liq_price = float(pos.get("liqPrice", "0")) if pos.get("liqPrice") else 0.0
                    mark_price = float(pos.get("markPrice", "0")) if pos.get("markPrice") else 0.0
                    leverage_val = float(pos.get("leverage", "1"))
                    side_str = pos.get("side", "Buy")
                    direction = "Bullish" if side_str == "Buy" else "Bearish"
                    sl_price = float(pos.get("stopLoss", "0")) if pos.get("stopLoss") else 0.0
                    tp_price = float(pos.get("takeProfit", "0")) if pos.get("takeProfit") else 0.0
                    position_value = float(pos.get("positionValue", "0"))
                    position_size_usd = position_value / leverage_val if leverage_val > 0 else position_value
                    qty_val = float(pos.get("size", "0"))
                    
                    # Retrieve the original target qty of the entry order
                    orig_qty, filled_qty, entry_order_id = get_bybit_entry_order_qty(symbol, side_str)
                    if orig_qty is not None and orig_qty > 0.0:
                        original_size = (orig_qty * avg_price) / leverage_val if leverage_val > 0 else (orig_qty * avg_price)
                        fill_pct = round((filled_qty / orig_qty) * 100.0, 2)
                        original_size_reconstructed = True
                        print(f"[Crash Recovery] Discovered target size for {symbol}: target={original_size:.2f}, filled={position_size_usd:.2f} ({fill_pct}%)")
                    else:
                        original_size = position_size_usd
                        fill_pct = 100.0
                        original_size_reconstructed = False
                    
                    limit_side = "Sell" if side_str == "Buy" else "Buy"
                    scale_out_order_id = get_bybit_active_limit_order_id(symbol, limit_side)
                    
                    import uuid
                    trade_uuid = str(uuid.uuid4())[:8]
                    
                    # Calculate proper ATR on recovery
                    calc_atr = abs(avg_price - sl_price) / 0.75 if sl_price > 0 else 0.015 * avg_price
                    if calc_atr > 0.05 * avg_price or calc_atr == 0:
                        calc_atr = 0.015 * avg_price
                    
                    # Sanitize TP and SL on recovery
                    if tp_price == 0.0:
                        if direction == "Bullish":
                            tp_price = max(mark_price + 1.25 * calc_atr, avg_price + 1.25 * calc_atr)
                        else:
                            tp_price = min(mark_price - 1.25 * calc_atr, avg_price - 1.25 * calc_atr)
                            
                    if sl_price == 0.0 or abs(avg_price - sl_price) > 3.0 * calc_atr:
                        if direction == "Bullish":
                            sl_price = avg_price - 0.75 * calc_atr
                            if liq_price > 0.0 and sl_price <= liq_price:
                                sl_price = liq_price + 0.2 * calc_atr
                        else:
                            sl_price = avg_price + 0.75 * calc_atr
                            if liq_price > 0.0 and sl_price >= liq_price:
                                sl_price = liq_price - 0.2 * calc_atr
                                
                    # Push the recovered/sanitized TP & SL to Bybit
                    if TRADE_MODE != "simulation":
                        update_bybit_take_profit(symbol, tp_price)
                        success = update_bybit_stop_loss(symbol, sl_price)
                        if not success:
                            sl_price = float(pos.get("stopLoss", "0")) if pos.get("stopLoss") else 0.0
     
                    # Dynamic timeframe and confidence resolution from prediction history matching symbol and direction
                    matched_tf = "1h"  # fallback default
                    matched_confidence = 0.0
                    for p in reversed(bot_state.get("prediction_history", [])):
                        if p.get("symbol") == symbol and p.get("direction") == direction:
                            # Verify if it was within the last 48 hours to avoid matching ancient predictions
                            if abs(p.get("timestamp", 0) - time.time()) < 86400 * 2:
                                token_val_not_used = None
                                matched_tf_interval = p.get("interval", "60")
                                tf_map_inv = {"60": "1h", "120": "2h", "240": "4h", "360": "6h"}
                                matched_tf = tf_map_inv.get(matched_tf_interval, "1h")
                                matched_confidence = float(p.get("calibrated_confidence", p.get("confidence", 0.63)))
                                break

                    recovered_trade = {
                        "trade_id": f"{symbol}_{trade_uuid}_recovered",
                        "bybit_order_id": entry_order_id,
                        "bybit_scale_out_order_id": scale_out_order_id,
                        "symbol": symbol,
                        "entry_price": avg_price,
                        "predicted_price": avg_price,
                        "stop_loss": sl_price,
                        "take_profit": tp_price,
                        "direction": direction,
                        "end_time": float(time.time() + 3600 * 48),
                        "entry_time": int(time.time() * 1000),
                        "atr_dollars": calc_atr,
                        "highest_price": max(avg_price, mark_price) if direction == "Bullish" else avg_price,
                        "lowest_price": min(avg_price, mark_price) if direction == "Bearish" else avg_price,
                        "break_even_triggered": False,
                        "half_closed": False,
                        "original_size": original_size,
                        "position_size_usd": position_size_usd,
                        "fill_pct": fill_pct,
                        "original_size_reconstructed": original_size_reconstructed,
                        "scaled_out_pnl": 0.0,
                        "kelly_fraction": 0.0,
                        "leverage": leverage_val,
                        "confidence": matched_confidence,
                        "qty": qty_val,
                        "original_qty": orig_qty if (orig_qty is not None and orig_qty > 0.0) else qty_val,
                        "liq_price": liq_price,
                        "mark_price": mark_price,
                        "recovered": True
                    }
                    
                    tf_key = matched_tf
                    # Cross-timeframe duplicate guard: skip if symbol already active in ANY timeframe
                    already_in_any_tf = any(
                        any(t.get("symbol") == symbol for t in bot_state.get(f"active_trade_{k}", []))
                        for k in ["1h", "2h", "4h", "6h"]
                    )
                    if already_in_any_tf:
                        print(f"[Crash Recovery] Skipped duplicate recovery for {symbol} — already tracked in an active timeframe.")
                        continue
                    active_trades_list = bot_state.get(f"active_trade_{tf_key}", [])
                    if not isinstance(active_trades_list, list):
                        active_trades_list = []
                    active_trades_list.append(recovered_trade)
                    bot_state[f"active_trade_{tf_key}"] = active_trades_list
                    recovered += 1
                    print(f"[Crash Recovery] Discovered/Recovered open position on Bybit: {symbol} {direction}")
                    
            if recovered > 0:
                save_history()
            return True
    except Exception as e:
        print(f"[Crash Recovery] Error checking Bybit: {e}")
        return False

def recover_missed_closed_trades():
    """Scan Bybit closed PnL records on startup to detect and alert on any trades closed while the bot was offline."""
    if TRADE_MODE == "simulation":
        return
    print("[Crash Recovery] Scanning recently closed trades on Bybit...")
    now_ms = int(time.time() * 1000)
    one_day_ms = 24 * 3600 * 1000
    
    for symbol in SUPPORTED_SYMBOLS:
        try:
            res = bybit_get_request("/v5/position/closed-pnl", {
                "category": "linear",
                "symbol": symbol,
                "limit": "10"
            })
            if res.get("retCode") == 0:
                pnl_list = res.get("result", {}).get("list", [])
                for item in pnl_list:
                    updated_time_ms = int(item.get("updatedTime", 0))
                    if now_ms - updated_time_ms < one_day_ms:
                        exit_price = float(item.get("avgExitPrice", 0.0))
                        entry_price = float(item.get("avgEntryPrice", 0.0))
                        closed_pnl = float(item.get("closedPnl", 0.0))
                        side = item.get("side")
                        direction = "Bullish" if side == "Sell" else "Bearish"
                        
                        exit_time_sec = updated_time_ms / 1000.0
                        already_logged = False
                        for t in bot_state.get("trade_history", []):
                            if t.get("symbol") == symbol and abs(t.get("exit_time", 0.0) - exit_time_sec) < 10.0:
                                already_logged = True
                                break
                                
                        if not already_logged:
                            print(f"[Crash Recovery] Discovered missed closed trade on Bybit: {symbol} at exit price {exit_price}")
                            new_bal = bot_state.get("simulated_balance", 0.0)
                            change_pct = ((exit_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
                            raw_return = change_pct if direction == "Bullish" else -change_pct
                            
                            trade_record = {
                                "symbol": symbol,
                                "exit_time": exit_time_sec,
                                "interval": "Unknown",
                                "direction": direction,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "change_pct": raw_return,
                                "success": closed_pnl > 0,
                                "reason": "CLOSED WHILE OFFLINE / RECOVERY SCAN",
                                "position_size_usd": float(item.get("qty", 0.0)) * entry_price,
                                "original_size": float(item.get("qty", 0.0)) * entry_price,
                                "pnl_usd": closed_pnl,
                                "balance": new_bal,
                                "leverage": 1.0,
                                "confidence": 0.0,
                                "take_profit": 0.0,
                                "stop_loss": 0.0,
                                "atr_dollars": 0.0,
                                "fill_pct": 100.0,
                                "bybit_order_id": "RECOVERED",
                                "bybit_scale_out_order_id": "RECOVERED"
                            }
                            
                            bot_state["trade_history"].append(trade_record)
                            save_history()
                            log_trade_journal(trade_record)
                            
                            emoji = "🚀" if closed_pnl > 0 else "🔴"
                            status_str = "*TAKE PROFIT HIT (RECOVERED)*" if closed_pnl > 0 else "*STOP LOSS HIT (RECOVERED)*"
                            send_telegram_alert(
                                f"{emoji} {status_str} 💻\n"
                                f"• *Asset*: {symbol}\n"
                                f"• *Status*: Trade closed while bot container was offline/restarting.\n"
                                f"• *Direction*: {direction}\n"
                                f"• *Entry Price*: ${entry_price:.2f}\n"
                                f"• *Exit Price*: ${exit_price:.2f}\n"
                                f"• *Realized PnL*: *${closed_pnl:+.2f}*\n"
                                f"• *Exit Time*: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(exit_time_sec))}"
                            )
        except Exception as err:
            print(f"[Crash Recovery Scan Error] for {symbol}: {err}")

def execute_bybit_trade_async(*args, **kwargs):
    symbol = args[0] if args else "Unknown"
    try:
        _execute_bybit_trade_async_inner(*args, **kwargs)
    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"[CRITICAL EXECUTION ERROR] Async trade execution failed for {symbol}: {err_msg}")
        traceback.print_exc()
        send_telegram_alert(f"🚨 *Async Trade Execution Error* 🚨\n• Symbol: `{symbol}`\n• Error: `{err_msg}`")
    finally:
        with active_execution_lock:
            active_execution_symbols.discard(symbol)

def _execute_bybit_trade_async_inner(symbol, iv, tf, ml_trend, leverage_val, qty_str, raw_qty, entry_price, stop_loss_price, take_profit_price, position_size_usd, kelly_fraction, calibrated_confidence, ml_confidence, dynamic_conf_threshold, latest_completed_ts, latest_candle, pred_change, predicted_price, atr_dollars, tp_multiplier_adjusted, sl_multiplier_adjusted, df_completed, trade_uuid, duration_seconds, active_trade_key):
    if latest_candle is None:
        latest_candle = {}
    if df_completed is None:
        df_completed = pd.DataFrame()
        
    bybit_success = True
    bybit_order_id = None
    bybit_scale_out_order_id = None
    actual_qty = raw_qty
    
    # 1. Live Exchange Position Guard
    try:
        pos_list = get_all_bybit_positions()
        if pos_list:
            existing_pos = next((p for p in pos_list if p.get("symbol") == symbol and float(p.get("size", "0")) > 0), None)
            if existing_pos:
                print(f"[{symbol} {iv}m API Block] Live order placement skipped: a live position already exists on Bybit.")
                sync_active_positions_from_bybit()
                return
    except Exception as pos_check_err:
        print(f"[{symbol} {iv}m API Warning] Live Position Guard check failed: {pos_check_err}")

    print(f"[{symbol} {iv}m API] Preparing to open live position on Bybit ({TRADE_MODE.upper()})...")
    leverage_ok = set_bybit_leverage(symbol, leverage_val)
    if leverage_ok:
        side = "Buy" if ml_trend == "Bullish" else "Sell"
        bybit_success = False
        
        rolling_atr = df_completed["ATR_norm"].tail(30)
        atr_mean = rolling_atr.mean()
        atr_std = rolling_atr.std()
        vol_z_score = (latest_candle["ATR_norm"] - atr_mean) / (atr_std + 1e-8) if atr_std > 0 else 0.0
        is_extreme_volatility = vol_z_score > 3.0
        
        if is_extreme_volatility:
            print(f"[{symbol} {iv}m API] Volatility Spiked! Z-score: {vol_z_score:.2f} > 3.0. Placing Taker IOC order immediately...")
            order_res = place_bybit_taker_ioc_order(symbol, side, qty_str, sl=stop_loss_price, tp=take_profit_price)
            if order_res.get("retCode") == 0:
                bybit_order_id = order_res.get("result", {}).get("orderId")
                bybit_success = True
                time.sleep(0.5)
                order_details = get_bybit_order_details(symbol, bybit_order_id)
                if order_details:
                    entry_price = float(order_details.get("avgPrice", entry_price))
                    actual_qty = float(order_details.get("cumExecQty", raw_qty))
                else:
                    fill_exec = get_bybit_last_execution(symbol)
                    if fill_exec:
                        entry_price = float(fill_exec.get("execPrice", entry_price))
                    actual_qty = raw_qty
            else:
                print(f"[{symbol} {iv}m API ERROR] Taker IOC order failed: {order_res.get('retMsg')}")
        else:
            # Normal Volatility: Place Limit Maker entry order with dynamic price chasing
            for chase in range(5):
                bid, ask, last = get_bybit_bid_ask(symbol)
                if bid is None or ask is None:
                    bid, ask = entry_price, entry_price
                limit_price = bid if side == "Buy" else ask
                print(f"[{symbol} {iv}m API] Placing Limit Maker order at ${limit_price:.4f} (Chase {chase+1}/5)...")
                order_res = place_bybit_limit_order(symbol, side, qty_str, limit_price)
                if order_res.get("retCode") == 0:
                    bybit_order_id = order_res.get("result", {}).get("orderId")
                    filled = False
                    for idx in range(24):
                        time.sleep(0.5)
                        with _ws_filled_orders_lock:
                            ws_details = _ws_filled_orders.get(bybit_order_id)
                        if ws_details:
                            entry_price = float(ws_details.get("avgPrice", limit_price))
                            actual_qty = float(ws_details.get("cumExecQty", raw_qty))
                            filled = True
                            bybit_success = True
                            print(f"[{symbol} {iv}m API] Order fill detected via WebSocket in {idx*0.5:.1f}s.")
                            break
                        if idx > 0 and idx % 6 == 0:
                            order_details = get_bybit_order_details(symbol, bybit_order_id)
                            if order_details:
                                status = order_details.get("orderStatus")
                                if status == "Filled":
                                    entry_price = float(order_details.get("avgPrice", limit_price))
                                    actual_qty = float(order_details.get("cumExecQty", raw_qty))
                                    filled = True
                                    bybit_success = True
                                    break
                                elif status in ["Cancelled", "Rejected"]:
                                    break
                    if filled:
                        print(f"[{symbol} {iv}m API] Success! Maker Limit Order filled at ${entry_price:.4f}.")
                        break
                    else:
                        print(f"[{symbol} {iv}m API] Order unfilled after 12s. Cancelling and re-quoting...")
                        cancel_bybit_order(symbol, bybit_order_id)
                        time.sleep(0.5)
                        final_details = get_bybit_order_details(symbol, bybit_order_id)
                        if final_details:
                            status = final_details.get("orderStatus")
                            cum_qty = float(final_details.get("cumExecQty", 0.0))
                            if status == "Filled" or cum_qty > 0:
                                entry_price = float(final_details.get("avgPrice", limit_price))
                                actual_qty = cum_qty if cum_qty > 0 else raw_qty
                                filled = True
                                bybit_success = True
                                print(f"[{symbol} {iv}m API] Success! Maker Limit Order filled/partially filled during cancel request at ${entry_price:.4f} (Qty: {actual_qty}).")
                                break
                else:
                    print(f"[{symbol} {iv}m API WARNING] Limit order placement failed: {order_res.get('retMsg')} (waiting 2s before retry)")
                    time.sleep(2)
                    
            # Fallback to Market order if limit chases failed (re-check WS fill cache first to prevent race condition)
            if not bybit_success:
                with _ws_filled_orders_lock:
                    if bybit_order_id and bybit_order_id in _ws_filled_orders:
                        bybit_success = True
                        print(f"[{symbol} {iv}m API] Order fill confirmed in WS cache. Skipping market order fallback.")
            
            if not bybit_success:
                atr_norm = float(latest_candle.get("ATR_norm", 0.0))
                if atr_norm >= 0.015:
                    print(f"[{symbol} {iv}m API BLOCK] Limit chases failed. Market fallback blocked due to extreme volatility (ATR_norm: {atr_norm:.4f} >= 0.015).")
                else:
                    print(f"[{symbol} {iv}m API] All Limit Maker chases failed. Falling back to Market order to guarantee entry...")
                    order_res = place_bybit_order(symbol, side, qty_str)
                    if order_res.get("retCode") == 0:
                        bybit_order_id = order_res.get("result", {}).get("orderId")
                        bybit_success = True
                        time.sleep(0.5)
                        order_details = get_bybit_order_details(symbol, bybit_order_id)
                        if order_details:
                            entry_price = float(order_details.get("avgPrice", entry_price))
                            actual_qty = float(order_details.get("cumExecQty", raw_qty))
                        else:
                            fill_exec = get_bybit_last_execution(symbol)
                            if fill_exec:
                                entry_price = float(fill_exec.get("execPrice", entry_price))
                            actual_qty = raw_qty
                            
        if bybit_success:
            # 3. Recalculate SL/TP targets based on actual entry_price
            min_sl_pct = 1.0 if symbol == "BTCUSDT" else (1.5 if symbol in ["ETHUSDT", "SOLUSDT", "BNBUSDT"] else 2.0)
            min_sl_dist = entry_price * (min_sl_pct / 100.0)
            raw_sl_dist = max(sl_multiplier_adjusted * atr_dollars, min_sl_dist)
            
            min_tp_dist = entry_price * 0.015
            if ml_trend == "Bullish":
                stop_loss_price = entry_price - raw_sl_dist
                est_tp_price = estimate_liquidation_pool(df_completed, "Bullish", entry_price)
                take_profit_price = max(entry_price + min_tp_dist, min(est_tp_price, entry_price + 1.5 * tp_multiplier_adjusted * atr_dollars))
            else:
                stop_loss_price = entry_price + raw_sl_dist
                est_tp_price = estimate_liquidation_pool(df_completed, "Bearish", entry_price)
                take_profit_price = min(entry_price - min_tp_dist, max(est_tp_price, entry_price - 1.5 * tp_multiplier_adjusted * atr_dollars))
                
            # 4. Set SL/TP on active position on Bybit
            temp_trade = {"qty": str(actual_qty), "direction": ml_trend}
            update_bybit_stop_loss(symbol, stop_loss_price, active_trade=temp_trade)
            update_bybit_take_profit(symbol, take_profit_price, active_trade=temp_trade)
            
            # Place scale-out limit order on Bybit
            limit_side = "Sell" if ml_trend == "Bullish" else "Buy"
            limit_price = entry_price + 1.0 * atr_dollars if ml_trend == "Bullish" else entry_price - 1.0 * atr_dollars
            limit_qty_str = format_bybit_qty(symbol, actual_qty * 0.5)
            limit_qty_val = float(limit_qty_str)
            scale_out_val = limit_qty_val * limit_price
            
            if scale_out_val >= 5.0:
                print(f"[{symbol} {iv}m API] Placing scale-out limit order for {limit_qty_str} at ${limit_price:.4f}...")
                limit_res = place_bybit_limit_order(symbol, limit_side, limit_qty_str, limit_price, reduce_only=True)
                if limit_res.get("retCode") == 0:
                    bybit_scale_out_order_id = limit_res.get("result", {}).get("orderId")
                    print(f"[{symbol} {iv}m API] Scale-out limit order placed successfully. Order ID: {bybit_scale_out_order_id}")
                else:
                    print(f"[{symbol} {iv}m API WARNING] Failed to place scale-out limit order: {limit_res.get('retMsg')}")
    else:
        bybit_success = False
        send_telegram_alert(
            f"🔴 *BYBIT LEVERAGE SETTING ERROR* 🔴\n"
            f"• *Asset*: {symbol}\n"
            f"• *Interval*: {iv}m\n"
            f"• *Target Leverage*: {leverage_val}x\n"
            f"• *Detail*: Failed to configure leverage on Bybit."
        )
        return
        
    if bybit_success:
        actual_size_usd = float(position_size_usd) * (actual_qty / raw_qty) if raw_qty > 0 else float(position_size_usd)
        active_trade = {
            "trade_id": f"{symbol}_{trade_uuid}",
            "bybit_order_id": bybit_order_id,
            "bybit_scale_out_order_id": bybit_scale_out_order_id,
            "symbol": symbol,
            "entry_price": float(entry_price),
            "predicted_price": float(predicted_price),
            "stop_loss": float(stop_loss_price),
            "take_profit": float(take_profit_price),
            "direction": str(ml_trend),
            "end_time": float(time.time() + duration_seconds),
            "entry_time": int(time.time() * 1000),
            "atr_dollars": float(atr_dollars),
            "highest_price": float(entry_price),
            "lowest_price": float(entry_price),
            "break_even_triggered": False,
            "half_closed": False,
            "original_size": float(position_size_usd),
            "position_size_usd": actual_size_usd,
            "scaled_out_pnl": 0.0,
            "kelly_fraction": float(kelly_fraction),
            "leverage": float(leverage_val),
            "confidence": float(calibrated_confidence),
            "qty": float(actual_qty),
            "original_qty": float(actual_qty),
            "fill_pct": round((actual_qty / raw_qty) * 100.0, 2) if raw_qty > 0 else 100.0
        }
        
        with active_trades_lock:
            current_trades = bot_state.get(active_trade_key, [])
            if not isinstance(current_trades, list):
                current_trades = []
            current_trades = list(current_trades)
            current_trades.append(active_trade)
            bot_state[active_trade_key] = current_trades
            
        sync_active_positions_from_bybit()
        
        entry_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        send_telegram_alert(
            f"🟢 *POSITION OPENED (SUCCESSFUL SIGNAL)* 🟢\n"
            f"• *Asset*: {symbol}\n"
            f"• *Interval*: {iv}m\n"
            f"• *Direction*: {ml_trend}\n"
            f"• *Entry Price*: ${float(entry_price):.4f}\n"
            f"• *Entry Time*: {entry_time_str}\n"
            f"• *Take Profit*: ${float(take_profit_price):.4f}\n"
            f"• *Stop Loss*: ${float(stop_loss_price):.4f}\n"
            f"• *Calibrated Confidence*: {calibrated_confidence * 100:.2f}%\n"
            f"• *Leverage*: {leverage_val:.1f}x\n"
            f"• *Position Size*: ${actual_size_usd:.2f} (Value: ${actual_qty * entry_price:.2f})\n"
            f"• *Execution Mode*: {TRADE_MODE.upper()}"
        )
        print(f"[{symbol} {iv}m] Trade Opened: {ml_trend} at price {entry_price:.2f} (SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f})")
    else:
        err_msg = order_res.get('retMsg') if 'order_res' in locals() else "Execution failed"
        err_code = order_res.get('retCode') if 'order_res' in locals() else "N/A"
        send_telegram_alert(
            f"🔴 *BYBIT API ORDER ERROR* 🔴\n"
            f"• *Asset*: {symbol}\n"
            f"• *Interval*: {iv}m\n"
            f"• *Direction*: {ml_trend}\n"
            f"• *Error Message*: {err_msg} (Code: {err_code})"
        )

def main():
    global live_price, last_ws_update_time
    # Load model weights here (deferred from module level)
    for iv in ["60", "120", "240", "360"]:
        load_model_weights(iv)
    load_history()
    print(f"{SYMBOL} LIVE BOT RUNNING...")
    send_telegram_alert(f"✅ *Bot started successfully on {TRADE_MODE.upper()} mode.*\n_Container boot detected — all systems nominal._")
    

    # Pre-load initial prices for all supported symbols
    load_initial_prices()

    # Crash Recovery: re-sync orphaned Bybit positions
    sync_active_positions_from_bybit()
    
    # Crash Recovery: scan for closed trades while offline
    recover_missed_closed_trades()

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
    last_position_sync_time = 0.0
    completed_this_hour = set()
    hour_check_complete = False
    last_candle_check_time = 0.0

    while True:
        current_time = time.time()
        
        # Sync active positions from Bybit periodically to save proxy bandwidth
        has_active_positions = any(len(bot_state.get(f"active_trade_{tf}", [])) > 0 for tf in ["1h", "2h", "4h", "6h"])
        sync_interval = POSITION_SYNC_INTERVAL_SECS if has_active_positions else POSITION_SYNC_IDLE_INTERVAL_SECS
        
        if (current_time - last_position_sync_time >= sync_interval):
            success = sync_active_positions_from_bybit()
            if success:
                last_position_sync_time = current_time
            else:
                # Retry in 5 seconds
                last_position_sync_time = current_time - sync_interval + 5.0
        
        # 1. Health check & current price update (Adaptive to save proxy bandwidth)
        # Rely on background run_fallback_price_updater. Only query directly if live_price is None
        # or has not been updated in over 10 minutes (600s) as a fail-safe.
        if live_price is None or (current_time - last_ws_update_time > 30.0):
            fallback_price = get_fallback_price()
            if fallback_price is not None:
                print(f"[{get_pkt_time().strftime('%H:%M:%S')}] WebSocket/Fallback price is stale or disconnected. Fetching price: {fallback_price:.2f}")
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
        with active_trades_lock:
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
    
                    # Bybit Live position query and state tracking
                    bybit_closed = False
                    bybit_scaled_out = False
                    bybit_exit_price = None
                    bybit_realized_pnl = None
                    
                    if TRADE_MODE != "simulation":
                        if active_trade.get("bybit_closed", False):
                            bybit_closed = True
                        else:
                            # Detect scale-out fill from cached and stored qty values
                            original_qty = active_trade.get("original_qty", active_trade.get("qty", 0.0))
                            current_qty = active_trade.get("qty", 0.0)
                            if original_qty > 0 and current_qty <= (original_qty * 0.6) and not active_trade.get("half_closed", False):
                                # Verify fill status of the scale-out limit order to prevent premature/false triggers
                                scale_out_order_id = active_trade.get("bybit_scale_out_order_id")
                                if scale_out_order_id:
                                    order_details = get_bybit_order_details(active_symbol, scale_out_order_id)
                                    if order_details and order_details.get("orderStatus") == "Filled":
                                        bybit_scaled_out = True
                                    else:
                                        status_msg = order_details.get("orderStatus") if order_details else "Unknown"
                                        stuck_since = active_trade.get("scale_out_stuck_since", time.time())
                                        active_trade.setdefault("scale_out_stuck_since", stuck_since)
                                        if time.time() - stuck_since > 600:  # 10-minute timeout
                                            print(f"[{active_symbol}] Scale-out stuck >10 min ({status_msg}). Cancelling stale order and marking half_closed.")
                                            cancel_bybit_order(active_symbol, scale_out_order_id)
                                            active_trade["half_closed"] = True
                                            active_trade.pop("scale_out_stuck_since", None)
                                        else:
                                            print(f"[{active_symbol}] Size check indicates scale-out, but limit order status is not Filled ({status_msg}). Waiting.")
                                else:
                                    bybit_scaled_out = True
    
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
                            # Trailing Stop: Trails highest price by dynamic multiplier
                            potential_sl = highest_price - trailing_multiplier * atr_dollars
                            if potential_sl > stop_loss:
                                if TRADE_MODE != "simulation":
                                    success = update_bybit_stop_loss(active_symbol, potential_sl, active_trade)
                                    if success:
                                        stop_loss = potential_sl
                                        active_trade["stop_loss"] = stop_loss
                                        active_trades_updated = True
                                        print(f"[{iv}m Trailing Stop] Moved SL up to {stop_loss:.2f} (trailing highest: {highest_price:.2f}, multiplier: {trailing_multiplier}x)")
                                else:
                                    stop_loss = potential_sl
                                    active_trade["stop_loss"] = stop_loss
                                    active_trades_updated = True
                                    print(f"[{iv}m Trailing Stop] Moved SL up to {stop_loss:.2f} (trailing highest: {highest_price:.2f}, multiplier: {trailing_multiplier}x)")
                        
                        # Break-Even Guard: if price goes up by 0.5 * ATR, move SL to entry
                        if not break_even_triggered and current_price >= entry_price + 0.5 * atr_dollars:
                            target_sl = max(stop_loss, entry_price)
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, target_sl, active_trade)
                                if success:
                                    break_even_triggered = True
                                    active_trade["break_even_triggered"] = True
                                    stop_loss = target_sl
                                    active_trade["stop_loss"] = stop_loss
                                    active_trades_updated = True
                                    print(f"[{iv}m Break-Even Guard] Triggered! SL moved to entry price: {entry_price:.2f}")
                            else:
                                break_even_triggered = True
                                active_trade["break_even_triggered"] = True
                                stop_loss = target_sl
                                active_trade["stop_loss"] = stop_loss
                                active_trades_updated = True
                                print(f"[{iv}m Break-Even Guard] Triggered! SL moved to entry price: {entry_price:.2f}")
                    else:
                        if current_price < lowest_price:
                            lowest_price = current_price
                            active_trade["lowest_price"] = lowest_price
                            # Trailing Stop: Trails lowest price by dynamic multiplier
                            potential_sl = lowest_price + trailing_multiplier * atr_dollars
                            if potential_sl < stop_loss:
                                if TRADE_MODE != "simulation":
                                    success = update_bybit_stop_loss(active_symbol, potential_sl, active_trade)
                                    if success:
                                        stop_loss = potential_sl
                                        active_trade["stop_loss"] = stop_loss
                                        active_trades_updated = True
                                        print(f"[{iv}m Trailing Stop] Moved SL down to {stop_loss:.2f} (trailing lowest: {lowest_price:.2f}, multiplier: {trailing_multiplier}x)")
                                else:
                                    stop_loss = potential_sl
                                    active_trade["stop_loss"] = stop_loss
                                    active_trades_updated = True
                                    print(f"[{iv}m Trailing Stop] Moved SL down to {stop_loss:.2f} (trailing lowest: {lowest_price:.2f}, multiplier: {trailing_multiplier}x)")
                                
                        # Break-Even Guard: if price goes down by 0.5 * ATR, move SL to entry
                        if not break_even_triggered and current_price <= entry_price - 0.5 * atr_dollars:
                            target_sl = min(stop_loss, entry_price)
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, target_sl, active_trade)
                                if success:
                                    break_even_triggered = True
                                    active_trade["break_even_triggered"] = True
                                    stop_loss = target_sl
                                    active_trade["stop_loss"] = stop_loss
                                    active_trades_updated = True
                                    print(f"[{iv}m Break-Even Guard] Triggered! SL moved to entry price: {entry_price:.2f}")
                            else:
                                break_even_triggered = True
                                active_trade["break_even_triggered"] = True
                                stop_loss = target_sl
                                active_trade["stop_loss"] = stop_loss
                                active_trades_updated = True
                                print(f"[{iv}m Break-Even Guard] Triggered! SL moved to entry price: {entry_price:.2f}")
    
                    # Scale-Out (50% partial profit taking at 1.0 * ATR)
                    half_closed = active_trade.get("half_closed", False)
                    trigger_scale_out = False
                    if not half_closed:
                        if TRADE_MODE != "simulation":
                            trigger_scale_out = bybit_scaled_out
                        else:
                            if direction == "Bullish" and current_price >= entry_price + 1.0 * atr_dollars:
                                trigger_scale_out = True
                            elif direction == "Bearish" and current_price <= entry_price - 1.0 * atr_dollars:
                                trigger_scale_out = True
    
                    if trigger_scale_out and not half_closed:
                        if direction == "Bullish":
                            # Scale-Out Triggered for Long
                            half_closed = True
                            active_trade["half_closed"] = True
                            
                            # Close 50% of the position
                            closed_size = round(position_size_usd * 0.5, 2)
                            remaining_size = round(position_size_usd - closed_size, 2)
                            
                            # Calculate profit on closed half (correct taker fee on leveraged size)
                            raw_return_pct = ((current_price - entry_price) / entry_price) * 100.0
                            lev = active_trade.get("leverage", 1.0)
                            gross_pnl = closed_size * (raw_return_pct * lev / 100.0)
                            taker_fee_cost = closed_size * lev * 0.00055  # exit side only
                            pnl_usd = round(gross_pnl - taker_fee_cost, 2)
                            if pnl_usd < -closed_size:
                                pnl_usd = -closed_size
                                net_return_pct = -100.0
                            else:
                                net_return_pct = round((pnl_usd / closed_size) * 100.0, 2) if closed_size > 0 else 0.0
                            
                            # Save scaled out pnl
                            active_trade["scaled_out_pnl"] = pnl_usd
                            
                            # Refund closed size + PnL to wallet balance (only in simulation)
                            if TRADE_MODE == "simulation":
                                bot_state["simulated_balance"] = round(bot_state["simulated_balance"] + closed_size + pnl_usd, 2)
                            
                            # Update position details
                            position_size_usd = remaining_size
                            active_trade["position_size_usd"] = remaining_size
                            
                            # Move stop loss to entry price (break-even)
                            target_sl = entry_price
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, target_sl, active_trade)
                                if success:
                                    stop_loss = target_sl
                                    active_trade["stop_loss"] = target_sl
                                    active_trade["break_even_triggered"] = True
                                    active_trades_updated = True
                                    print(f"[{active_symbol} {iv}m Scale-Out] 50% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to entry: {entry_price:.2f}")
                                    update_bybit_take_profit(active_symbol, take_profit, active_trade)
                                else:
                                    print(f"[{active_symbol} {iv}m Scale-Out ERROR] Failed to update Stop Loss to entry on Bybit. SL remains at {stop_loss:.2f}")
                            else:
                                stop_loss = target_sl
                                active_trade["stop_loss"] = target_sl
                                active_trade["break_even_triggered"] = True
                                active_trades_updated = True
                                print(f"[{active_symbol} {iv}m Scale-Out] 50% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to entry: {entry_price:.2f}")
                            
                        elif direction == "Bearish":
                            # Scale-Out Triggered for Short
                            half_closed = True
                            active_trade["half_closed"] = True
                            
                            # Close 50% of the position
                            closed_size = round(position_size_usd * 0.5, 2)
                            remaining_size = round(position_size_usd - closed_size, 2)
                            
                            # Calculate profit on closed half (correct taker fee on leveraged size)
                            raw_return_pct = ((entry_price - current_price) / entry_price) * 100.0
                            lev = active_trade.get("leverage", 1.0)
                            gross_pnl = closed_size * (raw_return_pct * lev / 100.0)
                            taker_fee_cost = closed_size * lev * 0.00055  # exit side only
                            pnl_usd = round(gross_pnl - taker_fee_cost, 2)
                            if pnl_usd < -closed_size:
                                pnl_usd = -closed_size
                                net_return_pct = -100.0
                            else:
                                net_return_pct = round((pnl_usd / closed_size) * 100.0, 2) if closed_size > 0 else 0.0
                            
                            # Save scaled out pnl
                            active_trade["scaled_out_pnl"] = pnl_usd
                            
                            # Refund closed size + PnL to wallet balance (only in simulation)
                            if TRADE_MODE == "simulation":
                                bot_state["simulated_balance"] = round(bot_state["simulated_balance"] + closed_size + pnl_usd, 2)
                            
                            # Update position details
                            position_size_usd = remaining_size
                            active_trade["position_size_usd"] = remaining_size
                            
                            # Move stop loss to entry price + fee offset (fee-free break-even for short)
                            fee_buffer = entry_price * 0.0005
                            target_sl = entry_price + fee_buffer
                            
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, target_sl, active_trade)
                                if success:
                                    stop_loss = target_sl
                                    active_trade["stop_loss"] = target_sl
                                    active_trade["break_even_triggered"] = True
                                    active_trades_updated = True
                                    print(f"[{active_symbol} {iv}m Scale-Out] 50% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to fee-adjusted entry: {stop_loss:.2f}")
                                    update_bybit_take_profit(active_symbol, take_profit, active_trade)
                                else:
                                    print(f"[{active_symbol} {iv}m Scale-Out ERROR] Failed to update Stop Loss to fee-adjusted entry on Bybit. SL remains at {stop_loss:.2f}")
                            else:
                                stop_loss = target_sl
                                active_trade["stop_loss"] = target_sl
                                active_trade["break_even_triggered"] = True
                                active_trades_updated = True
                                print(f"[{active_symbol} {iv}m Scale-Out] 50% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to fee-adjusted entry: {stop_loss:.2f}")
    
                    remaining_seconds = max(0, int(end_time - current_time))
                    mins, secs = divmod(remaining_seconds, 60)
                    countdown_str = f"{mins:02d}m {secs:02d}s"
                    print(f"[{active_symbol} {iv}m Active Trade] {direction} | Price: {current_price:.2f} (Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}) | Countdown: {countdown_str}")
                    exit_reason = None
                    half_closed = active_trade.get("half_closed", False)
                    
                    # 1. Check timer programmatic exit (applies to both simulation and live)
                    if current_time >= end_time and not half_closed:
                        cfg = TIMEFRAME_CONFIG.get(str(iv), {"lookahead": 10})
                        lookahead = cfg.get("lookahead", 10)
                        exit_reason = f"{int(iv)*lookahead}-MINUTE TIMER ELAPSED"
                    
                    # 2. Check stagnation programmatic exit (applies to both simulation and live)
                    if not exit_reason and not half_closed:
                        entry_time_ms = active_trade.get("entry_time")
                        if entry_time_ms:
                            trade_age_hours = (time.time() - (entry_time_ms / 1000.0)) / 3600.0
                            cfg = TIMEFRAME_CONFIG.get(str(iv), {"lookahead": 10})
                            lookahead = cfg.get("lookahead", 10)
                            lookahead_duration_hours = (int(iv) * lookahead) / 60.0
                            stagnation_age_hours = 0.6 * lookahead_duration_hours
                            
                            if trade_age_hours >= stagnation_age_hours:
                                atr_dollars = active_trade.get("atr_dollars") or (entry_price * 0.01)
                                price_dev = abs(current_price - entry_price)
                                if price_dev < (0.5 * atr_dollars):
                                    exit_reason = f"STAGNATION TIMEOUT (Age: {trade_age_hours:.1f}h, Price within 0.5 ATR)"
                    
                    # 3. Simulation mode SL/TP price checks
                    if TRADE_MODE == "simulation" and not exit_reason:
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
                    
                    if TRADE_MODE != "simulation":
                        if bybit_closed:
                            if bybit_realized_pnl is not None:
                                if bybit_realized_pnl >= 0:
                                    exit_reason = "TRAILING STOP HIT [SUCCESS]" if half_closed else "TAKE PROFIT HIT [SUCCESS]"
                                else:
                                    exit_reason = "STOP LOSS HIT [FAIL]"
                            else:
                                actual_exit_price = bybit_exit_price if bybit_exit_price is not None else current_price
                                actual_change = actual_exit_price - entry_price
                                is_profit = actual_change > 0 if direction == "Bullish" else actual_change < 0
                                if is_profit:
                                    exit_reason = "TRAILING STOP HIT [SUCCESS]" if half_closed else "TAKE PROFIT HIT [SUCCESS]"
                                else:
                                    exit_reason = "STOP LOSS HIT [FAIL]"
                        elif exit_reason is not None:
                            # Cancel scale-out limit order if it exists
                            scale_out_id = active_trade.get("bybit_scale_out_order_id")
                            if scale_out_id:
                                cancel_payload = {"category": "linear", "symbol": active_symbol, "orderId": scale_out_id}
                                bybit_post_request("/v5/order/cancel", cancel_payload)
                                print(f"[Bybit API] Canceled scale-out limit order {scale_out_id} due to programmatic exit.")
                            
                            # Place market close order
                            pos = get_bybit_position(active_symbol)
                            if pos:
                                qty_str = pos.get("size", "0")
                                if float(qty_str) > 0:
                                    close_side = "Sell" if direction == "Bullish" else "Buy"
                                    print(f"[Bybit API] Placing Market close order for {qty_str} {active_symbol} due to programmatic exit...")
                                    close_res = place_bybit_order(symbol=active_symbol, side=close_side, qty=qty_str, reduce_only=True)
                                    if close_res.get("retCode") == 0:
                                        bybit_closed = True
                                        time.sleep(0.5)
    
                    if exit_reason is not None:
                        # Query accumulated closed PnL transactions on Testnet/Live
                        bybit_pnl_data = None
                        if TRADE_MODE != "simulation":
                            entry_time_ms = active_trade.get("entry_time")
                            if not entry_time_ms:
                                entry_time_ms = int((end_time - (int(iv) * 60)) * 1000)
                            
                            bybit_pnl_data = get_bybit_accumulated_closed_pnl(active_symbol, entry_time_ms)
                            if bybit_pnl_data:
                                bybit_realized_pnl = bybit_pnl_data["total_pnl"]
                                if bybit_pnl_data["avg_exit_price"] is not None:
                                    bybit_exit_price = bybit_pnl_data["avg_exit_price"]
                                if bybit_pnl_data["total_entry_value"] is not None:
                                    lev = float(active_trade.get("leverage", 1.0))
                                    actual_margin = round(bybit_pnl_data["total_entry_value"] / lev, 2)
                                    active_trade["position_size_usd"] = actual_margin
                                    position_size_usd = actual_margin
    
                        # Maker vs Taker execution logic
                        is_stop_loss = "STOP LOSS" in str(exit_reason).upper()
                        
                        if is_stop_loss:
                            # Maker limit execution for Stop Loss exit (Post-Only model)
                            slippage_pct = 0.0
                            actual_price = bybit_exit_price if bybit_exit_price is not None else current_price
                            fee_rate_roundtrip = 0.04  # Maker Entry + Maker Exit roundtrip
                            exit_reason = str(exit_reason) + " [Limit order Maker close]"
                        else:
                            # Maker execution for Take Profit, Timer, etc.
                            slippage_pct = 0.0
                            actual_price = bybit_exit_price if bybit_exit_price is not None else current_price
                            fee_rate_roundtrip = 0.04  # Maker Entry + Maker Exit roundtrip
    
                        price_diff = actual_price - predicted_price
                        price_diff_pct = (price_diff / predicted_price) * 100
                        price_accuracy = max(0.0, 100.0 - abs((actual_price - predicted_price) / actual_price * 100))
                        actual_change = actual_price - entry_price
                        actual_change_pct = (actual_change / entry_price) * 100
                        
                        # Calculate PnL (long vs short) with correct taker fees on leveraged size
                        raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
                        leverage = active_trade.get("leverage", 1.0)
                        gross_pnl = position_size_usd * (raw_return_pct * leverage / 100.0)
                        entry_fee_rate = 0.0002  # Assumed Maker entry
                        exit_fee_rate = 0.0002 if not is_stop_loss else 0.00055  # Limit exit vs Stop Loss market close
                        roundtrip_fee_rate = entry_fee_rate + exit_fee_rate
                        fee_cost = position_size_usd * leverage * roundtrip_fee_rate
                        realized_pnl = gross_pnl - fee_cost
                        net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
                        
                        if realized_pnl < -position_size_usd:
                            realized_pnl = -position_size_usd
                            net_return_pct = -100.0
                        
                        # Aggregate PnL and size for trade history logging if scaled out
                        original_size = float(active_trade.get("original_size", position_size_usd))
                        scaled_out_pnl = float(active_trade.get("scaled_out_pnl", 0.0))
                        
                        if TRADE_MODE != "simulation" and bybit_realized_pnl is not None:
                            total_pnl = round(bybit_realized_pnl, 2)
                            realized_pnl = round(total_pnl - scaled_out_pnl, 2)
                            net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
                            total_net_return_pct = round((total_pnl / original_size) * 100.0, 4)
                        else:
                            total_pnl = round(realized_pnl + scaled_out_pnl, 2)
                            total_net_return_pct = round((total_pnl / original_size) * 100.0, 4)
                        
                        # Update simulated balance (only in simulation)
                        if TRADE_MODE == "simulation":
                            old_bal = bot_state.get("simulated_balance", 80.0)
                            new_bal = round(old_bal + position_size_usd + realized_pnl, 2)
                            bot_state["simulated_balance"] = new_bal
                        else:
                            new_bal = bot_state.get("simulated_balance", 0.0)
                        
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
                        print("==================================================")
                        
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
                            "position_size_usd": float(position_size_usd),
                            "original_size": float(original_size),
                            "pnl_usd": float(total_pnl),
                            "balance": float(new_bal),
                            "leverage": float(leverage),
                            "confidence": active_trade.get("confidence") if active_trade.get("confidence") == "MT" else float(active_trade.get("confidence") or 0.0),
                            "take_profit": float(active_trade.get("take_profit", 0.0)),
                            "stop_loss": float(active_trade.get("stop_loss", 0.0)),
                            "atr_dollars": float(active_trade.get("atr_dollars", 0.0)),
                            "fill_pct": float(active_trade.get("fill_pct", 100.0)),
                            "bybit_order_id": active_trade.get("bybit_order_id"),
                            "bybit_scale_out_order_id": active_trade.get("bybit_scale_out_order_id")
                        })
                        # Log to trade journal CSV
                        log_trade_journal(bot_state["trade_history"][-1])
                        
                        if total_pnl > 0:
                            exit_header = "🚀 *TAKE PROFIT HIT* 🚀" if "TAKE PROFIT" in str(exit_reason).upper() else "📈 *TRAILING STOP HIT (PROFITABLE)* 📈" if "TRAILING" in str(exit_reason).upper() else "🎉 *TRADE CLOSED WITH PROFIT* 🎉"
                            send_telegram_alert(
                                f"{exit_header}\n"
                                f"• *Asset*: {active_symbol}\n"
                                f"• *Interval*: {iv}m\n"
                                f"• *Direction*: {direction}\n"
                                f"• *Entry Price*: ${entry_price:.2f}\n"
                                f"• *Exit Price*: ${actual_price:.2f}\n"
                                f"• *Realized PnL*: *${total_pnl:+.2f}* (" + (f"{total_net_return_pct:+.2f}" if active_trade.get("half_closed", False) else f"{net_return_pct:+.2f}") + f"%)\n"
                                f"• *Exit Reason*: {exit_reason}" + (" (Scale-Out)" if active_trade.get("half_closed", False) else "") + "\n"
                                f"• *New Balance*: ${new_bal:.2f}"
                            )
                        else:
                            send_telegram_alert(
                                f"🔴 *POSITION CLOSED (AUTO)* 🔴\n"
                                f"• *Asset*: {active_symbol}\n"
                                f"• *Interval*: {iv}m\n"
                                f"• *Direction*: {direction}\n"
                                f"• *Exit Reason*: {exit_reason}" + (" (Scale-Out)" if active_trade.get("half_closed", False) else "") + "\n"
                                f"• *Entry Price*: ${entry_price:.2f}\n"
                                f"• *Exit Price*: ${actual_price:.2f}\n"
                                f"• *Realized PnL*: ${total_pnl:+.2f} (" + (f"{total_net_return_pct:+.2f}" if active_trade.get("half_closed", False) else f"{net_return_pct:+.2f}") + f"%)\n"
                                f"• *New Balance*: ${new_bal:.2f}"
                            )
                        
                        
                        # Send email alert on any profitable trade exit
                        if total_pnl > 0:
                            subject = f"🚀 [UBOTE Profit Target] {active_symbol} {iv}m Closed with Profit!"
                            invested_margin_usd = original_size
                            leveraged_position_usd = original_size * leverage if leverage > 0 else original_size
                            
                            # Dynamic header based on exit reason
                            exit_title = "🎉 Take Profit Hit!" if "TAKE PROFIT" in str(exit_reason).upper() else "📈 Trailing Stop Hit (Profitable Close)!" if "TRAILING" in str(exit_reason).upper() else "✅ Trade Closed with Profit!"
                            
                            body = f"""
                            <html>
                            <body style="font-family: Arial, sans-serif; background-color: #0b0e14; color: #f0f3fa; padding: 20px; border-radius: 8px;">
                                <h2 style="color: #00b0ff; margin-bottom: 20px;">{exit_title}</h2>
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
                                            <td style="padding: 6px 0; color: #8f9bb3;"><b>Exit Reason:</b></td>
                                            <td style="padding: 6px 0; font-family: monospace;">{exit_reason}</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 6px 0; color: #8f9bb3;"><b>Leveraged Position Size (USD):</b></td>
                                            <td style="padding: 6px 0; font-family: monospace;">${leveraged_position_usd:.2f} USD</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 6px 0; color: #8f9bb3;"><b>Leverage:</b></td>
                                            <td style="padding: 6px 0; font-family: monospace;">{leverage}x</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 6px 0; color: #8f9bb3;"><b>Actual Investment (USD):</b></td>
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
                                bot_state.save_prediction(p)
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
            # Enable Daily Drawdown Circuit Breaker at 7%
            if daily_dd_pct >= 7.0:
                if not bot_state.get("circuit_breaker_active", False):
                    send_telegram_alert(f"⚠️ *DAILY DRAWDOWN CIRCUIT BREAKER* ⚠️\n• Start Balance: ${start_bal:.2f}\n• Current Balance: ${curr_bal:.2f}\n• Daily Drawdown: *{daily_dd_pct:.2f}%* (>= 7%)\n• *Trading Halted* until reset.")
                bot_state["circuit_breaker_active"] = True
                print(f"[Circuit Breaker] TRIGGERED — Daily drawdown is {daily_dd_pct:.2f}% (>= 7.0%). Trading halted.")
            else:
                bot_state["circuit_breaker_active"] = False
            if daily_profit >= 1000.0 and not bot_state.get("daily_goal_reached", False):
                bot_state["daily_goal_reached"] = True
                send_telegram_alert(f"🎉 *DAILY PROFIT GOAL REACHED* 🎉\n• Daily Profit: *${daily_profit:.2f}* (Goal: $1,000)\n• Current Account Value: ${curr_bal:.2f}\n• Continuing trading to maximize gains.")
                print(f"[Daily Goal] REACHED — daily profit of ${daily_profit:.2f} >= $1000. Continuing trading to maximize gains (no maximum limit).")
            elif daily_profit < 1000.0:
                bot_state["daily_goal_reached"] = False

        # --- High-Impact News Window Guard ---
        def is_high_impact_news_window():
            """Returns True if within 15 minutes of a known high-impact event (CPI, FOMC, NFP)."""
            try:
                now_utc = datetime.utcnow()
                events = fetch_economic_calendar_cached()
                for ev_time in events:
                    diff = abs((now_utc - ev_time).total_seconds())
                    if diff <= 900:  # 15 minute window
                        return True, "High-Impact Economic Event"
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

        reloaded_intervals = check_and_hot_reload_models()
        
        # --- Intelligent Boundary Window Candle Polling ---
        current_time_utc = datetime.utcnow()
        current_utc_hour = current_time_utc.hour
        current_utc_minute = current_time_utc.minute
        
        # 1. Reset hourly state variables at a new UTC hour transition
        if current_utc_hour != last_check_hour:
            last_check_hour = current_utc_hour
            hour_check_complete = False
            completed_this_hour.clear()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] New UTC hour detected ({current_utc_hour}). Resetting boundary check status.")
            
        # 2. Determine if we are within the candle check window or if we are executing the startup check
        is_in_check_window = (current_utc_minute < CANDLE_CHECK_WINDOW_MINS) or (not startup_check_done)
        
        check_queue = []
        is_startup = not startup_check_done
        
        if is_in_check_window and not hour_check_complete:
            # Throttle requests to CANDLE_CHECK_INTERVAL_SECS (e.g. 20s)
            if current_time - last_candle_check_time >= CANDLE_CHECK_INTERVAL_SECS:
                last_candle_check_time = current_time
                
                if is_startup:
                    # Startup initial check: query all supported symbols for all intervals in parallel
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Startup initial fast check: checking all {len(SUPPORTED_SYMBOLS)} symbols across all timeframes...")
                    check_queue = [(symbol, iv) for iv in ["60", "120", "240", "360"] for symbol in SUPPORTED_SYMBOLS]
                    startup_check_done = True
                else:
                    # Regular hour transition checks: check active intervals
                    active_intervals = []
                    for iv_q in ["60", "120", "240", "360"]:
                        iv_hours = int(iv_q) // 60
                        if current_utc_hour % iv_hours == 0:
                            active_intervals.append(iv_q)
                    
                    # We check how many of the currently expected ones are completed
                    expected_pairs = [(symbol, iv) for iv in active_intervals for symbol in SUPPORTED_SYMBOLS]
                    missing_pairs = [pair for pair in expected_pairs if pair not in completed_this_hour]
                    
                    if not missing_pairs:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] All active candle checks for UTC hour {current_utc_hour} completed successfully. Polling stopped.")
                        hour_check_complete = True
                    else:
                        check_queue = missing_pairs

        forced_intervals = set()
        # 3. Handle hot-reload queue inject
        if reloaded_intervals:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Hot-reloaded intervals detected: {reloaded_intervals}. Resetting processed timestamps and forcing check...")
            for iv_hr in reloaded_intervals:
                forced_intervals.add(iv_hr)
                # Clear interval specific tracker
                last_processed_timestamps[f"last_processed_{iv_hr}_ts"] = None
                # Clear per-symbol tracker
                for sym in SUPPORTED_SYMBOLS:
                    last_ts_key = f"last_processed_{sym}_{iv_hr}_ts"
                    last_processed_timestamps.pop(last_ts_key, None)
                    
            # Inject to queue
            check_queue.extend([(symbol, iv_hr) for iv_hr in reloaded_intervals for symbol in SUPPORTED_SYMBOLS])
        
        # Calculate expected timestamp boundary
        current_hour_dt = datetime(current_time_utc.year, current_time_utc.month, current_time_utc.day, current_time_utc.hour)
        current_hour_ts = int(current_hour_dt.timestamp() * 1000)
        
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
                future_to_pair = {executor.submit(fetch_single_history, sym, iv): (sym, iv) for sym, iv in check_queue}
                for fut in future_to_pair:
                    sym, iv = future_to_pair[fut]
                    try:
                        _, _, df_raw_val, df_feat_val = fut.result(timeout=60)
                        if df_raw_val is not None:
                            fetched_data[(sym, iv)] = (df_raw_val, df_feat_val)
                    except Exception as e:
                        print(f"[Parallel Fetch] Error fetching {sym} {iv}: {type(e).__name__} {e}")
            print(f"[Parallel Fetch] Completed in {time.time() - t_start:.2f} seconds.")
 
        just_opened_symbols = set()  # Symbols opened this cycle — block duplicates regardless of Bybit sync latency
        for symbol, iv in check_queue:
            # Global active execution guard check
            with active_execution_lock:
                in_execution = symbol in active_execution_symbols
            if in_execution:
                print(f"[{symbol} {iv}m] Skip signal check: A live order placement is currently executing in the background.")
                continue

            tf = tf_map[iv]
            active_trade_key = f"active_trade_{tf}"
            with active_trades_lock:
                active_trades_list = bot_state.get(active_trade_key, [])
                if not isinstance(active_trades_list, list):
                    active_trades_list = [] if active_trades_list is None else [active_trades_list]
                    bot_state[active_trade_key] = active_trades_list
                active_trades_list = list(active_trades_list)
                
            if (symbol, iv) not in fetched_data:
                continue
            df_raw, df = fetched_data[(symbol, iv)]
            if df is None or len(df) == 0:
                continue
                
            try:
                df_completed = df.iloc[:-1].copy()
                latest_completed_ts = int(df_completed.iloc[-1]["timestamp"])
 
                last_ts_key = f"last_processed_{symbol}_{iv}_ts"
                if last_processed_timestamps.get(last_ts_key) is None:
                    last_processed_timestamps[last_ts_key] = 0
                    print(f"Initialized completed candle timestamp tracking for {symbol} on {iv}m: {get_local_time_str(latest_completed_ts/1000)}")
 
                # Validate if candle is up to date based on expected window boundary
                interval_ms = int(iv) * 60 * 1000
                expected_start_ms = current_hour_ts - interval_ms
                is_forced = iv in forced_intervals
                is_up_to_date = (latest_completed_ts >= expected_start_ms) or is_startup or is_forced
                
                if not is_up_to_date:
                    # Candle is stale, wait for exchange to finalize the new candle
                    continue
                    
                completed_this_hour.add((symbol, iv))
                
                if latest_completed_ts != last_processed_timestamps[last_ts_key]:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New completed {symbol} {iv}-minute candle detected (TS: {latest_completed_ts})")
                    
                    latest_candle = df.iloc[-1]
                    
                    # Dynamic Regime Routing based on GMM Unsupervised Classifier
                    regime = classify_market_regime(df, interval=iv)
                    adx_regime = latest_candle["ADX"]
                    
                    if iv in models_by_interval:
                        models_tf = models_by_interval[iv]
                        if regime == "Trending":
                            active_model_price = models_tf["trending"]["price"]
                            active_model_trend = models_tf["trending"]["trend"]
                            regime_name = "Trending (GMM)"
                            feat_list = models_tf.get("selected_features_trending")
                        else:
                            active_model_price = models_tf["ranging"]["price"]
                            active_model_trend = models_tf["ranging"]["trend"]
                            regime_name = "Ranging (GMM)"
                            feat_list = models_tf.get("selected_features_ranging")
                            
                        if active_model_price is None or active_model_trend is None:
                            print(f"[{symbol} {iv}m Warning] Models are not loaded (None). Skipping signal evaluation.")
                            continue
                            
                        if feat_list is not None:
                            X_live = latest_candle[feat_list].values.reshape(1, -1)
                        else:
                            X_live = latest_candle[features].values.reshape(1, -1)

                        # Refined regime switching voting weights (Trending: CatBoost dominant; Ranging: LightGBM dominant)
                        ensemble_weights = [0.3, 0.2, 0.5] if regime == "Trending" else [0.3, 0.5, 0.2]
                        
                        try:
                            pred_pct = float(active_model_price.predict(X_live, weights=ensemble_weights)[0])
                            pred_change = pred_pct * float(latest_candle["close"])
                            predicted_price = float(latest_candle["close"]) + pred_change
                            
                            # 3-class probabilities
                            probs = active_model_trend.predict_proba(X_live, weights=ensemble_weights)[0]
                        except Exception as pred_err:
                            print(f"[{symbol} {iv}m Prediction Error] Failed to run model prediction: {pred_err}. Skipping signal evaluation.")
                            continue
                        
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
                        calibrator = models_tf["trending"]["calibrator"] if regime == "Trending" else models_tf["ranging"]["calibrator"]
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
                        dynamic_conf_threshold = 0.58
                        
                        # 1. Regime Adjustment (GMM-based)
                        if "Trending" in regime_name:
                            dynamic_conf_threshold = 0.55
                        elif "Ranging" in regime_name:
                            dynamic_conf_threshold = 0.63
                            
                        # 2. Volatility Adjustment (ATR)
                        if atr_norm_val > 0.008:
                            dynamic_conf_threshold = min(0.68, dynamic_conf_threshold + 0.03)
                        elif atr_norm_val < 0.003:
                            dynamic_conf_threshold = min(0.68, dynamic_conf_threshold + 0.02)
                            
                        # 3. Sentiment-Adaptive Adjustment
                        with news_sentiment_lock:
                            current_sentiment = cached_news_sentiment
                        if current_sentiment == "Bullish":
                            if ml_trend == "Bullish":
                                dynamic_conf_threshold = max(0.50, dynamic_conf_threshold - 0.03)
                            elif ml_trend == "Bearish":
                                dynamic_conf_threshold = min(0.85, dynamic_conf_threshold + 0.05)
                        elif current_sentiment == "Bearish":
                            if ml_trend == "Bearish":
                                dynamic_conf_threshold = max(0.50, dynamic_conf_threshold - 0.03)
                            elif ml_trend == "Bullish":
                                dynamic_conf_threshold = min(0.85, dynamic_conf_threshold + 0.05)
                                
                        print(f"[{iv}m] Dynamic Confidence Threshold: {dynamic_conf_threshold * 100:.2f}% (Regime: {regime_name}, Volatility: {atr_norm_val * 100:.3f}%, Sentiment: {current_sentiment})")

                        # Meta-Classifier: Use as confidence MODIFIER instead of hard gate
                        meta_adjustment = 0.0
                        if ml_trend in ["Bullish", "Bearish"]:
                            active_meta_model = models_tf["trending"]["meta"] if adx_regime >= 20.0 else models_tf["ranging"]["meta"]
                            if active_meta_model is not None:
                                meta_pred = int(active_meta_model.predict(X_live)[0])
                                if meta_pred == 1:
                                    meta_adjustment = +0.05
                                    print(f"[{iv}m] Meta-Classifier: PASS (confidence boosted +5%)")
                                else:
                                    meta_adjustment = -0.07  # Middle ground penalty
                                    print(f"[{iv}m] Meta-Classifier: FAIL (confidence reduced -7%)")
                                calibrated_confidence = max(0.0, min(1.0, calibrated_confidence + meta_adjustment))
                                print(f"[{iv}m] Adjusted Calibrated Confidence: {calibrated_confidence*100:.2f}%")

                        # Determine tracking status
                        # Softened contradiction: only block if regressor predicts > 0.05% in OPPOSITE direction
                        pred_pct = (abs(pred_change) / latest_candle["close"]) * 100
                        strong_conflict = (ml_trend == "Bullish" and pred_change < 0 and pred_pct > 0.05) or \
                                          (ml_trend == "Bearish" and pred_change > 0 and pred_pct > 0.05)
                        
                        is_cooling, remaining_mins = is_symbol_interval_cooling_off(symbol, iv)
                        news_event = ""
                        
                        # Hierarchical Confluence Check (Trend Alignment)
                        confluence_blocked = False
                        htf_trend = "Neutral"
                        macro_tf = ""
                        htf_mapping = {"60": "240", "120": "360"}
                        if iv in htf_mapping:
                            macro_iv = htf_mapping[iv]
                            macro_tf = tf_map.get(str(macro_iv))
                            if macro_tf:
                                macro_pred = bot_state.get(f"latest_prediction_{macro_tf}")
                                if macro_pred and isinstance(macro_pred, dict):
                                    htf_trend = macro_pred.get("direction", "Neutral")
                                    if htf_trend in ["Bullish", "Bearish"]:
                                        if ml_trend == "Bullish" and htf_trend == "Bearish":
                                            confluence_blocked = True
                                        elif ml_trend == "Bearish" and htf_trend == "Bullish":
                                            confluence_blocked = True

                        # Funding Rate Carry Overlay
                        funding_rate = get_funding_rate(symbol)
                        funding_blocked = False
                        if funding_rate > 0.0005 and ml_trend == "Bullish":
                            dynamic_conf_threshold = min(0.85, dynamic_conf_threshold + 0.05)
                            print(f"[{symbol} {iv}m] Funding Carry Adjustment: Positive funding rate ({funding_rate*100:.3f}%) raised Long threshold to {dynamic_conf_threshold*100:.1f}%")
                        elif funding_rate < -0.0005 and ml_trend == "Bearish":
                            dynamic_conf_threshold = min(0.85, dynamic_conf_threshold + 0.05)
                            print(f"[{symbol} {iv}m] Funding Carry Adjustment: Negative funding rate ({funding_rate*100:.3f}%) raised Short threshold to {dynamic_conf_threshold*100:.1f}%")

                        if ml_trend == "Bullish" and funding_rate > 0.001:
                            funding_blocked = True
                        elif ml_trend == "Bearish" and funding_rate < -0.001:
                            funding_blocked = True
                            
                        # Open Interest Momentum Guard
                        try:
                            oi_delta = df.iloc[-1].get("open_interest_pct_change", 0.0) * 100.0
                            if oi_delta < 0.5:
                                dynamic_conf_threshold = min(0.85, dynamic_conf_threshold + 0.05)
                                print(f"[{symbol} {iv}m] OI Momentum Guard: Low Open Interest Delta ({oi_delta:+.2f}%) raised threshold to {dynamic_conf_threshold*100:.1f}%")
                        except Exception as e:
                            print(f"[{symbol} {iv}m] Exception in OI Momentum Guard: {e}")
                        
                        status_msg = "Pending"
                        active_trade_key = f"active_trade_{tf}"
                        active_trades_list = bot_state.get(active_trade_key, [])
                        
                        # Prevent duplicate parallel trades of the same symbol on ANY interval/timeframe
                        already_active = False
                        active_on_tf = None
                        if symbol in just_opened_symbols:
                            already_active = True
                            active_on_tf = "current_cycle"
                        else:
                            for tf_key in ["1h", "2h", "4h", "6h"]:
                                if any(t.get("symbol") == symbol for t in bot_state.get(f"active_trade_{tf_key}", [])):
                                    already_active = True
                                    active_on_tf = tf_key
                                    break
                        
                        # Session Filter: temporarily allow 24-hour trading
                        utc_hour = datetime.utcnow().hour
                        in_session = True

                        if not bot_state.get("bot_running", True):
                            status_msg = "Skipped (Bot Stopped)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Bot is currently stopped by the user.")
                        elif bot_state.get("circuit_breaker_active", False):
                            status_msg = "Skipped (Circuit Breaker)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Daily Drawdown Circuit Breaker is active.")
                        elif already_active:
                            status_msg = "Skipped (Already Active)"
                            print(f"[{symbol} {iv}m] Prediction skipped: A trade is already active for this symbol on the {active_on_tf} timeframe.")
                        elif not in_session:
                            status_msg = "Skipped (Off-Session)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Outside London/NY session (UTC hour: {utc_hour}).")
                        elif is_cooling:
                            status_msg = "Skipped (Cool-Off)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Interval is in a 6-hour cool-off period after consecutive losses ({remaining_mins} mins remaining).")
                        elif confluence_blocked:
                            status_msg = "Skipped (HTF Trend Block)"
                            print(f"[{symbol} {iv}m] Prediction skipped: Counter-trend relative to macro timeframe ({macro_tf} trend: {htf_trend}).")
                        elif funding_blocked:
                            status_msg = "Skipped (Funding Block)"
                            print(f"[{symbol} {iv}m] Prediction skipped: High funding fee payment risk (Funding: {funding_rate*100:.3f}%).")
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
                            # Check news window proximity status for logging/blocking purposes
                            in_news_window, news_event = is_high_impact_news_window()
                            if in_news_window:
                                print(f"[{iv}m News Block] Trade skipped: high-impact news event window active ({news_event}).")
                                status_msg = "Skipped (News Block)"
                                
                            with news_sentiment_lock:
                                news_sentiment = cached_news_sentiment
                                latest_titles = cached_news_titles
                                all_pass, confluence_results, confluence_score_pct = check_pre_trade_confluence(
                                    latest_candle["close"], df, ml_trend, news_sentiment, expected_pct_change, iv, symbol=symbol, htf_cache=htf_cache,
                                    calibrated_confidence=calibrated_confidence, dynamic_conf_threshold=dynamic_conf_threshold
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
                                    
                                    # Align stop loss and take profit multipliers dynamically from TIMEFRAME_CONFIG
                                    cfg = TIMEFRAME_CONFIG.get(str(iv), {
                                        "lookahead": 10,
                                        "sl_mult": 0.8,
                                        "tp_mult_ranging": 1.5,
                                        "tp_mult_trending": 2.5
                                    })
                                    sl_multiplier = cfg["sl_mult"]
                                    adx_val = latest_candle.get("ADX", 0.0)
                                    if adx_val >= 20.0:
                                        tp_multiplier_adjusted = cfg["tp_mult_trending"]
                                    else:
                                        tp_multiplier_adjusted = cfg["tp_mult_ranging"]
                                    print(f"[{iv}m Target Alignment] ADX: {adx_val:.1f} | Aligned multipliers: SL = {sl_multiplier}x, TP = {tp_multiplier_adjusted}x (from TIMEFRAME_CONFIG)")
                                    
                                    # Maker execution: zero entry slippage for limit orders
                                    slippage_pct = 0.0
                                    raw_entry_price = float(latest_candle["close"])
                                    entry_price = raw_entry_price

                                    # Enforce a minimum TP of 0.5%
                                    min_tp_change = entry_price * 0.005
                                    tp_change = max(min_tp_change, abs(pred_change))
                                    
                                    # Dynamically adjust Stop Loss multiplier based on prediction confidence
                                    sl_multiplier_adjusted = sl_multiplier
                                    if calibrated_confidence > dynamic_conf_threshold:
                                        confidence_ratio = (calibrated_confidence - dynamic_conf_threshold) / (1.0 - dynamic_conf_threshold)
                                        # Scale SL down by up to 30% for maximum confidence trades
                                        sl_multiplier_adjusted = sl_multiplier * (1.0 - 0.3 * confidence_ratio)
                                        
                                    min_sl_pct = 1.0 if symbol == "BTCUSDT" else (1.5 if symbol in ["ETHUSDT", "SOLUSDT", "BNBUSDT"] else 2.0)
                                    min_sl_dist = entry_price * (min_sl_pct / 100.0)
                                    raw_sl_dist = max(sl_multiplier_adjusted * atr_dollars, min_sl_dist)
                                    
                                    min_tp_dist = entry_price * 0.015
                                    if ml_trend == "Bullish":
                                        stop_loss_price = entry_price - raw_sl_dist
                                        atr_tp_price = entry_price + tp_change
                                        est_tp_price = estimate_liquidation_pool(df, "Bullish", entry_price)
                                        take_profit_price = max(entry_price + min_tp_dist, min(est_tp_price, entry_price + 1.5 * tp_change))
                                    else:
                                        stop_loss_price = entry_price + raw_sl_dist
                                        atr_tp_price = entry_price - tp_change
                                        est_tp_price = estimate_liquidation_pool(df, "Bearish", entry_price)
                                        take_profit_price = min(entry_price - min_tp_dist, max(est_tp_price, entry_price - 1.5 * tp_change))
                                    print(f"[{iv}m ML Targets] Entry: {entry_price:.2f} | Dynamic SL: {stop_loss_price:.2f} (Mult: {sl_multiplier_adjusted:.2f}x) | Regressor TP: {take_profit_price:.2f} (Expected: {pred_change:+.3f})")

                                    # Calibrated Position Sizing based on Isotonic Probability (Kelly scaling)
                                    c_prob = float(calibrated_confidence)
                                    current_hour_pkt = get_pkt_time().hour
                                    is_golden_hour = 18 <= current_hour_pkt < 21
                                    
                                    # Pre-calculate active trade stats needed for dynamic sizing
                                    total_active_size = sum(t.get("position_size_usd", 0.0) for tf_key in ["1h", "2h", "4h", "6h"] for t in bot_state.get(f"active_trade_{tf_key}", []))
                                    current_bal = bot_state.get("simulated_balance", 80.0)
                                    if TRADE_MODE != "simulation":
                                        real_bal = get_real_bybit_balance_cached(force=True)
                                        if isinstance(real_bal, (int, float)) and real_bal > 0:
                                            current_bal = real_bal
                                    cov_multiplier, net_risk = calculate_covariance_multiplier(symbol, ml_trend)
                                    
                                    # Base size dynamic calculation using Quarter-Kelly Criterion
                                    kelly_p = float(calibrated_confidence)
                                    kelly_b = float(tp_multiplier_adjusted / sl_multiplier) if sl_multiplier > 0 else 1.5
                                    raw_kelly = max(0.0, (kelly_p * (kelly_b + 1) - 1) / kelly_b) if kelly_b > 0 else 0.0
                                    
                                    # Quarter-Kelly scaling for safety and drawdown control
                                    scaled_kelly = 0.25 * raw_kelly
                                    
                                    # Enforce bounds on balance fraction (Min 2%, Max 15% per trade)
                                    f_clamped = max(0.02, min(0.15, scaled_kelly))
                                    
                                    # Sizing before leverage
                                    position_size_usd = current_bal * f_clamped
                                    
                                    # Covariance multiplier to account for existing correlations
                                    position_size_usd = position_size_usd * cov_multiplier
                                    
                                    # CVaR (Expected Shortfall) Risk Constraint
                                    try:
                                        hist_close = df["close"].values
                                        if len(hist_close) > 30:
                                            returns_pct = (hist_close[1:] - hist_close[:-1]) / hist_close[:-1]
                                            returns_sorted = np.sort(returns_pct)
                                            alpha_idx = max(1, int(len(returns_sorted) * 0.05))
                                            tail_losses = returns_sorted[:alpha_idx]
                                            cvar_95 = abs(float(np.mean(tail_losses))) if len(tail_losses) > 0 else 0.03
                                        else:
                                            cvar_95 = 0.03
                                        daily_loss_budget = current_bal * 0.05
                                        max_cvar_size = daily_loss_budget / (cvar_95 + 1e-8)
                                        print(f"[{iv}m CVaR Guard] 95% CVaR: {cvar_95*100:.2f}% | Max Risk Size Allowed: ${max_cvar_size:.2f}")
                                        position_size_usd = min(position_size_usd, max_cvar_size)
                                    except Exception as cvar_err:
                                        print(f"[CVaR Error] {cvar_err}")
                                    
                                    if is_golden_hour:
                                        # Golden Hour: Double the target slot allocation size
                                        position_size_usd = position_size_usd * 2.0
                                        print(f"[{iv}m Golden Hour Kelly Sizing] Kelly Fraction: {raw_kelly:.4f} -> Scaled: {scaled_kelly:.4f} -> Clamped: {f_clamped*100:.1f}% -> Golden Target: ${position_size_usd:.2f} (Covariance: {cov_multiplier:.2f}x)")
                                    else:
                                        print(f"[{iv}m Kelly Sizing] Kelly Fraction: {raw_kelly:.4f} -> Scaled: {scaled_kelly:.4f} -> Clamped: {f_clamped*100:.1f}% -> Final Size: ${position_size_usd:.2f} (Covariance: {cov_multiplier:.2f}x)")
                                        
                                    # Clip to minimum Bybit order requirement (e.g. $2.0)
                                    position_size_usd = max(2.0, position_size_usd)
                                    print(f"[{iv}m Trade Size Boundary Check] Final size before leverage (CVaR constrained): ${position_size_usd:.2f}")

                                    # Calculate Kelly parameters for logs and metadata (preserving variables for downstream use)
                                    kelly_fraction = raw_kelly

                                    # Ensure total size of active trades does not exceed the wallet balance
                                    min_bal_limit = 2.0
                                    min_size_limit = 2.0
                                    
                                    wallet_exceeded = False
                                    if current_bal <= min_bal_limit:
                                        print(f"[{symbol} {iv}m] Trade skipped: Wallet balance (${current_bal:.2f}) must be greater than ${min_bal_limit:.2f} to open new trades.")
                                        status_msg = "Skipped (Insufficient Balance)"
                                        wallet_exceeded = True
                                        send_telegram_alert(
                                            f"⚠️ *SIGNAL PASSED (SKIPPED - LOW BALANCE)* ⚠️\n"
                                            f"• *Asset*: {symbol}\n"
                                            f"• *Interval*: {iv}m\n"
                                            f"• *Direction*: {ml_trend}\n"
                                            f"• *Calibrated Confidence*: {calibrated_confidence * 100:.2f}%\n"
                                            f"• *Detail*: Wallet balance (${current_bal:.2f}) must be greater than ${min_bal_limit:.2f}."
                                        )
                                    elif total_active_size + position_size_usd > current_bal:
                                        remaining_bal = current_bal - total_active_size
                                        if remaining_bal >= min_size_limit:
                                            print(f"[{symbol} {iv}m] Sizing scaled down from ${position_size_usd:.2f} to ${remaining_bal:.2f} to fit remaining wallet balance (Total Active: ${total_active_size:.2f}, Wallet: ${current_bal:.2f}).")
                                            position_size_usd = remaining_bal
                                        else:
                                            print(f"[{symbol} {iv}m] Trade skipped: Insufficient wallet balance to maintain minimum ${min_size_limit:.2f} trade size (Total Active: ${total_active_size:.2f}, Wallet: ${current_bal:.2f}, Proposed: ${position_size_usd:.2f}).")
                                            status_msg = "Skipped (Exceeds Wallet)"
                                            wallet_exceeded = True
                                            send_telegram_alert(
                                                f"⚠️ *SIGNAL PASSED (SKIPPED - LOW BALANCE)* ⚠️\n"
                                                f"• *Asset*: {symbol}\n"
                                                f"• *Interval*: {iv}m\n"
                                                f"• *Direction*: {ml_trend}\n"
                                                f"• *Calibrated Confidence*: {calibrated_confidence * 100:.2f}%\n"
                                                f"• *Detail*: Insufficient wallet balance to maintain minimum ${min_size_limit:.2f} size (Total Active: ${total_active_size:.2f}, Wallet: ${current_bal:.2f}, Proposed: ${position_size_usd:.2f})."
                                            )

                                    if not wallet_exceeded:
                                        # Continuous Leverage Scaling: scale smoothly from 1x (at dynamic threshold) to 50x (at 100% confidence)
                                        c = float(calibrated_confidence)
                                        min_conf = dynamic_conf_threshold
                                        if c >= min_conf:
                                            leverage_val = 1.0 + (c - min_conf) / (1.0 - min_conf) * 49.0
                                        else:
                                            leverage_val = 1.0
                                        
                                        # Risk check: cap leverage so stop loss doesn't exceed 90% of capital, with absolute limit based on symbol volatility profile
                                        stop_loss_pct = (sl_multiplier * atr_dollars / entry_price) * 100
                                        max_safe_lev = 90.0 / stop_loss_pct if stop_loss_pct > 0 else 100.0
                                        
                                        if symbol == "BTCUSDT":
                                            lev_cap = 30.0
                                        elif symbol in ["ETHUSDT", "BNBUSDT", "SOLUSDT"]:
                                            lev_cap = 20.0
                                        else:
                                            lev_cap = 5.0
                                        # Volatility-based leverage scaling cap
                                        atr_pct_of_price = (atr_dollars / entry_price) * 100.0
                                        if atr_pct_of_price > 3.0:
                                            vol_lev_cap = 2.0 if symbol in ["BTCUSDT", "ETHUSDT"] else 1.0
                                            lev_cap = min(lev_cap, vol_lev_cap)
                                            print(f"[{symbol} {iv}m Volatility-Scaled Leverage] Extreme Volatility Detected (ATR = {atr_pct_of_price:.2f}% of price). Capped leverage to {lev_cap}x.")
                                        elif atr_pct_of_price > 1.5:
                                            lev_cap = min(lev_cap, lev_cap * 0.5)
                                            print(f"[{symbol} {iv}m Volatility-Scaled Leverage] High Volatility Detected (ATR = {atr_pct_of_price:.2f}% of price). Halved leverage cap to {lev_cap}x.")
                                        
                                            
                                        # Double leverage target and cap during Golden Hour (18:00 - 21:00 PKT)
                                        current_hour_pkt = get_pkt_time().hour
                                        if 18 <= current_hour_pkt < 21:
                                            leverage_val *= 2.0
                                            lev_cap *= 2.0
                                            print(f"[{symbol} {iv}m Golden Hour Boost] 18:00 - 21:00 PKT: Doubled leverage target to {leverage_val:.1f}x and cap to {lev_cap:.1f}x")
                                            
                                        # Sharpe-Adaptive Leverage Multiplier (Dynamic drawdown safety)
                                        sharpe_mult = calculate_recent_performance_leverage_multiplier(days=7)
                                        leverage_val = leverage_val * sharpe_mult
                                        lev_cap = lev_cap * sharpe_mult
                                            
                                        leverage_val = round(max(1.0, min(lev_cap, min(leverage_val, max_safe_lev))), 1)

                                        cfg = TIMEFRAME_CONFIG.get(str(iv), {"lookahead": 10})
                                        lookahead = cfg.get("lookahead", 10)
                                        duration_seconds = int(iv) * 60.0 * lookahead
                                        import uuid
                                        trade_uuid = str(uuid.uuid4())[:8]
                                        # Calculate quantity (qty) in coins rounded according to symbol requirements
                                        leveraged_size = position_size_usd * leverage_val
                                        raw_qty = leveraged_size / entry_price
                                        qty_str = format_bybit_qty(symbol, raw_qty)
                                        qty_val = float(qty_str)
                                        
                                        # Enforce minimum order value of 5.0 USDT (using 5.1 USDT as buffer)
                                        min_order_value = 5.1
                                        if qty_val * entry_price < min_order_value:
                                            step = get_bybit_min_qty_step(symbol)
                                            required_qty = min_order_value / entry_price
                                            import math
                                            if step > 0:
                                                qty_val = math.ceil(required_qty / step) * step
                                                qty_str = format_bybit_qty(symbol, qty_val)
                                                qty_val = float(qty_str)
                                                raw_qty = qty_val
                                                print(f"[{symbol} {iv}m API] Enforced minimum order value. Adjusted quantity to {qty_str} (Value: ${qty_val * entry_price:.2f})")

                                        # Balance Guard: if required margin exceeds 90% of available balance, dynamically scale up leverage
                                        required_margin = (qty_val * entry_price) / leverage_val
                                        if TRADE_MODE != "simulation" and required_margin > current_bal * 0.9:
                                            needed_leverage = (qty_val * entry_price) / (current_bal * 0.9)
                                            scaled_leverage = round(min(lev_cap, max(leverage_val, needed_leverage)), 1)
                                            if scaled_leverage > leverage_val:
                                                leverage_val = scaled_leverage
                                                print(f"[{symbol} {iv}m Leverage Scaling] Increased leverage to {leverage_val}x to meet minimum order size within balance limits.")
                                                
                                            # Recalculate margin after leverage scale-up
                                            required_margin = (qty_val * entry_price) / leverage_val
                                            if required_margin > current_bal * 0.95:
                                                # Downscale quantity to fit 90% of available balance
                                                max_margin = current_bal * 0.90
                                                new_raw_qty = (max_margin * leverage_val) / entry_price
                                                qty_str = format_bybit_qty(symbol, new_raw_qty)
                                                qty_val = float(qty_str)
                                                raw_qty = qty_val
                                                print(f"[{symbol} {iv}m Balance Guard] Downscaled order quantity to {qty_str} to fit within 90% of available balance (Margin: ${max_margin:.2f})")

                                        # Set Bybit Leverage and Place Order if in live/testnet mode
                                        bybit_success = True
                                        bybit_order_id = None
                                        bybit_scale_out_order_id = None
                                        
                                        if TRADE_MODE != "simulation":
                                            # Live trading execution offloaded to background thread to minimize latency
                                            just_opened_symbols.add(symbol)
                                            with active_execution_lock:
                                                active_execution_symbols.add(symbol)
                                            threading.Thread(
                                                target=execute_bybit_trade_async,
                                                args=(symbol, iv, tf, ml_trend, leverage_val, qty_str, raw_qty, entry_price, stop_loss_price, take_profit_price, position_size_usd, kelly_fraction, calibrated_confidence, ml_confidence, dynamic_conf_threshold, latest_completed_ts, latest_candle, pred_change, predicted_price, atr_dollars, tp_multiplier_adjusted, sl_multiplier_adjusted, df_completed, trade_uuid, duration_seconds, active_trade_key),
                                                daemon=True
                                            ).start()
                                            bybit_success = False # Skip the simulation path for this trade

                                        if bybit_success:
                                            actual_size_usd = float(position_size_usd) * (actual_qty / raw_qty) if raw_qty > 0 else float(position_size_usd)
                                            active_trade = {
                                                "trade_id": f"{symbol}_{trade_uuid}",
                                                "bybit_order_id": bybit_order_id,
                                                "bybit_scale_out_order_id": bybit_scale_out_order_id,
                                                "symbol": symbol,
                                                "entry_price": float(entry_price),
                                                "predicted_price": float(predicted_price),
                                                "stop_loss": float(stop_loss_price),
                                                "take_profit": float(take_profit_price),
                                                "direction": str(ml_trend),
                                                "end_time": float(time.time() + duration_seconds),
                                                "entry_time": int(time.time() * 1000),
                                                "atr_dollars": float(atr_dollars),
                                                "highest_price": float(entry_price),
                                                "lowest_price": float(entry_price),
                                                "break_even_triggered": False,
                                                "half_closed": False,
                                                "original_size": float(position_size_usd),
                                                "position_size_usd": actual_size_usd,
                                                "scaled_out_pnl": 0.0,
                                                "kelly_fraction": float(kelly_fraction),
                                                "leverage": float(leverage_val),
                                                "confidence": float(calibrated_confidence),
                                                "qty": float(actual_qty),
                                                "original_qty": float(actual_qty),
                                                "fill_pct": round((actual_qty / raw_qty) * 100.0, 2) if raw_qty > 0 else 100.0
                                            }
                                            
                                            # Send Telegram alert for successful prediction/trade entry
                                            entry_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                            send_telegram_alert(
                                                f"🟢 *POSITION OPENED (SUCCESSFUL SIGNAL)* 🟢\n"
                                                f"• *Asset*: {symbol}\n"
                                                f"• *Interval*: {iv}m\n"
                                                f"• *Direction*: {ml_trend}\n"
                                                f"• *Entry Price*: ${float(entry_price):.4f}\n"
                                                f"• *Entry Time*: {entry_time_str}\n"
                                                f"• *Take Profit*: ${float(take_profit_price):.4f}\n"
                                                f"• *Stop Loss*: ${float(stop_loss_price):.4f}\n"
                                                f"• *Calibrated Confidence*: {calibrated_confidence * 100:.2f}%\n"
                                                f"• *Leverage*: {leverage_val:.1f}x\n"
                                                f"• *Position Size*: ${actual_size_usd:.2f} (Value: ${actual_qty * entry_price:.2f})\n"
                                                f"• *Execution Mode*: {TRADE_MODE.upper()}"
                                            )
                                            
                                            with active_trades_lock:
                                                current_trades = bot_state.get(active_trade_key, [])
                                                if not isinstance(current_trades, list):
                                                    current_trades = []
                                                current_trades = list(current_trades)
                                                current_trades.append(active_trade)
                                                bot_state[active_trade_key] = current_trades
                                            
                                            # Mark symbol as opened this cycle — prevents duplicate opens due to Bybit sync latency
                                            just_opened_symbols.add(symbol)
                                            # Sync positions immediately to load live Bybit state parameters
                                            if TRADE_MODE != "simulation":
                                                sync_active_positions_from_bybit()
                                            
                                            # Deduct size from wallet balance immediately (only in simulation)
                                            if TRADE_MODE == "simulation":
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
                                "calibrated_confidence": float(calibrated_confidence),
                                "raw_confidence": float(ml_confidence),
                                "dynamic_threshold": float(dynamic_conf_threshold),
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
                traceback_str = traceback.format_exc()
                traceback.print_exc()
                print(f"Error checking {iv}m candle close signals: {e}")
                send_telegram_alert(
                    f"⚠️ *ERROR IN CANDLE CHECK SIGNAL CHECK* ⚠️\n"
                    f"• *Asset*: {symbol}\n"
                    f"• *Interval*: {iv}m\n"
                    f"• *Error*: {str(e)}\n"
                    f"• *Detail*: Failed during signal processing cycle."
                )

        time.sleep(10)

def run_order_flow_persister():
    print("[Order Flow] Persister background thread started.")
    import sqlite3
    from data import DB_PATH, db_write_lock
    while True:
        time.sleep(60) # Dump statistics every 60 seconds
        try:
            now_ts = time.time()
            # Align to minute mark
            minute_ts = float(int(now_ts / 60) * 60)
            
            with order_flow_lock:
                # Extract and clear/reset current CVD & OFI buffers
                to_write = []
                for sym, state in list(order_flow_data.items()):
                    cvd_val = state.get("cvd", 0.0)
                    ofi_val = state.get("ofi", 0.0)
                    ob_imb = state.get("ob_imbalance_L2", 0.0)
                    ob_spr = state.get("ob_spread_L2", 0.0)
                    liq_l = state.get("liq_long_1h", 0.0)
                    liq_s = state.get("liq_short_1h", 0.0)
                    
                    to_write.append((sym, minute_ts, cvd_val, ofi_val, ob_imb, ob_spr, liq_l, liq_s))
                    # Reset accumulators for the next minute
                    state["cvd"] = 0.0
                    state["ofi"] = 0.0
                    state["liq_long_1h"] = 0.0
                    state["liq_short_1h"] = 0.0
            
            if to_write:
                with db_write_lock:
                    conn = sqlite3.connect(DB_PATH, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.executemany(
                        "INSERT OR REPLACE INTO historical_order_flow (symbol, timestamp, cvd, ofi, ob_imbalance_L2, ob_spread_L2, liq_long_1h, liq_short_1h) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        to_write
                    )
                    conn.commit()
                    conn.close()
        except Exception as e:
            print(f"[Order Flow Persister Error] {e}")

def safe_main():
    while True:
        try:
            main()
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            traceback.print_exc()
            print(f"CRITICAL ERROR in main loop: {e}")
            send_telegram_alert(
                f"🔴 *CRITICAL RUNTIME ERROR* 🔴\n"
                f"• *Error*: {str(e)}\n"
                f"• *Action*: Restarting main bot loop...\n\n"
                f"```\n{traceback_str[:300]}...\n```"
            )
            time.sleep(15)  # wait before restarting main

if __name__ == "__main__":
    import threading
    # Start main bot loop in background thread
    threading.Thread(target=safe_main, daemon=True).start()
    # Start background Telegram command listener thread
    start_telegram_command_listener()
    send_telegram_alert(f"🤖 *BTC Trading Bot Started successfully on {TRADE_MODE.upper()} mode.*")
    # Start background news sentiment updater thread
    threading.Thread(target=run_news_sentiment_updater, daemon=True).start()
    # Start background Bybit balance updater thread
    threading.Thread(target=run_bybit_balance_updater, daemon=True).start()
    # Start Bybit WebSocket feed in a background thread
    threading.Thread(target=start_ws, daemon=True).start()
    # Start Bybit Private WebSocket feed in a background thread
    threading.Thread(target=start_private_ws, daemon=True).start()
    # Start WebSocket keep-alive watchdog thread in a background thread
    threading.Thread(target=run_websocket_watchdog, daemon=True).start()
    # Start Bybit REST API fallback price updater thread
    threading.Thread(target=run_fallback_price_updater, daemon=True).start()
    # Start automated rolling retraining scheduler in a background thread (Moved to retrain_worker.py)
    # threading.Thread(target=run_rolling_retrain_scheduler, daemon=True).start()
    # Start background order flow persister thread
    threading.Thread(target=run_order_flow_persister, daemon=True).start()
    # Start daily Telegram trade journal digest scheduler (Moved to retrain_worker.py / cron)
    # threading.Thread(target=run_daily_journal_scheduler, daemon=True).start()
    # Start funding rate arbitrage monitor thread
    threading.Thread(target=run_funding_rate_arbitrage_monitor, daemon=True).start()
    # Start daily database and trade journal backup thread
    threading.Thread(target=run_daily_backup_scheduler, daemon=True).start()
    # Run Flask on main thread so HF health check passes immediately
    run_flask()