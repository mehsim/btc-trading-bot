"""
dashboard_routes.py
-------------------
Flask API endpoints, security middleware, killswitch handlers, trade history migration & healing.
"""

import os
import time
import json
import threading
from functools import wraps
from flask import Blueprint, jsonify, request, render_template, make_response
from secret_manager import get_secure_env
import database

dashboard_bp = Blueprint("dashboard", __name__)
startup_time = time.time()

bot_logs = []
logs_lock = threading.Lock()
retraining_lock = threading.Lock()
active_trades_lock = threading.Lock()
bot_state_lock = threading.RLock()
ACTIVE_TRADE_TF_KEYS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h"]
HISTORY_FILE = "/data/dashboard_history.json" if os.path.exists("/data") and os.access("/data", os.W_OK) else "dashboard_history.json"


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        expected_key = get_secure_env("DASHBOARD_API_KEY", "").strip()
        if expected_key:
            client_key = request.headers.get("X-API-KEY") or request.args.get("api_key")
            if not client_key or client_key.strip() != expected_key:
                return jsonify({"error": "Unauthorized", "message": "Missing or invalid X-API-KEY header."}), 401
        return f(*args, **kwargs)
    return decorated_function


def require_ip_whitelist(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        allowed_ips = get_secure_env("ALLOWED_DASHBOARD_IPS", "").strip()
        if allowed_ips:
            ip_list = [ip.strip() for ip in allowed_ips.split(",") if ip.strip()]
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            else:
                client_ip = request.headers.get("X-Real-IP", request.remote_addr)
            if client_ip not in ip_list and client_ip != "127.0.0.1":
                return jsonify({"error": "Forbidden", "message": f"IP {client_ip} not allowed."}), 403
        return f(*args, **kwargs)
    return decorated_function


def trigger_emergency_kill_switch(bot_state, send_telegram_alert_func, reason: str = "Manual Trigger"):
    print(f"[EMERGENCY KILL SWITCH] Triggered! Reason: {reason}")
    bot_state["bot_running"] = False
    if send_telegram_alert_func:
        send_telegram_alert_func(f"🚨 *EMERGENCY KILL SWITCH ACTIVATED* 🚨\n• Reason: `{reason}`\n• Action: Halting bot & closing open positions at market.")
    try:
        from bybit_client import bybit_post_request, get_all_bybit_positions, TRADE_MODE
        if TRADE_MODE != "simulation":
            bybit_post_request("/v5/order/cancel-all", {"category": "linear", "settleCoin": "USDT"})
            positions = get_all_bybit_positions()
            for p in positions:
                sym = p.get("symbol")
                sz = float(p.get("size", "0"))
                side = p.get("side")
                if sz > 0 and sym:
                    close_side = "Sell" if side == "Buy" else "Buy"
                    bybit_post_request("/v5/order/create", {
                        "category": "linear",
                        "symbol": sym,
                        "side": close_side,
                        "orderType": "Market",
                        "qty": str(sz),
                        "timeInForce": "IOC",
                        "reduceOnly": True
                    })
    except Exception as err:
        print(f"[Kill Switch Error] Failed executing emergency close: {err}")


@dashboard_bp.route("/killswitch", methods=["GET", "POST"])
@require_api_key
def killswitch_endpoint():
    from state_manager import state_manager
    from telegram_bot import send_telegram_alert
    trigger_emergency_kill_switch(state_manager, send_telegram_alert, "HTTP /killswitch Request")
    return jsonify({"status": "KILL_SWITCH_ACTIVATED", "message": "All orders cancelled and bot halted."})


def save_history(bot_state):
    with bot_state_lock:
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
            bot_state["trade_history"] = deduped[-1000:]

        if len(bot_state.get("prediction_history", [])) > 500:
            bot_state["prediction_history"] = bot_state["prediction_history"][-500:]

        data = {
            "simulated_balance": bot_state.get("simulated_balance", 80.0),
            "trade_history": bot_state.get("trade_history", []),
            "prediction_history": bot_state.get("prediction_history", []),
            "active_trade_15m": bot_state.get("active_trade_15m", []),
            "active_trade_30m": bot_state.get("active_trade_30m", []),
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
                json.dump(data, f, indent=2)
            os.replace(temp_file, HISTORY_FILE)
            
            for tf_key in ACTIVE_TRADE_TF_KEYS:
                database.save_active_trades(tf_key, bot_state.get(f"active_trade_{tf_key}", []))
        except Exception as e:
            print(f"Error saving history to disk: {e}")


def migrate_active_trades(active_trades_list):
    if not isinstance(active_trades_list, list):
        return
    for t in active_trades_list:
        if "confidence" not in t:
            orig_size = t.get("original_size", t.get("position_size_usd", 9.5))
            if orig_size >= 11.0:
                t["confidence"] = 0.785
            elif orig_size >= 9.5:
                t["confidence"] = 0.685
            else:
                t["confidence"] = 0.585


def heal_completed_trades_bybit_order_ids(bot_state):
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
        save_history(bot_state)


def load_history(bot_state):
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                bot_state["simulated_balance"] = data.get("simulated_balance", 80.0)
                bot_state["trade_history"] = [t for t in data.get("trade_history", []) if str(t.get("interval", "60")) != "5"]
                bot_state["prediction_history"] = [p for p in data.get("prediction_history", []) if str(p.get("interval", "60")) != "5"]
                
                bot_state["active_trade_15m"] = data.get("active_trade_15m", [])
                bot_state["active_trade_30m"] = data.get("active_trade_30m", [])
                bot_state["active_trade_1h"] = data.get("active_trade_1h", [])
                bot_state["active_trade_2h"] = data.get("active_trade_2h", [])
                bot_state["active_trade_4h"] = data.get("active_trade_4h", [])
                bot_state["active_trade_6h"] = data.get("active_trade_6h", [])
                
                for tf_key in ACTIVE_TRADE_TF_KEYS:
                    migrate_active_trades(bot_state.get(f"active_trade_{tf_key}", []))
                    
                bot_state["bot_running"] = data.get("bot_running", True)
                bot_state["fresh_reset_v3"] = data.get("fresh_reset_v3", False)
                return
        except Exception as e:
            print(f"Error loading local history: {e}")


@dashboard_bp.route("/api/health")
def api_health():
    from bybit_client import get_real_bybit_balance_cached
    from websocket_client import get_ws_status
    ws_st = get_ws_status()
    return jsonify({
        "status": "ok",
        "uptime_seconds": int(time.time() - startup_time),
        "bybit_api": "connected" if get_real_bybit_balance_cached() != "API_KEYS_MISSING" else "API_KEYS_MISSING",
        "websocket": "live" if ws_st.get("public_connected") else "disconnected",
        "private_websocket": "live" if ws_st.get("private_connected") else "disconnected",
        "active_trades": 0
    })


@dashboard_bp.route("/metrics")
def prometheus_metrics():
    from state_manager import state_manager
    try:
        val = state_manager["simulated_balance"]
        sim_bal = float(val) if val is not None else 80.0
    except Exception:
        sim_bal = 80.0

    try:
        active_trades = database.get_active_trades()
        active_count = len(active_trades) if isinstance(active_trades, list) else 0
    except Exception:
        active_count = 0
        
    uptime = int(time.time() - startup_time)
    metrics_str = f"# HELP btc_bot_simulated_balance Simulated account cash balance in USD\n# TYPE btc_bot_simulated_balance gauge\nbtc_bot_simulated_balance {sim_bal:.2f}\n# HELP btc_bot_active_trades Count of currently active open trades\n# TYPE btc_bot_active_trades gauge\nbtc_bot_active_trades {active_count}\n# HELP btc_bot_uptime_seconds Total runtime of bot service in seconds\n# TYPE btc_bot_uptime_seconds counter\nbtc_bot_uptime_seconds {uptime}\n"
    return metrics_str, 200, {'Content-Type': 'text/plain; version=0.0.4'}
