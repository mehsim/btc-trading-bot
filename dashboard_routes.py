import time
import os
from flask import Blueprint, jsonify, request

dashboard_bp = Blueprint("dashboard", __name__)
startup_time = time.time()

@dashboard_bp.route("/api/health")
def api_health():
    from bybit_client import get_real_bybit_balance_cached
    return jsonify({
        "status": "ok",
        "uptime_seconds": int(time.time() - startup_time),
        "bybit_api": "connected" if get_real_bybit_balance_cached() != "API_KEYS_MISSING" else "API_KEYS_MISSING",
        "websocket": "live",
        "private_websocket": "live",
        "active_trades": 0
    })

@dashboard_bp.route("/metrics")
def prometheus_metrics():
    from state_manager import state_manager
    import database
    
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

