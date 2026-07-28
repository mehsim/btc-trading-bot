"""
main.py - BTC & Multi-Asset Trading Bot Entry Point
---------------------------------------------------
Lightweight orchestrator responsible for initializing global state, starting background
services (Flask dashboard, WebSocket streams, Telegram listener, background schedulers),
and running the main candle evaluation loop.
"""

import os
import sys
import time
import json
import signal
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()
from flask import Flask

# === MODULAR COMPONENT IMPORTS ===
import database
import state_manager
from secret_manager import get_secure_env
from trade_calculators import (
    compute_be_trigger_distance,
    validate_trade_structure,
    AdaptiveVolumeGate,
    MFEBreakEvenTrigger,
    choppiness_index,
    check_flash_crash,
    get_funding_adjustment,
    get_liquidity_score,
    estimate_liquidation_pool,
    calculate_covariance_multiplier,
    calculate_recent_performance_leverage_multiplier,
    adaptive_volume_gate,
    mfe_be_trigger
)
from bybit_client import (
    TRADE_MODE,
    BYBIT_BASE_URL,
    get_bybit_proxies,
    parse_proxy_url,
    get_bybit_time_offset,
    bybit_post_request,
    bybit_get_request,
    set_bybit_leverage,
    format_bybit_price,
    format_bybit_qty,
    get_bybit_min_qty_step,
    place_bybit_order,
    place_bybit_limit_order,
    place_bybit_taker_ioc_order,
    get_bybit_order_details,
    cancel_bybit_order,
    get_bybit_position,
    get_all_bybit_positions,
    get_bybit_closed_pnl,
    get_bybit_accumulated_closed_pnl,
    update_bybit_stop_loss,
    update_bybit_take_profit,
    get_bybit_bid_ask,
    get_bybit_last_execution,
    get_real_bybit_balance,
    get_real_bybit_balance_cached,
    run_bybit_balance_updater
)
from telegram_bot import (
    send_telegram_alert,
    execute_telegram_api_call,
    send_daily_summary,
    start_telegram_command_listener,
    run_manual_confluence_report
)
from dashboard_routes import (
    dashboard_bp,
    require_api_key,
    require_ip_whitelist,
    trigger_emergency_kill_switch,
    save_history,
    load_history,
    migrate_active_trades,
    heal_completed_trades_bybit_order_ids,
    bot_state_lock,
    logs_lock,
    active_trades_lock
)
from websocket_client import (
    init_bybit_websocket_listeners,
    get_ws_status,
    start_ws,
    start_private_ws,
    run_websocket_watchdog,
    order_flow_lock,
    order_flow_data
)
from news_monitor import (
    news_monitor,
    is_news_blackout,
    get_reddit_posts,
    get_cryptopanic_posts,
    get_x_tweets
)
from background_schedulers import (
    run_daily_journal_scheduler,
    run_funding_rate_arbitrage_monitor,
    run_daily_backup_scheduler,
    run_pain_feedback_verifier,
    run_daily_summary_scheduler,
    run_rolling_retrain_scheduler
)
from trading_engine import (
    execute_bybit_trade_async,
    active_execution_lock,
    active_execution_symbols
)

# === INITIALIZE GLOBAL APPLICATION & STATE ===
app = Flask(__name__, template_folder="templates")
app.register_blueprint(dashboard_bp)

startup_time = time.time()
bot_state = state_manager.StateManager()
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
ACTIVE_TRADE_TF_KEYS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h"]


def run_flask_server():
    """Runs the Flask dashboard server in a background thread."""
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[Flask Server] Starting web dashboard on {host}:{port}...")
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port, debug=False, use_reloader=False)


def handle_shutdown(signum, frame):
    """Graceful shutdown signal handler."""
    print("\n[System Shutdown] Received termination signal. Saving history and shutting down...")
    try:
        save_history(bot_state)
    except Exception as e:
        print(f"[Shutdown Error] {e}")
    sys.exit(0)


def main():
    """Main application entry point."""
    print("=" * 60)
    print(f"🚀 BTC & Multi-Asset Trading Bot (Mode: {TRADE_MODE.upper()}) Starting Up...")
    print("=" * 60)

    # Register OS signal handlers for graceful shutdown (main thread only)
    try:
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    except (ValueError, AttributeError):
        pass

    # Initialize database tables and load trade history
    database.init_db()
    load_history(bot_state)
    heal_completed_trades_bybit_order_ids(bot_state)

    # 1. Start Flask Web Dashboard Thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()

    # 2. Start Bybit WebSocket Streams & Connection Watchdog
    pub_ws_thread = threading.Thread(target=start_ws, kwargs={"bot_state": bot_state}, daemon=True)
    pub_ws_thread.start()

    if TRADE_MODE != "simulation":
        priv_ws_thread = threading.Thread(target=start_private_ws, kwargs={"bot_state": bot_state}, daemon=True)
        priv_ws_thread.start()

    ws_watchdog_thread = threading.Thread(target=run_websocket_watchdog, daemon=True)
    ws_watchdog_thread.start()

    # 3. Start Telegram Command Listener Thread
    tg_thread = threading.Thread(target=start_telegram_command_listener, kwargs={"bot_state": bot_state}, daemon=True)
    tg_thread.start()

    # 4. Start Real Balance Sync Worker Thread
    balance_thread = threading.Thread(target=run_bybit_balance_updater, kwargs={"bot_state": bot_state, "bot_state_lock": bot_state_lock}, daemon=True)
    balance_thread.start()

    # 4b. Start Signal Evaluator Worker Thread (Market Regimes & Predictions)
    from signal_evaluator import run_signal_evaluator_loop
    signal_eval_thread = threading.Thread(target=run_signal_evaluator_loop, kwargs={"bot_state": bot_state}, daemon=True)
    signal_eval_thread.start()

    # 5. Start Background Schedulers (Daily summary, journal, backup, funding rate, pain feedback)
    from bybit_client import place_bybit_order, format_bybit_qty
    from core import calculate_historical_thresholds
    
    def get_funding_rate_helper(symbol="BTCUSDT"):
        try:
            res = bybit_get_request("/v5/market/tickers", {"category": "linear", "symbol": symbol})
            if res.get("retCode") == 0 and res.get("result", {}).get("list"):
                return float(res["result"]["list"][0].get("fundingRate", 0.0))
        except Exception:
            pass
        return 0.0

    threading.Thread(target=run_daily_summary_scheduler, kwargs={"bot_state": bot_state, "bot_state_lock": bot_state_lock, "send_daily_summary_func": lambda: send_daily_summary(bot_state=bot_state)}, daemon=True).start()
    threading.Thread(target=run_daily_journal_scheduler, daemon=True).start()
    threading.Thread(target=run_daily_backup_scheduler, daemon=True).start()
    threading.Thread(target=run_pain_feedback_verifier, daemon=True).start()
    threading.Thread(
        target=run_funding_rate_arbitrage_monitor,
        kwargs={
            "bot_state": bot_state,
            "get_funding_rate_func": get_funding_rate_helper,
            "place_bybit_market_order_func": place_bybit_order,
            "format_bybit_qty_func": format_bybit_qty,
            "send_telegram_alert_func": send_telegram_alert,
            "trade_mode": TRADE_MODE
        },
        daemon=True
    ).start()

    send_telegram_alert(
        f"⚡ *BTC TRADING BOT STARTED* ⚡\n"
        f"• *Mode*: `{TRADE_MODE.upper()}`\n"
        f"• *Supported Assets*: `{', '.join(SUPPORTED_SYMBOLS)}`\n"
        f"• *Uptime*: Active & Monitoring Market Streams"
    )

    print("[Main Loop] Initialized successfully. Entering active monitoring loop...")
    while True:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()