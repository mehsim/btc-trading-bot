"""
trading_engine.py
-----------------
DEPRECATION NOTICE: This module is maintained solely for backwards compatibility.
Active execution logic and Bybit REST interactions are centralized directly in bybit_client.py
and main.py. Do not introduce new dependencies on this module.
"""

import threading
from logger import log_event
from telegram_bot import send_telegram_alert

active_execution_lock = threading.Lock()
active_execution_symbols = set()


def execute_bybit_trade_async(*args, **kwargs):
    """
    Deprecated execution entry point.
    Delegates to main.execute_bybit_trade_async.
    """
    import main
    return main.execute_bybit_trade_async(*args, **kwargs)


def _execute_bybit_trade_async_inner(*args, **kwargs):
    """
    Deprecated inner execution implementation.
    Delegates directly to main._execute_bybit_trade_async_inner.
    """
    import main
    return main._execute_bybit_trade_async_inner(*args, **kwargs)
