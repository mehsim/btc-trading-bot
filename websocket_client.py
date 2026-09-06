from logger import log_event
"""
websocket_client.py
-------------------
DEPRECATION NOTICE: This module is NOT a WebSocket client. It retains only the standalone,
side-effect-free helper parse_orderbook_depth plus two inert compatibility stubs.

Finding #169: the full public+private WebSocket implementation that used to live here was a dead
parallel copy of the live execution stack and has been removed. Real-time market-data and private
execution feeds are implemented exclusively in main.py (start_ws, start_private_ws,
run_websocket_watchdog, order_flow_data) and bybit_client.py. This module owns no socket, no
thread, no order-fill cache and no order-flow state, so it can never report a feed as healthy.
Do not introduce new dependencies on this module.
"""

from typing import Dict, Any, Tuple


def get_ws_status() -> Dict[str, Any]:
    """
    Finding #137/#169: This module holds no WebSocket connection, so it reports a permanently
    disconnected status rather than a stale value derived from globals no live code updates.
    Live feed health is owned by main.py's watchdog; query that, not this.
    """
    return {
        "public_connected": False,
        "private_connected": False,
        "last_update_age_sec": None,
        "source": "websocket_client_stub_no_live_feed",
    }


def init_bybit_websocket_listeners(symbols: list, callback_on_kline=None) -> bool:
    """
    Inert compatibility stub. Starts no thread and opens no socket. Returns False so any caller
    that checks the result fails closed instead of assuming a live feed was established.
    """
    log_event(
        "WARNING",
        f"[WebSocket Engine] init_bybit_websocket_listeners is an inert stub (symbols={symbols}). "
        f"No socket opened. Live WebSocket feeds are started by main.py."
    )
    return False


def parse_orderbook_depth(bids: list, asks: list) -> Tuple[float, float, float, float]:
    """
    Finding #158: Parses order book depth price and size levels with float casting.
    Handles fractional altcoin ticks (e.g. 0.0012) without raising ValueError from int cast.
    Returns (best_bid, best_ask, total_bid_depth_usd, total_ask_depth_usd).
    """
    try:
        best_bid = float(bids[0][0]) if bids and len(bids) > 0 and len(bids[0]) > 0 else 0.0
    except (ValueError, TypeError, IndexError):
        best_bid = 0.0

    try:
        best_ask = float(asks[0][0]) if asks and len(asks) > 0 and len(asks[0]) > 0 else 0.0
    except (ValueError, TypeError, IndexError):
        best_ask = 0.0

    bid_depth = 0.0
    if bids:
        for b in bids:
            try:
                if len(b) >= 2:
                    bid_depth += float(b[0]) * float(b[1])
            except (ValueError, TypeError, IndexError):
                continue

    ask_depth = 0.0
    if asks:
        for a in asks:
            try:
                if len(a) >= 2:
                    ask_depth += float(a[0]) * float(a[1])
            except (ValueError, TypeError, IndexError):
                continue

    return best_bid, best_ask, bid_depth, ask_depth
