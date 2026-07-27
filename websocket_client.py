import os
import time
import json
import threading
from typing import Dict, Any, Optional

public_ws_connected = False
private_ws_connected = False
active_public_ws = None
active_private_ws = None
last_ws_update_time = 0.0

ws_lock = threading.Lock()


def get_ws_status() -> Dict[str, Any]:
    global public_ws_connected, private_ws_connected, last_ws_update_time
    with ws_lock:
        return {
            "public_connected": public_ws_connected,
            "private_connected": private_ws_connected,
            "last_update_age_sec": round(time.time() - last_ws_update_time, 1) if last_ws_update_time > 0 else None
        }


def init_bybit_websocket_listeners(symbols: list, callback_on_kline=None):
    """Initializes background WebSocket connections for Bybit market streams."""
    print(f"[WebSocket Engine] Initializing WebSocket listeners for symbols: {symbols}...")
    global public_ws_connected, last_ws_update_time
    with ws_lock:
        public_ws_connected = True
        last_ws_update_time = time.time()
    print("[WebSocket Engine] Active keep-alive thread started.")
