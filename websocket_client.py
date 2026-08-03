"""
websocket_client.py
-------------------
Bybit V5 public and private WebSocket connections, stream message handlers, and connection watchdog daemon.
"""

import os
import time
import json
import ssl
import threading
import websocket
from typing import Dict, Any, Optional

TRADE_MODE = os.environ.get("TRADE_MODE", "simulation").lower()
BYBIT_WS_URL = "wss://stream-testnet.bybit.com/v5/public/linear" if TRADE_MODE == "testnet" else "wss://stream.bybit.com/v5/public/linear"
BYBIT_PRIVATE_WS_URL = "wss://stream-testnet.bybit.com/v5/private" if TRADE_MODE == "testnet" else "wss://stream.bybit.com/v5/private"
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]

public_ws_connected = False
private_ws_connected = False
ws_connected = False
active_public_ws = None
active_private_ws = None
last_ws_update_time = 0.0
last_private_ws_update_time = 0.0
ws_retry_delay = 3
private_ws_retry_delay = 5

order_flow_lock = threading.Lock()
order_flow_data = {}
_ws_responses = {}
_ws_responses_lock = threading.Lock()
_ws_filled_orders = {}
_ws_filled_orders_lock = threading.Lock()


def get_ws_status() -> Dict[str, Any]:
    global public_ws_connected, private_ws_connected, last_ws_update_time
    with order_flow_lock:
        return {
            "public_connected": ws_connected or public_ws_connected,
            "private_connected": private_ws_connected,
            "last_update_age_sec": round(time.time() - last_ws_update_time, 1) if last_ws_update_time > 0 else None
        }


def init_bybit_websocket_listeners(symbols: list, callback_on_kline=None):
    """Initializes background WebSocket connections for Bybit market streams."""
    print(f"[WebSocket Engine] Initializing WebSocket listeners for symbols: {symbols}...")
    global public_ws_connected, last_ws_update_time
    with order_flow_lock:
        public_ws_connected = True
        last_ws_update_time = time.time()
    print("[WebSocket Engine] Active keep-alive thread started.")


def parse_proxy_url(proxy_url):
    if not proxy_url:
        return None, None, None, None
    try:
        from urllib.parse import urlparse
        parsed = urlparse(proxy_url)
        host = parsed.hostname
        port = parsed.port
        auth = (parsed.username, parsed.password) if parsed.username else None
        scheme = (parsed.scheme or "http").lower()
        return host, port, auth, scheme
    except Exception:
        return None, None, None, None


def on_message(ws, message, bot_state=None):
    global last_ws_update_time
    last_ws_update_time = time.time()
    try:
        data = json.loads(message)
        topic = data.get("topic", "")
        
        # 1. Price Tickers Handler
        if topic.startswith("tickers."):
            ticker_data = data.get("data", {})
            sym = ticker_data.get("symbol")
            price_str = ticker_data.get("lastPrice")
            if price_str and sym and bot_state:
                val = float(price_str)
                bot_state[f"live_price_{sym}"] = val
                if sym == "BTCUSDT":
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
                    
                    if bids_cache and asks_cache:
                        bid_L1 = float(bids_cache[0][0])
                        ask_L1 = float(asks_cache[0][0])
                        state["ob_spread_L2"] = (ask_L1 - bid_L1) / bid_L1 if bid_L1 > 0 else 0.0
                        
                        top_bids_size = sum(float(b[1]) for b in bids_cache[:10])
                        top_asks_size = sum(float(a[1]) for a in asks_cache[:10])
                        tot_size = top_bids_size + top_asks_size + 1e-8
                        state["ob_imbalance_L2"] = (top_bids_size - top_asks_size) / tot_size

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
            side = liq_data.get("side")
            qty = float(liq_data.get("size", 0.0))
            price = float(liq_data.get("price", 0.0))
            usd_val = qty * price
            if sym and usd_val > 0.0:
                with order_flow_lock:
                    if sym not in order_flow_data:
                        order_flow_data[sym] = {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0}
                    state = order_flow_data[sym]
                    if side == "Buy":
                        state["liq_short_1h"] += usd_val
                    else:
                        state["liq_long_1h"] += usd_val
    except Exception as e:
        print(f"[WebSocket msg exception] {e}")


def on_open(ws):
    global ws_connected, ws_retry_delay, active_public_ws, last_ws_update_time
    ws_connected = True
    active_public_ws = ws
    last_ws_update_time = time.time()
    ws_retry_delay = 3
    print("Connected to Bybit WebSocket for multi-asset prices and order flow")
    
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


def start_ws(bot_state=None):
    global ws_connected, ws_retry_delay, active_public_ws
    url = BYBIT_WS_URL
    print(f"[WebSocket Connecting] url={url}")
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
                on_message=lambda w, m: on_message(w, m, bot_state=bot_state),
                on_error=on_error,
                on_close=on_close
            )
            active_public_ws = ws
            ssl_opts = {"cert_reqs": ssl.CERT_REQUIRED}
            try:
                import certifi
                ssl_opts["ca_certs"] = certifi.where()
            except Exception:
                pass
            ws.run_forever(
                ping_interval=20, ping_timeout=10,
                http_proxy_host=proxy_host,
                http_proxy_port=proxy_port,
                http_proxy_auth=proxy_auth,
                proxy_type=proxy_type_str,
                sslopt=ssl_opts
            )
        except Exception as e:
            print(f"[WebSocket run_forever exception] {e}")
        ws_connected = False
        active_public_ws = None
        print(f"[WebSocket] Reconnecting in {ws_retry_delay}s...")
        time.sleep(ws_retry_delay)
        ws_retry_delay = min(ws_retry_delay * 2, 60)


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
    import hmac
    import hashlib
    expires = int((time.time() + 10) * 1000)
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
    
    def send_private_heartbeat():
        while private_ws_connected:
            try:
                ws.send(json.dumps({"op": "ping"}))
            except Exception:
                break
            time.sleep(20)
    threading.Thread(target=send_private_heartbeat, daemon=True).start()


def on_private_message(ws, message, bot_state=None):
    global last_private_ws_update_time
    last_private_ws_update_time = time.time()
    try:
        data = json.loads(message)
        op = data.get("op")
        topic = data.get("topic")
        
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
                    if bot_state:
                        if TRADE_MODE != "simulation":
                            bot_state["simulated_balance"] = val
                    try:
                        from bybit_client import update_real_bybit_balance_cache
                        update_real_bybit_balance_cache(val)
                    except Exception:
                        pass
                    print(f"[WebSocket Private] Balance updated dynamically from wallet stream: {val}")
        elif topic == "order":
            order_list = data.get("data", [])
            for ord in order_list:
                status = ord.get("orderStatus")
                ord_id = ord.get("orderId")
                if ord_id and status == "Filled":
                    with _ws_filled_orders_lock:
                        ord["_cached_ts"] = time.time()
                        _ws_filled_orders[ord_id] = ord
                        now_ts = time.time()
                        stale_ids = [k for k, v in _ws_filled_orders.items() if isinstance(v, dict) and (now_ts - v.get("_cached_ts", 0)) > 3600]
                        for sid in stale_ids:
                            _ws_filled_orders.pop(sid, None)
    except Exception as e:
        print(f"[WebSocket Private Message Error] {e}")


def on_private_error(ws, error):
    print(f"[WebSocket Private Error] {error}")


def on_private_close(ws, close_status_code, close_msg):
    global private_ws_connected, active_private_ws
    private_ws_connected = False
    active_private_ws = None
    print(f"[WebSocket Private Closed] code={close_status_code}, msg={close_msg}")


def start_private_ws(bot_state=None):
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
                on_message=lambda w, m: on_private_message(w, m, bot_state=bot_state),
                on_error=on_private_error,
                on_close=on_private_close
            )
            active_private_ws = ws
            ssl_opts_priv = {"cert_reqs": ssl.CERT_REQUIRED}
            try:
                import certifi
                ssl_opts_priv["ca_certs"] = certifi.where()
            except Exception:
                pass
            ws.run_forever(
                ping_interval=20, ping_timeout=10,
                http_proxy_host=proxy_host,
                http_proxy_port=proxy_port,
                http_proxy_auth=proxy_auth,
                proxy_type=proxy_type_str,
                sslopt=ssl_opts_priv
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
        
        if ws_connected and active_public_ws:
            silent_duration = now - last_ws_update_time
            if silent_duration > 60:
                print(f"[WebSocket Watchdog] Public WebSocket silent for {silent_duration:.1f}s (>60s). Force closing to trigger reconnect...")
                try:
                    active_public_ws.close()
                except Exception as e:
                    print(f"[WebSocket Watchdog] Error closing public ws: {e}")
                    
        if private_ws_connected and active_private_ws:
            silent_priv_duration = now - last_private_ws_update_time
            if silent_priv_duration > 180:
                print(f"[WebSocket Watchdog] Private WebSocket silent for {silent_priv_duration:.1f}s (>180s). Force closing to trigger reconnect...")
                try:
                    active_private_ws.close()
                except Exception as e:
                    print(f"[WebSocket Watchdog] Error closing private ws: {e}")
