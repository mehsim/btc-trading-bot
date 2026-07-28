import os
import time
import json
import hmac
import hashlib
import urllib.parse
import threading
import asyncio
import aiohttp
from dotenv import load_dotenv
load_dotenv()

TRADE_MODE = os.environ.get("TRADE_MODE", "simulation").lower()
BYBIT_BASE_URL = "https://api-testnet.bybit.com" if TRADE_MODE == "testnet" else "https://api.bybit.com"

_cached_time_offset = 0
_last_time_sync = 0
_time_offset_lock = threading.Lock()
_aiohttp_session = None
_async_loop = None
_async_thread = None
_ws_responses = {}
_ws_responses_lock = threading.Lock()
_order_exec_lock = threading.Lock()

_real_balance_cache = None
_last_real_balance_sync = 0
_real_balance_lock = threading.Lock()


def get_bybit_proxies():
    if os.environ.get("SPACE_ID") and not os.environ.get("BYBIT_PROXY"):
        return None
    proxy = (
        os.environ.get("BYBIT_PROXY") or
        os.environ.get("HTTP_PROXY") or
        os.environ.get("HTTPS_PROXY")
    )
    if proxy:
        return {"http": proxy, "https": proxy}
    return None


def parse_proxy_url(proxy_url):
    """Parse proxy URL into a dictionary formatted for requests or aiohttp."""
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


_async_init_lock = threading.Lock()

def _ensure_async_loop():
    global _async_loop, _aiohttp_session, _async_thread
    with _async_init_lock:
        if _async_loop is None or not _async_loop.is_running():
            new_loop = asyncio.new_event_loop()
            _async_loop = new_loop
            def run_loop(l):
                asyncio.set_event_loop(l)
                l.run_forever()
            t = threading.Thread(target=run_loop, args=(new_loop,), daemon=True)
            t.start()
            _async_thread = t

        async def init_session():
            global _aiohttp_session
            if _aiohttp_session is None or _aiohttp_session.closed:
                _aiohttp_session = aiohttp.ClientSession()

        future = asyncio.run_coroutine_threadsafe(init_session(), _async_loop)
        future.result(timeout=5)


def get_bybit_time_offset() -> int:
    global _cached_time_offset, _last_time_sync
    now_t = time.time()
    with _time_offset_lock:
        if now_t - _last_time_sync < 300 and _last_time_sync > 0:
            return _cached_time_offset

    async def do_time_sync():
        url = f"{BYBIT_BASE_URL}/v5/market/time"
        proxy_dict = get_bybit_proxies()
        proxy_url = proxy_dict.get("https") or proxy_dict.get("http") if proxy_dict else None
        timeout = aiohttp.ClientTimeout(total=5)
        async with _aiohttp_session.get(url, proxy=proxy_url, timeout=timeout) as resp:
            data = await resp.json()
            return resp.status, data

    for attempt in range(3):
        try:
            _ensure_async_loop()
            future = asyncio.run_coroutine_threadsafe(do_time_sync(), _async_loop)
            status, res = future.result(timeout=7)
            if status == 200:
                server_time = int(res["result"]["timeNano"]) // 1000000
                local_time = int(time.time() * 1000)
                offset = server_time - local_time
                with _time_offset_lock:
                    _cached_time_offset = offset
                    _last_time_sync = time.time()
                return offset
        except Exception:
            time.sleep(1)
    return 0


from secret_manager import get_secure_env


def bybit_post_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = get_secure_env("BYBIT_API_KEY", "").strip()
    api_secret = get_secure_env("BYBIT_API_SECRET", "").strip()

    if not api_key or not api_secret:
        return {"retCode": -1, "retMsg": "API keys missing"}
        
    offset = get_bybit_time_offset()
    timestamp = str(int(time.time() * 1000) + offset)
    recv_window = "5000"
    
    payload_str = json.dumps(payload)
    val_str = timestamp + api_key + recv_window + payload_str
    sign = hmac.new(api_secret.encode("utf-8"), val_str.encode("utf-8"), hashlib.sha256).hexdigest()
    
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


def bybit_get_request(endpoint: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = get_secure_env("BYBIT_API_KEY", "").strip()
    api_secret = get_secure_env("BYBIT_API_SECRET", "").strip()

    if not api_key or not api_secret:
        return {"retCode": -1, "retMsg": "API keys missing"}
        
    offset = get_bybit_time_offset()
    timestamp = str(int(time.time() * 1000) + offset)
    recv_window = "5000"
    
    query_string = urllib.parse.urlencode(query_params)
    val_str = timestamp + api_key + recv_window + query_string
    sign = hmac.new(api_secret.encode("utf-8"), val_str.encode("utf-8"), hashlib.sha256).hexdigest()
    
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


def execute_bybit_order_ws_or_rest(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import uuid
    if endpoint == "/v5/order/create" and "orderLinkId" not in payload:
        payload["orderLinkId"] = f"cl_{payload.get('symbol', 'generic')}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        
    with _order_exec_lock:
        return bybit_post_request(endpoint, payload)


def set_bybit_leverage(symbol: str, leverage: float) -> Dict[str, Any]:
    lev_str = f"{float(leverage):.1f}"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": lev_str,
        "sellLeverage": lev_str
    }
    return bybit_post_request("/v5/position/set-leverage", payload)


def format_bybit_price(symbol: str, price: float) -> str:
    p_val = float(price)
    if "BTC" in symbol or "ETH" in symbol:
        return f"{p_val:.2f}"
    elif "SOL" in symbol or "BNB" in symbol:
        return f"{p_val:.3f}"
    else:
        return f"{p_val:.4f}"


def format_bybit_qty(symbol: str, qty: float) -> str:
    q_val = float(qty)
    if "BTC" in symbol:
        return f"{q_val:.3f}"
    elif "ETH" in symbol:
        return f"{q_val:.2f}"
    elif "SOL" in symbol or "BNB" in symbol:
        return f"{q_val:.2f}"
    else:
        return f"{int(round(q_val))}"


def get_bybit_min_qty_step(symbol: str) -> tuple:
    mins = {"BTCUSDT": 0.001, "ETHUSDT": 0.01, "SOLUSDT": 0.1, "BNBUSDT": 0.01, "ADAUSDT": 1.0, "XRPUSDT": 1.0}
    steps = {"BTCUSDT": 0.001, "ETHUSDT": 0.01, "SOLUSDT": 0.1, "BNBUSDT": 0.01, "ADAUSDT": 1.0, "XRPUSDT": 1.0}
    return mins.get(symbol, 0.001), steps.get(symbol, 0.001)


def place_bybit_order(symbol: str, side: str, qty: float, price: Optional[float] = None, sl: Optional[float] = None, tp: Optional[float] = None, reduce_only: bool = False, order_type: str = "Market", post_only: bool = False) -> Dict[str, Any]:
    if post_only and not reduce_only:
        return place_bybit_maker_chase_order(symbol=symbol, side=side, qty=qty, sl=sl, tp=tp)
        
    order_type_str = "Limit" if (order_type == "Limit" and price is not None) else "Market"
    tif_str = "PostOnly" if post_only else ("GTC" if order_type_str == "Limit" else "IOC")
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": order_type_str,
        "qty": format_bybit_qty(symbol, qty),
        "timeInForce": tif_str,
        "positionIdx": 0
    }
    if price is not None:
        payload["price"] = format_bybit_price(symbol, price)
    if reduce_only:
        payload["reduceOnly"] = True
    if sl:
        payload["stopLoss"] = format_bybit_price(symbol, sl)
    if tp:
        payload["takeProfit"] = format_bybit_price(symbol, tp)
        
    return execute_bybit_order_ws_or_rest("/v5/order/create", payload)


def place_bybit_limit_order(symbol: str, side: str, qty: float, price: float, sl: Optional[float] = None, tp: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
    return place_bybit_order(symbol=symbol, side=side, qty=qty, price=price, sl=sl, tp=tp, reduce_only=reduce_only, order_type="Limit")


def place_bybit_taker_ioc_order(symbol: str, side: str, qty: float, sl: Optional[float] = None, tp: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
    return place_bybit_order(symbol=symbol, side=side, qty=qty, sl=sl, tp=tp, reduce_only=reduce_only, order_type="Market", post_only=False)


def get_bybit_order_details(symbol: str, order_id: str) -> Dict[str, Any]:
    return bybit_get_request("/v5/order/realtime", {"category": "linear", "symbol": symbol, "orderId": order_id})


def cancel_bybit_order(symbol: str, order_id: str) -> Dict[str, Any]:
    return execute_bybit_order_ws_or_rest("/v5/order/cancel", {"category": "linear", "symbol": symbol, "orderId": order_id})


def get_bybit_position(symbol: str) -> Dict[str, Any]:
    res = bybit_get_request("/v5/position/list", {"category": "linear", "symbol": symbol})
    if res.get("retCode") == 0:
        p_list = res.get("result", {}).get("list", [])
        if p_list:
            return p_list[0]
    return {}


def get_all_bybit_positions() -> list:
    res = bybit_get_request("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
    if res.get("retCode") == 0:
        return res.get("result", {}).get("list", [])
    return []


def get_bybit_closed_pnl(symbol: str, limit: int = 1) -> float:
    res = bybit_get_request("/v5/position/closed-pnl", {"category": "linear", "symbol": symbol, "limit": limit})
    if res.get("retCode") == 0:
        p_list = res.get("result", {}).get("list", [])
        if p_list:
            return float(p_list[0].get("closedPnl", 0.0))
    return 0.0


def get_bybit_accumulated_closed_pnl(symbol: str, entry_time_ms: int) -> float:
    res = bybit_get_request("/v5/position/closed-pnl", {"category": "linear", "symbol": symbol, "limit": 20})
    if res.get("retCode") == 0:
        p_list = res.get("result", {}).get("list", [])
        tot = 0.0
        for p in p_list:
            created_t = int(p.get("createdTime", 0))
            if created_t >= entry_time_ms - 60000:
                tot += float(p.get("closedPnl", 0.0))
        return tot
    return 0.0


def update_bybit_stop_loss(symbol: str, sl_price: float, active_trade: Optional[Dict] = None) -> Dict[str, Any]:
    payload = {
        "category": "linear",
        "symbol": symbol,
        "stopLoss": format_bybit_price(symbol, sl_price),
        "positionIdx": 0
    }
    return bybit_post_request("/v5/position/trading-stop", payload)


def update_bybit_take_profit(symbol: str, tp_price: float, active_trade: Optional[Dict] = None) -> Dict[str, Any]:
    payload = {
        "category": "linear",
        "symbol": symbol,
        "takeProfit": format_bybit_price(symbol, tp_price),
        "positionIdx": 0
    }
    return bybit_post_request("/v5/position/trading-stop", payload)


def get_bybit_bid_ask(symbol: str) -> tuple:
    res = bybit_get_request("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if res.get("retCode") == 0:
        t_list = res.get("result", {}).get("list", [])
        if t_list:
            return float(t_list[0].get("bid1Price", 0.0)), float(t_list[0].get("ask1Price", 0.0))
    return 0.0, 0.0


def get_bybit_last_execution(symbol: str) -> Dict[str, Any]:
    res = bybit_get_request("/v5/execution/list", {"category": "linear", "symbol": symbol, "limit": 1})
    if res.get("retCode") == 0:
        e_list = res.get("result", {}).get("list", [])
        if e_list:
            return e_list[0]
    return {}


def place_bybit_maker_chase_order(symbol: str, side: str, qty: float, sl: Optional[float] = None, tp: Optional[float] = None, max_chase_seconds: float = 10.0) -> Dict[str, Any]:
    ticker_res = bybit_get_request("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    best_bid = 0.0
    best_ask = 0.0
    if ticker_res.get("retCode") == 0:
        t_list = ticker_res.get("result", {}).get("list", [])
        if t_list:
            best_bid = float(t_list[0].get("bid1Price", 0.0))
            best_ask = float(t_list[0].get("ask1Price", 0.0))
            
    if best_bid > 0 and best_ask > 0:
        spread_pct = (best_ask - best_bid) / best_bid
        if spread_pct > 0.003:
            print(f"[{symbol}] Wide spread detected ({spread_pct*100:.2f}% > 0.30%). Bypassing Post-Only to execute via Taker Market.")
            return place_bybit_order(symbol=symbol, side=side, qty=qty, sl=sl, tp=tp, reduce_only=False, order_type="Market", post_only=False)

    limit_price = best_bid if side == "Buy" and best_bid > 0 else (best_ask if side == "Sell" and best_ask > 0 else None)
    if limit_price is None:
        return place_bybit_order(symbol=symbol, side=side, qty=qty, sl=sl, tp=tp, reduce_only=False, order_type="Market", post_only=False)

    post_payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Limit",
        "qty": format_bybit_qty(symbol, qty),
        "price": format_bybit_price(symbol, limit_price),
        "timeInForce": "PostOnly",
        "positionIdx": 0
    }
    if sl:
        post_payload["stopLoss"] = format_bybit_price(symbol, sl)
    if tp:
        post_payload["takeProfit"] = format_bybit_price(symbol, tp)
        
    res = execute_bybit_order_ws_or_rest("/v5/order/create", post_payload)
    if res.get("retCode") != 0:
        return place_bybit_order(symbol=symbol, side=side, qty=qty, sl=sl, tp=tp, reduce_only=False, order_type="Market", post_only=False)

    order_id = res.get("result", {}).get("orderId")
    if not order_id:
        return res
        
    start_t = time.time()
    while time.time() - start_t < max_chase_seconds:
        time.sleep(2.0)
        chk = bybit_get_request("/v5/order/realtime", {"category": "linear", "symbol": symbol, "orderId": order_id})
        if chk.get("retCode") == 0:
            o_list = chk.get("result", {}).get("list", [])
            if o_list:
                o_status = o_list[0].get("orderStatus")
                if o_status in ["Filled"]:
                    return chk
                elif o_status in ["Cancelled", "Rejected"]:
                    break

    cancel_payload = {"category": "linear", "symbol": symbol, "orderId": order_id}
    execute_bybit_order_ws_or_rest("/v5/order/cancel", cancel_payload)
    
    chk_final = bybit_get_request("/v5/order/realtime", {"category": "linear", "symbol": symbol, "orderId": order_id})
    filled_qty = 0.0
    if chk_final.get("retCode") == 0 and chk_final.get("result", {}).get("list"):
        filled_qty = float(chk_final["result"]["list"][0].get("cumExecQty", 0.0))
        
    rem_qty = float(qty) - filled_qty
    if rem_qty > 0.0001:
        return place_bybit_order(symbol=symbol, side=side, qty=rem_qty, sl=sl, tp=tp, reduce_only=False, order_type="Market", post_only=False)
    return chk_final


def get_real_bybit_balance() -> float:
    res = bybit_get_request("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
    if res.get("retCode") == 0:
        l_data = res.get("result", {}).get("list", [])
        if l_data:
            return float(l_data[0].get("totalEquity") or l_data[0].get("totalWalletBalance") or 0.0)
    return 0.0


def get_real_bybit_balance_cached(force: bool = False):
    global _real_balance_cache, _last_real_balance_sync
    now = time.time()
    with _real_balance_lock:
        if not force and _real_balance_cache is not None and (now - _last_real_balance_sync) < 60:
            return _real_balance_cache

    api_key = get_secure_env("BYBIT_API_KEY", "").strip()
    api_secret = get_secure_env("BYBIT_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return "API_KEYS_MISSING"

    res = bybit_get_request("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
    if res.get("retCode") == 0:
        list_data = res.get("result", {}).get("list", [])
        if list_data:
            total_equity = float(list_data[0].get("totalEquity") or list_data[0].get("totalWalletBalance") or 0.0)
            with _real_balance_lock:
                _real_balance_cache = total_equity
                _last_real_balance_sync = now
            return total_equity
    return _real_balance_cache


def run_bybit_balance_updater(bot_state=None, bot_state_lock=None):
    """
    Background worker thread running every 120s to sync wallet balance from Bybit API.
    """
    print("[Balance Sync] Bybit real wallet balance sync thread started.")
    while True:
        try:
            bal = get_real_bybit_balance_cached(force=True)
            if isinstance(bal, (int, float)) and bal > 0 and bot_state:
                if bot_state_lock:
                    with bot_state_lock:
                        bot_state["wallet_balance"] = bal
                        bot_state["live_balance"] = bal
                else:
                    bot_state["wallet_balance"] = bal
                    bot_state["live_balance"] = bal
        except Exception as e:
            print(f"[Balance Sync Error] Failed to update Bybit balance: {e}")
        time.sleep(120)
