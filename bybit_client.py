import os
import time
import json
import hmac
import hashlib
import urllib.parse
import threading
import asyncio
import aiohttp
from typing import Dict, Any, Optional

BYBIT_BASE_URL = "https://api.bybit.com"

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


def _ensure_async_loop():
    global _async_loop, _aiohttp_session, _async_thread
    if _async_loop is None or not _async_loop.is_running():
        def run_loop():
            nonlocal loop
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop = asyncio.new_event_loop()
        _async_loop = loop
        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        _async_thread = t

        async def init_session():
            global _aiohttp_session
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
        url = "https://api.bybit.com/v5/market/time"
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


def format_bybit_price(symbol: str, price: float) -> str:
    p_val = float(price)
    if "BTC" in symbol or "ETH" in symbol:
        return f"{p_val:.2f}"
    elif "SOL" in symbol or "BNB" in symbol:
        return f"{p_val:.3f}"
    else:
        return f"{p_val:.4f}"


def place_bybit_order(symbol: str, side: str, qty: float, price: Optional[float] = None, sl: Optional[float] = None, tp: Optional[float] = None, reduce_only: bool = False, order_type: str = "Market") -> Dict[str, Any]:
    order_type_str = "Limit" if (order_type == "Limit" and price is not None) else "Market"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": order_type_str,
        "qty": str(qty),
        "timeInForce": "GTC" if order_type_str == "Limit" else "IOC",
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


def get_all_bybit_positions() -> list:
    res = bybit_get_request("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
    if res.get("retCode") == 0:
        return res.get("result", {}).get("list", [])
    return []


def get_real_bybit_balance_cached(force: bool = False):
    global _real_balance_cache, _last_real_balance_sync
    now = time.time()
    with _real_balance_lock:
        if not force and _real_balance_cache is not None and (now - _last_real_balance_sync) < 60:
            return _real_balance_cache

    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
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
    return _real_balance_cache if _real_balance_cache is not None else 80.0
