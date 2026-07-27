import os
import time
import requests
import threading
from typing import Dict, Any, Optional

cached_telegram_token = None
cached_chat_ids = None

from secret_manager import get_secure_env

def get_telegram_config():
    global cached_telegram_token, cached_chat_ids
    if cached_telegram_token is not None and cached_chat_ids is not None:
        return cached_telegram_token, cached_chat_ids

    token = get_secure_env("TELEGRAM_BOT_TOKEN", "").strip()
    chat_ids_str = get_secure_env("TELEGRAM_CHAT_ID", "").strip()


    chat_ids = []
    if chat_ids_str:
        for cid in chat_ids_str.split(","):
            cid = cid.strip()
            if cid:
                chat_ids.append(cid)

    cached_telegram_token = token
    cached_chat_ids = chat_ids
    return token, chat_ids


def execute_telegram_api_call(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    token, allowed_chat_ids = get_telegram_config()
    if not token:
        return {}

    url = f"https://api.telegram.org/bot{token}/{method}"
    tg_proxy = os.environ.get("TELEGRAM_PROXY") or os.environ.get("BYBIT_PROXY")
    proxies_dict = None
    if tg_proxy:
        proxies_dict = {"http": tg_proxy, "https": tg_proxy}

    headers = {"Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15, proxies=proxies_dict)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
    return {}


def send_telegram_alert(message: str, disable_web_page_preview: bool = True) -> bool:
    token, chat_ids = get_telegram_config()
    if not token or not chat_ids:
        return False

    success = False
    for cid in chat_ids:
        payload = {
            "chat_id": cid,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": disable_web_page_preview
        }
        res = execute_telegram_api_call("sendMessage", payload)
        if res.get("ok"):
            success = True
    return success
