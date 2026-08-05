import requests
import time
import os
import sys
from datetime import datetime, timezone

TARGET_URL = os.environ.get("HEARTBEAT_TARGET_URL", "http://127.0.0.1:5001/api/status")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MAX_FAILURES = 3

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[Alert Output] {message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

def run_heartbeat_check():
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Checking heartbeat at {TARGET_URL}...")
    consecutive_failures = 0
    
    for attempt in range(1, MAX_FAILURES + 1):
        try:
            resp = requests.get(TARGET_URL, timeout=8)
            if resp.status_code == 200:
                print(f"✅ Heartbeat OK: Server status endpoint responding (HTTP 200).")
                return True
            else:
                consecutive_failures += 1
                print(f"⚠️ Heartbeat Warning: HTTP {resp.status_code} (Attempt {attempt}/{MAX_FAILURES})")
        except Exception as e:
            consecutive_failures += 1
            print(f"⚠️ Heartbeat Error: {e} (Attempt {attempt}/{MAX_FAILURES})")
        time.sleep(2)
        
    if consecutive_failures >= MAX_FAILURES:
        alert_msg = (
            f"🚨 *CRITICAL ALERT: AWS BOT UNRESPONSIVE* 🚨\n\n"
            f"• *Target Endpoint*: `{TARGET_URL}`\n"
            f"• *Status*: Failed {MAX_FAILURES} consecutive heartbeat checks.\n"
            f"• *Timestamp*: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`\n\n"
            f"⚠️ *Action Required*: Verify AWS server status (`47.129.153.199`) and active Bybit positions immediately."
        )
        send_telegram_alert(alert_msg)
        return False

if __name__ == "__main__":
    run_heartbeat_check()
