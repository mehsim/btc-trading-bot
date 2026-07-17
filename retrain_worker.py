import os
import sys
import time
import subprocess
import requests

def send_telegram_alert(message):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

def main():
    print("[retrain_worker] Starting weekly model retraining...")
    intervals = ["60", "120", "240", "360"]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    for iv in intervals:
        print(f"[retrain_worker] Spawning subprocess for interval {iv}m...")
        # Use nice -n 19 to keep process priority low
        cmd = ["nice", "-n", "19", sys.executable, "train.py", "--interval", iv, "--pages", "20", "--live-feedback"]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = p.communicate()
        if p.returncode == 0:
            print(f"[retrain_worker] Retraining for interval {iv}m finished successfully.")
        else:
            print(f"[retrain_worker Error] Retraining for interval {iv}m failed:\n{stderr}")
            send_telegram_alert(f"❌ *retrain_worker error* ❌\nRetraining failed for {iv}m:\n`{stderr[:200]}`")

    print("[retrain_worker] Weekly retraining complete.")
    send_telegram_alert("🔄 *MODEL RETRAINING COMPLETE* 🔄\n• retrain_worker finished successfully.\n• Ensemble and meta-classifiers updated on disk.")

if __name__ == "__main__":
    main()
