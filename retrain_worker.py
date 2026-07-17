import os
import sys
import time
import subprocess
import requests
import re

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
    send_telegram_alert("🔄 *MODEL RETRAINING STARTED* 🔄\n• Weekly retraining script has started on Singapore instance.\n• Main execution bot continues running normally.")

    intervals = ["60", "120", "240", "360"]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    results_summary = []

    for iv in intervals:
        print(f"[retrain_worker] Spawning subprocess for interval {iv}m...")
        cmd = ["nice", "-n", "19", sys.executable, "train.py", "--interval", iv, "--pages", "5", "--live-feedback"]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = p.communicate()
        
        if p.returncode != 0:
            print(f"[retrain_worker Error] Retraining for interval {iv}m failed:\n{stderr}")
            send_telegram_alert(f"❌ *retrain_worker error* ❌\nRetraining failed for {iv}m:\n`{stderr[:200]}`")
            continue
            
        print(f"[retrain_worker] Retraining for interval {iv}m finished successfully.")
        
        # Parse stdout for summary metrics
        feedback_count = 0
        total_rows = 0
        accuracies = []
        
        for line in stdout.splitlines():
            # Match feedback samples: "[Live Feedback] Injecting X feedback samples..."
            fb_match = re.search(r"Injecting (\d+) feedback samples", line)
            if fb_match:
                feedback_count = int(fb_match.group(1))
                
            # Match total rows: "=== Combined Training Dataset: Y total rows"
            row_match = re.search(r"Combined Training Dataset: (\d+) total rows", line)
            if row_match:
                total_rows = int(row_match.group(1))
                
            # Match accuracy: "Validation Out-of-Sample Accuracy (Ensemble Trend): Z%"
            acc_match = re.search(r"Validation Out-of-Sample Accuracy \(Ensemble Trend\):\s*([\d\.]+)%", line)
            if acc_match:
                accuracies.append(float(acc_match.group(1)))
                
        avg_acc_str = f"{sum(accuracies)/len(accuracies):.1f}%" if accuracies else "N/A"
        results_summary.append(
            f"• *{iv}m*: {total_rows} rows | {feedback_count} feedback samples | Avg Accuracy: {avg_acc_str}"
        )

    print("[retrain_worker] Weekly retraining complete.")
    summary_msg = "\n".join(results_summary)
    send_telegram_alert(
        f"🔄 *MODEL RETRAINING COMPLETE* 🔄\n\n"
        f"Model retraining completed successfully. Updates applied to disk:\n\n"
        f"{summary_msg}"
    )

if __name__ == "__main__":
    main()
