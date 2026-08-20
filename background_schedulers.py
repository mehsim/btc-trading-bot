"""
background_schedulers.py
------------------------
Background periodic schedulers and thread routines (journal, funding rate monitor, backup, pain feedback, summary, weekly retraining).
"""

import os
import time
from datetime import datetime, timezone
import database

try:
    import config
    SUPPORTED_SYMBOLS = getattr(config, "SUPPORTED_SYMBOLS", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT", "LINKUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "DOGEUSDT"])
except Exception:
    SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT", "LINKUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "DOGEUSDT"]
JOURNAL_PATH = "trade_journal.csv"
ACTIVE_TRADE_TF_KEYS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h"]
FUNDING_ARB_THRESHOLD = 0.001
FUNDING_ARB_SIZE_USD = 20.0


def run_daily_journal_scheduler(send_daily_journal_digest_func=None):
    """Send daily Telegram digest at 00:00 UTC every day."""
    print("[Journal Scheduler] Daily digest scheduler started.")
    while True:
        now = time.gmtime()
        seconds_to_midnight = 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
        time.sleep(max(1, seconds_to_midnight))
        try:
            if send_daily_journal_digest_func:
                send_daily_journal_digest_func()
        except Exception as e:
            print(f"[Journal Scheduler] Error: {e}")


def run_funding_rate_arbitrage_monitor(bot_state=None, get_funding_rate_func=None, place_bybit_market_order_func=None, format_bybit_qty_func=None, send_telegram_alert_func=None, trade_mode="simulation"):
    """Monitor funding rates. When rate > 0.1%, open a small short to collect funding income."""
    print("[Funding Arb] Monitor started.")
    time.sleep(60)
    while True:
        try:
            for sym in SUPPORTED_SYMBOLS:
                if not get_funding_rate_func:
                    continue
                rate = get_funding_rate_func(sym)
                arb_key = f"funding_arb_{sym}"
                existing = bot_state.get(arb_key) if bot_state else None

                if rate > FUNDING_ARB_THRESHOLD and not existing:
                    print(f"[Funding Arb] {sym} funding rate {rate*100:.4f}% > 0.1%. Opening arb short.")
                    if trade_mode == "live" and place_bybit_market_order_func and format_bybit_qty_func and bot_state:
                        cur_price = bot_state.get(f"live_price_{sym}", 0.0) or 50000.0
                        qty_str = format_bybit_qty_func(sym, FUNDING_ARB_SIZE_USD / cur_price)
                        res = place_bybit_market_order_func(sym, "Sell", qty_str, reduce_only=False)
                        if res and res.get("retCode") == 0:
                            bot_state[arb_key] = {"qty": qty_str, "open_rate": rate}
                            if send_telegram_alert_func:
                                send_telegram_alert_func(
                                    f"💰 *FUNDING ARB OPENED*\n"
                                    f"• Asset: {sym}\n"
                                    f"• Funding Rate: {rate*100:.4f}%\n"
                                    f"• Side: Short (collecting funding)\n"
                                    f"• Size: ${FUNDING_ARB_SIZE_USD}"
                                )
                elif existing and rate < FUNDING_ARB_THRESHOLD * 0.3:
                    print(f"[Funding Arb] {sym} funding rate normalized ({rate*100:.4f}%). Closing arb short.")
                    if trade_mode == "live" and place_bybit_market_order_func:
                        qty_str = existing["qty"]
                        res = place_bybit_market_order_func(sym, "Buy", qty_str, reduce_only=True)
                        if res and res.get("retCode") == 0:
                            if bot_state:
                                bot_state.pop(arb_key, None)
                            if send_telegram_alert_func:
                                send_telegram_alert_func(
                                    f"✅ *FUNDING ARB CLOSED*\n"
                                    f"• Asset: {sym}\n"
                                    f"• Current Rate: {rate*100:.4f}%"
                                )
        except Exception as e:
            print(f"[Funding Arb] Error: {e}")
        time.sleep(300)


def run_daily_backup_scheduler():
    """
    Background scheduler that runs daily at 00:00 UTC.
    Calculates time to UTC midnight, sleeps, then creates a compressed zip file
    of trading_bot.db and trade_journal.csv, uploading to AWS S3 if credentials are set.
    """
    print("[Backup Scheduler] Daily database backup scheduler started.")
    import zipfile
    from database import DB_FILE
    
    while True:
        now = time.gmtime()
        seconds_to_midnight = 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
        time.sleep(max(1, seconds_to_midnight))
        
        try:
            print("[Backup Scheduler] Triggering daily backup...")
            backup_dir = "backups"
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            zip_filename = os.path.join(backup_dir, f"backup_{timestamp_str}.zip")
            
            current_db = database.get_db_path()
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(current_db):
                    zipf.write(current_db, os.path.basename(current_db))
                if os.path.exists(current_db + "-wal"):
                    zipf.write(current_db + "-wal", os.path.basename(current_db + "-wal"))
                if os.path.exists(current_db + "-shm"):
                    zipf.write(current_db + "-shm", os.path.basename(current_db + "-shm"))
                if os.path.exists(JOURNAL_PATH):
                    zipf.write(JOURNAL_PATH, os.path.basename(JOURNAL_PATH))
                    
            print(f"[Backup Scheduler] Created local compressed backup: {zip_filename}")
            
            s3_bucket = os.environ.get("AWS_S3_BUCKET")
            if s3_bucket:
                try:
                    import boto3
                    s3_client = boto3.client('s3')
                    s3_key = f"backups/{os.path.basename(zip_filename)}"
                    s3_client.upload_file(zip_filename, s3_bucket, s3_key)
                    print(f"[Backup Scheduler] Successfully uploaded backup to S3: s3://{s3_bucket}/{s3_key}")
                except Exception as s3_err:
                    print(f"[Backup Scheduler Warning] S3 upload failed (boto3 or credentials missing): {s3_err}")
            
            # Prune local backups older than 14 days to prevent disk space leaks
            now_sec = time.time()
            for old_file in os.listdir(backup_dir):
                if old_file.endswith(".zip"):
                    file_path = os.path.join(backup_dir, old_file)
                    if (now_sec - os.path.getmtime(file_path)) > 14 * 86400:
                        try:
                            os.remove(file_path)
                            print(f"[Backup Scheduler] Pruned old backup: {old_file}")
                        except Exception:
                            pass
        except Exception as e:
            print(f"[Backup Scheduler Error] Daily backup failed: {e}")


def run_pain_feedback_verifier():
    """
    Background worker that runs hourly to verify whether closed pain trades hit TP within 24h post-exit.
    """
    print("[Pain Feedback Verifier] Hourly 24h post-exit verification scheduler started.")
    while True:
        try:
            time.sleep(3600)
            from data import get_history
            import pain_feedback
            pain_feedback.verify_pending_pain_trades(database_module=database, fetch_kline_func=get_history)
        except Exception as e:
            print(f"[Pain Feedback Verifier Error] Exception in verification loop: {e}")


def run_daily_summary_scheduler(bot_state=None, bot_state_lock=None, send_daily_summary_func=None):
    """
    Background scheduler that guarantees daily 00:00:00 UTC report execution.
    """
    print("[Daily Summary Scheduler] Dedicated 00:00 UTC summary report scheduler started.")
    while True:
        try:
            now_gm = time.gmtime()
            today_date_str = time.strftime("%Y-%m-%d", now_gm)
            seconds_to_midnight = 86400 - (now_gm.tm_hour * 3600 + now_gm.tm_min * 60 + now_gm.tm_sec)
            time.sleep(max(1, seconds_to_midnight))
            
            if bot_state and bot_state_lock:
                with bot_state_lock:
                    last_date = bot_state.get("last_daily_summary_date", "")
                    if last_date != today_date_str:
                        bot_state["last_daily_summary_date"] = today_date_str
                        print(f"[Daily Summary Scheduler] Midnight UTC detected ({today_date_str}). Sending daily summary...")
                        if send_daily_summary_func:
                            send_daily_summary_func()
            elif send_daily_summary_func:
                send_daily_summary_func()
        except Exception as e:
            print(f"[Daily Summary Scheduler Error] Exception in scheduler loop: {e}")
            time.sleep(60)


def run_rolling_retrain_scheduler(retrain_models_thread_func=None):
    """
    Background scheduler that runs weekly on Sundays at 00:00 UTC.
    Checks time every 15 minutes.
    """
    print("[Scheduler] Automated weekly Sunday retraining scheduler started.")
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            if now_utc.weekday() == 6 and now_utc.hour == 0 and now_utc.minute < 20:
                print(f"[Scheduler] Sunday 00:00 UTC detected. Triggering weekly model retraining...")
                if retrain_models_thread_func:
                    retrain_models_thread_func(is_manual=False)
                time.sleep(2700)
        except Exception as e:
            print(f"[Scheduler] Error in weekly retraining scheduler: {e}")
        
        time.sleep(900)


def run_statistical_governance_scheduler():
    """
    Periodic background scheduler that evaluates live trade distributions per interval slot.
    If a slot produces statistically confirmed negative expectancy (REJECT), it persists the slot
    to governance_denylist.json to automatically offline the model on next execution cycle.
    """
    print("[Scheduler] Automated statistical governance monitoring scheduler started.")
    intervals_to_monitor = ["15", "30", "60", "120", "240", "360"]
    while True:
        try:
            import database
            from statistical_validation import statistical_validation
            from train import _record_to_governance_denylist
            
            all_trades = database.get_completed_trades(limit=1000)
            if all_trades:
                for iv in intervals_to_monitor:
                    slot_trades = [t for t in all_trades if str(t.get("interval", "")) == iv]
                    n_trades = len(slot_trades)
                    if n_trades >= 5:
                        returns = [float(t.get("change_pct", t.get("pnl_pct", 0.0))) for t in slot_trades]
                        baseline_rets = [0.0] * n_trades
                        matrix_res = statistical_validation.calculate_governed_validation_matrix(
                            component_name=f"live_{iv}",
                            baseline_returns=baseline_rets,
                            component_returns=returns,
                            completed_trades=n_trades,
                            module_uuid=f"LIVE_{iv}",
                            num_trials=1
                        )
                        decision = matrix_res.get("governance", {}).get("decision")
                        if decision == "REJECT":
                            reasons_str = "; ".join(matrix_res.get("governance", {}).get("reasons", ["Statistical rejection"]))
                            print(f"[Statistical Governance Live Gate] Denylisting trending_{iv} due to statistical rejection: {reasons_str}")
                            _record_to_governance_denylist(f"trending_{iv}", reason=f"Live statistical rejection: {reasons_str}")
        except Exception as e:
            print(f"[Statistical Governance Scheduler Error] {e}")

        time.sleep(3600)  # Check hourly
