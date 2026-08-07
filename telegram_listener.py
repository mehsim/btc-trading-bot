"""
telegram_listener.py
---------------------
Standalone Telegram command listener and interactive report generator extracted from main.py.
Handles polling getUpdates, commands (/status, /balance, /trades, /skipped, /confluence, /help, etc.),
and authenticating users.
"""

import os
import time
import json
import threading
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd

from logger import log_event
from telegram_bot import execute_telegram_api_call, send_telegram_alert


def run_manual_confluence_report(symbol, interval, bot_state=None, bot_state_lock=None):
    try:
        from data import get_history, merge_derivatives_sentiment_features, classify_market_regime
        from core import add_features, features
        from confluence_engine import check_pre_trade_confluence
        
        iv = str(interval)
        df_raw = get_history(symbol=symbol, interval=iv, limit=300)
        if df_raw is None or len(df_raw) < 2:
            return f"❌ Failed to fetch price history from Bybit/Binance/Kraken for *{symbol}*."
            
        # Ensure close_btc is populated (required for features calculations)
        if symbol == "BTCUSDT":
            df_raw["close_btc"] = df_raw["close"]
        else:
            df_btc = get_history(symbol="BTCUSDT", interval=iv, limit=300)
            if df_btc is not None and len(df_btc) > 0:
                df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                df_raw = pd.merge(df_raw, df_btc_sub, on="timestamp", how="left")
                df_raw["close_btc"] = df_raw["close_btc"].ffill().bfill().fillna(df_raw["close"])
            else:
                df_raw["close_btc"] = df_raw["close"]

        df_raw = merge_derivatives_sentiment_features(df_raw, symbol=symbol, interval=iv)
        df_features = add_features(df_raw)
        latest_candle = df_features.iloc[-1]
        
        models_by_interval = bot_state.get("models_by_interval", {}) if bot_state else {}
        models_tf = models_by_interval.get(iv)
        if not models_tf or not (models_tf.get("trending", {}).get("price") or models_tf.get("ranging", {}).get("price")):
            return "❌ Models are currently not fully loaded or active."
            
        # Unsupervised GMM Market Regime Classification
        regime = classify_market_regime(df_features, interval=iv)
        regime_key = regime.lower() if regime in ["Trending", "Ranging"] else "trending"
        m_price = models_tf.get(regime_key, {}).get("price")
        m_trend = models_tf.get(regime_key, {}).get("trend")
        calibrator = models_tf.get(regime_key, {}).get("calibrator")
        feat_list = models_tf.get(f"selected_features_{regime_key}") or models_tf.get("selected_features")

        if m_price is None or m_trend is None or not feat_list:
            alt_key = "ranging" if regime_key == "trending" else "trending"
            alt_price = models_tf.get(alt_key, {}).get("price")
            alt_trend = models_tf.get(alt_key, {}).get("trend")
            alt_cal = models_tf.get(alt_key, {}).get("calibrator")
            alt_feats = models_tf.get(f"selected_features_{alt_key}") or models_tf.get("selected_features")
            if alt_price is not None and alt_trend is not None and alt_feats:
                m_price, m_trend, calibrator, feat_list = alt_price, alt_trend, alt_cal, alt_feats
            else:
                return "❌ Models are currently not fully loaded or active."

        active_model_price = m_price
        active_model_trend = m_trend
            
        if feat_list is not None:
            X_live = latest_candle[feat_list].values.reshape(1, -1)
        else:
            X_live = latest_candle[features].values.reshape(1, -1)
            
        ensemble_weights = [0.3, 0.2, 0.5] if regime == "Trending" else [0.3, 0.5, 0.2]
        pred_pct = float(active_model_price.predict(X_live, weights=ensemble_weights)[0])
        pred_change = pred_pct * float(latest_candle["close"])
        expected_pct_change = (abs(pred_change) / latest_candle["close"]) * 100
        probs = active_model_trend.predict_proba(X_live, weights=ensemble_weights)[0]
        if len(probs) >= 3:
            prob_bearish = float(probs[0])
            prob_neutral = float(probs[1])
            prob_bullish = float(probs[2])
        elif len(probs) == 2:
            prob_bearish = float(probs[0])
            prob_neutral = 0.0
            prob_bullish = float(probs[1])
        else:
            prob_bearish = float(probs[0]) if float(probs[0]) < 0.5 else 0.0
            prob_neutral = 0.0
            prob_bullish = float(probs[0]) if float(probs[0]) >= 0.5 else 0.0

        if prob_bullish > max(prob_bearish, prob_neutral) and prob_bullish >= 0.50:
            ml_trend = "Bullish"
            ml_confidence = prob_bullish
        elif prob_bearish > max(prob_bullish, prob_neutral) and prob_bearish >= 0.50:
            ml_trend = "Bearish"
            ml_confidence = prob_bearish
        else:
            ml_trend = "Neutral"
            ml_confidence = max(prob_bullish, prob_bearish, prob_neutral)
            
        if calibrator is not None and "X" in calibrator and "y" in calibrator and ml_trend in ["Bullish", "Bearish"]:
            calibrated_confidence = float(np.interp(ml_confidence, calibrator["X"], calibrator["y"]))
        else:
            calibrated_confidence = float(np.clip(ml_confidence, 0.0, 1.0))
            
        news_sentiment = bot_state.get("news_sentiment", "Neutral") if bot_state else "Neutral"
            
        all_pass, confluence_results, confluence_score_pct = check_pre_trade_confluence(
            latest_candle["close"], df_features, ml_trend, news_sentiment, expected_pct_change, iv, symbol=symbol,
            calibrated_confidence=calibrated_confidence, dynamic_conf_threshold=0.58, get_history_fn=get_history
        )
        
        adx_regime = float(latest_candle.get("ADX", 0.0))
        atr_val = latest_candle.get("ATR_norm", 0.005) * latest_candle["close"]
        est_tp_val = latest_candle["close"] + (1.5 * atr_val if ml_trend == "Bullish" else -1.5 * atr_val)
        
        report = (
            f"🔍 *CONFLUENCE REPORT: {symbol} ({iv.replace('60','1H').replace('120','2H').replace('240','4H').replace('360','6H')})*\n"
            f"• *Signal*: {ml_trend} ({calibrated_confidence*100:.1f}% confidence)\n"
            f"• *Regime*: {regime} (ADX: {adx_regime:.1f})\n"
            f"• *Expected Move*: {pred_change:+.4f} ({expected_pct_change:.2f}%)\n"
            f"• *Liquidation TP Target*: ${est_tp_val:.2f}\n"
            f"• *Decision*: *{'APPROVED' if all_pass else 'REJECTED'}*\n\n"
            f"*Check Details:*\n"
        )
        for idx, (check_name, res_val) in enumerate(confluence_results.items(), 1):
            if check_name == "_Score_Summary" or not isinstance(res_val, dict):
                continue
            circle = "🟢" if res_val.get("pass", False) else "🔴"
            detail_str = res_val.get("detail", "")
            report += f"{circle} *{check_name.replace('_', ' ')}*: {detail_str}\n"
            
        report += f"\n📊 *{confluence_results.get('_Score_Summary', {}).get('detail', '')}*"
        return report
    except Exception as e:
        return f"❌ *Error running manual check:* {str(e)}"


def start_telegram_command_listener(bot_state, bot_state_lock):
    """Starts the background thread to poll and handle incoming Telegram commands."""
    from secret_manager import get_secure_env
    token = get_secure_env("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
    raw_chat_id = get_secure_env("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not raw_chat_id:
        print("[Telegram Command Listener] Unconfigured credentials. Listener skipped.")
        return
        
    allowed_chat_ids = [cid.strip() for cid in raw_chat_id.split(",") if cid.strip()]

    with bot_state_lock:
        dyn_list = bot_state.get("telegram_allowed_ids", [])
        for dyn_id in dyn_list:
            if dyn_id not in allowed_chat_ids:
                allowed_chat_ids.append(dyn_id)

    pending_auth = {}
    pending_confluence = {}
    pending_manual_trade = {}
    pending_skipped = {}

    TF_MAP_SKIPPED = {
        "15": "15", "15m": "15", "15min": "15", "15-min": "15", "15 min": "15",
        "30": "30", "30m": "30", "30min": "30", "30-min": "30", "30 min": "30",
        "60": "60", "1h": "60", "1hr": "60", "1-hr": "60", "1 hour": "60",
        "120": "120", "2h": "120", "2hr": "120", "2-hr": "120", "2 hour": "120"
    }
    TF_DISPLAY = {
        "15": "15m",
        "30": "30m",
        "60": "1h",
        "120": "2h"
    }

    def get_skipped_trades_report(target_iv):
        tf_disp = TF_DISPLAY.get(str(target_iv), f"{target_iv}m")
        with bot_state_lock:
            all_preds = list(bot_state.get("prediction_history", []))
            
            tf_preds = [
                p for p in all_preds 
                if str(p.get("interval", "")).replace("m", "") == str(target_iv).replace("m", "")
            ]
            if len(tf_preds) < 10:
                try:
                    import database
                    db_preds = database.get_prediction_history(limit=500)
                    if db_preds:
                        existing_keys = {(p.get("symbol"), p.get("candle_timestamp"), str(p.get("interval"))) for p in all_preds}
                        for p in db_preds:
                            k = (p.get("symbol"), p.get("candle_timestamp"), str(p.get("interval")))
                            if k not in existing_keys:
                                all_preds.append(p)
                                existing_keys.add(k)
                        tf_preds = [
                            p for p in all_preds 
                            if str(p.get("interval", "")).replace("m", "") == str(target_iv).replace("m", "")
                        ]
                except Exception as e:
                    print(f"[Skipped Report] DB fallback error: {e}")
            
            if not tf_preds:
                return f"ℹ️ *No prediction history found for the {tf_disp} timeframe.*"
                
            def _norm_ms(val):
                if val is None:
                    return 0
                try:
                    v = float(val)
                    return int(v * 1000) if v < 1e11 else int(v)
                except Exception:
                    return 0

            for p in tf_preds:
                if isinstance(p, dict):
                    raw_c_ts = p.get("candle_timestamp") or p.get("timestamp")
                    if raw_c_ts is not None:
                        p["candle_timestamp"] = _norm_ms(raw_c_ts)

            candle_timestamps = [p.get("candle_timestamp") for p in tf_preds if p.get("candle_timestamp") and p.get("candle_timestamp") > 0]
            if not candle_timestamps:
                return f"ℹ️ *No candle timestamp data found for the {tf_disp} timeframe.*"
                
            latest_candle_ts = max(candle_timestamps)
            candle_dt_str = datetime.fromtimestamp(latest_candle_ts / 1000.0, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            
            def _is_skipped_trade(p):
                st = str(p.get("status", ""))
                dir_str = str(p.get("direction", ""))
                if st.startswith("Skipped (") and st not in ["Skipped (Neutral)", "Skipped (Bot Stopped)"]:
                    return True
                if dir_str in ["Bullish", "Bearish"] and (st.startswith("Evaluated (") or st.startswith("Fallback (")) and not p.get("trade_executed", False):
                    return True
                return False

            latest_skipped = [
                p for p in tf_preds 
                if _norm_ms(p.get("candle_timestamp")) == latest_candle_ts 
                and _is_skipped_trade(p)
            ]
            
            is_previous_candle = False
            if not latest_skipped:
                two_hours_ms = 2 * 3600 * 1000
                recent_skipped_preds = [
                    p for p in tf_preds 
                    if _is_skipped_trade(p)
                    and (latest_candle_ts - _norm_ms(p.get("candle_timestamp"))) <= two_hours_ms
                ]
                if recent_skipped_preds:
                    latest_skipped_ts = max(_norm_ms(p.get("candle_timestamp")) for p in recent_skipped_preds)
                    latest_skipped = [p for p in recent_skipped_preds if _norm_ms(p.get("candle_timestamp")) == latest_skipped_ts]
                    candle_dt_str = datetime.fromtimestamp(latest_skipped_ts / 1000.0, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    is_previous_candle = True
                    
            if not latest_skipped:
                return (
                    f"🚫 *SKIPPED TRADES — {tf_disp.upper()} TIMEFRAME* 🚫\n\n"
                    f"📅 *Latest Opened Candle*: `{candle_dt_str}`\n"
                    f"ℹ️ *No skipped trades logged on the latest open candle.*\n\n"
                    f"_All signals on this timeframe were either Neutral or executed successfully._"
                )
                
            header_note = " (Previous Candle with Skipped Signals)" if is_previous_candle else " (Latest Opened Candle)"
            report_lines = [
                f"🚫 *SKIPPED TRADES — {tf_disp.upper()} TIMEFRAME* 🚫\n",
                f"📅 *Candle Time*: `{candle_dt_str}`{header_note}\n"
            ]
            
            for p in latest_skipped:
                symbol = p.get("symbol", "N/A")
                direction = p.get("direction", "N/A")
                raw_status = str(p.get("status", ""))
                status = raw_status.replace("Skipped (", "").replace("Evaluated (", "").rstrip(")").strip()
                conf = p.get("calibrated_confidence", 0.0)
                ref_p = p.get("ref_price", 0.0)
                thresh = p.get("dynamic_threshold", 0.60)
                
                detail_line = f"• *{symbol}* | Signal: *{direction}*\n"
                detail_line += f"  - *Reason*: `{status.upper()}`\n"
                detail_line += f"  - *Confidence*: `{conf*100:.1f}%` (Threshold: `{thresh*100:.1f}%`)\n"
                if ref_p > 0:
                    detail_line += f"  - *Price at Evaluation*: `${ref_p:.2f}`\n"
                    
                report_lines.append(detail_line)
                
            return "\n".join(report_lines)

    def listener_loop():
        offset = 0
        init_res = execute_telegram_api_call("getUpdates", {"limit": 1})
        if init_res and init_res.get("ok") and init_res.get("result"):
            offset = init_res["result"][-1]["update_id"] + 1

        commands_payload = {
            "commands": [
                {"command": "status", "description": "Current bot status, balance & active regime"},
                {"command": "balance", "description": "Wallet balance, margin used & available funds"},
                {"command": "trades", "description": "View all open active trades across timeframes"},
                {"command": "history", "description": "View recent completed trades PnL summary"},
                {"command": "performance", "description": "Win rate, total PnL & Sharpe metrics"},
                {"command": "skipped", "description": "View skipped trades for 15m/30m/1h/2h"},
                {"command": "confluence", "description": "Run manual confluence report for BTC/ETH"},
                {"command": "regime", "description": "Current market regime & volatility breakdown"},
                {"command": "start", "description": "Start automatic trading bot"},
                {"command": "stop", "description": "Pause automatic trading bot"},
                {"command": "kill", "description": "Emergency kill switch - halt & close all positions"},
                {"command": "auth", "description": "Authenticate session with pin code"},
                {"command": "help", "description": "List all interactive bot commands"}
            ]
        }
        execute_telegram_api_call("setMyCommands", commands_payload)

        print(f"[Telegram Command Listener] Active. Monitoring authorized chats: {allowed_chat_ids}")

        while True:
            try:
                updates_res = execute_telegram_api_call("getUpdates", {"offset": offset, "timeout": 5})
                if not updates_res or not updates_res.get("ok"):
                    time.sleep(3)
                    continue

                for update in updates_res.get("result", []):
                    offset = update["update_id"] + 1
                    message = update.get("message") or update.get("edited_message") or update.get("channel_post")
                    if not message:
                        continue

                    chat_id = str(message.get("chat", {}).get("id"))
                    text = message.get("text", "").strip()

                    if chat_id not in allowed_chat_ids and len(allowed_chat_ids) > 0:
                        print(f"[Telegram Security] Unauthorized command attempt from chat_id {chat_id}: '{text}'")
                        execute_telegram_api_call("sendMessage", {
                            "chat_id": chat_id,
                            "text": f"⛔ *Access Denied*\nYour chat ID ({chat_id}) is not authorized to interact with this trading bot.",
                            "parse_mode": "Markdown"
                        })
                        continue

                    if not text:
                        continue

                    parts = text.split()
                    raw_cmd = parts[0] if parts else ""
                    cmd = raw_cmd.split("@")[0].lower().lstrip("/")
                    args = parts[1:] if len(parts) > 1 else []

                    if cmd in ["status"]:
                        with bot_state_lock:
                            running = bot_state.get("bot_running", True)
                            bal = bot_state.get("wallet_balance", 0.0)
                            active_trades = sum(len(bot_state.get(f"active_trade_{tf}", [])) for tf in ["15m", "30m", "1h", "2h", "4h", "6h"])
                            regime_15 = bot_state.get("regime_15m", "N/A")
                            regime_1h = bot_state.get("regime_1h", "N/A")
                            cb_active = bot_state.get("circuit_breaker_active", False)

                        status_text = (
                            f"🤖 *BTC TRADING BOT STATUS*\n\n"
                            f"• *Status*: {'🟢 RUNNING' if running else '🔴 PAUSED'}\n"
                            f"• *Wallet Balance*: `${bal:.2f} USDT`\n"
                            f"• *Active Trades*: `{active_trades}`\n"
                            f"• *15m Regime*: `{regime_15}`\n"
                            f"• *1h Regime*: `{regime_1h}`\n"
                            f"• *Circuit Breaker*: {'🚨 ACTIVE' if cb_active else '✅ NORMAL'}\n"
                        )
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": status_text, "parse_mode": "Markdown"})

                    elif cmd in ["balance"]:
                        with bot_state_lock:
                            bal = bot_state.get("wallet_balance", 0.0)
                            avail = bot_state.get("available_balance", bal)
                        bal_text = (
                            f"💰 *WALLET BALANCE REPORT*\n\n"
                            f"• *Total Wallet Balance*: `${bal:.2f} USDT`\n"
                            f"• *Available Margin*: `${avail:.2f} USDT`\n"
                        )
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": bal_text, "parse_mode": "Markdown"})

                    elif cmd in ["trades", "positions"]:
                        trade_lines = ["📈 *ACTIVE OPEN POSITIONS*\n"]
                        total_trades = 0
                        with bot_state_lock:
                            for tf in ["15m", "30m", "1h", "2h", "4h", "6h"]:
                                t_list = bot_state.get(f"active_trade_{tf}", [])
                                for t in t_list:
                                    total_trades += 1
                                    sym = t.get("symbol", "BTCUSDT")
                                    d = t.get("direction", "BUY")
                                    ep = t.get("entry_price", 0.0)
                                    trade_lines.append(f"• *{sym}* ({tf}) | {d} @ `${ep:.2f}`")
                        if total_trades == 0:
                            trade_lines.append("ℹ️ *No active positions open currently.*")
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": "\n".join(trade_lines), "parse_mode": "Markdown"})

                    elif cmd in ["history"]:
                        try:
                            import database
                            c_trades = database.get_completed_trades(limit=10)
                            if not c_trades:
                                hist_text = "ℹ️ *No completed trades in database history.*"
                            else:
                                hist_lines = ["📜 *RECENT COMPLETED TRADES*\n"]
                                for t in c_trades:
                                    sym = t.get("symbol", "BTCUSDT")
                                    d = t.get("direction", "BUY")
                                    pnl = float(t.get("pnl", 0.0) or 0.0)
                                    pnl_pct = float(t.get("pnl_pct", 0.0) or 0.0)
                                    reason = t.get("reason", "TP/SL")
                                    icon = "🟢" if pnl >= 0 else "🔴"
                                    hist_lines.append(f"• {icon} *{sym}* ({d}) | PnL: `${pnl:+.2f}` (`{pnl_pct:+.2f}%`) | `{reason}`")
                                hist_text = "\n".join(hist_lines)
                        except Exception as ex:
                            hist_text = f"❌ *Error fetching trade history:* {ex}"
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": hist_text, "parse_mode": "Markdown"})

                    elif cmd in ["performance"]:
                        try:
                            import database
                            c_trades = database.get_completed_trades(limit=500)
                            if not c_trades:
                                perf_text = "ℹ️ *No completed trades found to calculate performance.*"
                            else:
                                total_cnt = len(c_trades)
                                wins = [t for t in c_trades if float(t.get("pnl", 0.0) or 0.0) > 0]
                                losses = [t for t in c_trades if float(t.get("pnl", 0.0) or 0.0) <= 0]
                                win_rate = (len(wins) / total_cnt) * 100.0 if total_cnt > 0 else 0.0
                                total_pnl = sum(float(t.get("pnl", 0.0) or 0.0) for t in c_trades)
                                total_win_pnl = sum(float(t.get("pnl", 0.0) or 0.0) for t in wins)
                                total_loss_pnl = abs(sum(float(t.get("pnl", 0.0) or 0.0) for t in losses))
                                profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else total_win_pnl
                                perf_text = (
                                    f"📊 *PERFORMANCE METRICS REPORT*\n\n"
                                    f"• *Total Completed Trades*: `{total_cnt}`\n"
                                    f"• *Win Rate*: `{win_rate:.1f}%` ({len(wins)}W / {len(losses)}L)\n"
                                    f"• *Total Realized PnL*: `${total_pnl:+.2f} USDT`\n"
                                    f"• *Profit Factor*: `{profit_factor:.2f}`\n"
                                )
                        except Exception as ex:
                            perf_text = f"❌ *Error calculating performance:* {ex}"
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": perf_text, "parse_mode": "Markdown"})

                    elif cmd in ["regime"]:
                        with bot_state_lock:
                            r15 = bot_state.get("regime_15m", "N/A")
                            r30 = bot_state.get("regime_30m", "N/A")
                            r1h = bot_state.get("regime_1h", "N/A")
                            sent = bot_state.get("news_sentiment", "Neutral")
                        reg_text = (
                            f"🌐 *MARKET REGIME & VOLATILITY REPORT*\n\n"
                            f"• *15m Regime*: `{r15}`\n"
                            f"• *30m Regime*: `{r30}`\n"
                            f"• *1h Regime*: `{r1h}`\n"
                            f"• *News Sentiment*: `{sent}`\n"
                        )
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": reg_text, "parse_mode": "Markdown"})

                    elif cmd in ["skipped"]:
                        target_tf = "15"
                        if args:
                            raw_arg = args[0].lower().replace("m", "").replace("h", "")
                            target_tf = TF_MAP_SKIPPED.get(args[0].lower(), TF_MAP_SKIPPED.get(raw_arg, "15"))
                        skipped_msg = get_skipped_trades_report(target_tf)
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": skipped_msg, "parse_mode": "Markdown"})

                    elif cmd in ["confluence"]:
                        target_sym = "BTCUSDT"
                        target_tf = "15"
                        if args:
                            for a in args:
                                a_clean = a.upper().strip()
                                if a_clean in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT", "BTC", "ETH", "SOL"]:
                                    target_sym = a_clean if a_clean.endswith("USDT") else f"{a_clean}USDT"
                                elif a.lower() in TF_MAP_SKIPPED:
                                    target_tf = TF_MAP_SKIPPED[a.lower()]
                        conf_msg = run_manual_confluence_report(target_sym, target_tf, bot_state=bot_state, bot_state_lock=bot_state_lock)
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": conf_msg, "parse_mode": "Markdown"})

                    elif cmd in ["start"]:
                        with bot_state_lock:
                            bot_state["bot_running"] = True
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": "🟢 *Trading Bot Resumed.* Automatic trade execution enabled.", "parse_mode": "Markdown"})

                    elif cmd in ["stop", "pause"]:
                        with bot_state_lock:
                            bot_state["bot_running"] = False
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": "🔴 *Trading Bot Paused.* Automatic trade execution suspended.", "parse_mode": "Markdown"})

                    elif cmd in ["kill"]:
                        with bot_state_lock:
                            bot_state["bot_running"] = False
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": "🚨 *EMERGENCY KILL SWITCH ENGAGED.* Automatic trading suspended.", "parse_mode": "Markdown"})

                    elif cmd in ["help"]:
                        help_text = (
                            "📖 *BTC TRADING BOT COMMANDS*\n\n"
                            "• `/status` - Bot running state & active regime\n"
                            "• `/balance` - Wallet balance & available margin\n"
                            "• `/trades` - Active open positions\n"
                            "• `/history` - Recent completed trades PnL summary\n"
                            "• `/performance` - Win rate & total PnL report\n"
                            "• `/regime` - Current market regime & sentiment\n"
                            "• `/skipped [tf]` - View skipped signals (e.g. `/skipped 30`)\n"
                            "• `/confluence [sym] [tf]` - Run confluence check (e.g. `/confluence ETH 30`)\n"
                            "• `/start` - Start automatic trading\n"
                            "• `/stop` - Pause automatic trading\n"
                            "• `/kill` - Emergency halt trading\n"
                        )
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": help_text, "parse_mode": "Markdown"})

            except Exception as e:
                print(f"[Telegram Listener Error] {e}")
                time.sleep(3)

    threading.Thread(target=listener_loop, daemon=True).start()
