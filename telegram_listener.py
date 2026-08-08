"""
telegram_listener.py
---------------------
Standalone Telegram command listener and interactive report generator extracted from main.py.
Handles polling getUpdates, commands (/status, /balance, /trades, /openmanualtrade, /skipped, /regime, /help, etc.),
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
    return "ℹ️ Confluence report command deprecated."


def get_live_bybit_wallet_details():
    try:
        from bybit_client import bybit_get_request
        for acct_type in ["UNIFIED", "CONTRACT", "SPOT"]:
            res = bybit_get_request("/v5/account/wallet-balance", {"accountType": acct_type})
            if isinstance(res, dict) and res.get("retCode") == 0:
                list_data = res.get("result", {}).get("list", [])
                if list_data:
                    first = list_data[0]
                    total_eq = float(first.get("totalEquity") or first.get("totalWalletBalance") or 0.0)
                    total_wb = float(first.get("totalWalletBalance") or total_eq)
                    total_upl = float(first.get("totalPerpUPL") or 0.0)
                    coins = first.get("coin", [])
                    avail_bal = total_wb
                    used_margin = 0.0
                    if coins and isinstance(coins, list):
                        usdt_coin = next((c for c in coins if c.get("coin") == "USDT"), coins[0])
                        wb = float(usdt_coin.get("walletBalance") or total_wb)
                        eq = float(usdt_coin.get("equity") or total_eq)
                        pos_im = float(usdt_coin.get("totalPositionIM") or 0.0)
                        order_im = float(usdt_coin.get("totalOrderIM") or 0.0)
                        upl = float(usdt_coin.get("unrealisedPnl") or total_upl)
                        used_margin = pos_im + order_im
                        avail_bal = max(0.0, wb - used_margin)
                        return {
                            "wallet_balance": wb,
                            "equity": eq,
                            "available_balance": avail_bal,
                            "used_margin": used_margin,
                            "unrealized_pnl": upl,
                            "account_type": acct_type
                        }
                    return {
                        "wallet_balance": total_wb,
                        "equity": total_eq,
                        "available_balance": max(0.0, total_wb - used_margin),
                        "used_margin": used_margin,
                        "unrealized_pnl": total_upl,
                        "account_type": acct_type
                    }
    except Exception as ex:
        print(f"[Balance Detail Error] {ex}")
    return None


def get_live_trades_report(bot_state=None, bot_state_lock=None):
    from bybit_client import bybit_get_request
    
    active_map = {}
    if bot_state and bot_state_lock:
        with bot_state_lock:
            for tf in ["15m", "30m", "1h", "2h", "4h", "6h"]:
                t_list = bot_state.get(f"active_trade_{tf}", [])
                if isinstance(t_list, list):
                    for t in t_list:
                        sym = t.get("symbol")
                        if sym:
                            active_map[sym] = (tf, t)

    pos_res = bybit_get_request("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
    bybit_positions = []
    if isinstance(pos_res, dict) and pos_res.get("retCode") == 0:
        raw_list = pos_res.get("result", {}).get("list", [])
        for p in raw_list:
            if float(p.get("size", "0") or 0.0) > 0:
                bybit_positions.append(p)

    if not bybit_positions and not active_map:
        return "📈 *ACTIVE OPEN POSITIONS*\n\nℹ️ *No active positions open currently.*"

    report_lines = [f"📈 *ACTIVE OPEN POSITIONS ({len(bybit_positions)})*\n"]

    for pos in bybit_positions:
        sym = pos.get("symbol", "N/A")
        side_raw = pos.get("side", "Buy")
        direction = "Bullish (Long)" if side_raw in ["Buy", "Long"] else "Bearish (Short)"
        size_qty = float(pos.get("size", 0.0))
        entry_p = float(pos.get("avgPrice", 0.0) or 0.0)
        mark_p = float(pos.get("markPrice", 0.0) or entry_p)
        sl_p = float(pos.get("stopLoss", 0.0) or 0.0)
        tp_p = float(pos.get("takeProfit", 0.0) or 0.0)
        upnl = float(pos.get("unrealisedPnl", 0.0) or 0.0)
        lev = float(pos.get("leverage", 1.0) or 1.0)
        pos_val_usd = size_qty * mark_p
        margin_usd = pos_val_usd / lev if lev > 0 else pos_val_usd
        upnl_pct = (upnl / margin_usd * 100.0) if margin_usd > 0 else 0.0

        tf_str = "1h"
        if sym in active_map:
            tf_str = active_map[sym][0]

        pnl_icon = "🟢" if upnl >= 0 else "🔴"
        sl_str = f"${sl_p:.4f}" if sl_p > 0 else "None"
        tp_str = f"${tp_p:.4f}" if tp_p > 0 else "None"

        block = (
            f"{pnl_icon} *{sym}* ({tf_str}) | *{direction}*\n"
            f"• *Entry*: `${entry_p:.4f}` | *Mark*: `${mark_p:.4f}`\n"
            f"• *Stop Loss*: `{sl_str}` | *Take Profit*: `{tp_str}`\n"
            f"• *Leverage*: `{lev:.1f}x` | *Margin*: `${margin_usd:.2f}` (Val: `${pos_val_usd:.2f}`)\n"
            f"• *Live Unrealized PnL*: *${upnl:+.2f} USDT* (`{upnl_pct:+.2f}%`)\n"
        )
        report_lines.append(block)

    return "\n".join(report_lines)


def execute_manual_trade(symbol, interval, direction, bot_state=None, bot_state_lock=None):
    try:
        from data import get_history
        from core import add_features
        from bybit_client import place_bybit_taker_ioc_order, format_bybit_qty, set_bybit_leverage, get_bybit_min_qty_step
        from trade_calculators import assert_valid_geometry
        import uuid

        sym = symbol.upper().strip()
        if not sym.endswith("USDT"):
            sym += "USDT"

        iv = str(interval).replace("m", "").replace("h", "")
        if iv == "1":
            iv = "60"
        elif iv == "2":
            iv = "120"
        elif iv == "4":
            iv = "240"
        elif iv == "6":
            iv = "360"

        if iv not in ["15", "30", "60", "120", "240", "360"]:
            return f"❌ Invalid timeframe: `{interval}`. Supported: 15, 30, 60 (1h), 120 (2h), 240 (4h)."

        dir_clean = str(direction).strip().title()
        if dir_clean in ["Bullish", "Long", "Buy"]:
            ml_trend = "Bullish"
            side = "Buy"
        elif dir_clean in ["Bearish", "Short", "Sell"]:
            ml_trend = "Bearish"
            side = "Sell"
        else:
            return f"❌ Invalid direction: `{direction}`. Supported: Bullish/Long/Buy or Bearish/Short/Sell."

        df_raw = get_history(symbol=sym, interval=iv, limit=100)
        if df_raw is None or len(df_raw) < 10:
            return f"❌ Failed to fetch market data for `{sym}` ({iv}m)."

        df = add_features(df_raw)
        latest = df.iloc[-1]
        entry_price = float(latest["close"])
        atr_dollars = float(latest.get("ATR_norm", 0.005) * entry_price)

        if ml_trend == "Bullish":
            stop_loss_price = entry_price - (1.0 * atr_dollars)
            take_profit_price = entry_price + (2.0 * atr_dollars)
        else:
            stop_loss_price = entry_price + (1.0 * atr_dollars)
            take_profit_price = entry_price - (2.0 * atr_dollars)

        assert_valid_geometry(ml_trend, entry_price, stop_loss_price, take_profit_price, symbol=sym)

        leverage_val = 5.0
        position_size_usd = 2.0
        set_bybit_leverage(sym, leverage_val)

        leveraged_size = position_size_usd * leverage_val
        raw_qty = leveraged_size / entry_price
        qty_str = format_bybit_qty(sym, raw_qty)
        qty_val = float(qty_str)

        if qty_val * entry_price < 5.1:
            step = get_bybit_min_qty_step(sym)
            import math
            if step > 0:
                qty_val = math.ceil(5.1 / (entry_price * step)) * step
                qty_str = format_bybit_qty(sym, qty_val)

        order_res = place_bybit_taker_ioc_order(sym, side, qty_str, sl=stop_loss_price, tp=take_profit_price)
        if order_res.get("retCode") == 0:
            bybit_order_id = order_res.get("result", {}).get("orderId")
            
            tf_map_inv = {"15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h", "360": "6h"}
            tf_key = tf_map_inv.get(iv, f"{iv}m")
            trade_uuid = str(uuid.uuid4())
            trade_record = {
                "trade_id": f"{sym}_{trade_uuid}",
                "bybit_order_id": bybit_order_id,
                "symbol": sym,
                "direction": ml_trend,
                "entry_price": entry_price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "leverage": leverage_val,
                "position_size_usd": position_size_usd,
                "qty": float(qty_str),
                "entry_time": int(time.time() * 1000),
                "end_time": time.time() + (int(iv) * 60 * 10),
                "atr_dollars": atr_dollars
            }
            if bot_state and bot_state_lock:
                with bot_state_lock:
                    key = f"active_trade_{tf_key}"
                    curr_list = bot_state.get(key, [])
                    if not isinstance(curr_list, list):
                        curr_list = []
                    curr_list.append(trade_record)
                    bot_state[key] = curr_list

            return (
                f"🟢 *MANUAL TRADE EXECUTED SUCCESSFULLY*\n\n"
                f"• *Asset*: `{sym}` ({tf_key})\n"
                f"• *Direction*: `{ml_trend}`\n"
                f"• *Entry Price*: `${entry_price:.4f}`\n"
                f"• *Stop Loss*: `${stop_loss_price:.4f}`\n"
                f"• *Take Profit*: `${take_profit_price:.4f}`\n"
                f"• *Leverage*: `{leverage_val:.1f}x` | *Margin*: `${position_size_usd:.2f}`\n"
                f"• *Bybit Order ID*: `{bybit_order_id}`"
            )
        else:
            return f"🔴 *Manual Trade Execution Failed*: {order_res.get('retMsg')}"

    except Exception as ex:
        return f"🔴 *Manual Trade Error*: {str(ex)}"


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
                {"command": "balance", "description": "Live Bybit wallet balance, margin & equity"},
                {"command": "trades", "description": "View all open active trades with live PnL"},
                {"command": "skipped", "description": "View skipped trades for 15m/30m/1h/2h"},
                {"command": "regime", "description": "Current market regime & volatility breakdown"},
                {"command": "openmanualtrade", "description": "Execute manual trade (e.g. /openmanualtrade BTC 15 Long)"},
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
                        bal_info = get_live_bybit_wallet_details()
                        if bal_info:
                            wb = bal_info["wallet_balance"]
                            eq = bal_info["equity"]
                            avail = bal_info["available_balance"]
                            used = bal_info["used_margin"]
                            upl = bal_info["unrealized_pnl"]
                            pnl_icon = "🟢" if upl >= 0 else "🔴"
                            bal_text = (
                                f"💰 *REAL LIVE BYBIT WALLET BALANCE*\n\n"
                                f"• *Total Wallet Balance*: `${wb:.2f} USDT`\n"
                                f"• *Account Equity*: `${eq:.2f} USDT`\n"
                                f"• *Available Margin*: `${avail:.2f} USDT`\n"
                                f"• *Used Position Margin*: `${used:.2f} USDT`\n"
                                f"• *Live Unrealized PnL*: {pnl_icon} `${upl:+.4f} USDT`\n"
                                f"• *Account Type*: `{bal_info['account_type']}`"
                            )
                        else:
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
                        trades_text = get_live_trades_report(bot_state=bot_state, bot_state_lock=bot_state_lock)
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": trades_text, "parse_mode": "Markdown"})

                    elif cmd in ["openmanualtrade", "manualtrade"]:
                        if not args or len(args) < 3:
                            usage_text = (
                                "⚠️ *Usage for Manual Trade Execution*:\n"
                                "`/openmanualtrade <SYMBOL> <TIMEFRAME> <DIRECTION>`\n\n"
                                "*Examples*:\n"
                                "• `/openmanualtrade BTC 15 Bullish`\n"
                                "• `/openmanualtrade ETH 60 Bearish`\n"
                                "• `/openmanualtrade SOL 30 Long`"
                            )
                            execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": usage_text, "parse_mode": "Markdown"})
                        else:
                            m_sym = args[0]
                            m_tf = args[1]
                            m_dir = args[2]
                            res_msg = execute_manual_trade(m_sym, m_tf, m_dir, bot_state=bot_state, bot_state_lock=bot_state_lock)
                            execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": res_msg, "parse_mode": "Markdown"})

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
                            "• `/balance` - Real live Bybit wallet balance, margin & equity\n"
                            "• `/trades` - Active open positions with live PnL\n"
                            "• `/openmanualtrade [sym] [tf] [dir]` - Execute manual trade (e.g. `/openmanualtrade BTC 15 Long`)\n"
                            "• `/skipped [tf]` - View skipped signals (e.g. `/skipped 30`)\n"
                            "• `/regime` - Current market regime & sentiment\n"
                            "• `/start` - Start automatic trading\n"
                            "• `/stop` - Pause automatic trading\n"
                            "• `/kill` - Emergency halt trading\n"
                        )
                        execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": help_text, "parse_mode": "Markdown"})

            except Exception as e:
                print(f"[Telegram Listener Error] {e}")
                time.sleep(3)

    threading.Thread(target=listener_loop, daemon=True).start()
