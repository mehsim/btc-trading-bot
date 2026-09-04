"""
telegram_listener.py
---------------------
Standalone Telegram command listener and interactive report generator extracted from main.py.
Handles polling getUpdates, commands (/status, /balance, /trades, /openmanualtrade, /skipped, /regime, /help, etc.),
inline keyboards, parameter auto-suggestions, and authenticating users.
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
    import database
    
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

    pos_res = bybit_get_request("/v5/position/list", {"category": "linear", "settleCoin": "USDT", "limit": 50})
    bybit_positions = []
    if isinstance(pos_res, dict) and pos_res.get("retCode") == 0:
        raw_list = pos_res.get("result", {}).get("list", [])
        for p in raw_list:
            if float(p.get("size", "0") or 0.0) > 0:
                bybit_positions.append(p)

    if not bybit_positions:
        # Fallback to database active trades
        try:
            db_trades = database.get_active_trades()
            if db_trades:
                report_lines = [f"📈 *ACTIVE OPEN POSITIONS ({len(db_trades)})*\n"]
                for t in db_trades:
                    sym = t.get("symbol", "N/A")
                    direction = t.get("direction", "Bullish")
                    entry_p = float(t.get("entry_price", 0.0) or 0.0)
                    sl_p = float(t.get("stop_loss", 0.0) or 0.0)
                    tp_p = float(t.get("take_profit", 0.0) or 0.0)
                    lev = float(t.get("leverage", 1.0) or 1.0)
                    pos_val_usd = float(t.get("position_size_usd", 0.0) or 0.0)
                    upnl = float(t.get("bybit_unrealized_pnl", 0.0) or 0.0)
                    pnl_icon = "🟢" if upnl >= 0 else "🔴"
                    block = (
                        f"{pnl_icon} *{sym}* ({t.get('tf', '15m')}) | *{direction}*\n"
                        f"• *Entry*: `${entry_p:.4f}`\n"
                        f"• *Stop Loss*: `${sl_p:.4f}` | *Take Profit*: `${tp_p:.4f}`\n"
                        f"• *Leverage*: `{lev:.1f}x` | *Margin*: `${pos_val_usd:.2f}`\n"
                        f"• *Unrealized PnL*: *${upnl:+.2f} USDT*\n"
                    )
                    report_lines.append(block)
                return "\n".join(report_lines)
        except Exception as ex_db:
            log_event("WARNING", f"db active trades fallback error: {ex_db}")

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

        tf_str = "15m"
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


def get_manual_trade_suggestion(symbol, interval, direction="Bullish"):
    try:
        from data import get_history
        from core import add_features

        sym = symbol.upper().strip()
        if not sym.endswith("USDT"):
            sym += "USDT"

        iv = str(interval).replace("m", "").replace("h", "")
        if iv not in ["15", "30", "60", "120", "240", "360"]:
            iv = "15"

        df_raw = get_history(symbol=sym, interval=iv, limit=100)
        if df_raw is None or len(df_raw) < 10:
            return None

        df = add_features(df_raw)
        latest = df.iloc[-1]
        entry_price = float(latest["close"])
        atr_dollars = float(latest.get("ATR_norm", 0.005) * entry_price)

        dir_clean = str(direction).strip().title()
        ml_trend = "Bullish" if dir_clean in ["Bullish", "Long", "Buy"] else "Bearish"

        if ml_trend == "Bullish":
            sl_price = entry_price - (1.0 * atr_dollars)
            tp_price = entry_price + (2.0 * atr_dollars)
        else:
            sl_price = entry_price + (1.0 * atr_dollars)
            tp_price = entry_price - (2.0 * atr_dollars)

        return {
            "symbol": sym,
            "interval": iv,
            "direction": ml_trend,
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "leverage": 5.0,
            "atr_dollars": atr_dollars
        }
    except Exception as ex:
        print(f"[Manual Suggestion Error] {ex}")
        return None


def execute_manual_trade(symbol, interval, direction, entry_price=None, stop_loss=None, take_profit=None, leverage=5.0, bot_state=None, bot_state_lock=None, active_trades_lock=None):
    try:
        from data import get_history
        from core import add_features
        from bybit_client import place_bybit_taker_ioc_order, format_bybit_qty, set_bybit_leverage, get_bybit_min_qty_step
        from trade_calculators import assert_valid_geometry
        import risk_engine
        from decision_journal import DecisionRecord, write_decision, ReasonCode
        import database
        import uuid
        import math
        import threading

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

        rec = DecisionRecord(
            symbol=sym,
            interval=str(iv),
            signal_source="TELEGRAM_MANUAL",
            direction=ml_trend,
            regime="MANUAL",
            raw_confidence=1.0,
            calibrated_conf=1.0
        )

        df_raw = get_history(symbol=sym, interval=iv, limit=100)
        if df_raw is None or len(df_raw) < 10:
            rec.outcome = "REJECTED"
            rec.reject_reason = "Failed to fetch market data"
            rec.reason_code = ReasonCode.PREDICTION_ERROR
            write_decision(rec)
            return f"❌ Failed to fetch market data for `{sym}` ({iv}m)."

        df = add_features(df_raw)
        latest = df.iloc[-1]
        live_price = float(latest["close"])
        atr_dollars = float(latest.get("ATR_norm", 0.005) * live_price)

        final_entry = float(entry_price) if entry_price is not None and float(entry_price) > 0 else live_price
        if stop_loss is not None and float(stop_loss) > 0:
            final_sl = float(stop_loss)
        else:
            final_sl = final_entry - (1.0 * atr_dollars) if ml_trend == "Bullish" else final_entry + (1.0 * atr_dollars)

        if take_profit is not None and float(take_profit) > 0:
            final_tp = float(take_profit)
        else:
            final_tp = final_entry + (2.0 * atr_dollars) if ml_trend == "Bullish" else final_entry - (2.0 * atr_dollars)

        from risk_limits import HARD_TIMEFRAME_MAX_LEVERAGE_CAPS
        tf_cap = HARD_TIMEFRAME_MAX_LEVERAGE_CAPS.get(str(iv), 5.0)
        req_lev = float(leverage) if leverage is not None and float(leverage) > 0 else tf_cap
        final_lev = min(req_lev, tf_cap)
        rec.leverage = final_lev

        # Check circuit breaker and halt states
        if bot_state:
            with (bot_state_lock if bot_state_lock else threading.Lock()):
                if bot_state.get("circuit_breaker_active", False):
                    rec.outcome = "REJECTED"
                    rec.reject_reason = "Circuit breaker active"
                    rec.reason_code = ReasonCode.CIRCUIT_BREAKER_ACTIVE
                    write_decision(rec)
                    return "❌ Order Rejected: Circuit Breaker is active."
                if bot_state.get("bot_stopped", False):
                    rec.outcome = "REJECTED"
                    rec.reject_reason = "Bot stopped"
                    rec.reason_code = ReasonCode.BOT_STOPPED
                    write_decision(rec)
                    return "❌ Order Rejected: Bot is in STOPPED / Emergency Halt state."

        # Validate geometry
        try:
            assert_valid_geometry(ml_trend, final_entry, final_sl, final_tp, symbol=sym)
        except Exception as geom_err:
            rec.outcome = "REJECTED"
            rec.reject_reason = f"Invalid geometry: {geom_err}"
            rec.reason_code = ReasonCode.GEOMETRY_INVALID
            write_decision(rec)
            return f"❌ Order Rejected: Invalid Geometry — {geom_err}"

        position_size_usd = 2.0
        rec.position_size_usd = position_size_usd

        # Evaluate pre-trade risk checklist
        active_trades_list = [t for tf_k in ["15m", "30m", "1h", "2h", "4h", "6h"] for t in (bot_state.get(f"active_trade_{tf_k}", []) if bot_state else [])]
        df_dict = {sym: df}
        passed_checklist, checklist_msg, dd_mult, capped_size = risk_engine.evaluate_pre_trade_checklist(
            sym, position_size_usd, final_lev, active_trades_list, bot_state or {}, df_dict, interval=str(iv), direction=ml_trend, journal=rec
        )
        if not passed_checklist:
            rec.outcome = "REJECTED"
            rec.reject_reason = checklist_msg
            rec.reason_code = ReasonCode.RISK_CHECKLIST_BLOCKED
            write_decision(rec)
            return f"❌ Order Rejected by Risk Engine: {checklist_msg}"

        position_size_usd = min(position_size_usd, capped_size)
        set_bybit_leverage(sym, final_lev)

        leveraged_size = position_size_usd * final_lev
        raw_qty = leveraged_size / final_entry
        qty_str = format_bybit_qty(sym, raw_qty)
        qty_val = float(qty_str)

        if qty_val * final_entry < 5.1:
            step = get_bybit_min_qty_step(sym)
            if step > 0:
                qty_val = math.ceil(5.1 / (final_entry * step)) * step
                qty_str = format_bybit_qty(sym, qty_val)

        order_res = place_bybit_taker_ioc_order(sym, side, qty_str, sl=final_sl, tp=final_tp)
        if order_res.get("retCode") == 0:
            bybit_order_id = order_res.get("result", {}).get("orderId")
            
            tf_map_inv = {"15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h", "360": "6h"}
            tf_key = tf_map_inv.get(iv, f"{iv}m")
            trade_uuid = str(uuid.uuid4())
            trade_id_str = f"{sym}_{trade_uuid}"
            trade_record = {
                "trade_id": trade_id_str,
                "bybit_order_id": bybit_order_id,
                "symbol": sym,
                "direction": ml_trend,
                "entry_price": final_entry,
                "predicted_price": final_entry,
                "stop_loss": final_sl,
                "take_profit": final_tp,
                "leverage": final_lev,
                "position_size_usd": position_size_usd,
                "qty": float(qty_str),
                "confidence": 1.0,
                "regime": "MANUAL",
                "entry_time": int(time.time() * 1000),
                "end_time": time.time() + (int(iv) * 60 * 10),
                "atr_dollars": atr_dollars
            }

            lock_to_use = active_trades_lock if active_trades_lock is not None else (bot_state_lock if bot_state_lock else threading.Lock())
            with lock_to_use:
                key = f"active_trade_{tf_key}"
                curr_list = bot_state.get(key, []) if bot_state else []
                if not isinstance(curr_list, list):
                    curr_list = []
                curr_list.append(trade_record)
                if bot_state:
                    bot_state[key] = curr_list
                database.save_active_trades(tf_key, curr_list)

            rec.outcome = "EXECUTED"
            rec.trade_id = trade_id_str
            write_decision(rec)

            return (
                f"🟢 *MANUAL TRADE EXECUTED SUCCESSFULLY*\n\n"
                f"• *Asset*: `{sym}` ({tf_key})\n"
                f"• *Direction*: `{ml_trend}`\n"
                f"• *Entry Price*: `${final_entry:.4f}`\n"
                f"• *Stop Loss*: `${final_sl:.4f}`\n"
                f"• *Take Profit*: `${final_tp:.4f}`\n"
                f"• *Leverage*: `{final_lev:.1f}x` | *Margin*: `${position_size_usd:.2f}`\n"
                f"• *Bybit Order ID*: `{bybit_order_id}`"
            )
        else:
            rec.outcome = "REJECTED"
            rec.reject_reason = order_res.get("retMsg", "Order rejected by Bybit")
            rec.reason_code = ReasonCode.EXECUTION_VALIDATION_FAILED
            write_decision(rec)
            return f"🔴 *Manual Trade Execution Failed*: {order_res.get('retMsg')}"

    except Exception as ex:
        return f"🔴 *Manual Trade Error*: {str(ex)}"


def start_telegram_command_listener(bot_state, bot_state_lock, active_trades_lock=None):
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
                {"command": "skipped", "description": "View skipped trades by timeframe"},
                {"command": "regime", "description": "Current market regime & volatility breakdown"},
                {"command": "openmanualtrade", "description": "Execute manual trade with live suggestions"},
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

                    # Handle Callback Queries (Inline Keyboard Clicks)
                    cb_query = update.get("callback_query")
                    if cb_query:
                        cb_chat_id = str(cb_query.get("message", {}).get("chat", {}).get("id"))
                        cb_from_id = str(cb_query.get("from", {}).get("id"))
                        cb_data = str(cb_query.get("data", ""))

                        is_cb_auth = bool(allowed_chat_ids and (cb_chat_id in allowed_chat_ids or cb_from_id in allowed_chat_ids))
                        if not is_cb_auth:
                            print(f"[Telegram Security] Unauthorized callback attempt: chat_id {cb_chat_id}, from_id {cb_from_id}")
                            execute_telegram_api_call("answerCallbackQuery", {
                                "callback_query_id": cb_query.get("id"),
                                "text": "⛔ Access Denied: Unauthorized user",
                                "show_alert": True
                            })
                            continue

                        if cb_data.startswith("skipped_"):
                            tf_choice = cb_data.replace("skipped_", "")
                            rep = get_skipped_trades_report(tf_choice)
                            execute_telegram_api_call("sendMessage", {"chat_id": cb_chat_id, "text": rep, "parse_mode": "Markdown"})
                        elif cb_data.startswith("quick_trade_"):
                            # Format: quick_trade_BTC_15_Bullish
                            cb_parts = cb_data.split("_")
                            if len(cb_parts) >= 4:
                                q_sym, q_tf, q_dir = cb_parts[2], cb_parts[3], cb_parts[4] if len(cb_parts) > 4 else "Bullish"
                                res_msg = execute_manual_trade(q_sym, q_tf, q_dir, bot_state=bot_state, bot_state_lock=bot_state_lock, active_trades_lock=active_trades_lock)
                                execute_telegram_api_call("sendMessage", {"chat_id": cb_chat_id, "text": res_msg, "parse_mode": "Markdown"})
                        elif cb_data.startswith("exec_manual_"):
                            # Format: exec_manual_{sym}_{tf}_{direction}_{entry}_{sl}_{tp}_{lev}
                            cb_parts = cb_data.split("_")
                            if len(cb_parts) >= 8:
                                e_sym, e_tf, e_dir = cb_parts[2], cb_parts[3], cb_parts[4]
                                e_entry, e_sl, e_tp, e_lev = cb_parts[5], cb_parts[6], cb_parts[7], cb_parts[8] if len(cb_parts) > 8 else 5.0
                                res_msg = execute_manual_trade(e_sym, e_tf, e_dir, entry_price=e_entry, stop_loss=e_sl, take_profit=e_tp, leverage=e_lev, bot_state=bot_state, bot_state_lock=bot_state_lock, active_trades_lock=active_trades_lock)
                                execute_telegram_api_call("sendMessage", {"chat_id": cb_chat_id, "text": res_msg, "parse_mode": "Markdown"})

                    message = update.get("message") or update.get("edited_message") or update.get("channel_post")
                    if not message:
                        continue

                    chat_id = str(message.get("chat", {}).get("id"))
                    text = message.get("text", "").strip()

                    if not allowed_chat_ids or chat_id not in allowed_chat_ids:
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
                        bal_info = get_live_bybit_wallet_details()
                        with bot_state_lock:
                            running = bot_state.get("bot_running", True)
                            if bal_info:
                                bal = bal_info["wallet_balance"]
                                equity = bal_info["equity"]
                            else:
                                bal = float(bot_state.get("live_balance", bot_state.get("wallet_balance", 0.0)))
                                equity = float(bot_state.get("live_balance", bot_state.get("wallet_balance", 0.0)))
                            active_trades = sum(len(bot_state.get(f"active_trade_{tf}", [])) for tf in ["15m", "30m", "1h", "2h", "4h", "6h"])
                            regime_15 = bot_state.get("regime_15m", "N/A")
                            regime_1h = bot_state.get("regime_1h", "N/A")
                            cb_active = bot_state.get("circuit_breaker_active", False)

                        status_text = (
                            f"🤖 *BTC TRADING BOT STATUS*\n\n"
                            f"• *Status*: {'🟢 RUNNING' if running else '🔴 PAUSED'}\n"
                            f"• *Real Wallet Balance*: `${bal:.2f} USDT`\n"
                            f"• *Account Equity*: `${equity:.2f} USDT`\n"
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
                        if not args:
                            btc_sug = get_manual_trade_suggestion("BTCUSDT", "15", "Bullish")
                            eth_sug = get_manual_trade_suggestion("ETHUSDT", "15", "Bullish")
                            
                            prompt_text = (
                                "🛠️ *MANUAL TRADE SETUP & AUTO-SUGGESTIONS*\n\n"
                                "Specify custom parameters or review live ATR suggestions:\n\n"
                                "*Full Command Format*:\n"
                                "`/openmanualtrade <SYMBOL> <TF> <DIR> [ENTRY] [SL] [TP] [LEV]`\n\n"
                                "*Examples*:\n"
                                "• Auto-suggested defaults: `/openmanualtrade BTC 15 Long`\n"
                                "• Custom parameters: `/openmanualtrade BTC 15 Long 68500 67800 70000 10`\n\n"
                            )
                            if btc_sug:
                                prompt_text += (
                                    f"💡 *Live Suggested Setup (BTCUSDT 15m)*:\n"
                                    f"• Entry: `${btc_sug['entry_price']:.2f}` | SL: `${btc_sug['stop_loss']:.2f}` | TP: `${btc_sug['take_profit']:.2f}` | Lev: `5x`\n\n"
                                )
                            if eth_sug:
                                prompt_text += (
                                    f"💡 *Live Suggested Setup (ETHUSDT 15m)*:\n"
                                    f"• Entry: `${eth_sug['entry_price']:.2f}` | SL: `${eth_sug['stop_loss']:.2f}` | TP: `${eth_sug['take_profit']:.2f}` | Lev: `5x`\n\n"
                                )

                            reply_markup = {
                                "inline_keyboard": [
                                    [
                                        {"text": "🚀 Trade BTC 15m Long (Suggested)", "callback_data": "quick_trade_BTC_15_Bullish"},
                                        {"text": "🚀 Trade ETH 15m Long (Suggested)", "callback_data": "quick_trade_ETH_15_Bullish"}
                                    ]
                                ]
                            }
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": chat_id,
                                "text": prompt_text,
                                "parse_mode": "Markdown",
                                "reply_markup": reply_markup
                            })
                        elif len(args) <= 3:
                            m_sym, m_tf = args[0], args[1]
                            m_dir = args[2] if len(args) >= 3 else "Bullish"
                            sug = get_manual_trade_suggestion(m_sym, m_tf, m_dir)
                            if sug:
                                confirm_text = (
                                    f"🎯 *PROPOSED MANUAL TRADE SETUP*\n\n"
                                    f"• *Asset*: `{sug['symbol']}` ({sug['interval']}m)\n"
                                    f"• *Direction*: `{sug['direction']}`\n"
                                    f"• *Entry Price (Live)*: `${sug['entry_price']:.4f}`\n"
                                    f"• *Suggested Stop Loss*: `${sug['stop_loss']:.4f}` (1.0x ATR)\n"
                                    f"• *Suggested Take Profit*: `${sug['take_profit']:.4f}` (2.0x ATR)\n"
                                    f"• *Suggested Leverage*: `{sug['leverage']:.1f}x`\n\n"
                                    f"Click *Confirm & Execute Trade* below, or pass custom values:\n"
                                    f"`/openmanualtrade {m_sym} {m_tf} {m_dir} <ENTRY> <SL> <TP> [LEVERAGE]`"
                                )
                                cb_payload = f"exec_manual_{sug['symbol']}_{sug['interval']}_{sug['direction']}_{sug['entry_price']:.4f}_{sug['stop_loss']:.4f}_{sug['take_profit']:.4f}_{sug['leverage']:.0f}"
                                reply_markup = {
                                    "inline_keyboard": [
                                        [
                                            {"text": "✅ Confirm & Execute Trade", "callback_data": cb_payload}
                                        ]
                                    ]
                                }
                                execute_telegram_api_call("sendMessage", {
                                    "chat_id": chat_id,
                                    "text": confirm_text,
                                    "parse_mode": "Markdown",
                                    "reply_markup": reply_markup
                                })
                            else:
                                res_msg = execute_manual_trade(m_sym, m_tf, m_dir, bot_state=bot_state, bot_state_lock=bot_state_lock, active_trades_lock=active_trades_lock)
                                execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": res_msg, "parse_mode": "Markdown"})
                        else:
                            m_sym = args[0]
                            m_tf = args[1]
                            m_dir = args[2]
                            c_entry = args[3] if len(args) > 3 else None
                            c_sl = args[4] if len(args) > 4 else None
                            c_tp = args[5] if len(args) > 5 else None
                            c_lev = args[6] if len(args) > 6 else 5.0
                            res_msg = execute_manual_trade(m_sym, m_tf, m_dir, entry_price=c_entry, stop_loss=c_sl, take_profit=c_tp, leverage=c_lev, bot_state=bot_state, bot_state_lock=bot_state_lock, active_trades_lock=active_trades_lock)
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
                        if not args:
                            menu_text = (
                                "🔍 *SKIPPED TRADES — SELECT TIMEFRAME / MODEL*\n\n"
                                "Please select which model timeframe you want to inspect:\n\n"
                                "• `/skipped 15m` — 15-Minute Scalping Model\n"
                                "• `/skipped 30m` — 30-Minute Model\n"
                                "• `/skipped 1h` — 1-Hour Model\n"
                                "• `/skipped 2h` — 2-Hour Model\n"
                                "• `/skipped 4h` — 4-Hour Model\n"
                                "• `/skipped 6h` — 6-Hour Model\n"
                            )
                            reply_markup = {
                                "inline_keyboard": [
                                    [
                                        {"text": "⚡ 15m", "callback_data": "skipped_15"},
                                        {"text": "⏱️ 30m", "callback_data": "skipped_30"},
                                        {"text": "📈 1h", "callback_data": "skipped_60"}
                                    ],
                                    [
                                        {"text": "📊 2h", "callback_data": "skipped_120"},
                                        {"text": "⏳ 4h", "callback_data": "skipped_240"},
                                        {"text": "🏆 6h", "callback_data": "skipped_360"}
                                    ]
                                ]
                            }
                            execute_telegram_api_call("sendMessage", {
                                "chat_id": chat_id,
                                "text": menu_text,
                                "parse_mode": "Markdown",
                                "reply_markup": reply_markup
                            })
                        else:
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
                        try:
                            from dashboard_routes import trigger_emergency_kill_switch
                            from telegram_bot import send_telegram_alert
                            success, errors = trigger_emergency_kill_switch(bot_state, None, f"Telegram /kill command from chat_id={chat_id}")
                            if success and not errors:
                                resp_text = "🚨 *EMERGENCY KILL SWITCH ENGAGED.* Bot halted, working orders cancelled, and all open positions closed at market."
                            else:
                                err_str = "; ".join(errors) if errors else "partial error"
                                resp_text = f"🚨 *EMERGENCY KILL SWITCH ENGAGED.* Bot halted (close/cancel warnings: {err_str})."
                            execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": resp_text, "parse_mode": "Markdown"})
                        except Exception as kill_err:
                            with bot_state_lock:
                                bot_state["bot_running"] = False
                            try:
                                import database
                                database.set_setting("bot_running", "False")
                            except Exception as ex_db:
                                log_event("WARNING", f"[Telegram Kill Database Error] {ex_db}")
                            execute_telegram_api_call("sendMessage", {"chat_id": chat_id, "text": f"🚨 *EMERGENCY KILL SWITCH ENGAGED.* Bot halted (error: {kill_err}).", "parse_mode": "Markdown"})

                    elif cmd in ["help"]:
                        help_text = (
                            "📖 *BTC TRADING BOT COMMANDS*\n\n"
                            "• `/status` - Bot running state & active regime\n"
                            "• `/balance` - Real live Bybit wallet balance, margin & equity\n"
                            "• `/trades` - Active open positions with live PnL\n"
                            "• `/openmanualtrade [sym] [tf] [dir] [entry] [sl] [tp] [lev]` - Execute manual trade with live suggestions\n"
                            "• `/skipped [tf]` - View skipped signals by model timeframe\n"
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
