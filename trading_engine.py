"""
trading_engine.py
-----------------
Core trade execution and async order placement engine for Bybit.
"""

import os
import time
import uuid
import json
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import config

from bybit_client import (
    TRADE_MODE,
    set_bybit_leverage,
    place_bybit_order,
    place_bybit_limit_order,
    place_bybit_taker_ioc_order,
    cancel_bybit_order,
    get_bybit_order_details,
    get_all_bybit_positions,
    get_bybit_bid_ask,
    get_bybit_last_execution,
    format_bybit_qty
)
from websocket_client import (
    _ws_filled_orders,
    _ws_filled_orders_lock
)
from telegram_bot import send_telegram_alert

active_execution_lock = threading.Lock()
active_execution_symbols = set()


def execute_bybit_trade_async(*args, **kwargs):
    symbol = args[0] if args else "Unknown"
    try:
        _execute_bybit_trade_async_inner(*args, **kwargs)
    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"[CRITICAL EXECUTION ERROR] Async trade execution failed for {symbol}: {err_msg}")
        traceback.print_exc()
        send_telegram_alert(f"🚨 *Async Trade Execution Error* 🚨\n• Symbol: `{symbol}`\n• Error: `{err_msg}`")
    finally:
        with active_execution_lock:
            active_execution_symbols.discard(symbol)


def _execute_bybit_trade_async_inner(symbol, iv, tf, ml_trend, leverage_val, qty_str, raw_qty, entry_price, stop_loss_price, take_profit_price, position_size_usd, kelly_fraction, calibrated_confidence, ml_confidence, dynamic_conf_threshold, latest_completed_ts, latest_candle, pred_change, predicted_price, atr_dollars, tp_multiplier_adjusted, sl_multiplier_adjusted, df_completed, trade_uuid, duration_seconds, active_trade_key, is_oversized=False, bot_state=None, save_history_func=None):

    if latest_candle is None:
        latest_candle = {}
    if df_completed is None:
        df_completed = pd.DataFrame()
        
    bybit_success = True
    bybit_order_id = None
    bybit_scale_out_order_id = None
    actual_qty = raw_qty
    
    try:
        pos_list = get_all_bybit_positions()
        if pos_list:
            existing_pos = next((p for p in pos_list if p.get("symbol") == symbol and float(p.get("size", "0")) > 0), None)
            if existing_pos:
                print(f"[{symbol} {iv}m API Block] Live order placement skipped: a live position already exists on Bybit.")
                return
    except Exception as pos_check_err:
        print(f"[{symbol} {iv}m API Warning] Live Position Guard check failed: {pos_check_err}")

    print(f"[{symbol} {iv}m API] Preparing to open live position on Bybit ({TRADE_MODE.upper()})...")
    leverage_ok = set_bybit_leverage(symbol, leverage_val)
    if leverage_ok:
        side = "Buy" if ml_trend in ["Bullish", "BUY", "LONG", "UP"] else "Sell"
        bybit_success = False
        
        rolling_atr = df_completed["ATR_norm"].tail(30) if "ATR_norm" in df_completed.columns else pd.Series([0.01])
        atr_mean = rolling_atr.mean()
        atr_std = rolling_atr.std()
        vol_z_score = (latest_candle.get("ATR_norm", 0.01) - atr_mean) / (atr_std + 1e-8) if atr_std > 0 else 0.0
        is_extreme_volatility = vol_z_score > 3.0
        
        if is_extreme_volatility:
            print(f"[{symbol} {iv}m API] Volatility Spiked! Z-score: {vol_z_score:.2f} > 3.0. Placing Taker IOC order immediately...")
            order_res = place_bybit_taker_ioc_order(symbol, side, qty_str, sl=stop_loss_price, tp=take_profit_price)
            if order_res.get("retCode") == 0:
                bybit_order_id = order_res.get("result", {}).get("orderId")
                bybit_success = True
                time.sleep(0.5)
                order_details = get_bybit_order_details(symbol, bybit_order_id)
                if order_details:
                    entry_price = float(order_details.get("avgPrice", entry_price))
                    actual_qty = float(order_details.get("cumExecQty", raw_qty))
                else:
                    fill_exec = get_bybit_last_execution(symbol)
                    if fill_exec:
                        entry_price = float(fill_exec.get("execPrice", entry_price))
                    actual_qty = raw_qty
            else:
                print(f"[{symbol} {iv}m API ERROR] Taker IOC order failed: {order_res.get('retMsg')}")
        else:
            for chase in range(5):
                bid, ask = get_bybit_bid_ask(symbol)
                if bid == 0.0 or ask == 0.0:
                    bid, ask = entry_price, entry_price
                limit_price = bid if side == "Buy" else ask
                print(f"[{symbol} {iv}m API] Placing Limit Maker order at ${limit_price:.4f} (Chase {chase+1}/5)...")
                order_res = place_bybit_limit_order(symbol, side, qty_str, limit_price)
                if order_res.get("retCode") == 0:
                    bybit_order_id = order_res.get("result", {}).get("orderId")
                    filled = False
                    for idx in range(24):
                        time.sleep(0.5)
                        with _ws_filled_orders_lock:
                            ws_details = _ws_filled_orders.get(bybit_order_id)
                        if ws_details:
                            entry_price = float(ws_details.get("avgPrice", limit_price))
                            actual_qty = float(ws_details.get("cumExecQty", raw_qty))
                            filled = True
                            bybit_success = True
                            print(f"[{symbol} {iv}m API] Order fill detected via WebSocket in {idx*0.5:.1f}s.")
                            break
                        if idx > 0 and idx % 6 == 0:
                            order_details = get_bybit_order_details(symbol, bybit_order_id)
                            if order_details:
                                status = order_details.get("orderStatus")
                                if status == "Filled":
                                    entry_price = float(order_details.get("avgPrice", limit_price))
                                    actual_qty = float(order_details.get("cumExecQty", raw_qty))
                                    filled = True
                                    bybit_success = True
                                    break
                                elif status in ["Cancelled", "Rejected"]:
                                    break
                    if filled:
                        print(f"[{symbol} {iv}m API] Success! Maker Limit Order filled at ${entry_price:.4f}.")
                        break
                    else:
                        print(f"[{symbol} {iv}m API] Order unfilled after 12s. Cancelling and re-quoting...")
                        cancel_bybit_order(symbol, bybit_order_id)
                        time.sleep(0.5)
                        final_details = get_bybit_order_details(symbol, bybit_order_id)
                        if final_details:
                            status = final_details.get("orderStatus")
                            cum_qty = float(final_details.get("cumExecQty", 0.0))
                            if status == "Filled" or cum_qty > 0:
                                entry_price = float(final_details.get("avgPrice", limit_price))
                                actual_qty = cum_qty if cum_qty > 0 else raw_qty
                                filled = True
                                bybit_success = True
                                print(f"[{symbol} {iv}m API] Success! Maker Limit Order filled/partially filled during cancel request at ${entry_price:.4f} (Qty: {actual_qty}).")
                                break
                else:
                    print(f"[{symbol} {iv}m API WARNING] Limit order placement failed: {order_res.get('retMsg')} (waiting 2s before retry)")
                    time.sleep(2)
                    
            if not bybit_success:
                with _ws_filled_orders_lock:
                    if bybit_order_id and bybit_order_id in _ws_filled_orders:
                        bybit_success = True
                        print(f"[{symbol} {iv}m API] Order fill confirmed in WS cache. Skipping market order fallback.")
            
            if not bybit_success:
                atr_norm = float(latest_candle.get("ATR_norm", 0.0))
                if atr_norm >= 0.015:
                    print(f"[{symbol} {iv}m API BLOCK] Limit chases failed. Market fallback blocked due to extreme volatility (ATR_norm: {atr_norm:.4f} >= 0.015).")
                else:
                    print(f"[{symbol} {iv}m API] All Limit Maker chases failed. Falling back to Market order to guarantee entry...")
                    order_res = place_bybit_order(symbol, side, qty_str)
                    if order_res.get("retCode") == 0:
                        bybit_order_id = order_res.get("result", {}).get("orderId")
                        bybit_success = True
                        time.sleep(0.5)
                        order_details = get_bybit_order_details(symbol, bybit_order_id)
                        if order_details:
                            entry_price = float(order_details.get("avgPrice", entry_price))
                            actual_qty = float(order_details.get("cumExecQty", raw_qty))
                        else:
                            fill_exec = get_bybit_last_execution(symbol)
                            if fill_exec:
                                entry_price = float(fill_exec.get("execPrice", entry_price))
                            actual_qty = raw_qty
                    else:
                        print(f"[{symbol} {iv}m API ERROR] Market order placement failed: {order_res.get('retMsg')}")

    min_fill_pct = getattr(config, "MIN_ACCEPTABLE_FILL_PCT", 0.60)
    fill_ratio = (actual_qty / raw_qty) if (raw_qty > 0 and actual_qty > 0) else (1.0 if bybit_success else 0.0)
    if bybit_success and fill_ratio < min_fill_pct:
        print(f"[{symbol} {iv}m API WARNING] Fill ratio {fill_ratio*100:.1f}% below {min_fill_pct*100:.0f}% threshold. Reversing partial fill...")
        if getattr(config, "RESIDUAL_ACTION", "CLOSE") == "CLOSE":
            opp_side = "Sell" if side == "Buy" else "Buy"
            place_bybit_taker_ioc_order(symbol, opp_side, format_bybit_qty(symbol, actual_qty))
            bybit_success = False

    if not bybit_success and TRADE_MODE != "simulation":
        print(f"[{symbol} {iv}m API CRITICAL BLOCK] Position NOT registered in bot state because order failed on Bybit.")
        send_telegram_alert(f"⚠️ *Live Order Placement Failed* ⚠️\n• Symbol: `{symbol}`\n• Interval: `{iv}m`\n• Reason: API rejected order execution on Bybit.")
        return

    # Create active trade payload in bot state
    trade_payload = {
        "trade_id": trade_uuid,
        "symbol": symbol,
        "entry_price": float(entry_price),
        "stop_loss": float(stop_loss_price),
        "stop_price": float(stop_loss_price),
        "take_profit": float(take_profit_price),
        "target_tp_price": float(take_profit_price),
        "tp_price": float(take_profit_price),
        "initial_stop_loss": float(stop_loss_price),
        "initial_take_profit": float(take_profit_price),
        "stop_state": "INITIAL",
        "stop_state_meta": {
            "stop_state": "INITIAL",
            "stop_version": "v3.2",
            "transition_reason": "InitialTradeOpened",
            "locked_r": 0.0,
            "expected_net_pnl": 0.0,
            "updated_at": time.time()
        },
        "sl_multiplier": float(sl_multiplier_adjusted),
        "tp_multiplier": float(tp_multiplier_adjusted),
        "highest_price": float(entry_price),
        "lowest_price": float(entry_price),
        "swing_low_3b": float(df_completed["low"].tail(3).min()) if (df_completed is not None and not df_completed.empty and "low" in df_completed.columns) else float(entry_price),
        "swing_high_3b": float(df_completed["high"].tail(3).max()) if (df_completed is not None and not df_completed.empty and "high" in df_completed.columns) else float(entry_price),
        "break_even_triggered": False,
        "half_closed": False,
        "direction": ml_trend,
        "entry_time": float(latest_completed_ts),
        "start_time": time.time(),
        "position_size_usd": float(position_size_usd),
        "original_size": float(position_size_usd),
        "intended_size_usd": float(position_size_usd),
        "leverage": float(leverage_val),
        "confidence": float(calibrated_confidence),
        "raw_data": json.dumps({
            "vol_pctile": float(latest_candle.get("vol_pctile", 1.0)),
            "mfe": 0.0
        }),
        "interval": iv,
        "duration_seconds": duration_seconds,
        "bybit_order_id": bybit_order_id,
        "bybit_scale_out_order_id": bybit_scale_out_order_id
    }
    
    if bot_state:
        bot_state[active_trade_key].append(trade_payload)
        if save_history_func:
            save_history_func(bot_state)

    oversized_flag = " ⚠️ (OVERSIZED HIGH CONFIDENCE)" if is_oversized else ""
    send_telegram_alert(
        f"⚡ *TRADE EXECUTED* ⚡{oversized_flag}\n"
        f"• *Asset*: {symbol}\n"
        f"• *Interval*: {iv}m\n"
        f"• *Direction*: {ml_trend}\n"
        f"• *Leverage*: {leverage_val:.1f}x\n"
        f"• *Entry Price*: ${entry_price:.2f}\n"
        f"• *Stop Loss*: ${stop_loss_price:.2f}\n"
        f"• *Take Profit*: ${take_profit_price:.2f}\n"
        f"• *Position Size*: ${position_size_usd:.2f} (Kelly: {kelly_fraction:.1f}%)\n"
        f"• *Confidence*: {calibrated_confidence*100:.1f}% (Threshold: {dynamic_conf_threshold*100:.1f}%)"
    )
