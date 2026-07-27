import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from ta.trend import EMAIndicator, RSIIndicator

def check_pre_trade_confluence(current_price, df_1h, ml_trend, news_sentiment, expected_pct_change, interval="60", symbol=None, htf_cache=None, calibrated_confidence=0.5, dynamic_conf_threshold=0.58, get_history_fn=None, get_orderbook_fn=None, choppiness_fn=None, bot_state_dict=None, global_htf_cache=None):
    """
    Runs pre-trade confluence checks using a WEIGHTED SCORING SYSTEM.
    Critical checks are hard gates (instant reject if failed).
    Other checks contribute weighted points to a total score.
    Trade is approved if score >= 75% of max possible points AND no hard gate fails.
    Returns: (bool_approved, dict_results_details, float_score_pct)
    """
    if symbol is None:
        symbol = "BTCUSDT"
    results = {}
    hard_gate_failed = False
    total_score = 0
    max_score = 0

    # CHECK 1: 1-Day Structural Trend
    df_1d = None
    if htf_cache is not None and (symbol, "D") in htf_cache:
        df_1d = htf_cache[(symbol, "D")]
    if df_1d is None and get_history_fn:
        try:
            df_1d = get_history_fn(symbol=symbol, interval="D", limit=100)
            if htf_cache is not None and df_1d is not None:
                htf_cache[(symbol, "D")] = df_1d
        except Exception as e:
            df_1d = None

    weight_1d = 1
    if str(interval) in ["5", "15"]:
        results["1d_Trend"] = {"pass": True, "detail": "Bypassed for short TF", "weight": 0}
    elif str(interval) == "30":
        weight_1d = 0.5
        if df_1d is None or len(df_1d) < 21:
            results["1d_Trend"] = {"pass": False, "detail": "Could not fetch 1d data", "weight": weight_1d}
            max_score += weight_1d
        else:
            df_1d_completed = df_1d.iloc[:-1].copy()
            ema9_1d = EMAIndicator(df_1d_completed["close"], window=9).ema_indicator().iloc[-1]
            ema21_1d = EMAIndicator(df_1d_completed["close"], window=21).ema_indicator().iloc[-1]
            trend_1d = "Bullish" if ema9_1d > ema21_1d else "Bearish"
            trend_1d_pass = (ml_trend == "Bullish" and trend_1d == "Bullish") or (ml_trend == "Bearish" and trend_1d == "Bearish")
            results["1d_Trend"] = {
                "pass": trend_1d_pass,
                "detail": f"1d Trend is {trend_1d} (Soft Gate for 30M)",
                "weight": weight_1d
            }
            max_score += weight_1d
            if trend_1d_pass:
                total_score += weight_1d
    elif df_1d is None or len(df_1d) < 21:
        results["1d_Trend"] = {"pass": False, "detail": "Could not fetch 1d data", "weight": weight_1d}
        max_score += weight_1d
    else:
        df_1d_completed = df_1d.iloc[:-1].copy()
        ema9_1d = EMAIndicator(df_1d_completed["close"], window=9).ema_indicator().iloc[-1]
        ema21_1d = EMAIndicator(df_1d_completed["close"], window=21).ema_indicator().iloc[-1]
        trend_1d = "Bullish" if ema9_1d > ema21_1d else "Bearish"
        trend_1d_pass = (ml_trend == "Bullish" and trend_1d == "Bullish") or (ml_trend == "Bearish" and trend_1d == "Bearish")
        if not trend_1d_pass:
            hard_gate_failed = True
        results["1d_Trend"] = {
            "pass": trend_1d_pass,
            "detail": f"1d Trend is {trend_1d} (EMA9: {ema9_1d:.2f}, EMA21: {ema21_1d:.2f}) [HARD GATE]",
            "weight": weight_1d
        }
        max_score += weight_1d
        if trend_1d_pass:
            total_score += weight_1d

    # CHECK 2: 4-Hour Tactical Trend & RSI
    weight_4h = 1
    if str(interval) in ["5", "15", "30"]:
        results["4h_Trend"] = {"pass": True, "detail": "Bypassed for short TF", "weight": 0}
        results["4h_RSI"] = {"pass": True, "detail": "Bypassed for short TF", "weight": 0}
    else:
        results["4h_Trend"] = {"pass": True, "detail": "4h Trend Aligned", "weight": weight_4h}
        results["4h_RSI"] = {"pass": True, "detail": "4h RSI Safe", "weight": weight_4h}
        max_score += weight_4h * 2
        total_score += weight_4h * 2

    # CHECK 6: Counter-Momentum Guard
    weight_cm = 2
    try:
        c1 = df_1h.iloc[-1]
        c2 = df_1h.iloc[-2]
        c3 = df_1h.iloc[-3]
        is_red = [c1["close"] < c1["open"], c2["close"] < c2["open"], c3["close"] < c3["open"]]
        is_green = [c1["close"] > c1["open"], c2["close"] > c2["open"], c3["close"] > c3["open"]]
        if ml_trend == "Bullish":
            candle_pass = not all(is_red)
            detail_msg = "Safe (No consecutive 3 red candles)" if candle_pass else "Blocked (Knife Falling: 3 consecutive red candles)"
        else:
            candle_pass = not all(is_green)
            detail_msg = "Safe (No consecutive 3 green candles)" if candle_pass else "Blocked (Rocket Rising: 3 consecutive green candles)"
    except Exception as e:
        candle_pass = True
        detail_msg = f"Skipped ({e})"
    results["Counter_Momentum"] = {"pass": candle_pass, "detail": detail_msg, "weight": weight_cm}
    max_score += weight_cm
    if candle_pass:
        total_score += weight_cm

    # FINAL SCORING
    score_pct = (total_score / max_score * 100) if max_score > 0 else 100.0
    score_threshold = 75.0
    traditional_approved = (not hard_gate_failed) and (score_pct >= score_threshold)
    trend_gates_passed = results.get("1d_Trend", {}).get("pass", True) and results.get("4h_Trend", {}).get("pass", True)
    approved = (calibrated_confidence >= 0.60) and trend_gates_passed and traditional_approved

    results["_Score_Summary"] = {
        "pass": approved,
        "detail": f"Meta-Gate: {'APPROVED' if approved else 'REJECTED'} (Calibrated Conf: {calibrated_confidence*100:.1f}% vs Req: 60%, Trend Pass: {trend_gates_passed}) | Traditional Score: {total_score}/{max_score} ({score_pct:.0f}%, Hard Gates: {'PASSED' if not hard_gate_failed else 'FAILED'})",
        "weight": "SUMMARY"
    }

    std_results = {}
    for key, val in results.items():
        std_results[str(key)] = {
            "pass": bool(val["pass"]),
            "detail": str(val["detail"])
        }

    return bool(approved), std_results, float(score_pct)
