import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator


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
    if get_history_fn is None:
        try:
            from data import get_history
            get_history_fn = get_history
        except Exception:
            pass
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
        df_4h = None
        if htf_cache is not None and (symbol, "240") in htf_cache:
            df_4h = htf_cache[(symbol, "240")]
        if df_4h is None and get_history_fn:
            try:
                df_4h = get_history_fn(symbol=symbol, interval="240", limit=100)
                if htf_cache is not None and df_4h is not None:
                    htf_cache[(symbol, "240")] = df_4h
            except Exception:
                df_4h = None

        if df_4h is None or len(df_4h) < 21:
            results["4h_Trend"] = {"pass": False, "detail": "Could not fetch 4h data", "weight": weight_4h}
            results["4h_RSI"] = {"pass": False, "detail": "Could not fetch 4h data", "weight": weight_4h}
            max_score += weight_4h * 2
        else:
            df_4h_completed = df_4h.iloc[:-1].copy()
            ema9_4h = EMAIndicator(df_4h_completed["close"], window=9).ema_indicator().iloc[-1]
            ema21_4h = EMAIndicator(df_4h_completed["close"], window=21).ema_indicator().iloc[-1]
            rsi_4h = RSIIndicator(df_4h_completed["close"], window=14).rsi().iloc[-1]

            trend_4h = "Bullish" if ema9_4h > ema21_4h else "Bearish"
            trend_4h_pass = (ml_trend == "Bullish" and trend_4h == "Bullish") or (ml_trend == "Bearish" and trend_4h == "Bearish")
            rsi_4h_pass = (rsi_4h < 75.0) if ml_trend == "Bullish" else (rsi_4h > 25.0)

            results["4h_Trend"] = {
                "pass": trend_4h_pass,
                "detail": f"4h Trend is {trend_4h} (EMA9: {ema9_4h:.2f}, EMA21: {ema21_4h:.2f})",
                "weight": weight_4h
            }
            results["4h_RSI"] = {
                "pass": rsi_4h_pass,
                "detail": f"4h RSI is {rsi_4h:.1f}",
                "weight": weight_4h
            }
            max_score += weight_4h * 2
            if trend_4h_pass:
                total_score += weight_4h
            if rsi_4h_pass:
                total_score += weight_4h

    # CHECK 3: Orderbook Imbalance & L2 Depth
    weight_ob = 1
    if get_orderbook_fn:
        try:
            ob_data = get_orderbook_fn(symbol=symbol)
            ob_imbalance = ob_data.get("imbalance", 0.0)
            ob_pass = (ml_trend == "Bullish" and ob_imbalance >= -0.2) or (ml_trend == "Bearish" and ob_imbalance <= 0.2)
            results["Orderbook_L2"] = {"pass": ob_pass, "detail": f"Imbalance: {ob_imbalance:+.2f}", "weight": weight_ob}
            max_score += weight_ob
            if ob_pass:
                total_score += weight_ob
        except Exception as e:
            results["Orderbook_L2"] = {"pass": True, "detail": f"Bypassed ({e})", "weight": 0}
    else:
        results["Orderbook_L2"] = {"pass": True, "detail": "Orderbook check bypassed", "weight": 0}

    # CHECK 4: Choppiness Index Gate
    weight_chop = 1
    if choppiness_fn and df_1h is not None and len(df_1h) >= 14:
        try:
            ci_val = choppiness_fn(df_1h)
            chop_pass = ci_val < 61.8
            results["Choppiness_Gate"] = {"pass": chop_pass, "detail": f"CI: {ci_val:.1f} (<61.8 threshold)", "weight": weight_chop}
            max_score += weight_chop
            if chop_pass:
                total_score += weight_chop
        except Exception as e:
            results["Choppiness_Gate"] = {"pass": True, "detail": f"Bypassed ({e})", "weight": 0}
    else:
        results["Choppiness_Gate"] = {"pass": True, "detail": "Choppiness check bypassed", "weight": 0}

    # CHECK 5: News Blackout & Sentiment Check
    weight_news = 1
    news_pass = True
    news_detail = "News Safe"
    if news_sentiment:
        if isinstance(news_sentiment, str) and "BLACKOUT" in news_sentiment:
            news_pass = False
            news_detail = news_sentiment
        elif isinstance(news_sentiment, dict) and news_sentiment.get("blackout"):
            news_pass = False
            news_detail = news_sentiment.get("reason", "News Blackout Active")
    results["News_Blackout"] = {"pass": news_pass, "detail": news_detail, "weight": weight_news}
    max_score += weight_news
    if news_pass:
        total_score += weight_news

    # CHECK 6: Counter-Momentum Guard
    weight_cm = 2
    try:
        c1 = df_1h.iloc[-1]
        c2 = df_1h.iloc[-2]
        c3 = df_1h.iloc[-3]
        is_red = [c1["close"] < c1["open"], c2["close"] < c2["open"], c3["close"] < c3["open"]]
        is_green = [c1["close"] > c1["open"], c2["close"] > c2["open"], c3["close"] > c3["open"]]
        is_bullish = ml_trend in ["Bullish", "BUY", "LONG", "UP"]
        is_bearish = ml_trend in ["Bearish", "SELL", "SHORT", "DOWN"]
        if is_bullish:
            candle_pass = not all(is_red)
            detail_msg = "Safe (No consecutive 3 red candles)" if candle_pass else "Blocked (Knife Falling: 3 consecutive red candles)"
        elif is_bearish:
            candle_pass = not all(is_green)
            detail_msg = "Safe (No consecutive 3 green candles)" if candle_pass else "Blocked (Rocket Rising: 3 consecutive green candles)"
        else:
            candle_pass = True
            detail_msg = "Safe (Neutral trend)"

    except Exception as e:
        candle_pass = True
        detail_msg = f"Skipped ({e})"
    results["Counter_Momentum"] = {"pass": candle_pass, "detail": detail_msg, "weight": weight_cm}
    max_score += weight_cm
    if candle_pass:
        total_score += weight_cm

    # FINAL SCORING
    score_pct = (total_score / max(1.0, float(max_score)) * 100.0) if max_score > 0 else 100.0
    score_threshold = 75.0
    traditional_approved = (not hard_gate_failed) and (score_pct >= score_threshold)
    trend_gates_passed = results.get("1d_Trend", {}).get("pass", True) and results.get("4h_Trend", {}).get("pass", True)

    try:
        from trade_frequency_optimizer import trade_frequency_optimizer
        adx_val = 20.0
        if df_1h is not None and "ADX" in df_1h.columns and len(df_1h) > 0:
            adx_val = float(df_1h["ADX"].iloc[-1])
        effective_conf_threshold = trade_frequency_optimizer.calculate_regime_adaptive_confidence_threshold(adx_val, dynamic_conf_threshold)
    except Exception:
        effective_conf_threshold = dynamic_conf_threshold

    approved = (calibrated_confidence >= effective_conf_threshold) and trend_gates_passed and traditional_approved

    results["_Score_Summary"] = {
        "pass": approved,
        "detail": f"Meta-Gate: {'APPROVED' if approved else 'REJECTED'} (Calibrated Conf: {calibrated_confidence*100:.1f}% vs Req: {effective_conf_threshold*100:.1f}%, Trend Pass: {trend_gates_passed}) | Traditional Score: {total_score}/{max_score} ({score_pct:.0f}%, Hard Gates: {'PASSED' if not hard_gate_failed else 'FAILED'})",
        "weight": "SUMMARY"
    }

    std_results = {}
    for key, val in results.items():
        std_results[str(key)] = {
            "pass": bool(val["pass"]),
            "detail": str(val["detail"])
        }

    return bool(approved), std_results, float(score_pct)
