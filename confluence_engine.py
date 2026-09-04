from logger import log_event
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from cycle_detector import detect_market_regime_with_hysteresis


def get_valid_htf_cache(htf_cache: dict, key: tuple, ttl_seconds: float = 900.0):
    """
    Retrieves HTF DataFrame from htf_cache with a 15-minute TTL eviction rule and freshness validation.
    """
    if htf_cache is None or key not in htf_cache:
        return None
    val = htf_cache[key]
    df_cached = None
    if isinstance(val, tuple) and len(val) == 2:
        df_cand, timestamp = val
        if time.time() - timestamp < ttl_seconds:
            df_cached = df_cand
        else:
            return None
    elif isinstance(val, pd.DataFrame):
        df_cached = val

    if df_cached is not None:
        if df_cached.empty or not df_cached.attrs.get("fetch_ok", True):
            return None
        return df_cached
    return None

def set_valid_htf_cache(htf_cache: dict, key: tuple, df: pd.DataFrame):
    if htf_cache is not None and df is not None and not df.empty and df.attrs.get("fetch_ok", True):
        htf_cache[key] = (df, time.time())


def calculate_exit_quality_score(
    structure_pass: bool,
    liquidity_pass: bool,
    expected_move_pct: float,
    spread_pct: float = 0.0003,
    funding_rate: float = 0.0001,
    atr_norm: float = 0.0040,
    regime: str = "MODERATE_TREND"
) -> float:
    """
    Component 5: Exit Quality Score (EQS 0 - 100)
    Evaluates Structure (20), Liquidity (15), Expected Move (20), Spread (15), Funding (10), Volatility (10), Regime (10).
    """
    import config
    policy = getattr(config, "CONFLUENCE_POLICY", {})
    w_struct = policy.get("structure_weight", 20.0)
    w_liq = policy.get("liquidity_weight", 15.0)
    w_move = policy.get("move_weight", 20.0)
    w_spread = policy.get("spread_weight", 15.0)
    w_fund = policy.get("funding_weight", 10.0)
    w_vol = policy.get("volatility_weight", 10.0)
    w_reg = policy.get("regime_weight", 10.0)

    target_move = policy.get("strong_move_pct", 0.015)
    max_spread = policy.get("max_spread_pct", 0.0010)
    max_fund = policy.get("max_funding_rate", 0.0003)
    min_atr = policy.get("min_atr_norm", 0.0035)

    score = 0.0
    if structure_pass:
        score += w_struct
    if liquidity_pass:
        score += w_liq

    # Continuous proportional move score
    score += w_move * float(np.clip(float(expected_move_pct) / max(1e-6, target_move), 0.0, 1.0))

    # Continuous proportional spread score (lower spread = higher score)
    score += w_spread * float(np.clip(1.0 - (float(spread_pct) / max(1e-6, max_spread)), 0.0, 1.0))

    # Continuous proportional funding score
    score += w_fund * float(np.clip(1.0 - (abs(float(funding_rate)) / max(1e-6, max_fund)), 0.0, 1.0))

    # Continuous volatility score
    score += w_vol * float(np.clip(float(atr_norm) / max(1e-6, min_atr), 0.0, 1.0))

    reg_u = regime.upper()
    if "STRONG" in reg_u or "MODERATE" in reg_u:
        score += w_reg
    elif "RANGING" in reg_u:
        score += w_reg * 0.5

    return round(score, 1)


def evaluate_expectancy_gate(historical_win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> Tuple[bool, float]:
    """
    Component 9: Expectancy Gate
    Expected Value = (WinRate * Reward) - (LossRate * Risk)
    """
    wr = float(historical_win_rate) if float(historical_win_rate) <= 1.0 else float(historical_win_rate) / 100.0
    lr = 1.0 - wr
    ev = (wr * abs(avg_win_pct)) - (lr * abs(avg_loss_pct))
    return ev > 0.0, ev



def check_pre_trade_confluence(current_price, df_1h, ml_trend, news_sentiment, expected_pct_change, interval="60", symbol=None, htf_cache=None, calibrated_confidence=0.5, dynamic_conf_threshold=0.58, get_history_fn=None, get_orderbook_fn=None, choppiness_fn=None, bot_state_dict=None, global_htf_cache=None, current_regime=None):
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
        except Exception as ex_confluence_engine:
            log_event("WARNING", f"confluence_engine notice: {ex_confluence_engine}")
    results = {}
    hard_gate_failed = False
    total_score = 0
    max_score = 0

    if current_regime is None:
        current_regime = detect_market_regime_with_hysteresis(df_1h, symbol=symbol, interval=str(interval))
    is_ranging_regime = "ranging" in str(current_regime).lower()

    # CHECK 1: 1-Day Structural Trend
    df_1d = get_valid_htf_cache(htf_cache, (symbol, "D"))
    if df_1d is None and get_history_fn:
        try:
            df_1d = get_history_fn(symbol=symbol, interval="D", limit=100)
            if df_1d is not None and (df_1d.empty or not df_1d.attrs.get("fetch_ok", True)):
                df_1d = None
            if df_1d is not None:
                set_valid_htf_cache(htf_cache, (symbol, "D"), df_1d)
        except Exception as ex_confluence_engine:
            log_event("WARNING", f"confluence_engine notice: {ex_confluence_engine}")
            df_1d = None

    weight_1d = 1
    if str(interval) in ["5", "15"]:
        results["1d_Trend"] = {"pass": True, "detail": "Bypassed for short TF", "weight": 0}
    elif str(interval) == "30":
        weight_1d = 0.5
        if df_1d is None or len(df_1d) < 21 or not df_1d.attrs.get("fetch_ok", True):
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
        if not trend_1d_pass and not is_ranging_regime:
            hard_gate_failed = True
        results["1d_Trend"] = {
            "pass": trend_1d_pass,
            "detail": f"1d Trend is {trend_1d} (EMA9: {ema9_1d:.2f}, EMA21: {ema21_1d:.2f}) [{'SOFT GATE (Ranging)' if is_ranging_regime else 'HARD GATE'}]",
            "weight": weight_1d
        }
        max_score += weight_1d
        if trend_1d_pass:
            total_score += weight_1d

    # CHECK 2: 1-Hour & 4-Hour Tactical Trend & RSI
    weight_4h = 1
    if str(interval) in ["5", "15", "30"]:
        if df_1h is not None and len(df_1h) >= 21:
            df_1h_comp = df_1h.iloc[:-1].copy()
            s_ema9 = EMAIndicator(df_1h_comp["close"], window=9).ema_indicator()
            s_ema21 = EMAIndicator(df_1h_comp["close"], window=21).ema_indicator()
            ema9_1h = float(s_ema9.iloc[-1]) if len(s_ema9) > 0 and pd.notna(s_ema9.iloc[-1]) else 0.0
            ema21_1h = float(s_ema21.iloc[-1]) if len(s_ema21) > 0 and pd.notna(s_ema21.iloc[-1]) else 0.0
            trend_1h = "Bullish" if ema9_1h > ema21_1h else "Bearish"
            trend_1h_pass = (ml_trend == "Bullish" and trend_1h == "Bullish") or (ml_trend == "Bearish" and trend_1h == "Bearish")
            is_soft_intraday = str(interval) in ["5", "15", "30"]
            results["1h_Trend"] = {
                "pass": trend_1h_pass,
                "detail": f"1h Trend is {trend_1h} (EMA9: {ema9_1h:.2f}, EMA21: {ema21_1h:.2f}) [{'SOFT GATE (Intraday/Ranging)' if (is_ranging_regime or is_soft_intraday) else 'HARD GATE'}]",
                "weight": 1
            }
            if not trend_1h_pass and not is_ranging_regime and not is_soft_intraday:
                hard_gate_failed = True
            max_score += 1
            if trend_1h_pass:
                total_score += 1
        else:
            results["1h_Trend"] = {"pass": False, "detail": "Rejected (Insufficient 1h data)", "weight": 0}

    # Evaluate 4h Trend & RSI Hard Gates for ALL timeframes
    df_4h = get_valid_htf_cache(htf_cache, (symbol, "240"))
    if df_4h is None and get_history_fn:
        try:
            df_4h = get_history_fn(symbol=symbol, interval="240", limit=100)
            if df_4h is not None and (df_4h.empty or not df_4h.attrs.get("fetch_ok", True)):
                df_4h = None
            if df_4h is not None:
                set_valid_htf_cache(htf_cache, (symbol, "240"), df_4h)
        except Exception as ex_confluence_engine:
            log_event("WARNING", f"confluence_engine notice: {ex_confluence_engine}")
            df_4h = None

    is_soft_intraday = str(interval) in ["5", "15", "30"]
    if df_4h is None or len(df_4h) < 21 or not df_4h.attrs.get("fetch_ok", True):
        results["4h_Trend"] = {"pass": False, "detail": f"Could not fetch 4h data [{'SOFT GATE (Intraday/Ranging)' if (is_ranging_regime or is_soft_intraday) else 'HARD GATE'}]", "weight": weight_4h}
        results["4h_RSI"] = {"pass": False, "detail": f"Could not fetch 4h data [{'SOFT GATE (Intraday/Ranging)' if (is_ranging_regime or is_soft_intraday) else 'HARD GATE'}]", "weight": weight_4h}
        if not is_ranging_regime and not is_soft_intraday:
            hard_gate_failed = True
        max_score += weight_4h * 2
    else:
        df_4h_completed = df_4h.iloc[:-1].copy()
        s_ema9_4h = EMAIndicator(df_4h_completed["close"], window=9).ema_indicator()
        s_ema21_4h = EMAIndicator(df_4h_completed["close"], window=21).ema_indicator()
        s_rsi_4h = RSIIndicator(df_4h_completed["close"], window=14).rsi()
        
        ema9_4h = float(s_ema9_4h.iloc[-1]) if len(s_ema9_4h) > 0 and pd.notna(s_ema9_4h.iloc[-1]) else 0.0
        ema21_4h = float(s_ema21_4h.iloc[-1]) if len(s_ema21_4h) > 0 and pd.notna(s_ema21_4h.iloc[-1]) else 0.0
        rsi_4h = float(s_rsi_4h.iloc[-1]) if len(s_rsi_4h) > 0 and pd.notna(s_rsi_4h.iloc[-1]) else 50.0

        trend_4h = "Bullish" if ema9_4h > ema21_4h else "Bearish"
        trend_4h_pass = (ml_trend == "Bullish" and trend_4h == "Bullish") or (ml_trend == "Bearish" and trend_4h == "Bearish")
        rsi_4h_pass = (rsi_4h < 75.0) if ml_trend == "Bullish" else (rsi_4h > 25.0)

        if not trend_4h_pass and not is_ranging_regime and not is_soft_intraday:
            hard_gate_failed = True
        if not rsi_4h_pass and not is_ranging_regime and not is_soft_intraday:
            hard_gate_failed = True

        results["4h_Trend"] = {
            "pass": trend_4h_pass,
            "detail": f"4h Trend is {trend_4h} (EMA9: {ema9_4h:.2f}, EMA21: {ema21_4h:.2f}) [{'SOFT GATE (Intraday/Ranging)' if (is_ranging_regime or is_soft_intraday) else 'HARD GATE'}]",
            "weight": weight_4h
        }
        results["4h_RSI"] = {
            "pass": rsi_4h_pass,
            "detail": f"4h RSI is {rsi_4h:.1f} [{'SOFT GATE (Intraday/Ranging)' if (is_ranging_regime or is_soft_intraday) else 'HARD GATE'}]",
            "weight": weight_4h
        }
        max_score += weight_4h * 2
        if trend_4h_pass:
            total_score += weight_4h
        if rsi_4h_pass:
            total_score += weight_4h

    # CHECK 3: Orderbook Imbalance & L2 Depth
    weight_ob = 1
    _ob_fn = get_orderbook_fn
    if _ob_fn is None:
        try:
            from data import get_orderbook_imbalance
            _ob_fn = get_orderbook_imbalance
        except Exception:
            _ob_fn = None

    if _ob_fn:
        try:
            ob_data = _ob_fn(symbol=symbol)
            ob_imbalance = ob_data.get("imbalance", 0.0) if isinstance(ob_data, dict) else (float(ob_data) if isinstance(ob_data, (int, float)) else 0.0)
            ob_pass = (ml_trend == "Bullish" and ob_imbalance >= -0.2) or (ml_trend == "Bearish" and ob_imbalance <= 0.2)
            results["Orderbook_L2"] = {"pass": ob_pass, "detail": f"Imbalance: {ob_imbalance:+.2f}", "weight": weight_ob}
            max_score += weight_ob
            if ob_pass:
                total_score += weight_ob
        except Exception as ex_confluence_engine:
            log_event("WARNING", f"confluence_engine notice: {ex_confluence_engine}")
            results["Orderbook_L2"] = {"pass": True, "detail": f"Orderbook check fallback ({ex_confluence_engine}) [EXCLUDED]", "weight": 0}
    else:
        results["Orderbook_L2"] = {"pass": True, "detail": "Orderbook unavailable [EXCLUDED]", "weight": 0}

    # CHECK 4: Choppiness Index Gate
    weight_chop = 1
    _chop_fn = choppiness_fn
    if _chop_fn is None:
        try:
            from trade_calculators import choppiness_index
            _chop_fn = choppiness_index
        except Exception:
            _chop_fn = None

    if _chop_fn and df_1h is not None and len(df_1h) >= 14:
        try:
            ci_val = _chop_fn(df_1h)
            chop_pass = ci_val < 61.8
            results["Choppiness_Gate"] = {"pass": chop_pass, "detail": f"CI: {ci_val:.1f} (<61.8 threshold)", "weight": weight_chop}
            max_score += weight_chop
            if chop_pass:
                total_score += weight_chop
        except Exception as ex_confluence_engine:
            log_event("WARNING", f"confluence_engine notice: {ex_confluence_engine}")
            results["Choppiness_Gate"] = {"pass": True, "detail": f"Choppiness check fallback ({ex_confluence_engine}) [EXCLUDED]", "weight": 0}
    else:
        results["Choppiness_Gate"] = {"pass": True, "detail": "Choppiness check unavailable [EXCLUDED]", "weight": 0}

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
        c1_c, c1_o = float(c1["close"]), float(c1["open"])
        c2_c, c2_o = float(c2["close"]), float(c2["open"])
        c3_c, c3_o = float(c3["close"]), float(c3["open"])
        is_red = [c1_c < c1_o, c2_c < c2_o, c3_c < c3_o]
        is_green = [c1_c > c1_o, c2_c > c2_o, c3_c > c3_o]
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

    except Exception as ex_confluence_engine:
        log_event("WARNING", f"confluence_engine notice: {ex_confluence_engine}")
        candle_pass = True
        detail_msg = f"Skipped ({ex_confluence_engine})"
    results["Counter_Momentum"] = {"pass": candle_pass, "detail": detail_msg, "weight": weight_cm}
    max_score += weight_cm
    if candle_pass:
        total_score += weight_cm

    # CHECK 7: Regime-Aware RSI Guard (Rec 2)
    weight_regime_rsi = 2
    rsi_val = 50.0
    if df_1h is not None and "RSI" in df_1h.columns and len(df_1h) > 0:
        rsi_val = float(df_1h["RSI"].iloc[-1])
    
    from production_regime_engine import production_regime_engine
    macro_blackout = (
        news_sentiment == "BLACKOUT" or
        (isinstance(news_sentiment, dict) and bool(news_sentiment.get("blackout") or news_sentiment.get("is_blackout") or news_sentiment.get("macro_guard_active"))) or
        (df_1h is not None and "hours_to_news" in df_1h.columns and len(df_1h) > 0 and float(df_1h["hours_to_news"].iloc[-1]) <= 0.5)
    )
    regime_res = production_regime_engine.evaluate_confluence(ml_trend, current_regime, rsi_val, macro_guard_active=macro_blackout)
    
    regime_rsi_pass = regime_res["execute"]
    regime_rsi_detail = regime_res["reason"] + f" [Regime: {current_regime}, RSI: {rsi_val:.1f}]"
    
    if not regime_rsi_pass:
        hard_gate_failed = True
        
    results["Regime_RSI_Guard"] = {"pass": regime_rsi_pass, "detail": regime_rsi_detail, "weight": weight_regime_rsi}
    max_score += weight_regime_rsi
    if regime_rsi_pass:
        total_score += weight_regime_rsi

    # FINAL SCORING
    score_pct = (total_score / max(1.0, float(max_score)) * 100.0) if max_score > 0 else 100.0
    score_threshold = 75.0
    traditional_approved = (not hard_gate_failed) and (score_pct >= score_threshold)
    is_ranging_regime = "ranging" in str(current_regime).lower()
    pass_1h = results.get("1h_Trend", {}).get("pass", False)
    pass_4h = results.get("4h_Trend", {}).get("pass", False)

    if is_soft_intraday or is_ranging_regime:
        # Neither ranging nor soft intraday may enter when BOTH 1h and 4h HTF structures oppose the signal
        trend_gates_passed = not (not pass_1h and not pass_4h)
    else:
        pass_1d = results.get("1d_Trend", {}).get("pass", False)
        trend_gates_passed = pass_1d and pass_4h and pass_1h

    if not pass_1h and not pass_4h:
        hard_gate_failed = True
        traditional_approved = False

    effective_conf_threshold = float(dynamic_conf_threshold)
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


def calculate_entry_signal_attribution(
    confluence_results: dict,
    calibrated_confidence: float,
    regime_name: str = "Trending"
) -> dict:
    """
    Decomposes entry decision into 7 sub-scores (0.0 to 1.0) and final confidence.
    """
    if not isinstance(confluence_results, dict):
        confluence_results = {}

    def _check_pass(*keys):
        for k in keys:
            if k in confluence_results:
                item = confluence_results.get(k, {})
                if isinstance(item, dict) and "pass" in item:
                    return 1.0 if item.get("pass") else 0.4
        return 0.7  # neutral default when check not evaluated for timeframe

    t1 = _check_pass("1d_Trend")
    t2 = _check_pass("4h_Trend", "1h_Trend")
    trend_score = round(float((t1 + t2) / 2.0), 2)

    r1 = _check_pass("4h_RSI", "1h_RSI", "Regime_RSI_Guard")
    r2 = _check_pass("Counter_Momentum")
    momentum_score = round(float((r1 + r2) / 2.0), 2)

    v1 = _check_pass("Volume_Participation", "Choppiness_Gate")
    volume_score = round(float(v1), 2)

    l1 = _check_pass("BB_Edge_Guard", "Choppiness_Gate")
    l2 = _check_pass("Volatility_Guard", "Regime_RSI_Guard")
    liquidity_score = round(float((l1 + l2) / 2.0), 2)

    regime_score = round(0.95 if "Trending" in str(regime_name) else 0.75, 2)

    m1 = _check_pass("News_Blackout", "News_Sentiment")
    macro_score = round(float(m1), 2)

    o1 = _check_pass("Orderbook_L2", "Orderbook_Imbalance")
    orderbook_score = round(float(o1), 2)

    return {
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "liquidity_score": liquidity_score,
        "regime_score": regime_score,
        "macro_score": macro_score,
        "orderbook_score": orderbook_score,
        "final_confidence": round(float(calibrated_confidence), 4),
        "attribution_pct": calculate_confidence_attribution_breakdown(trend_score, momentum_score, volume_score, liquidity_score, macro_score, orderbook_score, regime_score)
    }


def calculate_confidence_attribution_breakdown(
    trend_score: float,
    momentum_score: float,
    volume_score: float,
    liquidity_score: float,
    macro_score: float,
    orderbook_score: float,
    regime_score: float
) -> dict:
    """
    Computes percentage contribution of each feature component to final confidence.
    """
    total = (trend_score * 0.25) + (momentum_score * 0.20) + (volume_score * 0.15) + (liquidity_score * 0.10) + (macro_score * 0.10) + (orderbook_score * 0.10) + (regime_score * 0.10)
    total_val = max(1e-6, total)

    return {
        "trend_pct": round(((trend_score * 0.25) / total_val) * 100.0, 1),
        "momentum_pct": round(((momentum_score * 0.20) / total_val) * 100.0, 1),
        "volume_pct": round(((volume_score * 0.15) / total_val) * 100.0, 1),
        "liquidity_pct": round(((liquidity_score * 0.10) / total_val) * 100.0, 1),
        "macro_pct": round(((macro_score * 0.10) / total_val) * 100.0, 1),
        "orderbook_pct": round(((orderbook_score * 0.10) / total_val) * 100.0, 1),
        "regime_pct": round(((regime_score * 0.10) / total_val) * 100.0, 1)
    }


