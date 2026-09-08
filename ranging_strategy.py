"""
ranging_strategy.py
-------------------
Institutional Mean-Reversion Ranging Strategy Engine (Option B).
Activated exclusively when the market regime is confirmed Ranging (ADX < 25).
Fades overextended Bollinger Band boundaries (2.0 std dev) when confirmed
by extreme RSI readings and liquidity absorption pin-bar wicks.
Generates maker post-only limit orders with tight structural stops to eliminate taker fee drag.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from logger import log_event


@dataclass
class RangingSignal:
    is_signal: bool
    direction: str  # "Bullish" (fade lower band support), "Bearish" (fade upper band resistance), "Neutral"
    confidence: float  # Calibrated probability 0.0 to 1.0
    entry_price: float
    take_profit: float
    stop_loss: float
    expected_move: float
    reason: str
    order_type: str = "POST_ONLY_LIMIT"


def evaluate_ranging_mean_reversion(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    interval: str = "15"
) -> RangingSignal:
    """
    Evaluates candle DataFrame for high-conviction Bollinger Band mean-reversion setups.
    Returns RangingSignal dataclass instance.
    """
    if str(interval) in ("240", "4h"):
        return RangingSignal(
            is_signal=False,
            direction="Neutral",
            confidence=0.0,
            entry_price=0.0,
            take_profit=0.0,
            stop_loss=0.0,
            expected_move=0.0,
            reason="240m (4h) interval is decoupled from Option B bridge; executes on ML ensemble"
        )

    if df is None or len(df) < 25:
        return RangingSignal(
            is_signal=False,
            direction="Neutral",
            confidence=0.0,
            entry_price=0.0,
            take_profit=0.0,
            stop_loss=0.0,
            expected_move=0.0,
            reason="Insufficient candle history for ranging evaluation"
        )

    last_row = df.iloc[-1]
    close = float(last_row.get("close", 0.0))
    high = float(last_row.get("high", close))
    low = float(last_row.get("low", close))
    open_p = float(last_row.get("open", close))

    if close <= 0.0:
        return RangingSignal(
            is_signal=False,
            direction="Neutral",
            confidence=0.0,
            entry_price=0.0,
            take_profit=0.0,
            stop_loss=0.0,
            expected_move=0.0,
            reason="Zero close price detected"
        )

    adx = float(last_row.get("ADX", 20.0)) if ("ADX" in df.columns and pd.notna(last_row.get("ADX"))) else 20.0
    if adx >= 25.0:
        return RangingSignal(
            is_signal=False,
            direction="Neutral",
            confidence=0.0,
            entry_price=0.0,
            take_profit=0.0,
            stop_loss=0.0,
            expected_move=0.0,
            reason=f"Market is Trending (ADX {adx:.1f} >= 25.0) — Ranging engine inactive"
        )

    # Calculate or retrieve Bollinger Bands
    if "BB_high" in df.columns and "BB_low" in df.columns and "BB_mid" in df.columns:
        bb_high = float(last_row.get("BB_high", close * 1.015))
        bb_low = float(last_row.get("BB_low", close * 0.985))
        bb_mid = float(last_row.get("BB_mid", close))
    else:
        roll_mean = df["close"].rolling(20, min_periods=10).mean().iloc[-1]
        roll_std = df["close"].rolling(20, min_periods=10).std().iloc[-1]
        bb_mid = float(roll_mean) if pd.notna(roll_mean) else close
        std_val = float(roll_std) if (pd.notna(roll_std) and roll_std > 0) else (close * 0.01)
        bb_high = bb_mid + (2.0 * std_val)
        bb_low = bb_mid - (2.0 * std_val)

    rsi = float(last_row.get("RSI", 50.0)) if ("RSI" in df.columns and pd.notna(last_row.get("RSI"))) else 50.0
    atr = float(last_row.get("ATR", close * 0.008)) if ("ATR" in df.columns and pd.notna(last_row.get("ATR"))) else (close * 0.008)
    atr = max(atr, close * 0.002)

    lower_wick_ratio = float(last_row.get("lower_wick_volume_ratio", 1.0)) if "lower_wick_volume_ratio" in df.columns else 1.0
    upper_wick_ratio = float(last_row.get("upper_wick_volume_ratio", 1.0)) if "upper_wick_volume_ratio" in df.columns else 1.0

    bar_range = max(1e-6, high - low)
    lower_wick = max(0.0, min(open_p, close) - low)
    upper_wick = max(0.0, high - max(open_p, close))
    lower_wick_pct = lower_wick / bar_range
    upper_wick_pct = upper_wick / bar_range

    precision = 4 if close < 10.0 else 2

    # --- Setup 1: Bullish Mean Reversion (Fade Lower Band Support) ---
    is_lower_touched = bool(low <= bb_low * 1.002 or close <= bb_low * 1.001)
    is_oversold = bool(rsi <= 36.0)
    has_bull_rejection = bool(lower_wick_ratio >= 1.25 or lower_wick_pct >= 0.25 or close > low + (bar_range * 0.20))

    if is_lower_touched and is_oversold and has_bull_rejection:
        entry_p = round(max(low, bb_low), precision)
        tp_p = round(bb_mid, precision)
        sl_p = round(low - (0.90 * atr), precision)
        
        # Ensure positive geometry
        if tp_p > entry_p and sl_p < entry_p:
            conf_bonus = max(0.0, (36.0 - rsi) / 100.0) + min(0.06, (lower_wick_ratio - 1.0) * 0.03)
            calibrated_conf = round(min(0.72, 0.60 + conf_bonus), 4)
            expected_change = round(tp_p - entry_p, precision)
            log_event("INFO", f"[Ranging Strategy] {symbol} {interval}m Bullish BB Fade detected: entry={entry_p}, tp={tp_p}, sl={sl_p}, conf={calibrated_conf*100:.1f}%, RSI={rsi:.1f}")
            return RangingSignal(
                is_signal=True,
                direction="Bullish",
                confidence=calibrated_conf,
                entry_price=entry_p,
                take_profit=tp_p,
                stop_loss=sl_p,
                expected_move=expected_change,
                reason=f"Lower Bollinger Band Exhaustion Fade (RSI {rsi:.1f}, WickRatio {lower_wick_ratio:.2f})",
                order_type="POST_ONLY_LIMIT"
            )

    # --- Setup 2: Bearish Mean Reversion (Fade Upper Band Resistance) ---
    is_upper_touched = bool(high >= bb_high * 0.998 or close >= bb_high * 0.999)
    is_overbought = bool(rsi >= 64.0)
    has_bear_rejection = bool(upper_wick_ratio >= 1.25 or upper_wick_pct >= 0.25 or close < high - (bar_range * 0.20))

    if is_upper_touched and is_overbought and has_bear_rejection:
        entry_p = round(min(high, bb_high), precision)
        tp_p = round(bb_mid, precision)
        sl_p = round(high + (0.90 * atr), precision)

        # Ensure positive geometry
        if tp_p < entry_p and sl_p > entry_p:
            conf_bonus = max(0.0, (rsi - 64.0) / 100.0) + min(0.06, (upper_wick_ratio - 1.0) * 0.03)
            calibrated_conf = round(min(0.72, 0.60 + conf_bonus), 4)
            expected_change = round(tp_p - entry_p, precision)
            log_event("INFO", f"[Ranging Strategy] {symbol} {interval}m Bearish BB Fade detected: entry={entry_p}, tp={tp_p}, sl={sl_p}, conf={calibrated_conf*100:.1f}%, RSI={rsi:.1f}")
            return RangingSignal(
                is_signal=True,
                direction="Bearish",
                confidence=calibrated_conf,
                entry_price=entry_p,
                take_profit=tp_p,
                stop_loss=sl_p,
                expected_move=expected_change,
                reason=f"Upper Bollinger Band Exhaustion Fade (RSI {rsi:.1f}, WickRatio {upper_wick_ratio:.2f})",
                order_type="POST_ONLY_LIMIT"
            )

    # Inside mid-band chop: Abstain safely
    return RangingSignal(
        is_signal=False,
        direction="Neutral",
        confidence=0.0,
        entry_price=0.0,
        take_profit=0.0,
        stop_loss=0.0,
        expected_move=0.0,
        reason=f"Inside Bollinger Mid-Range [BB_Low={bb_low:.2f} < {close:.2f} < BB_High={bb_high:.2f}] (RSI: {rsi:.1f})"
    )
