"""
trade_calculators.py
--------------------
Contains standalone trade structure, risk, volume, and market condition calculation helpers.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import database

ACTIVE_TRADE_TF_KEYS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h"]

MAX_RR_RATIO = {
    "5m": 3.0,
    "15": 4.0, "15m": 4.0,
    "30": 5.0, "30m": 5.0,
    "60": 6.0, "1h": 6.0,
    "120": 8.0, "2h": 8.0,
    "240": 8.0, "4h": 8.0,
    "360": 8.0, "6h": 8.0
}

MIN_RR_RATIO = {
    "5m": 1.8,
    "15": 2.0, "15m": 2.0,
    "30": 2.5, "30m": 2.5,
    "60": 3.0, "1h": 3.0,
    "120": 4.0, "2h": 4.0,
    "240": 4.0, "4h": 4.0,
    "360": 4.0, "6h": 4.0
}


def compute_be_trigger_distance(atr_dollars, leverage, interval, mfe_trigger_atr_multiple, entry_price=0.0, min_pct_floor=0.0):
    """
    Compute minimum favorable move before break-even activates.
    Enforces a minimum 1.0x ATR distance for leverage > 10x to prevent premature BE chop-outs.
    """
    base_be_dist = max(mfe_trigger_atr_multiple * atr_dollars, entry_price * min_pct_floor)
    if leverage > 10.0:
        min_be_dist = 1.0 * atr_dollars
        final_be_dist = max(base_be_dist, min_be_dist)
        return final_be_dist
    return base_be_dist


def validate_trade_structure(entry_price, stop_price, tp_price, atr_dollars, leverage, interval, symbol, direction):
    """
    UNIVERSAL TRADE STRUCTURE SANITIZER: Pre-flight gate before order placement.
    Validates & adjusts R:R ratio, minimum stop width, and leverage compatibility.
    Returns: (is_valid, adjusted_dict, log_reason_str)
    """
    stop_dist = abs(entry_price - stop_price)
    tp_dist = abs(tp_price - entry_price)
    
    adjusted = {
        "stop_price": stop_price,
        "tp_price": tp_price,
        "leverage": leverage,
        "stop_dist": stop_dist,
        "tp_dist": tp_dist
    }
    logs = []
    
    # 1. Enforce minimum stop width for ALL trades
    min_stop = atr_dollars * 1.0 if leverage > 10.0 else atr_dollars * 0.75
    if stop_dist < min_stop:
        if leverage > 10.0:
            adjusted["leverage"] = 10.0
            logs.append(f"[LEVERAGE_CAPPED] {symbol} {interval} leverage reduced from {leverage:.1f}x to 10.0x & SL widened from ${stop_dist:.4f} to 1.0x ATR (${min_stop:.4f})")
        else:
            logs.append(f"[STOP_WIDENED] {symbol} {interval} SL widened from ${stop_dist:.4f} to 0.75x ATR (${min_stop:.4f})")
            
        required_stop = min_stop
        if direction == "Bearish":
            adjusted["stop_price"] = entry_price + required_stop
        else:
            adjusted["stop_price"] = entry_price - required_stop
        adjusted["stop_dist"] = required_stop

    # 2. Universal R:R Ratio Capping by timeframe (Max Cap)
    iv_str = str(interval).replace("m", "")
    max_rr = MAX_RR_RATIO.get(str(interval), MAX_RR_RATIO.get(iv_str, 4.0))
    current_rr = adjusted["tp_dist"] / adjusted["stop_dist"] if adjusted["stop_dist"] > 0 else 0.0
    
    if current_rr > max_rr:
        max_allowed_tp_dist = adjusted["stop_dist"] * max_rr
        if direction == "Bearish":
            adjusted["tp_price"] = entry_price - max_allowed_tp_dist
        else:
            adjusted["tp_price"] = entry_price + max_allowed_tp_dist
        adjusted["tp_dist"] = max_allowed_tp_dist
        current_rr = max_rr
        orig_rr_str = f"{tp_dist/adjusted['stop_dist']:.1f}:1" if adjusted['stop_dist'] > 0 else "N/A"
        logs.append(f"[TP_CAPPED_UNIVERSAL] {symbol} {interval} R:R capped from {orig_rr_str} to {max_rr:.1f}:1 (TP dist reduced from ${tp_dist:.4f} to ${max_allowed_tp_dist:.4f})")
        
    # 3. Minimum R:R Ratio Floor Gate (Reject trades below minimum viable R:R)
    min_rr = MIN_RR_RATIO.get(str(interval), MIN_RR_RATIO.get(iv_str, 2.0))
    if current_rr < min_rr:
        logs.append(f"[REJECT_MIN_RR] {symbol} {interval} R:R {current_rr:.1f}:1 is below minimum floor {min_rr:.1f}:1")
        return False, adjusted, "; ".join(logs)
        
    return True, adjusted, "; ".join(logs) if logs else "OK"


class AdaptiveVolumeGate:
    def __init__(self, lookback_days=30, optimization_window=500):
        self.lookback_days = lookback_days
        self.optimization_window = optimization_window
        self.threshold_cache = {}
        self.last_optimized = {}

    def get_volume_percentile(self, symbol, kline_df=None):
        try:
            if kline_df is not None and "volume" in kline_df.columns and len(kline_df) >= 10:
                volumes = kline_df["volume"].values
                current_vol = volumes[-1]
                percentile = float(np.mean(volumes <= current_vol))
                return percentile
        except Exception:
            pass
        return 1.0

    def optimize_threshold(self, symbol):
        try:
            trades = database.get_trade_history(limit=self.optimization_window)
            sym_trades = [t for t in trades if isinstance(t, dict) and t.get("symbol") == symbol]
            if len(sym_trades) < 20:
                return 0.25
            
            def parse_vol(t_obj):
                try:
                    if t_obj.get("raw_data"):
                        return float(json.loads(t_obj["raw_data"]).get("vol_pctile", 1.0))
                except Exception:
                    pass
                return 1.0

            best_threshold = 0.25
            best_profit = -float('inf')
            for threshold in np.arange(0.10, 0.51, 0.05):
                allowed = [t for t in sym_trades if parse_vol(t) >= threshold]
                if len(allowed) < 5:
                    continue
                pnl_sum = sum(float(t.get("pnl_usd", 0.0)) for t in allowed)
                if pnl_sum > best_profit:
                    best_profit = pnl_sum
                    best_threshold = threshold
            self.threshold_cache[symbol] = float(best_threshold)
            self.last_optimized[symbol] = time.time()
            return float(best_threshold)
        except Exception:
            return 0.25

    def check(self, symbol, kline_df=None):
        current_pct = self.get_volume_percentile(symbol, kline_df=kline_df)
        last_opt = self.last_optimized.get(symbol, 0)
        if time.time() - last_opt > 86400 * 7 or symbol not in self.threshold_cache:
            threshold = self.optimize_threshold(symbol)
        else:
            threshold = self.threshold_cache[symbol]
            
        if current_pct < threshold:
            return False, f"VOLUME_GATE_BLOCKED: {symbol} 4H volume at {current_pct:.1%} (Threshold: {threshold:.1%})", current_pct
        return True, f"VOLUME_GATE_PASSED: {symbol} 4H volume at {current_pct:.1%}", current_pct


class MFEBreakEvenTrigger:
    def __init__(self, lookback_trades=150, min_sample_size=15):
        self.lookback_trades = lookback_trades
        self.min_sample_size = min_sample_size
        self.trigger_cache = {}

    def get_trigger_multiple(self, symbol, timeframe="60"):
        key = (symbol, str(timeframe))
        if key in self.trigger_cache:
            return self.trigger_cache[key]
            
        try:
            trades = database.get_trade_history(limit=self.lookback_trades)
            sym_winning_trades = [
                t for t in trades
                if isinstance(t, dict) and t.get("symbol") == symbol and float(t.get("pnl_usd", 0.0)) > 0
            ]
            mfe_ratios = []
            for t in sym_winning_trades:
                atr = float(t.get("atr_dollars", 0.0))
                if atr > 0:
                    raw = json.loads(t.get("raw_data", "{}")) if t.get("raw_data") else {}
                    mfe_val = float(raw.get("mfe", 0.0))
                    if mfe_val > 0:
                        mfe_ratios.append(mfe_val / atr)
            if len(mfe_ratios) >= self.min_sample_size:
                trig = float(np.percentile(mfe_ratios, 25))
                trig = float(np.clip(trig, 0.8, 2.0))
                self.trigger_cache[key] = trig
                return trig
        except Exception:
            pass
        return 0.85 if str(timeframe) not in ["15", "30"] else 0.65


adaptive_volume_gate = AdaptiveVolumeGate()
mfe_be_trigger = MFEBreakEvenTrigger()


def choppiness_index(df, window=14):
    """0-100 scale. >61.8 = choppy, <38.2 = trending"""
    if df is None or len(df) < window:
        return 50.0
    high_max = df['high'].rolling(window).max()
    low_min = df['low'].rolling(window).min()
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    atr_sum = tr.rolling(window).sum()
    price_range = high_max - low_min
    ci = 100 * np.log10(atr_sum / (price_range + 1e-8)) / np.log10(window)
    return float(ci.iloc[-1]) if not np.isnan(ci.iloc[-1]) else 50.0


def check_flash_crash(symbol: str, max_drop_pct: float = 3.0, window_minutes: int = 5) -> bool:
    """Block 15M/30M entries if price dropped >3% in last 5 minutes"""
    try:
        from data import get_history
        df_1m = get_history(symbol=symbol, interval="1", limit=window_minutes + 2)
        if df_1m is None or len(df_1m) < window_minutes:
            return False
        recent_high = df_1m["high"].iloc[-window_minutes:].max()
        current_low = df_1m["low"].iloc[-1]
        drop_pct = ((recent_high - current_low) / (recent_high + 1e-8)) * 100.0
        return drop_pct > max_drop_pct
    except Exception:
        return False


def get_funding_adjustment(symbol: str, direction: str, funding_rate: float) -> float:
    """Bias confidence toward funded side (+0.03 boost) and penalize expensive side (-0.05)"""
    if funding_rate < -0.001:  # -0.1% funding: shorts get paid yield
        return +0.03 if direction == "Bearish" else -0.05
    elif funding_rate > 0.001: # +0.1% funding: longs get paid yield
        return +0.03 if direction == "Bullish" else -0.05
    return 0.0


def get_liquidity_score(symbol: str, orderbook_depth: int = 10) -> float:
    """Score 0-1 based on L2 orderbook depth"""
    try:
        from data import get_orderbook_imbalance
        ob = get_orderbook_imbalance(symbol=symbol)
        depth_est = ob.get("total_depth", 500000000)
        score = min(float(depth_est) / 500000000.0, 1.0)
        return max(0.1, score)
    except Exception:
        return 1.0


def estimate_liquidation_pool(df_history, direction, entry_price):
    """
    Estimates the location of the nearest high-leverage liquidation pool
    based on historical swing highs/lows (support and resistance levels).
    """
    lookback = min(len(df_history), 60)
    df_recent = df_history.iloc[-lookback:]
    
    if direction == "Bullish":
        swing_high = float(df_recent["high"].max())
        liq_pool_target = swing_high * 1.012
        return max(liq_pool_target, entry_price * 1.005)
    elif direction == "Bearish":
        swing_low = float(df_recent["low"].min())
        liq_pool_target = swing_low * 0.988
        return min(liq_pool_target, entry_price * 0.995)
    else:
        return entry_price


def calculate_covariance_multiplier(new_symbol, new_direction, bot_state=None):
    """
    Calculates a position sizing multiplier based on portfolio covariance.
    Penalizes highly correlated assets in the same direction.
    Allows offsetting/hedging for assets in opposite directions.
    """
    CORRELATION_MAP = {
        ("BTCUSDT", "BTCUSDT"): 1.0,
        ("ETHUSDT", "ETHUSDT"): 1.0,
        ("SOLUSDT", "SOLUSDT"): 1.0,
        ("BNBUSDT", "BNBUSDT"): 1.0,
        ("ADAUSDT", "ADAUSDT"): 1.0,
        ("XRPUSDT", "XRPUSDT"): 1.0,
        
        ("BTCUSDT", "ETHUSDT"): 0.85,
        ("BTCUSDT", "SOLUSDT"): 0.75,
        ("BTCUSDT", "BNBUSDT"): 0.70,
        ("BTCUSDT", "ADAUSDT"): 0.70,
        ("BTCUSDT", "XRPUSDT"): 0.65,
        
        ("ETHUSDT", "SOLUSDT"): 0.80,
        ("ETHUSDT", "BNBUSDT"): 0.75,
        ("ETHUSDT", "ADAUSDT"): 0.75,
        ("ETHUSDT", "XRPUSDT"): 0.65,
        
        ("SOLUSDT", "BNBUSDT"): 0.70,
        ("SOLUSDT", "ADAUSDT"): 0.70,
        ("SOLUSDT", "XRPUSDT"): 0.60,
        
        ("BNBUSDT", "ADAUSDT"): 0.70,
        ("BNBUSDT", "XRPUSDT"): 0.60,
        
        ("ADAUSDT", "XRPUSDT"): 0.65
    }

    is_stressed = False
    try:
        from data import get_history
        df_vol = get_history(symbol=new_symbol, interval="60", limit=30)
        if df_vol is not None and not df_vol.empty and "ATR_norm" in df_vol.columns:
            rolling_atr = df_vol["ATR_norm"].tail(30)
            atr_mean = rolling_atr.mean()
            atr_std = rolling_atr.std()
            vol_z_score = (df_vol["ATR_norm"].iloc[-1] - atr_mean) / (atr_std + 1e-8) if atr_std > 0 else 0.0
            is_stressed = vol_z_score > 2.0
            if is_stressed:
                print(f"[Stress Covariance] Volatility Z-score: {vol_z_score:.2f} > 2.0. Stressed correlation mode active.")
    except Exception as e:
        print(f"[Stress Covariance Warning] Could not calculate volatility z-score: {e}")

    def get_correlation(s1, s2):
        if s1 == s2:
            return 1.0
        if is_stressed:
            return 0.95
        return CORRELATION_MAP.get((s1, s2)) or CORRELATION_MAP.get((s2, s1)) or 0.70

    open_trades = []
    if bot_state:
        for tf_key in ACTIVE_TRADE_TF_KEYS:
            open_trades.extend(bot_state.get(f"active_trade_{tf_key}", []))

    if not open_trades:
        return 1.0, 0.0

    total_risk = 0.0
    breakdown = []
    
    for t in open_trades:
        open_sym = t.get("symbol")
        open_dir = t.get("direction")
        if not open_sym or not open_dir:
            continue
        r = get_correlation(new_symbol, open_sym)
        
        is_new_bull = new_direction in ["Bullish", "BUY", "LONG", "UP"]
        is_open_bull = open_dir in ["Bullish", "BUY", "LONG", "UP"]
        if is_new_bull == is_open_bull:
            impact = r
            risk_type = "CONCENTRATION"
        else:
            impact = -r
            risk_type = "HEDGE"
            
        total_risk += impact
        breakdown.append(f"  - Active: {open_sym} {open_dir} | Correlation: {r:.2f} | Risk impact: {impact:+.2f} ({risk_type})")

    if total_risk <= 0:
        multiplier = 1.0
    else:
        multiplier = 1.0 / (1.0 + total_risk)
        multiplier = max(0.20, min(1.0, multiplier))

    print(f"\n[Portfolio Covariance Analysis] New Entry: {new_symbol} {new_direction}")
    for item in breakdown:
        print(item)
    print(f"  - Total Net Correlation Risk: {total_risk:+.2f} -> Covariance Multiplier: {multiplier:.2f}x\n")

    return float(multiplier), float(total_risk)


def calculate_recent_performance_leverage_multiplier(bot_state=None, days=7):
    """
    Calculates a leverage multiplier based on the rolling Sharpe ratio of completed trades.
    Reduces max leverage during drawdowns to manage risk.
    """
    try:
        trades = bot_state.get("trade_history", []) if bot_state else []
        if len(trades) < 5:
            return 1.0
            
        cutoff = time.time() - days * 86400
        recent_trades = [t for t in trades if float(t.get("exit_time", 0.0)) >= cutoff]
        
        if len(recent_trades) < 3:
            return 1.0
            
        pnls = [float(t.get("pnl_usd", 0.0)) for t in recent_trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        
        if std_pnl < 1e-4:
            return 1.0
            
        sharpe = (mean_pnl / std_pnl) * np.sqrt(len(recent_trades))
        if sharpe < 0:
            multiplier = max(0.5, 1.0 + sharpe * 0.2)
        elif sharpe > 1.5:
            multiplier = min(1.5, 1.0 + (sharpe - 1.5) * 0.2)
        else:
            multiplier = 1.0
            
        return float(multiplier)
    except Exception:
        return 1.0
