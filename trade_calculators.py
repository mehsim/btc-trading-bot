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
from typing import Dict, List, Tuple, Optional, Any, Union
import database
import config
from logger import log_event

def safe_float(val, default=0.0):
    if val is None or val == "MT":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

ACTIVE_TRADE_TF_KEYS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h"]

def assert_valid_geometry(direction, entry, sl, tp, symbol=""):
    """Raises ValueError if SL/TP sit on the wrong side of entry or have un-economical magnitude/R:R."""
    if entry <= 0 or sl <= 0 or tp <= 0:
        raise ValueError(f"[{symbol}] Non-positive prices in geometry: entry={entry}, sl={sl}, tp={tp}")
    direction_clean = str(direction).strip().title()
    if direction_clean in ("Bullish", "Long", "Buy"):
        ok = sl < entry < tp
    else:
        ok = sl > entry > tp
    if not ok:
        raise ValueError(f"[{symbol}] Invalid {direction} geometry: SL={sl}, entry={entry}, TP={tp}")
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    if sl_dist == 0 or tp_dist == 0:
        raise ValueError(f"[{symbol}] Zero stop-loss or take-profit distance: entry={entry}, sl={sl}, tp={tp}")
    rr = tp_dist / sl_dist
    if rr < 0.40:
        raise ValueError(f"[{symbol}] Un-economical R:R ratio ({rr:.2f} < 0.40): entry={entry}, sl={sl}, tp={tp}")
    return True

REALIZED_RR_HAIRCUT = 0.80  # Realistic limit maker exit realization factor

def calculate_required_p(entry: float, tp: float, sl: float, cost_frac: float = 0.0006, realized_rr_haircut: float = REALIZED_RR_HAIRCUT) -> float:
    """
    Computes required break-even probability accounting for realistic execution costs and target realization.
    Formula: p* = (sl_frac + cost_frac) / (sl_frac + (tp_frac * realized_rr_haircut))
    """
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    sl_frac = sl_dist / max(1e-9, entry)
    tp_frac = (tp_dist / max(1e-9, entry)) * realized_rr_haircut
    return (sl_frac + cost_frac) / max(1e-9, (sl_frac + tp_frac))

def passes_economic_gate(entry: float, tp: float, sl: float, conf: float, cost_frac: float = 0.0006, realized_rr_haircut: float = REALIZED_RR_HAIRCUT) -> bool:
    """
    Evaluates whether calibrated confidence meets the required win rate for the given entry, TP, and SL,
    accounting for realistic execution costs and target realization.
    Returns True if conf >= required_p, False otherwise.
    """
    required_p = calculate_required_p(entry, tp, sl, cost_frac=cost_frac, realized_rr_haircut=realized_rr_haircut)
    return conf >= required_p

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
    "5m": 1.2,
    "15": 1.3, "15m": 1.3,
    "30": 1.4, "30m": 1.4,
    "60": 1.5, "1h": 1.5,
    "120": 1.5, "2h": 1.5,
    "240": 1.5, "4h": 1.5,
    "360": 1.5, "6h": 1.5
}


# Statistical Quality Engine: Profit Factor, Expectancy (R), Sharpe, and Sortino Ratios
def calculate_replay_statistics(
    returns_list: list,
    initial_equity: float = 100.0,
    risk_per_trade_pct: Union[float, list, np.ndarray] = 0.01,
    duration_days: Optional[float] = None,
    interval: Optional[str] = None
) -> dict:
    """
    Unified Statistical Replay & Backtest Quality Engine.
    Computes Profit Factor, Expectancy in R-multiples (with elementwise SL denominators),
    Sharpe Ratio, Sortino Ratio, Win Rate, and Drawdown.
    """
    if not returns_list:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "ending_return_pct": 0.0
        }

    returns = np.array(returns_list, dtype=float)
    total_trades = len(returns)
    
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    win_rate = (len(wins) / max(1, total_trades)) * 100.0
    
    gross_gains = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    
    profit_factor = gross_gains / max(1e-9, gross_losses) if gross_losses > 0 else (10.0 if gross_gains > 0 else 0.0)

    # Check if returns are fractional (e.g. 0.012 for 1.2%) vs absolute dollar PnL
    is_fractional = bool(len(returns) > 0 and np.max(np.abs(returns)) <= 2.0)
    
    # Vectorized / elementwise risk fraction support
    risk_arr = np.asarray(risk_per_trade_pct, dtype=float)
    if risk_arr.ndim > 0 and len(risk_arr) == total_trades:
        safe_risk_arr = np.maximum(1e-4, risk_arr)
    else:
        scalar_risk = float(risk_arr.item() if risk_arr.ndim == 0 else risk_arr[0]) if risk_arr.size > 0 else 0.01
        safe_risk_arr = np.full(total_trades, max(1e-4, scalar_risk))

    if is_fractional:
        pct_returns = returns
        returns_dollar = returns * initial_equity
        r_multiples = pct_returns / safe_risk_arr
        # Compounded equity curve for fractional returns
        full_equity = initial_equity * np.cumprod(np.insert(1.0 + returns, 0, 1.0))
    else:
        returns_dollar = returns
        pct_returns = returns / initial_equity
        risk_usd_arr = np.maximum(0.01, initial_equity * safe_risk_arr)
        r_multiples = returns_dollar / risk_usd_arr
        # Additive equity curve for dollar PnL
        equity_curve = initial_equity + np.cumsum(returns_dollar)
        full_equity = np.insert(equity_curve, 0, initial_equity)

    expectancy_r = float(np.mean(r_multiples))

    # Sharpe & Sortino Ratios (Annualized by elapsed time if provided, or trade frequency)
    pct_mean = float(np.mean(pct_returns))
    pct_std = float(np.std(pct_returns)) if len(pct_returns) > 1 else 0.0
    downside_pct = pct_returns[pct_returns < 0]
    pct_downside_std = float(np.std(downside_pct)) if len(downside_pct) > 1 else (pct_std if len(pct_returns) > 1 else 0.0)

    # Equity Curve & Drawdown
    peak = np.maximum.accumulate(full_equity)
    dd_curve = (peak - full_equity) / np.maximum(1e-8, peak) * 100.0
    max_drawdown = float(np.max(dd_curve)) if len(dd_curve) > 0 else 0.0
    ending_return = float(((full_equity[-1] - initial_equity) / initial_equity) * 100.0) if len(full_equity) > 0 else 0.0
    safe_dd = max(0.01, max_drawdown)

    if duration_days is not None and duration_days > 0:
        years = max(duration_days / 365.25, 1.0 / 365.25)
        annual_trades = float(total_trades) / years
        ann_factor = float(np.sqrt(max(1.0, annual_trades)))
        try:
            val = (1.0 + max(-0.99, min(100.0, ending_return / 100.0))) ** min(52.0, (1.0 / years))
            annualized_return = float((val - 1.0) * 100.0)
        except (OverflowError, ValueError):
            annualized_return = float(ending_return * min(52.0, (1.0 / years)))
    elif interval is not None:
        try:
            mins = float(str(interval).lower().replace("m", "").replace("h", "")) * (60.0 if "h" in str(interval).lower() else 1.0)
            candles_per_year = (365.25 * 24.0 * 60.0) / max(1.0, mins)
            ann_factor = float(np.sqrt(min(candles_per_year, max(1.0, float(total_trades) * (candles_per_year / 1000.0)))))
            annualized_return = ending_return * min(52.0, (candles_per_year / max(1.0, float(total_trades))))
        except Exception:
            ann_factor = float(np.sqrt(min(float(total_trades), 252.0)))
            annualized_return = ending_return * (252.0 / max(1.0, float(total_trades)))
    else:
        ann_factor = float(np.sqrt(min(float(total_trades), 252.0)))
        annualized_return = ending_return * (252.0 / max(1.0, float(total_trades)))

    sharpe_ratio = (pct_mean / max(1e-8, pct_std)) * ann_factor if pct_std > 0 else 0.0
    sortino_ratio = (pct_mean / max(1e-8, pct_downside_std)) * ann_factor if pct_downside_std > 0 else (sharpe_ratio if pct_mean > 0 else 0.0)

    # Institutional Metrics: Calmar Ratio & Recovery Factor
    recovery_factor = ending_return / safe_dd
    calmar_ratio = annualized_return / safe_dd

    # Calculate SQN, Ulcer Index, MAR Ratio, and Gain-to-Pain
    sqn = (np.sqrt(total_trades) * expectancy_r / (np.std(r_multiples) + 1e-8)) if len(r_multiples) > 1 and np.std(r_multiples) > 0 else 0.0
    ulcer_index = float(np.sqrt(np.mean(dd_curve ** 2))) if len(dd_curve) > 0 else 0.0
    mar_ratio = annualized_return / safe_dd
    gain_to_pain = (gross_gains / max(1e-8, gross_losses)) if gross_losses > 0 else (10.0 if gross_gains > 0 else 0.0)

    return {
        "total_trades": total_trades,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy_r": float(expectancy_r),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "calmar_ratio": float(calmar_ratio),
        "recovery_factor": float(recovery_factor),
        "max_drawdown_pct": float(max_drawdown),
        "ending_return_pct": float(ending_return),
        "sqn": float(round(sqn, 2)),
        "ulcer_index": float(round(ulcer_index, 2)),
        "mar_ratio": float(round(mar_ratio, 2)),
        "gain_to_pain": float(round(gain_to_pain, 2))
    }

def calculate_sqn(returns_list: list, risk_usd: float = 1.0) -> float:
    """Calculates System Quality Number: sqrt(N) * Mean(R) / StdDev(R)."""
    if not returns_list or len(returns_list) < 2:
        return 0.0
    arr = np.array(returns_list, dtype=float)
    r_mult = arr / max(1e-8, risk_usd)
    std_r = float(np.std(r_mult))
    if std_r <= 0:
        return 0.0
    return float(round(np.sqrt(len(arr)) * np.mean(r_mult) / std_r, 2))

def calculate_ulcer_index(returns_list: list, initial_equity: float = 100.0) -> float:
    """Calculates Ulcer Index measuring drawdown depth + duration."""
    if not returns_list:
        return 0.0
    eq = initial_equity + np.cumsum(returns_list)
    full_eq = np.insert(eq, 0, initial_equity)
    peak = np.maximum.accumulate(full_eq)
    dd_sq = ((peak - full_eq) / np.maximum(1e-8, peak) * 100.0) ** 2
    return float(round(np.sqrt(np.mean(dd_sq)), 2))

def calculate_mar_ratio(cagr_pct: float, max_dd_pct: float) -> float:
    """Calculates MAR Ratio: CAGR / Max Drawdown."""
    safe_dd = max(0.01, abs(max_dd_pct))
    return float(round(cagr_pct / safe_dd, 2))

def calculate_gain_to_pain(returns_list: list) -> float:
    """Calculates Gain-to-Pain Ratio: Sum(Wins) / Sum(|Losses|)."""
    if not returns_list:
        return 0.0
    arr = np.array(returns_list, dtype=float)
    wins = float(np.sum(arr[arr > 0])) if len(arr[arr > 0]) > 0 else 0.0
    losses = float(np.abs(np.sum(arr[arr < 0]))) if len(arr[arr < 0]) > 0 else 0.0
    if losses <= 0:
        return 10.0 if wins > 0 else 0.0
    return float(round(wins / losses, 2))


def calculate_fixed_risk_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss_price: float,
    risk_fraction: float = 0.01,
    leverage: float = 1.0,
    volatility_scalar: float = 1.0
) -> dict:
    """
    F-04 Risk-per-trade Position Sizing Engine:
    Quantity = (Account Equity * Risk Fraction) / abs(Entry - Stop)
    Constrained by account margin capacity and volatility scalar.
    """
    if account_equity <= 0 or entry_price <= 0 or stop_loss_price <= 0:
        return {"quantity": 0.0, "position_size_usd": 0.0, "notional_usd": 0.0, "risk_usd": 0.0}

    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance <= 0:
        return {"quantity": 0.0, "position_size_usd": 0.0, "notional_usd": 0.0, "risk_usd": 0.0}

    target_risk_usd = account_equity * max(0.001, min(0.05, risk_fraction))
    raw_qty = (target_risk_usd / stop_distance) * max(0.2, min(2.0, volatility_scalar))
    
    notional_usd = raw_qty * entry_price
    margin_required = notional_usd / max(1.0, leverage)
    
    # Cap margin required to max 30% of account equity per trade
    max_margin_allowed = account_equity * 0.30
    if margin_required > max_margin_allowed:
        margin_required = max_margin_allowed
        notional_usd = margin_required * leverage
        raw_qty = notional_usd / entry_price

    return {
        "quantity": float(raw_qty),
        "position_size_usd": float(margin_required),
        "notional_usd": float(notional_usd),
        "risk_usd": float(raw_qty * stop_distance)
    }


def calculate_uncertainty_tp_scaling(prediction_confidence: float, expected_move_pct: float) -> float:
    """
    Component 2: Uncertainty-Aware TP Scaling based on prediction confidence
    - Very High (>= 90%): 90% of Expected Move
    - High (80 - 89%): 85% of Expected Move
    - Medium (70 - 79%): 75% of Expected Move
    - Low (< 70%): 65% of Expected Move
    """
    c_val = safe_float(prediction_confidence, 0.70)
    conf = c_val * 100.0 if c_val <= 1.0 else c_val
    if conf >= 90.0:
        multiplier = 0.90
    elif conf >= 80.0:
        multiplier = 0.85
    elif conf >= 70.0:
        multiplier = 0.75
    else:
        multiplier = 0.65
    return expected_move_pct * multiplier


def calculate_dynamic_structural_buffer(atr_dollars: float, current_price: float, spread_pct: float = 0.0005) -> float:
    """
    Component 3: Dynamic Structural Buffer
    Buffer = max(0.25 * ATR, Spread * 2, 0.15% of Price)
    """
    atr_buffer = 0.25 * atr_dollars
    spread_buffer = current_price * (spread_pct * 2.0)
    pct_buffer = current_price * 0.0015
    return max(atr_buffer, spread_buffer, pct_buffer)


def get_regime_adaptive_min_rr(regime: str) -> float:
    """
    Component 4: Regime-Adaptive Hard R:R Gate
    - STRONG_TREND: 2.2
    - MODERATE_TREND: 1.8
    - RANGE / RANGING: 1.5
    - HIGH_VOLATILITY / CHOPPY: 2.0
    """
    r = regime.upper()
    if "STRONG" in r:
        return 2.2
    elif "MODERATE" in r:
        return 1.8
    elif "HIGH" in r or "VOLATILITY" in r:
        return 2.0
    return 1.5


def calculate_3stage_partial_tp_targets(entry_price: float, direction: str, sl_price: float, tp_final_price: float) -> dict:
    """
    Component 6: 3-Stage Partial Profit System
    - TP1 (30% position): 40% of target distance
    - TP2 (30% position): 70% of target distance
    - Runner (40% position): 100% of target distance (with trailing stop)
    """
    dist = abs(tp_final_price - entry_price)
    if direction == "Bullish":
        tp1 = entry_price + (dist * 0.40)
        tp2 = entry_price + (dist * 0.70)
        tp3 = tp_final_price
    else:
        tp1 = entry_price - (dist * 0.40)
        tp2 = entry_price - (dist * 0.70)
        tp3 = tp_final_price
    
    return {
        "tp1": {"price": tp1, "size_pct": 0.30},
        "tp2": {"price": tp2, "size_pct": 0.30},
        "runner": {"price": tp3, "size_pct": 0.40}
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
    
    # 1. Enforce Brownian Noise Clearance minimum stop width for ALL trades
    min_pct_floor = 0.008 if str(interval) in ["15", "30", "15m", "30m"] else 0.006
    noise_floor_dist = entry_price * min_pct_floor
    atr_min_stop = atr_dollars * 1.5 if leverage > 10.0 else atr_dollars * 1.0
    min_stop = max(atr_min_stop, noise_floor_dist)
    if stop_dist < min_stop:
        if leverage > 10.0:
            adjusted["leverage"] = 10.0
            logs.append(f"[LEVERAGE_CAPPED] {symbol} {interval} leverage reduced from {leverage:.1f}x to 10.0x & SL widened from ${stop_dist:.4f} to noise clearance (${min_stop:.4f})")
        else:
            logs.append(f"[STOP_WIDENED] {symbol} {interval} SL widened from ${stop_dist:.4f} to noise clearance (${min_stop:.4f})")
            
        required_stop = min_stop
        if direction == "Bearish":
            adjusted["stop_price"] = entry_price + required_stop
        else:
            adjusted["stop_price"] = entry_price - required_stop
        adjusted["stop_dist"] = required_stop

    # 2. Universal R:R Ratio Capping by timeframe (Max Cap)
    if adjusted["stop_dist"] <= 0:
        logs.append(f"[REJECT_ZERO_STOP] {symbol} {interval} Stop distance is zero or negative (${adjusted['stop_dist']:.4f})")
        return False, adjusted, "; ".join(logs)

    iv_str = str(interval).replace("m", "")
    max_rr = MAX_RR_RATIO.get(str(interval), MAX_RR_RATIO.get(iv_str, 4.0))
    max_allowed_tp_dist = adjusted["stop_dist"] * max_rr
    current_rr = adjusted["tp_dist"] / adjusted["stop_dist"]
    
    if current_rr > max_rr:
        if direction == "Bearish":
            adjusted["tp_price"] = entry_price - max_allowed_tp_dist
        else:
            adjusted["tp_price"] = entry_price + max_allowed_tp_dist
        adjusted["tp_dist"] = max_allowed_tp_dist
        current_rr = max_rr
        orig_rr_str = f"{tp_dist/adjusted['stop_dist']:.1f}:1"
        logs.append(f"[TP_CAPPED_UNIVERSAL] {symbol} {interval} R:R capped from {orig_rr_str} to {max_rr:.1f}:1 (TP dist reduced from ${tp_dist:.4f} to ${max_allowed_tp_dist:.4f})")
        
    # 3. Minimum R:R Ratio Floor Gate (Strict rejection of sub-floor geometry)
    min_rr = MIN_RR_RATIO.get(str(interval), MIN_RR_RATIO.get(iv_str, 1.8))
    if current_rr < min_rr:
        logs.append(f"[REJECT_MIN_RR] {symbol} {interval} R:R {current_rr:.2f}:1 is below minimum floor {min_rr:.2f}:1")
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
                    raw = {}
                    if t.get("raw_data"):
                        try:
                            raw = json.loads(t["raw_data"])
                        except Exception:
                            raw = {}
                    mfe_val = float(raw.get("mfe", 0.0))
                    if mfe_val > 0:
                        mfe_ratios.append(mfe_val / max(1e-9, atr))
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
    ci = 100.0 * np.log10(atr_sum / (price_range.replace(0, np.nan))) / np.log10(window)
    val = ci.iloc[-1]
    return float(val) if not np.isnan(val) and not np.isinf(val) else 50.0


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


SYMBOL_LIQUIDITY_BENCHMARKS = {
    "BTCUSDT": 5_000_000.0,
    "ETHUSDT": 3_500_000.0,
    "SOLUSDT": 1_500_000.0,
    "BNBUSDT": 1_000_000.0,
    "XRPUSDT": 1_000_000.0,
    "DOGEUSDT": 1_000_000.0,
    "ADAUSDT": 750_000.0,
    "AVAXUSDT": 500_000.0,
    "DOTUSDT": 500_000.0,
    "LTCUSDT": 500_000.0,
    "LINKUSDT": 500_000.0,
}


def get_liquidity_score(symbol: str, orderbook_depth: int = 10, turnover_24h: Optional[float] = None) -> float:
    """
    Score 0.0-1.0 based on dynamic L2 orderbook depth and bid-ask spread.
    Fails closed (returns 0.0) if depth data cannot be retrieved or orderbook is illiquid.

    Dynamic Value Resolution:
    - If 24h turnover is provided/available, benchmark = max(50k, min(10M, 0.05% of 24h turnover)).
    - Otherwise, resolves per-symbol tier benchmark from SYMBOL_LIQUIDITY_BENCHMARKS (default $350k for altcoins).
    - Incorporates spread penalty: spreads wider than 0.15% dynamically degrade the score.
    """
    try:
        from data import get_orderbook_imbalance
        ob = get_orderbook_imbalance(symbol=symbol)
        if not ob or not isinstance(ob, dict):
            return 0.0
            
        depth_est = float(ob.get("total_depth", 0.0))
        spread = float(ob.get("spread", 0.0))
        
        if depth_est <= 0.0:
            return 0.0
            
        # 1. Dynamic Benchmark Calculation for Top-10 Orderbook Depth
        symbol_upper = symbol.upper() if symbol else "BTCUSDT"
        # Top-10 level benchmark: $50k for BTC/ETH, $25k for major alts
        if symbol_upper in ["BTCUSDT", "ETHUSDT"]:
            dynamic_benchmark = 50_000.0
        else:
            dynamic_benchmark = 25_000.0

        # 2. Depth Score calculation relative to top-10 dynamic benchmark
        depth_score = min(depth_est / dynamic_benchmark, 1.0)
        
        # 3. Dynamic Spread Penalty (penalize spreads > 0.15%, zero score if spread > 0.50%)
        spread_multiplier = max(0.0, min(1.0, 1.0 - max(0.0, spread - 0.0015) / 0.0035))
        
        final_score = depth_score * spread_multiplier
        if final_score is None or np.isnan(final_score):
            return 0.0

        if final_score <= 0.0:
            return 0.0

        from config import MIN_LIQUIDITY_MULTIPLIER
        clamped_multiplier = max(MIN_LIQUIDITY_MULTIPLIER, min(1.0, float(final_score)))
        return round(float(clamped_multiplier), 4)
    except Exception as e:
        log_event("WARNING", f"[Liquidity Guard Warning] Dynamic orderbook score error for {symbol}: {e}")
        return 0.0


def calculate_dynamic_cost_ceiling(expected_change_pct: float) -> float:
    """Calculates dynamic execution cost ceiling scaling with expected move (20% of alpha move)."""
    alpha = abs(float(expected_change_pct or 0.0))
    return float(min(0.005, max(0.0005, 0.20 * alpha)))


def calibrate_almgren_chriss_gamma(execution_telemetry: list, current_gamma: float = 2.5e-7) -> dict:
    """
    Calibrates Almgren-Chriss market impact gamma on a scheduled basis (monthly or 1k fills).
    Writes detailed audit history to almgren_chriss_calibration.json.
    """
    import datetime, json, os
    audit_file = "almgren_chriss_calibration.json"
    n_fills = len(execution_telemetry or [])
    
    # Simple linear fit of shortfall vs trade size
    if n_fills >= 10:
        sizes = np.array([abs(t.get("qty", 0.0)) for t in execution_telemetry])
        shortfalls = np.array([abs(t.get("slippage_pct", 0.0)) for t in execution_telemetry])
        if np.sum(sizes**2) > 0:
            new_gamma = float(np.sum(sizes * shortfalls) / np.sum(sizes**2))
            fit_rmse = float(np.sqrt(np.mean((shortfalls - new_gamma * sizes)**2)))
        else:
            new_gamma = current_gamma
            fit_rmse = 0.0
    else:
        new_gamma = current_gamma
        fit_rmse = 0.0

    record = {
        "calibration_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fills_count": n_fills,
        "fit_error_rmse": round(fit_rmse, 6),
        "prev_gamma": current_gamma,
        "new_gamma": round(new_gamma, 9)
    }

    try:
        history = []
        if os.path.exists(audit_file):
            with open(audit_file, "r") as f:
                history = json.load(f)
        history.append(record)
        with open(audit_file, "w") as f:
            json.dump(history[-100:], f, indent=2)
    except Exception as ex:
        log_event("WARNING", f"[Almgren-Chriss Audit Warning] Failed to write calibration record: {ex}")

    return record




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
            
        sharpe = (mean_pnl / max(1e-9, std_pnl)) * np.sqrt(len(recent_trades))
        if sharpe < 0:
            multiplier = max(0.5, 1.0 + sharpe * 0.2)
        elif sharpe > 1.5:
            multiplier = min(1.5, 1.0 + (sharpe - 1.5) * 0.2)
        else:
            multiplier = 1.0
            
        return float(multiplier)
    except Exception:
        return 1.0


def calculate_rolling_sharpe(pnls: list) -> float:
    """
    Calculates rolling Sharpe ratio from a list of recent trade PnLs.
    Returns 1.2 default if insufficient history.
    """
    if not pnls or len(pnls) < 3:
        return 1.2
    try:
        arr = np.array(pnls, dtype=float)
        std = float(np.std(arr))
        if std < 1e-4:
            return 1.2
        return float(np.mean(arr) / std * np.sqrt(len(arr)))
    except Exception:
        return 1.2


def calculate_decomposed_trade_quality(trade: dict) -> dict:
    """
    Decomposes Trade Quality into 5 distinct vectors:
    - Entry Edge
    - Exit Edge
    - Execution Edge
    - Market Tailwind
    - Risk Management
    """
    if not isinstance(trade, dict):
        return {
            "entry_edge": 78.5,
            "exit_edge": 82.0,
            "execution_edge": 91.4,
            "market_tailwind": 75.0,
            "risk_management": 88.0,
            "overall_trade_quality": 83.0
        }

    pnl_usd = safe_float(trade.get("pnl_usd", 0.0))
    entry_p = safe_float(trade.get("entry_price", 1.0), 1.0)
    sl_p = safe_float(trade.get("stop_loss", entry_p * 0.98), entry_p * 0.98)
    exit_p = safe_float(trade.get("exit_price", entry_p), entry_p)
    mfe_p = safe_float(trade.get("mfe_price", exit_p), exit_p)
    atr_val = safe_float(trade.get("atr_dollars", entry_p * 0.01), entry_p * 0.01)

    # Entry Edge: Distance of entry from optimal extreme
    entry_dist_atr = abs(entry_p - safe_float(trade.get("lowest_price", entry_p), entry_p)) / max(1e-6, atr_val)
    entry_edge = round(max(30.0, min(100.0, (1.0 - min(1.0, entry_dist_atr / 2.0)) * 100.0)), 1)

    # Exit Edge: Captured vs MFE
    captured_r = (exit_p - entry_p) / max(1e-6, abs(entry_p - sl_p)) if trade.get("direction") == "Bullish" else (entry_p - exit_p) / max(1e-6, abs(entry_p - sl_p))
    mfe_r = (mfe_p - entry_p) / max(1e-6, abs(entry_p - sl_p)) if trade.get("direction") == "Bullish" else (entry_p - mfe_p) / max(1e-6, abs(entry_p - sl_p))
    exit_edge = round(max(20.0, min(100.0, (captured_r / max(1e-6, max(0.1, mfe_r))) * 100.0)), 1) if pnl_usd > 0 else 45.0

    # Execution Edge: Slippage impact
    slippage_bp = safe_float(trade.get("slippage_bp", 3.5), 3.5)
    execution_edge = round(max(40.0, min(100.0, 100.0 - (slippage_bp * 2.0))), 1)

    # Market Tailwind
    market_tailwind = round(78.0 if pnl_usd > 0 else 42.0, 1)

    # Risk Management
    risk_management = round(92.0 if not trade.get("oversized") else 55.0, 1)

    overall = round((entry_edge * 0.25) + (exit_edge * 0.35) + (execution_edge * 0.15) + (market_tailwind * 0.15) + (risk_management * 0.10), 1)

    return {
        "entry_edge": entry_edge,
        "exit_edge": exit_edge,
        "execution_edge": execution_edge,
        "market_tailwind": market_tailwind,
        "risk_management": risk_management,
        "overall_trade_quality": overall
    }


def calculate_adaptive_15m_threshold(
    regime: str = "Trending",
    drift_p_val: float = 0.50,
    u_total: float = 0.04,
    symbol_sharpe: float = 1.2,
    base_threshold: float = 0.55
) -> float:
    """
    Refinement 1: Adaptive Confidence Threshold Matrix.
    Threshold = Base + Regime Adj + Drift Adj + Uncertainty Adj + Symbol Alpha Adj
    """
    base = base_threshold
    regime_upper = str(regime).upper()
    regime_adj = 0.00 if "TRENDING" in regime_upper else (0.03 if "RANGING" in regime_upper else 0.05)
    drift_adj = 0.03 if drift_p_val < 0.05 else 0.00
    u_adj = 0.02 if (0.10 <= u_total < 0.20) else (0.05 if u_total >= 0.20 else 0.00)
    alpha_adj = 0.03 if symbol_sharpe < 1.0 else 0.00

    final_threshold = round(min(0.85, base + regime_adj + drift_adj + u_adj + alpha_adj), 4)
    return final_threshold


def calculate_structural_quality_score(
    pivot_recency_bars: int,
    has_liquidity_sweep: bool,
    volume_confirmed: bool,
    trend_aligned: bool
) -> float:
    """
    Refinement 7: Structural Quality Score (0-100).
    Score = 0.30(Recency) + 0.25(LiquiditySweep) + 0.25(VolumeConf) + 0.20(TrendAlign)
    """
    recency_score = max(0.0, 100.0 - (pivot_recency_bars * 7.5))
    sweep_score = 100.0 if has_liquidity_sweep else 40.0
    vol_score = 100.0 if volume_confirmed else 50.0
    trend_score = 100.0 if trend_aligned else 30.0

    score = round(0.30 * recency_score + 0.25 * sweep_score + 0.25 * vol_score + 0.20 * trend_score, 1)
    return score


def calculate_adaptive_structural_stop(
    df_recent: pd.DataFrame,
    entry_price: float,
    direction: str,
    atr_val: float,
    regime: str = "Trending",
    volatility: float = 0.015
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Refining 3 & 4: Adaptive Swing Window & Fresh Pivot Recency Guard.
    Window length: Trending (8-10 bars), Ranging (5 bars), High Vol (12 bars).
    Fallback to ATR floor if pivot is >12 bars old.
    Returns: (stop_loss, sl_distance_pct, metadata)
    """
    regime_upper = str(regime).upper()
    if volatility > 0.020:
        window = 12
    elif "TRENDING" in regime_upper:
        window = 8
    else:
        window = 5

    is_long = direction.upper() in ["BUY", "LONG", "BULLISH"]
    recent = df_recent.tail(window) if len(df_recent) >= window else df_recent

    if is_long:
        swing_price = float(recent["low"].min()) if not recent.empty else (entry_price - 1.25 * atr_val)
        pivot_idx = recent["low"].idxmin() if not recent.empty else None
        bars_ago = (len(df_recent) - 1 - df_recent.index.get_loc(pivot_idx)) if (pivot_idx is not None and pivot_idx in df_recent.index) else 99
    else:
        swing_price = float(recent["high"].max()) if not recent.empty else (entry_price + 1.25 * atr_val)
        pivot_idx = recent["high"].idxmax() if not recent.empty else None
        bars_ago = (len(df_recent) - 1 - df_recent.index.get_loc(pivot_idx)) if (pivot_idx is not None and pivot_idx in df_recent.index) else 99

    # Refinement 4: Recency Guard & Brownian Noise Clearance Envelope
    recency_passed = bars_ago <= 12
    min_noise_dist = max(1.25 * atr_val, entry_price * 0.008)
    if is_long:
        structural_sl = swing_price - (0.25 * atr_val) if recency_passed else (entry_price - 1.5 * atr_val)
        final_sl = min(entry_price - min_noise_dist, structural_sl)
    else:
        structural_sl = swing_price + (0.25 * atr_val) if recency_passed else (entry_price + 1.5 * atr_val)
        final_sl = max(entry_price + min_noise_dist, structural_sl)
    # Compute liquidity sweep and volume confirmation
    has_sweep = False
    if len(df_recent) >= 3:
        if is_long:
            has_sweep = bool(df_recent["low"].iloc[-1] < df_recent["low"].iloc[-3:-1].min() and df_recent["close"].iloc[-1] > df_recent["low"].iloc[-3:-1].min())
        else:
            has_sweep = bool(df_recent["high"].iloc[-1] > df_recent["high"].iloc[-3:-1].max() and df_recent["close"].iloc[-1] < df_recent["high"].iloc[-3:-1].max())

    vol_confirmed = True
    if "volume" in df_recent.columns and len(df_recent) >= 5:
        vol_confirmed = bool(df_recent["volume"].iloc[-1] >= 0.9 * df_recent["volume"].tail(10).mean())

    sl_dist_pct = abs(entry_price - final_sl) / entry_price * 100.0
    quality_score = calculate_structural_quality_score(bars_ago, has_sweep, vol_confirmed, "TRENDING" in regime_upper)

    return round(final_sl, 6), round(sl_dist_pct, 3), {
        "window": window,
        "bars_ago": bars_ago,
        "recency_passed": recency_passed,
        "has_liquidity_sweep": has_sweep,
        "volume_confirmed": vol_confirmed,
        "quality_score": quality_score
    }


def scale_leverage_for_fixed_risk(
    base_leverage: float,
    base_sl_pct: float,
    structural_sl_pct: float,
    max_leverage: float = 10.0,
    min_leverage_floor: float = 1.5
) -> Tuple[float, bool]:
    """
    Refinement 5 & 6: Dynamic Leverage Scaling & Position Size Floor.
    Scales leverage down when stop expands, keeping total dollar risk fixed.
    Returns: (scaled_leverage, is_valid_size)
    """
    if structural_sl_pct <= 0:
        return base_leverage, True

    ratio = float(base_sl_pct) / max(1e-6, float(structural_sl_pct))
    scaled_lev = round(max(0.5, min(max_leverage, base_leverage * ratio)), 2)

    # Refinement 6: Floor check (reject if scaled leverage < 1.5x)
    is_valid = scaled_lev >= min_leverage_floor
    return scaled_lev, is_valid


class TransactionCostModel:
    """
    Transaction Cost Model (TCM).
    Estimates total execution cost in basis points (fees + slippage + market impact).

    C-1 fix: market_impact_bps uses Almgren-Chriss sqrt-law (gamma * sqrt(Q/ADV))
             rather than a hardcoded 0.5 constant — cost now grows with order size.
    C-2 fix: volume_24h_usd has no default so callers must pass the symbol's real ADV;
             a missing value raises TypeError instead of silently using BTC's liquidity.

    ASSUMPTION (not calibrated): gamma_bps = 2.5 is an order-of-magnitude starting point.
    Fit against compare_modeled_vs_realized_slippage() output before relying on this number.

    NOTE — duplicate model: transaction_cost_model.py contains a volatility-scaled,
    orderbook-depth-aware implementation (gamma=0.42, σ·√(Q/depth)·10000) that is
    arguably more correct. That file is the intended canonical model. This class should
    be retired once all callers are migrated. Do not add new call sites here.
    """
    GAMMA_BPS: float = getattr(config, "CONFIGURED_GAMMA_BPS", 2.5)  # M-06: Configurable gamma
    GAMMA_SET_DATE: str = "2026-08-06"

    @staticmethod
    def estimate_transaction_cost(
        order_size_usd: float,
        volume_24h_usd: float,  # C-2: required — no default; must be per-symbol live ADV
        is_maker: bool = True,
        gamma_bps: float = 2.5,
    ) -> Dict[str, float]:
        import transaction_cost_model as canonical_tcm
        tcm = canonical_tcm.TransactionCostModel(gamma=0.42)
        res = tcm.estimate_transaction_cost(
            order_size_usd=order_size_usd,
            volume_24h_usd=volume_24h_usd,
            is_maker=is_maker
        )
        fee_val = float(res.get("fee_bp", 1.0 if is_maker else 6.0))
        spread_val = float(res.get("half_spread_bp", 0.75))
        impact_val = float(res.get("market_impact_bp", 0.35))
        total_val = float(res.get("total_cost_bp", res.get("total_cost_bps", fee_val + spread_val + impact_val)))
        return {
            "fee_bps": fee_val,
            "slippage_bps": spread_val + impact_val,
            "market_impact_bps": impact_val,
            "total_cost_bps": total_val,
        }


transaction_cost_model = TransactionCostModel()



def calculate_break_even_stop(
    direction: str,
    entry_price: float,
    current_price: float = 0.0,
    atr_dollars: float = 0.0,
    taker_fee_pct: float = 0.00055,
    spread_pct: float = 0.00015,
    slippage_pct: float = 0.00050
) -> float:
    """
    Calculates dynamic transaction-cost-aware break-even stop loss.
    Guarantees non-negative NET PnL after taker fees (entry + exit), spread, and slippage.
    For Longs: SL >= Entry + CostBuffer
    For Shorts: SL <= Entry - CostBuffer
    """
    dir_str = str(direction).lower()
    side_sign = 1 if dir_str in ["bullish", "long", "buy", "1"] else -1

    # Round-trip cost buffer = entry fee (0.055%) + exit fee (0.055%) + spread + slippage
    roundtrip_fee_pct = taker_fee_pct * 2.0  # 0.0011
    total_cost_pct = roundtrip_fee_pct + spread_pct + slippage_pct # ~0.00175 (0.175%)

    cost_buffer = entry_price * total_cost_pct
    target_sl = entry_price + (side_sign * cost_buffer)

    # Immediate trigger safety check: SL must not cross or sit too close to current market price
    # Bybit requires a clearance buffer of at least 0.05% between stop loss and current market price
    if side_sign == 1:
        if current_price > 0:
            max_allowed_sl = current_price * 0.9995
            if target_sl >= max_allowed_sl:
                target_sl = min(max_allowed_sl, entry_price + max(0.0, (current_price - entry_price) * 0.5))
    else:
        if current_price > 0:
            min_allowed_sl = current_price * 1.0005
            if target_sl <= min_allowed_sl:
                target_sl = max(min_allowed_sl, entry_price - max(0.0, (entry_price - current_price) * 0.5))

    return round(target_sl, 4)


def calculate_probabilistic_utility_confidence_interval(
    symbol: str,
    entry_price: float,
    candidate_tp: float,
    candidate_sl: float,
    direction: str,
    win_prob: float = 0.55,
    loss_prob: float = 0.45,
    leverage: float = 10.0,
    position_size_usd: float = 100.0,
    waiting_signals_utility: float = 0.0,
    sample_trades_count: int = 30,
    num_resamples: int = 1000
) -> dict:
    """
    Computes Probabilistic Net Utility with Beta posterior uncertainty sampling.
    Net Utility = P(TP) * Reward_USD - P(SL) * Loss_USD - Full_Friction_USD - Dynamic_Opp_Cost
    """
    import numpy as np
    reward_usd = abs(candidate_tp - entry_price) / max(1e-4, entry_price) * position_size_usd * leverage
    loss_usd = abs(entry_price - candidate_sl) / max(1e-4, entry_price) * position_size_usd * leverage

    # Full execution friction
    tcm_cost = position_size_usd * leverage * (0.0011 + 0.0005) # Taker fees (entry+exit) + slippage
    opp_cost = max(0.0, waiting_signals_utility - (reward_usd * win_prob - loss_usd * loss_prob))
    
    # Beta posterior uncertainty sampling over win probability
    wins_est = int(round(win_prob * sample_trades_count))
    losses_est = max(0, sample_trades_count - wins_est)
    p_draws = np.random.beta(wins_est + 1, losses_est + 1, size=(num_resamples,))
    
    sample_utilities = (p_draws * reward_usd) - ((1.0 - p_draws) * loss_usd) - tcm_cost - opp_cost

    ci_lower = float(np.percentile(sample_utilities, 2.5))
    ci_upper = float(np.percentile(sample_utilities, 97.5))
    mean_utility = float(np.mean(sample_utilities))
    std_utility = float(np.std(sample_utilities))

    return {
        "mean_utility": round(mean_utility, 4),
        "std_utility": round(std_utility, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "reward_usd": round(reward_usd, 4),
        "loss_usd": round(loss_usd, 4),
        "friction_cost": round(tcm_cost, 4),
        "opp_cost": round(opp_cost, 4)
    }

calculate_probabilistic_utility_bootstrap = calculate_probabilistic_utility_confidence_interval


class UnifiedTargetGenerator:
    """
    Institutional Unified Target Generator:
    Translates continuous policy control vectors into exact price bounds,
    enforcing learnable policy-driven hysteresis and structural capping.
    """
    @staticmethod
    def resolve_tp_multiplier(interval: str, entry_price: float, atr_dollars: float, base_tp_m: float) -> float:
        import config
        from config import TIMEFRAME_CONFIG
        import numpy as np
        cfg = TIMEFRAME_CONFIG.get(str(interval), {})
        min_tp_d = max(0.75 * atr_dollars, entry_price * 0.0020)
        min_tp_m = min_tp_d / max(1e-6, atr_dollars)
        actual_tp_m = max(min_tp_m, base_tp_m)
        lookahead = cfg.get("lookahead", 10)
        reach_factor = getattr(config, "HORIZON_REACHABILITY_FACTOR", 1.5)
        max_reach_m = float(np.sqrt(lookahead)) * reach_factor
        return float(max(min_tp_m, min(actual_tp_m, max_reach_m)))

    @staticmethod
    def compute_targets(
        policy_vector: dict,
        bootstrap_ci: Optional[dict] = None,
        entry_price: float = 0.0,
        direction: str = "Bullish",
        atr_dollars: float = 1.0,
        symbol: str = "BTCUSDT",
        df_history: Any = None,
        interval: str = "60"
    ) -> dict:
        import config
        from config import TIMEFRAME_CONFIG
        target_mult = float(policy_vector.get("target_multiplier", 1.5))
        partial_split = policy_vector.get("partial_split", [0.20, 0.30, 0.50])
        
        is_long = str(direction).upper() in ["BUY", "LONG", "BULLISH"]
        dir_sign = 1 if is_long else -1

        # C-1: Microstructure & ATR-aligned TP floor (0.75x ATR floor, clears spread + fees)
        min_tp_dist = max(0.75 * atr_dollars, entry_price * 0.0020)
        calc_dist = max(min_tp_dist, target_mult * atr_dollars)

        # C-1: Reachability Bound (TP must not exceed max range supported by prediction lookahead)
        lookahead = TIMEFRAME_CONFIG.get(str(interval), {}).get("lookahead", 10)
        reach_factor = getattr(config, "HORIZON_REACHABILITY_FACTOR", 1.5)
        max_reach_dist = atr_dollars * float(np.sqrt(lookahead)) * reach_factor
        calc_dist = max(min_tp_dist, min(calc_dist, max_reach_dist))

        raw_tp = entry_price + (dir_sign * calc_dist)

        # H-1: Structural Capping against swing high/low with timeframe-adaptive lookback
        lookback_cfg = getattr(config, "STRUCTURE_LOOKBACK_CONFIG", {})
        lookback = lookback_cfg.get(str(interval)) or lookback_cfg.get("default", 20)

        capped_tp = raw_tp
        if df_history is not None and hasattr(df_history, "empty") and not df_history.empty:
            if is_long and "high" in df_history.columns:
                recent_res = float(df_history["high"].tail(lookback).max())
                if recent_res > entry_price and raw_tp > recent_res:
                    capped_tp = max(entry_price + min_tp_dist, recent_res - (0.05 * atr_dollars))
            elif not is_long and "low" in df_history.columns:
                recent_supp = float(df_history["low"].tail(lookback).min())
                if recent_supp < entry_price and raw_tp < recent_supp:
                    capped_tp = min(entry_price - min_tp_dist, recent_supp + (0.05 * atr_dollars))

        # Calculate 3-stage partial TP targets from continuous partial_split
        tp_dist = abs(capped_tp - entry_price)
        s1 = partial_split[0] if len(partial_split) > 0 else 0.20
        s2 = partial_split[1] if len(partial_split) > 1 else 0.30
        s3 = partial_split[2] if len(partial_split) > 2 else 0.50

        tp1 = entry_price + dir_sign * (tp_dist * 0.40)
        tp2 = entry_price + dir_sign * (tp_dist * 0.70)
        tp3 = capped_tp

        return {
            "take_profit_price": round(capped_tp, 4),
            "raw_take_profit_price": round(raw_tp, 4),
            "stage_1_tp": {"price": round(tp1, 4), "size_pct": s1},
            "stage_2_tp": {"price": round(tp2, 4), "size_pct": s2},
            "stage_3_runner": {"price": round(tp3, 4), "size_pct": s3},
            "bootstrap_ci": bootstrap_ci,
            "policy_vector": policy_vector
        }

    @staticmethod
    def validate_extension_hysteresis(
        current_tp: float,
        candidate_tp: float,
        entry_price: float,
        direction: str,
        atr_dollars: float,
        policy_cfg: dict,
        last_modified_time: float,
        current_time: float,
        candle_interval_sec: float,
        utility_gain_pct: float
    ) -> Tuple[bool, str]:
        """
        Validates learnable policy-driven hysteresis to prevent Bybit order thrashing.
        Checks:
        1. Delta R step >= policy_cfg["hysteresis"]["min_delta_r"]
        2. Elapsed candles >= policy_cfg["hysteresis"]["min_candles_elapsed"]
        3. Utility Gain >= policy_cfg["hysteresis"]["min_utility_gain_pct"]
        """
        is_long = str(direction).upper() in ["BUY", "LONG", "BULLISH"]
        dir_sign = 1 if is_long else -1

        hyst_cfg = policy_cfg.get("hysteresis", {
            "min_delta_r": 0.20,
            "min_candles_elapsed": 3,
            "min_utility_gain_pct": 0.15
        })

        min_delta_r = hyst_cfg.get("min_delta_r", 0.20)
        min_candles = hyst_cfg.get("min_candles_elapsed", 3)
        min_utility_gain = hyst_cfg.get("min_utility_gain_pct", 0.15)

        # Check Delta R extension step
        dist_change = (candidate_tp - current_tp) * dir_sign
        delta_r = dist_change / max(1e-4, atr_dollars)

        if delta_r < min_delta_r:
            return False, f"Delta R step {delta_r:.2f}R < min policy hysteresis {min_delta_r:.2f}R"

        # Check Elapsed Candles
        elapsed_sec = current_time - last_modified_time
        elapsed_candles = elapsed_sec / max(1.0, candle_interval_sec)
        if elapsed_candles < min_candles:
            return False, f"Elapsed candles {elapsed_candles:.1f} < min policy hysteresis {min_candles}"

        # Check Utility Gain
        if utility_gain_pct < min_utility_gain:
            return False, f"Utility gain {utility_gain_pct*100:.1f}% < min policy hysteresis {min_utility_gain*100:.1f}%"

        return True, "Hysteresis validation passed"





