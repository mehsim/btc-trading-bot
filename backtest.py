import os
import config
import sys
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# Add workspace path to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logger import log_event
from data import get_history, merge_derivatives_sentiment_features
from core import (
    add_features,
    calibrate_confidence,
    calculate_historical_thresholds,
    features,
    TIMEFRAME_CONFIG
)
import argparse

parser = argparse.ArgumentParser(description="BTC Trading Bot Backtester")
parser.add_argument("--interval", default="60", choices=["15", "30", "60", "120", "240", "360"], help="Trading interval in minutes")
parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
parser.add_argument("--fee-rate", type=float, default=0.002, help="Trading fee rate")
parser.add_argument("--min-confidence", type=float, default=None, help="Minimum confidence threshold (default: None for production dynamic economic p*)")
parser.add_argument("--pages", type=int, default=40, help="History pages count")
parser.add_argument("--pessimistic", action="store_true", default=True, help="Use pessimistic fill model (next-bar open + spread/slippage)")
parser.add_argument("--optimistic", action="store_true", default=False, help="Use optimistic fill model (signal close price)")
parser.add_argument("--sl-mult", type=float, default=None, help="Override SL multiplier for simulation")
parser.add_argument("--tp-mult", type=float, default=None, help="Override TP multiplier for simulation")
parser.add_argument("--lookahead", type=int, default=None, help="Override lookahead bars for simulation")
parser.add_argument("--rule-feature", type=str, default=None, help="Direct decile rule mode on a single feature")
parser.add_argument("--challenger", action="store_true", default=False, help="Evaluate newly trained challenger candidate models")
parser.add_argument("--bypass-denylist", action="store_true", default=False, help="Bypass governance denylist check for historical backtesting")

args, _ = parser.parse_known_args()
INTERVAL = args.interval
SYMBOL = args.symbol
FEE_RATE = args.fee_rate
MIN_CONFIDENCE = args.min_confidence
PAGES = args.pages
PESSIMISTIC_MODE = not args.optimistic
OVERRIDE_SL_MULT = args.sl_mult
OVERRIDE_TP_MULT = args.tp_mult
OVERRIDE_LOOKAHEAD = args.lookahead
RULE_FEATURE = args.rule_feature
USE_CHALLENGER = args.challenger
BYPASS_DENYLIST = args.bypass_denylist

INTERVAL_SLIPPAGE = {
    "5": 0.0005,   # 0.05% base slippage for 5m
    "15": 0.0004,  # 0.04% base slippage for 15m
    "30": 0.00035, # 0.035% base slippage for 30m
    "60": 0.0003,  # 0.03% base slippage for 60m
    "120": 0.0003  # 0.03% base slippage for 120m
}

def calculate_backtest_slippage(interval: str, atr_norm: float = 0.0) -> float:
    base = INTERVAL_SLIPPAGE.get(str(interval), 0.0003)
    volatility_premium = (atr_norm * 0.07) if atr_norm >= 0.01 else 0.0
    return base + volatility_premium


def run_single_backtest(df, models_trending, models_ranging, p95, max_conf, min_confidence=0.40, use_regressor_fee_check=True, require_trend_alignment=True, fee_rate=0.002, interval="60", pessimistic_mode=True, rule_feature=None, return_trades=False):
    df = df.reset_index(drop=True)
    trades = []
    equity_compounded = 100.0
    equity_simple = 0.0
    peak_equity = 100.0
    max_drawdown = 0.0
    
    feat_ranks = None
    if rule_feature and rule_feature in df.columns:
        feat_ranks = df[rule_feature].rank(pct=True)
    
    import json
    feat_trending = None
    feat_ranging = None
    trending_file = f"selected_features_{interval}_trending.json"
    ranging_file = f"selected_features_{interval}_ranging.json"
    selected_features_filename = f"selected_features_{interval}.json"

    if models_trending is not None and models_trending.get("feature_names"):
        feat_trending = models_trending["feature_names"]
    elif getattr(models_trending.get("trend") if models_trending else None, "feature_names", None):
        feat_trending = models_trending["trend"].feature_names
    elif os.path.exists(trending_file):
        with open(trending_file, "r") as f:
            feat_trending = json.load(f)
    elif os.path.exists(selected_features_filename):
        with open(selected_features_filename, "r") as f:
            feat_dict = json.load(f)
            feat_trending = feat_dict.get("trending") if isinstance(feat_dict, dict) else feat_dict

    if models_ranging is not None and models_ranging.get("feature_names"):
        feat_ranging = models_ranging["feature_names"]
    elif getattr(models_ranging.get("trend") if models_ranging else None, "feature_names", None):
        feat_ranging = models_ranging["trend"].feature_names
    elif os.path.exists(ranging_file):
        with open(ranging_file, "r") as f:
            feat_ranging = json.load(f)
    elif os.path.exists(selected_features_filename):
        with open(selected_features_filename, "r") as f:
            feat_dict = json.load(f)
            feat_ranging = feat_dict.get("ranging") if isinstance(feat_dict, dict) else feat_dict

    if feat_trending is None:
        feat_trending = features
    if feat_ranging is None:
        feat_ranging = features

    X_matrix_trending = df[feat_trending].values if all(c in df.columns for c in feat_trending) else df[[c for c in feat_trending if c in df.columns]].values
    X_matrix_ranging = df[feat_ranging].values if all(c in df.columns for c in feat_ranging) else df[[c for c in feat_ranging if c in df.columns]].values

    probs_tr_all = None
    pred_pct_tr_all = None
    if models_trending is not None:
        probs_tr_all = models_trending["trend"].predict_proba(X_matrix_trending, weights=models_trending.get("weights"))
        pred_pct_tr_all = models_trending["price"].predict(X_matrix_trending)

    probs_rn_all = None
    pred_pct_rn_all = None
    if models_ranging is not None:
        probs_rn_all = models_ranging["trend"].predict_proba(X_matrix_ranging, weights=models_ranging.get("weights"))
        pred_pct_rn_all = models_ranging["price"].predict(X_matrix_ranging)

    adx_enter_map = getattr(config, "REGIME_ADX_ENTER_BY_INTERVAL", {})
    adx_exit_map = getattr(config, "REGIME_ADX_EXIT_BY_INTERVAL", {})
    adx_enter = adx_enter_map.get(str(interval), getattr(config, "STRONG_TREND_ADX_ENTER", 28.0))
    adx_exit = adx_exit_map.get(str(interval), getattr(config, "STRONG_TREND_ADX_EXIT", 24.0))
    _routing = getattr(config, "ENABLE_DYNAMIC_REGIME_ROUTING", False) or \
               str(interval) in getattr(config, "DYNAMIC_REGIME_ROUTING_INTERVALS", set())

    is_trending_state = False
    active_until_idx = -1
    i = 3
    total_candles = len(df)
    while i < total_candles - 1:
        if i <= active_until_idx:
            i += 1
            continue
        adx_val = df.loc[i, "ADX"]
        close_price = df.loc[i, "close"]
        
        if not is_trending_state and adx_val >= adx_enter:
            is_trending_state = True
        elif is_trending_state and adx_val <= adx_exit:
            is_trending_state = False
            
        if (not _routing) or is_trending_state:
            if probs_tr_all is None:
                i += 1
                continue
            pred_pct = float(pred_pct_tr_all[i])
            probs = probs_tr_all[i]
            calibrator = models_trending.get("calibrator")
        else:
            if probs_rn_all is None or (getattr(config, "MODEL_SLOT_DENYLIST", set()) and f"ranging_{interval}" in config.MODEL_SLOT_DENYLIST):
                i += 1
                continue
            pred_pct = float(pred_pct_rn_all[i])
            probs = probs_rn_all[i]
            calibrator = models_ranging.get("calibrator")

        pred_change = pred_pct * close_price

        from ensemble import resolve_direction
        ml_trend, ml_confidence = resolve_direction(probs)
        prob_neutral = float(probs[1]) if len(probs) >= 2 else 0.0
        neutral_coeff = getattr(config, "NEUTRAL_PENALTY_COEFFICIENT", 0.0)
        raw_confidence = ml_confidence
        if ml_trend in ("Bullish", "Bearish") and neutral_coeff > 0.0:
            raw_confidence = min(0.95, raw_confidence * (1.0 - prob_neutral * neutral_coeff))

        # Live Calibration alignment using Beta / Isotonic engine (matches main.py:6242-6250)
        calibrated_confidence = raw_confidence
        if calibrator is not None and ml_trend in ["Bullish", "Bearish"]:
            from tools.beta_calibrator import calibrate_probability
            calibrated_confidence = calibrate_probability(raw_confidence, calibrator)
        calibrated_confidence = float(np.clip(calibrated_confidence, 1e-3, 1.0 - 1e-3))

        if feat_ranks is not None:
            r_val = float(feat_ranks.iloc[i])
            if r_val >= 0.90:
                ml_trend = "Bullish"
                calibrated_confidence = 1.0
            elif r_val <= 0.10:
                ml_trend = "Bearish"
                calibrated_confidence = 1.0
            else:
                ml_trend = "Neutral"
                calibrated_confidence = 0.0

        if ml_trend == "Neutral":
            i += 1
            continue

        expected_pct_change = (abs(pred_change) / max(1e-9, close_price)) * 100

        # Resolve target multipliers for economic break-even calculation
        cfg = TIMEFRAME_CONFIG.get(str(interval), {})
        sl_multiplier = OVERRIDE_SL_MULT if OVERRIDE_SL_MULT is not None else cfg.get("sl_mult", 0.8)
        tp_multiplier = OVERRIDE_TP_MULT if OVERRIDE_TP_MULT is not None else cfg.get("tp_mult_trending" if is_trending_state else "tp_mult_ranging", 1.85)

        # 1. Economic Break-Even Threshold (p*) + Production Dynamic Confidence Gate (mirrors main.py:6571-6625)
        atr_norm = float(df.loc[i, "ATR_norm"]) if "ATR_norm" in df.columns else 0.01
        cost_bps = (fee_rate * 2.0) * 10000.0  # round-trip in bps
        effective_tp_m = tp_multiplier * 0.80  # REALIZED_RR_HAIRCUT
        p_star = sl_multiplier / max(1e-6, (effective_tp_m + sl_multiplier))
        cost_adj = (cost_bps / 1e4) / max(1e-6, (effective_tp_m + sl_multiplier) * max(1e-4, atr_norm))
        economic_base_threshold = float(round(p_star + cost_adj, 4))
        
        base_cfg_thresh = float(cfg.get("base_confidence_threshold", 0.0))
        dynamic_conf_threshold = max(economic_base_threshold, base_cfg_thresh)

        # Production additive adjustments (Ranging, High Volatility, Session)
        if not is_trending_state:
            regime_delta = 0.03 if str(interval) in ["15", "30"] else 0.05
            dynamic_conf_threshold += regime_delta
        
        if atr_norm > 0.015:
            dynamic_conf_threshold += 0.05
            
        candle_hour = 0
        if "timestamp" in df.columns:
            try:
                candle_hour = datetime.utcfromtimestamp(df.loc[i, "timestamp"] / 1000.0).hour
            except Exception:
                candle_hour = 0
        if 0 <= candle_hour < 8:
            dynamic_conf_threshold += 0.02

        if min_confidence == "dynamic_plus_3":
            effective_min_conf = dynamic_conf_threshold + 0.03
        elif isinstance(min_confidence, (int, float)):
            effective_min_conf = float(min_confidence)
        else:
            effective_min_conf = dynamic_conf_threshold

        if calibrated_confidence < effective_min_conf:
            i += 1
            continue

        # 2. ADX Regime Floor Filter (mirrors live production main.py)
        min_adx_thresh = float(cfg.get("min_adx", 20.0))
        adx_val = float(df.loc[i, "ADX"]) if "ADX" in df.columns else 25.0
        if adx_val < min_adx_thresh:
            i += 1
            continue

        # 3. Macro Trend Alignment (when enabled)
        if require_trend_alignment and "EMA_9_1d" in df.columns and "EMA_21_1d" in df.columns:
            trend_1d = "Bullish" if df.loc[i, "EMA_9_1d"] > df.loc[i, "EMA_21_1d"] else "Bearish"
            if ml_trend != trend_1d:
                i += 1
                continue

        # 4. Volatility & Fee check
        if use_regressor_fee_check:
            if expected_pct_change < 0.25:
                i += 1
                continue
        else:
            if atr_norm < 0.0025:
                i += 1
                continue

        # Trade execution
        if pessimistic_mode:
            # F-01 Fill Model: Next-bar open for market orders
            if i + 1 >= total_candles:
                break
            raw_entry = df.loc[i + 1, "open"]
            half_spread = (fee_rate / 4.0) * raw_entry
            entry_price = raw_entry + half_spread if ml_trend == "Bullish" else raw_entry - half_spread
        else:
            entry_price = close_price

        atr_dollars = atr_norm * entry_price

        # ATR-based Stop Loss and Take Profit with production 4-layer geometry (mirrors main.py:7147-7205)
        regime_detected = "trending" if is_trending_state else "ranging"

        vol_factor = 1.0
        if atr_norm > 0:
            vol_factor = 1.5 - ((atr_norm - 0.003) / 0.005) * 0.75
            vol_factor = max(0.75, min(1.5, vol_factor))

        min_target = max(getattr(config, "MIN_TARGET_ATR_MULT", {}).get(str(interval), 1.5), 1.20 * sl_multiplier)
        base_tp_target = max(tp_multiplier, min_target)
        tp_multiplier_adjusted = round(base_tp_target * vol_factor, 3)

        # (a) Volatility (ATR Percentile) Adjustment (±5%)
        vol_adj = 1.00
        if "ATR" in df.columns and i >= 10:
            hist_atr = df["ATR"].iloc[max(0, i-100):i].dropna()
            if len(hist_atr) > 0:
                atr_percentile = float((hist_atr < df.loc[i, "ATR"]).mean() * 100.0)
                if atr_percentile > 90.0:
                    vol_adj = 0.95
                elif atr_percentile < 20.0:
                    vol_adj = 1.05
        tp_multiplier_adjusted *= vol_adj

        # (b) Session Liquidity Adjustment
        if 6 <= candle_hour < 8:
            session_factor = 0.95
        elif 12 <= candle_hour < 16:
            session_factor = 1.00
        else:
            session_factor = 0.98
        tp_multiplier_adjusted *= session_factor

        # (c) Walk-Forward Unified Target Resolution
        from trade_calculators import UnifiedTargetGenerator
        tp_m = UnifiedTargetGenerator.resolve_tp_multiplier(
            interval=str(interval), entry_price=entry_price,
            atr_dollars=atr_dollars, base_tp_m=tp_multiplier_adjusted
        )

        # (d) Dynamic Stop Loss Multiplier based on prediction confidence
        sl_multiplier_adjusted = sl_multiplier
        if calibrated_confidence > effective_min_conf:
            confidence_ratio = (calibrated_confidence - effective_min_conf) / max(1e-9, 1.0 - effective_min_conf)
            sl_multiplier_adjusted = sl_multiplier * (1.0 - 0.3 * confidence_ratio)

        sl_dist = sl_multiplier_adjusted * atr_dollars
        tp_dist = tp_m * atr_dollars

        # S-01: Timeframe-Adaptive Minimum Stop Loss Floor
        min_sl_cfg = getattr(config, "MIN_SL_PCT_CONFIG", {})
        min_sl_pct = min_sl_cfg.get(str(interval), min_sl_cfg.get("default", 0.008))
        min_sl_dist = entry_price * min_sl_pct
        target_rr = tp_m / max(1e-9, sl_multiplier_adjusted)
        if sl_dist < min_sl_dist:
            sl_dist = min_sl_dist
            tp_dist = sl_dist * target_rr
        _sl_frac = sl_dist / max(1e-9, entry_price)

        if ml_trend == "Bullish":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        else:
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist

        # Post-floor economic gate (mirrors live production abort with REALIZED_RR_HAIRCUT)
        from trade_calculators import passes_economic_gate
        if not passes_economic_gate(entry=entry_price, tp=take_profit, sl=stop_loss, conf=calibrated_confidence):
            i += 1
            continue

        # Look up to lookahead candles
        lookahead = OVERRIDE_LOOKAHEAD if OVERRIDE_LOOKAHEAD is not None else cfg.get("lookahead", 10)
        
        start_step = 1 if pessimistic_mode else 1
        exit_price = df.iloc[min(i + lookahead, total_candles - 1)]["close"]
        exit_reason = "Timer Elapsed"
        candles_elapsed = lookahead

        for step in range(start_step, lookahead + 1):
            if i + step >= total_candles:
                break
            high = df.iloc[i + step]["high"]
            low = df.iloc[i + step]["low"]

            if ml_trend == "Bullish":
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss"
                    candles_elapsed = step
                    break
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "Take Profit"
                    candles_elapsed = step
                    break
            else:
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss"
                    candles_elapsed = step
                    break
                elif low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "Take Profit"
                    candles_elapsed = step
                    break

        if ml_trend == "Bullish":
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price
            
        if pessimistic_mode:
            # Full round-trip taker fee (entry + exit) + half spread
            total_trading_cost = (2.0 * fee_rate) + (fee_rate / 4.0)
            net_return = gross_return - total_trading_cost
        else:
            vol_slippage = (atr_norm * 0.05) if atr_norm is not None else 0.0005
            net_return = gross_return - fee_rate - vol_slippage
        
        # Deduct 8h perpetual funding rate for holds >= 8 hours
        try:
            iv_num = int(str(interval).replace("m", "").replace("h", "0")) if str(interval).isdigit() else 60
        except Exception:
            iv_num = 60
        hours_held = (candles_elapsed * iv_num) / 60.0
        funding_cost = (hours_held / 8.0) * 0.0001 if hours_held >= 8.0 else 0.0
        net_return = net_return - funding_cost
        
        equity_compounded = equity_compounded * (1.0 + net_return)
        equity_simple += net_return
        
        peak_equity = max(peak_equity, equity_compounded)
        current_drawdown = (peak_equity - equity_compounded) / max(1e-9, peak_equity)
        max_drawdown = max(max_drawdown, current_drawdown)

        trades.append({
            "exit_reason": exit_reason,
            "gross_return": gross_return,
            "net_return": net_return,
            "sl_frac": _sl_frac
        })
        active_until_idx = i + max(1, candles_elapsed)
        i = active_until_idx

    if trades:
        import collections
        _mix = collections.Counter(t["exit_reason"] for t in trades)
        _n = max(1, len(trades))
        _sl_frac_avg = (sum(t.get("sl_frac", 0.01) for t in trades) / _n) or 0.01
        print("\n  --- Exit mix ---")
        for _r in ("Take Profit", "Stop Loss", "Timer Elapsed"):
            _b = [t for t in trades if t["exit_reason"] == _r]
            _meanR = (sum(t["net_return"] for t in _b) / len(_b) / _sl_frac_avg) if _b else 0.0
            print(f"  {_r:<15} {_mix[_r]:>5}  {100*_mix[_r]/_n:5.1f}%   mean {_meanR:+.3f}R")

    returns = [t["net_return"] for t in trades]
    sl_fracs = [t.get("sl_frac", 0.01) for t in trades]
    duration_days = None
    try:
        if "timestamp" in df.columns and len(df) > 1:
            ts_start = float(df["timestamp"].iloc[0])
            ts_end = float(df["timestamp"].iloc[-1])
            duration_days = max(0.1, (ts_end - ts_start) / (86400.0 * (1000.0 if ts_end > 1e11 else 1.0)))
    except Exception as ex_ts:
        log_event("WARNING", f"Timestamp duration calculation notice: {ex_ts}")
    from trade_calculators import calculate_replay_statistics
    stats = calculate_replay_statistics(returns, initial_equity=100.0, risk_per_trade_pct=sl_fracs, duration_days=duration_days, interval=str(INTERVAL))
    
    stat_tuple = (
        stats.get("total_trades", 0),
        stats.get("win_rate", 0.0),
        stats.get("profit_factor", 0.0),
        stats.get("max_drawdown_pct", 0.0),
        stats.get("ending_return_pct", 0.0),
        stats.get("expectancy_r", 0.0),
        stats.get("sharpe_ratio", 0.0),
        stats.get("sortino_ratio", 0.0),
        stats.get("calmar_ratio", 0.0),
        stats.get("recovery_factor", 0.0)
    )
    if return_trades:
        return {"trades": trades, "stats": stats, "metrics": stat_tuple}
    return stat_tuple

def run_backtest():
    print("=" * 60)
    print("BTC TRADING BOT - HISTORICAL BACKTESTING SIMULATOR")
    print("=" * 60)

    # 1. Load trained models
    print("Loading trained models...")
    try:
        from ensemble import load_ensemble_classifier, load_ensemble_regressor
        import json
        sfx = "_challenger" if USE_CHALLENGER else ""
        trending_manifest_file = f"ensemble_trending_trend_{INTERVAL}{sfx}_manifest.json"
        ranging_manifest_file = f"ensemble_ranging_trend_{INTERVAL}{sfx}_manifest.json"
        trending_file = f"selected_features_{INTERVAL}_trending.json"
        ranging_file = f"selected_features_{INTERVAL}_ranging.json"
        selected_features_filename = f"selected_features_{INTERVAL}.json"

        feat_tr = None
        feat_rn = None
        if os.path.exists(trending_manifest_file):
            with open(trending_manifest_file, "r") as f:
                feat_tr = json.load(f).get("feature_names")
        elif os.path.exists(trending_file):
            with open(trending_file, "r") as f:
                feat_tr = json.load(f)
        elif os.path.exists(selected_features_filename):
            with open(selected_features_filename, "r") as f:
                d_f = json.load(f)
                feat_tr = d_f.get("trending") if isinstance(d_f, dict) else d_f

        if os.path.exists(ranging_manifest_file):
            with open(ranging_manifest_file, "r") as f:
                feat_rn = json.load(f).get("feature_names")
        elif os.path.exists(ranging_file):
            with open(ranging_file, "r") as f:
                feat_rn = json.load(f)
        elif os.path.exists(selected_features_filename):
            with open(selected_features_filename, "r") as f:
                d_f = json.load(f)
                feat_rn = d_f.get("ranging") if isinstance(d_f, dict) else d_f

        if feat_tr is None:
            feat_tr = features
        if feat_rn is None:
            feat_rn = features

        weights_tr_file = f"ensemble_trending_trend_{INTERVAL}{sfx}_weights.json"
        weights_rn_file = f"ensemble_ranging_trend_{INTERVAL}{sfx}_weights.json"
        weights_tr = [0.10, 0.45, 0.45] if str(INTERVAL) == "15" else ([0.15, 0.42, 0.43] if str(INTERVAL) == "30" else [0.30, 0.20, 0.50])
        weights_rn = [0.10, 0.45, 0.45] if str(INTERVAL) == "15" else ([0.15, 0.42, 0.43] if str(INTERVAL) == "30" else [0.30, 0.50, 0.20])
        if os.path.exists(weights_tr_file):
            with open(weights_tr_file, "r") as f:
                weights_tr = json.load(f).get("classifier_weights", weights_tr)
        if os.path.exists(weights_rn_file):
            with open(weights_rn_file, "r") as f:
                weights_rn = json.load(f).get("classifier_weights", weights_rn)

        trending_cal_file = f"calibrator_trending_{INTERVAL}{sfx}.json"
        ranging_cal_file = f"calibrator_ranging_{INTERVAL}{sfx}.json"
        cal_tr = None
        cal_rn = None
        if os.path.exists(trending_cal_file):
            with open(trending_cal_file, "r") as f:
                cal_tr = json.load(f)
        if os.path.exists(ranging_cal_file):
            with open(ranging_cal_file, "r") as f:
                cal_rn = json.load(f)

        models_trending = None
        if USE_CHALLENGER or BYPASS_DENYLIST or f"trending_{INTERVAL}" not in getattr(config, "MODEL_SLOT_DENYLIST", set()):
            try:
                models_trending = {
                    "trend": load_ensemble_classifier(f"ensemble_trending_trend_{INTERVAL}{sfx}", len(feat_tr), feature_names=feat_tr),
                    "price": load_ensemble_regressor(f"ensemble_trending_price_{INTERVAL}{sfx}", len(feat_tr), feature_names=feat_tr),
                    "weights": weights_tr,
                    "calibrator": cal_tr,
                    "feature_names": feat_tr
                }
            except Exception as e:
                print(f"[Warning] Could not load trending models: {e}")

        models_ranging = None
        if USE_CHALLENGER or BYPASS_DENYLIST or f"ranging_{INTERVAL}" not in getattr(config, "MODEL_SLOT_DENYLIST", set()):
            try:
                models_ranging = {
                    "trend": load_ensemble_classifier(f"ensemble_ranging_trend_{INTERVAL}{sfx}", len(feat_rn), feature_names=feat_rn),
                    "price": load_ensemble_regressor(f"ensemble_ranging_price_{INTERVAL}{sfx}", len(feat_rn), feature_names=feat_rn),
                    "weights": weights_rn,
                    "calibrator": cal_rn,
                    "feature_names": feat_rn
                }
            except Exception as e:
                print(f"[Warning] Could not load ranging models: {e}")

        if models_trending is None and models_ranging is None:
            raise RuntimeError(f"Both trending and ranging models are unavailable for interval {INTERVAL}.")
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}. Please run 'train.py' first.")
        sys.exit(1)

    # 2. Fetch Historical Data
    pages = PAGES
    print(f"\n[Step 1] Fetching historical hourly klines ({pages * 1000} candles)...")
    try:
        df = get_history(symbol=SYMBOL, interval=INTERVAL, limit=1000, pages=pages)
        if df is None or len(df) == 0:
            raise ValueError("No historical data returned.")
        print(f"Loaded {len(df)} hourly candles.")
    except Exception as e:
        print(f"Error fetching history: {e}")
        sys.exit(1)

    # 3. Multi-Timeframe Resampling & Alignment (Look-ahead bias free)
    print("\n[Step 2] Resampling and aligning multi-timeframe indicators...")
    
    # Create datetime index for resampling
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    # Daily trend indicators (Shift by 1 row to prevent look-ahead bias)
    df_daily = df["close"].resample("D").last().to_frame()
    df_daily["EMA_9_1d"] = EMAIndicator(df_daily["close"], window=9).ema_indicator()
    df_daily["EMA_21_1d"] = EMAIndicator(df_daily["close"], window=21).ema_indicator()
    df_daily_shifted = df_daily[["EMA_9_1d", "EMA_21_1d"]].shift(1)

    # 4-hour trend/RSI indicators (Shift by 1 row to prevent look-ahead bias)
    df_4h = df["close"].resample("4h").last().to_frame()
    df_4h["EMA_9_4h"] = EMAIndicator(df_4h["close"], window=9).ema_indicator()
    df_4h["EMA_21_4h"] = EMAIndicator(df_4h["close"], window=21).ema_indicator()
    df_4h["RSI_4h"] = RSIIndicator(df_4h["close"], window=14).rsi()
    df_4h_shifted = df_4h[["EMA_9_4h", "EMA_21_4h", "RSI_4h"]].shift(1)

    # Reset indices for pd.merge_asof
    df.reset_index(inplace=True)
    df_daily_shifted.reset_index(inplace=True)
    df_4h_shifted.reset_index(inplace=True)

    # Merge daily indicators backward
    df = pd.merge_asof(
        df.sort_values("datetime"),
        df_daily_shifted.sort_values("datetime"),
        on="datetime",
        direction="backward"
    )

    # Merge 4-hour indicators backward
    df = pd.merge_asof(
        df.sort_values("datetime"),
        df_4h_shifted.sort_values("datetime"),
        on="datetime",
        direction="backward"
    )

    # Add 1-hour indicators (Features list)
    print("Engineering 1-hour candle indicators...")
    df["close_btc"] = df["close"]
    df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=INTERVAL)
    df = add_features(df)
    df.reset_index(drop=True, inplace=True)

    # 4. Volatility Guard Boundaries & Volume Averages
    df["avg_vol_20"] = df["volume"].rolling(20).mean().shift(1)
    df["volume_pass"] = df["volume"] >= 0.8 * df["avg_vol_20"]

    df["p10_atr"] = df["ATR_norm"].expanding(min_periods=100).quantile(0.10)
    df["p90_atr"] = df["ATR_norm"].expanding(min_periods=100).quantile(0.90)

    # 5. Fetch Calibration limits
    print("\n[Step 3] Calibrating confidence thresholds...")
    p95, max_conf = calculate_historical_thresholds(models_trending["trend"], INTERVAL)

    # Drop any rows that still have NaNs due to lookbacks
    df.dropna(subset=["EMA_9_1d", "EMA_21_1d", "EMA_9_4h", "EMA_21_4h", "RSI_4h", "p10_atr", "p90_atr"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Clean backtest dataset has {len(df)} candles.")

    # 6. Run Scenarios (Filters Comparison)
    print("\n[Step 4] Simulating backtest scenario comparisons...")
    
    scenarios = {
        "A (Production Baseline: Dynamic Economic Gate p*, Trend Align)": {
            "min_confidence": MIN_CONFIDENCE, "use_regressor_fee_check": True, "require_trend_alignment": True
        },
        "B (Pure Dynamic Economic Gate p*, Trend Align)": {
            "min_confidence": MIN_CONFIDENCE, "use_regressor_fee_check": False, "require_trend_alignment": True
        },
        "C (High Conviction: Base Dynamic + 3%, Trend Align)": {
            "min_confidence": ("dynamic_plus_3" if MIN_CONFIDENCE is None else MIN_CONFIDENCE + 0.03), "use_regressor_fee_check": False, "require_trend_alignment": True
        },
        "D (Pure Model Signals: Production Floor, No HTF Filter)": {
            "min_confidence": MIN_CONFIDENCE, "use_regressor_fee_check": False, "require_trend_alignment": False
        },
        "E (High Conviction: Base Dynamic + 3%, No HTF Filter)": {
            "min_confidence": ("dynamic_plus_3" if MIN_CONFIDENCE is None else MIN_CONFIDENCE + 0.03), "use_regressor_fee_check": False, "require_trend_alignment": False
        }
    }

    results = []
    for name, cfg in scenarios.items():
        # Execute pessimistic (realistic) run
        res_p = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=cfg["min_confidence"],
            use_regressor_fee_check=cfg["use_regressor_fee_check"],
            require_trend_alignment=cfg["require_trend_alignment"],
            fee_rate=0.002,
            interval=INTERVAL,
            pessimistic_mode=True,
            rule_feature=RULE_FEATURE
        )
        t_count_p, win_rate_p, pf_p, mdd_p, ret_p, exp_r_p = res_p[:6]

        # Execute optimistic (signal close) run for comparison
        res_o = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=cfg["min_confidence"],
            use_regressor_fee_check=cfg["use_regressor_fee_check"],
            require_trend_alignment=cfg["require_trend_alignment"],
            fee_rate=0.002,
            interval=INTERVAL,
            pessimistic_mode=False,
            rule_feature=RULE_FEATURE
        )
        t_count_o, win_rate_o, pf_o, mdd_o, ret_o, exp_r_o = res_o[:6]
        avg_ret_p = (ret_p / max(1, t_count_p)) if t_count_p > 0 else 0.0

        results.append({
            "Scenario": name,
            "Trades": t_count_p,
            "Pessimistic Return": f"{ret_p:+.2f}%" if t_count_p > 0 else "0.00%",
            "Avg Net / Trade": f"{avg_ret_p:+.2f}%" if t_count_p > 0 else "0.00%",
            "Expectancy (R)": f"{exp_r_p:+.2f}R" if t_count_p > 0 else "0.00R",
            "Optimistic Return": f"{ret_o:+.2f}%" if t_count_o > 0 else "0.00%",
            "Pessimistic MDD": f"{mdd_p:.2f}%" if t_count_p > 0 else "N/A",
            "Pessimistic WinRate": f"{win_rate_p:.2f}%" if t_count_p > 0 else "N/A"
        })

    # Print Comparison Table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("F-01 REALISM COMPARISON: OPTIMISTIC VS PESSIMISTIC FILL MODEL")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90 + "\n")

    # 7. Fee Sensitivity Analysis on Scenario D (Active trade population)
    print("\n[Step 5] Simulating fee sensitivity analysis on Scenario D (Pure Model Signals)...")
    fee_structures = {
        "1. Spot Taker Fee (0.20% roundtrip)": 0.0020,
        "2. Futures Taker Fee (0.10% roundtrip)": 0.0010,
        "3. Futures Limit/Maker Fee (0.04% roundtrip)": 0.0004
    }
    
    fee_results = []
    for structure_name, rate in fee_structures.items():
        res_fee = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=MIN_CONFIDENCE,
            use_regressor_fee_check=False,
            require_trend_alignment=False,
            fee_rate=rate,
            interval=INTERVAL,
            rule_feature=RULE_FEATURE
        )
        t_count, win_rate, pf, mdd, ret = res_fee[:5]
        fee_results.append({
            "Fee Structure": structure_name,
            "Trades": t_count,
            "Win Rate": f"{win_rate:.2f}%",
            "Profit Factor": f"{pf:.2f}",
            "Max Drawdown": f"{mdd:.2f}%",
            "Net Cumulative Return": f"{ret:+.2f}%"
        })
        
    fee_df = pd.DataFrame(fee_results)
    print("=" * 90)
    print("FEE SENSITIVITY COMPARISON TABLE (SCENARIO B)")
    print("=" * 90)
    print(fee_df.to_string(index=False))
    print("=" * 90 + "\n")

    # Export backtest results to JSON for auditing and comparison
    import json
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": results_df.to_dict(orient="records"),
        "fee_sensitivity": fee_results
    }
    try:
        from walk_forward_engine import run_walk_forward_backtest
        from functools import partial
        sim_fn = partial(
            run_single_backtest,
            models_trending=models_trending,
            models_ranging=models_ranging,
            p95=p95,
            max_conf=max_conf,
            min_confidence=MIN_CONFIDENCE,
            fee_rate=FEE_RATE,
            interval=INTERVAL,
            rule_feature=RULE_FEATURE,
            pessimistic_mode=True,
            return_trades=True
        )
        wf_summary = run_walk_forward_backtest(df, trade_simulator_fn=sim_fn)
        if wf_summary.get("status") == "success":
            print("=" * 90)
            print("ROLLING WINDOW WALK-FORWARD VALIDATION SUMMARY")
            print("=" * 90)
            print(f"Total Windows Evaluated        : {wf_summary.get('window_count')}")
            print(f"Mean Window Win Rate           : {wf_summary.get('mean_win_rate'):.2f}%")
            print(f"Mean Window Return             : {wf_summary.get('mean_return'):+.2f}%")
            print(f"Worst Window Drawdown          : {wf_summary.get('max_drawdown'):.2f}%")
            print("=" * 90 + "\n")
            export_data["walk_forward_validation"] = wf_summary
    except Exception as wf_err:
        print(f"[Walk-Forward Engine] Info: {wf_err}")

    with open("backtest_results.json", "w") as f:
        json.dump(export_data, f, indent=2)
    print("[Backtest] Results exported to backtest_results.json successfully.")

if __name__ == "__main__":
    run_backtest()
