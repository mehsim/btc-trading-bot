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
from typing import Tuple, Optional, Dict, List, Any, Union

# Add workspace path to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logger import log_event
from pattern_miner import wilson_score_interval
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
parser.add_argument("--fee-rate", type=float, default=getattr(config, "TAKER_FEE_PCT", 0.00055), help="Trading fee rate (default: Bybit VIP0 taker)")
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
parser.add_argument("--output", type=str, default=None, help="Target JSON output path for backtest results")

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
OUTPUT_FILE = args.output or ("backtest_results_challenger.json" if USE_CHALLENGER else "backtest_results.json")

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


def export_backtest_trades(all_scenario_trades: list, output_file: str = "backtest_trades.jsonl", archive_dir: str = "backtest_runs") -> bool:
    """Safely and atomically exports per-trade backtest context records."""
    if not all_scenario_trades or len(all_scenario_trades) == 0:
        log_event("WARNING", f"[Backtest Warning] Zero scenario trades generated across all scenarios. Refusing to overwrite {output_file}.")
        return False
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(archive_dir, f"backtest_trades_{ts_str}.jsonl")
    temp_path = f"{output_file}_{ts_str}.tmp"
    import shutil
    import json
    try:
        with open(temp_path, "w") as f_tmp:
            for tr in all_scenario_trades:
                f_tmp.write(json.dumps(tr) + "\n")
        shutil.copyfile(temp_path, archive_path)
        os.replace(temp_path, output_file)
        print(f"[Backtest] Emitted {len(all_scenario_trades)} rich per-trade context records to {output_file} and {archive_path}.")
        return True
    except Exception as e_export:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        log_event("WARNING", f"[Backtest Warning] Failed to export JSONL trades: {e_export}")
        return False


PARTITION_FEATURES = [
    "ADX", "ATR_norm", "CHOP", "BB_width", "RSI_z", "volatility_vts",
    "htf_4h_trend_dir", "hour_sin", "hour_cos", "volume_ratio",
    "funding_rate", "oi_change_1h", "CVD_norm"
]


def run_single_backtest(df, models_trending, models_ranging, p95, max_conf, min_confidence=0.40, use_regressor_fee_check=True, require_trend_alignment=True, fee_rate=FEE_RATE, interval="60", pessimistic_mode=True, rule_feature=None, return_trades=False):
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

    def _safe_predict_proba(model, X_matrix, weights=None):
        if model is None:
            return None
        if weights is not None:
            try:
                return model.predict_proba(X_matrix, weights=weights)
            except (TypeError, ValueError):
                pass
        return model.predict_proba(X_matrix)

    def _safe_predict(model, X_matrix):
        if model is None:
            return None
        return model.predict(X_matrix)

    probs_tr_all = None
    pred_pct_tr_all = None
    if models_trending is not None and models_trending.get("trend") is not None:
        probs_tr_all = _safe_predict_proba(models_trending["trend"], X_matrix_trending, weights=models_trending.get("weights"))
        pred_pct_tr_all = _safe_predict(models_trending.get("price"), X_matrix_trending)

    probs_rn_all = None
    pred_pct_rn_all = None
    if models_ranging is not None and models_ranging.get("trend") is not None:
        probs_rn_all = _safe_predict_proba(models_ranging["trend"], X_matrix_ranging, weights=models_ranging.get("weights"))
        pred_pct_rn_all = _safe_predict(models_ranging.get("price"), X_matrix_ranging)

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
        adx_val = float(df.loc[i, "ADX"]) if "ADX" in df.columns else 25.0
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
        ml_trend, ml_confidence = resolve_direction(probs, interval=str(interval))
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
        from trade_calculators import REALIZED_RR_HAIRCUT, get_realized_rr_haircut
        nominal_rr = tp_multiplier / max(1e-6, sl_multiplier)
        realized_haircut = get_realized_rr_haircut(interval=str(interval), regime="trending" if is_trending_state else "ranging", nominal_rr=nominal_rr)
        effective_tp_m = tp_multiplier * realized_haircut
        p_star = sl_multiplier / max(1e-6, (effective_tp_m + sl_multiplier))
        cost_adj = (cost_bps / 1e4) / max(1e-6, (effective_tp_m + sl_multiplier) * max(1e-4, atr_norm))
        economic_base_threshold = float(round(p_star + cost_adj, 4))

        # Calibrator Economic Viability Guard (Finding #31, mirrors main.py:7895-7899)
        if calibrator is not None:
            from tools.beta_calibrator import is_calibrator_viable
            if not is_calibrator_viable(calibrator, min_required_p_star=economic_base_threshold):
                i += 1
                continue
        
        base_cfg_thresh = float(cfg.get("base_confidence_threshold", 0.0))
        dynamic_conf_threshold = max(economic_base_threshold, base_cfg_thresh)

        # Production additive adjustments (Ranging, High Volatility, Session) - mirrors main.py:7912-7967
        if not is_trending_state:
            regime_delta = 0.02 if str(interval) in ["15", "30"] else 0.04
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

        # Recent 50-Trade Performance Decay Filter (mirrors main.py:7944-7953)
        if len(trades) >= 10:
            recent_trades = trades[-50:]
            win_count = sum(1 for t in recent_trades if float(t.get("net_return", 0.0)) > 0)
            recent_wr = (win_count / len(recent_trades)) * 100.0
            if recent_wr < 45.0:
                dynamic_conf_threshold += 0.04

        # HTF Contradiction Decay Penalty (mirrors main.py:7930-7941)
        if str(interval) in ["5", "15"] and "EMA_9_1d" in df.columns and "EMA_21_1d" in df.columns:
            htf_trend_1d = "Bullish" if df.loc[i, "EMA_9_1d"] > df.loc[i, "EMA_21_1d"] else "Bearish"
            if htf_trend_1d != ml_trend:
                dynamic_conf_threshold += 0.03

        # Cap additive penalties so 3-class models are not pushed to unreachable thresholds (mirrors main.py:7965-7967)
        effective_base = max(float(economic_base_threshold), float(base_cfg_thresh))
        max_conf_cap = max(effective_base, 0.50 if str(interval) in ["15", "30", "60"] else 0.55)
        dynamic_conf_threshold = max(effective_base, min(max_conf_cap, dynamic_conf_threshold))

        # Adaptive Confidence Threshold Matrix for 15m (mirrors main.py:8056-8068)
        if str(interval) == "15":
            import trade_calculators
            adaptive_val = trade_calculators.calculate_adaptive_15m_threshold(
                regime="Ranging" if not is_trending_state else "Trending",
                drift_p_val=0.5,
                u_total=0.1,
                symbol_sharpe=1.0,
                base_threshold=economic_base_threshold
            )
            if adaptive_val > dynamic_conf_threshold:
                dynamic_conf_threshold = adaptive_val

        # Bayesian Cold-Start Adjustment (Trades 3-9) (mirrors main.py:8069-8075)
        if 3 <= len(trades) < 10:
            from mlops_engine import mlops_engine
            bayesian_res = mlops_engine.get_bayesian_adjusted_threshold(interval, trades)
            if bayesian_res.get("confidence_boost", 0) > 0:
                dynamic_conf_threshold += bayesian_res["confidence_boost"]

        if min_confidence == "dynamic_plus_3":
            effective_min_conf = dynamic_conf_threshold + 0.03
        elif isinstance(min_confidence, (int, float)):
            effective_min_conf = float(min_confidence)
        else:
            effective_min_conf = dynamic_conf_threshold

        if calibrated_confidence < effective_min_conf:
            i += 1
            continue

        # 2. ADX Regime Floor Filter (mirrors live production main.py:7050-7056)
        if is_trending_state:
            min_adx_thresh = float(cfg.get("min_adx", 16.0 if str(interval) in ["15", "30"] else 24.0))
        else:
            min_adx_thresh = float(cfg.get("min_adx_ranging", 10.0 if str(interval) in ["15", "30"] else 12.0))
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

        # 4. Pre-Trade Confluence Checks (Finding #31, mirrors main.py:8453, confluence_engine.py:107)
        try:
            from confluence_engine import check_pre_trade_confluence
            sym_name = str(getattr(df, "symbol", "BTCUSDT") if hasattr(df, "symbol") else df["symbol"].iloc[0] if "symbol" in df.columns else "BTCUSDT")
            conf_ok, _, _ = check_pre_trade_confluence(
                current_price=close_price,
                df_1h=df.iloc[max(0, i-100):i+1],
                ml_trend=ml_trend,
                news_sentiment="Neutral",
                expected_pct_change=expected_pct_change,
                interval=str(interval),
                symbol=sym_name,
                calibrated_confidence=calibrated_confidence,
                dynamic_conf_threshold=dynamic_conf_threshold,
                current_regime="trending" if is_trending_state else "ranging",
                get_history_fn=lambda *args, **kwargs: None,
                get_orderbook_fn=lambda *args, **kwargs: {"imbalance": 0.0}
            )
            if not conf_ok:
                i += 1
                continue
        except Exception as ex_conf:
            log_event("WARNING", f"Backtest confluence check error (Fail-Closed): {ex_conf}")
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
        # (c) Walk-Forward Unified Target Resolution (Finding #91 / Finding #159 Parity)
        from trade_calculators import resolve_trade_geometry
        geom = resolve_trade_geometry(
            entry_price=entry_price,
            direction=ml_trend,
            interval=str(interval),
            atr_dollars=atr_dollars,
            base_sl_multiplier=sl_multiplier,
            base_tp_multiplier=tp_multiplier_adjusted,
            df=df.iloc[:i+1] if hasattr(df, "iloc") else None,
            symbol="BTCUSDT",
            regime="Trending" if is_trending_state else "Ranging",
            volatility=atr_norm,
            database_module=None
        )
        stop_loss = geom["stop_loss_price"]
        take_profit = geom["take_profit_price"]
        sl_dist = geom["sl_dist"]
        tp_dist = geom["tp_dist"]
        sl_multiplier_adjusted = geom["sl_multiplier_adjusted"]
        tp_m = geom["tp_multiplier_adjusted"]
        _sl_frac = sl_dist / max(1e-9, entry_price)

        # Post-floor economic gate (mirrors live production abort with REALIZED_RR_HAIRCUT)
        from trade_calculators import passes_economic_gate
        if not passes_economic_gate(entry=entry_price, tp=take_profit, sl=stop_loss, conf=calibrated_confidence):
            i += 1
            continue

        # Conservative Kelly edge gate & sizing (mirrors live risk_engine)
        import risk_engine
        from config import MIN_POSITION_BALANCE_FRAC, MAX_POSITION_BALANCE_FRAC
        scaled_kelly = risk_engine.compute_conservative_kelly(
            calibrated_confidence=calibrated_confidence,
            tp_multiplier=tp_dist / max(1e-6, sl_dist),
            sl_multiplier=1.0,
            interval=str(interval),
            trade_history=trades[-30:] if trades else [],
            mcc_val=0.10,
            haircut=0.28,
            atr_norm=atr_norm
        )
        if scaled_kelly <= 0.0:
            i += 1
            continue
        position_frac = min(MAX_POSITION_BALANCE_FRAC, scaled_kelly)

        # Look up to lookahead candles
        lookahead = OVERRIDE_LOOKAHEAD if OVERRIDE_LOOKAHEAD is not None else cfg.get("lookahead", 10)

        start_step = 1 if pessimistic_mode else 1
        exit_price = df.iloc[min(i + lookahead, total_candles - 1)]["close"]
        exit_reason = "Timer Elapsed"
        candles_elapsed = lookahead

        # Scale-Out Configuration (mirrors live main.py:5885-5920)
        from config import SCALE_OUT_CONFIG
        scale_out_portion = SCALE_OUT_CONFIG.get("position_portion", 0.50)
        adx_val = float(df.iloc[i].get("ADX", 25.0))
        if str(interval) in ["240", "360"]:
            scale_out_mult = 1.60 if adx_val >= 35.0 else (1.20 if adx_val < 22.0 else 1.40)
        elif str(interval) in ["60", "120"]:
            scale_out_mult = 1.40 if adx_val >= 35.0 else (1.00 if adx_val < 22.0 else 1.20)
        else:
            scale_out_mult = 1.20 if adx_val >= 35.0 else (0.80 if adx_val < 22.0 else 1.00)
        scale_out_target = entry_price + (scale_out_mult * atr_dollars) if ml_trend == "Bullish" else entry_price - (scale_out_mult * atr_dollars)
        half_closed = False
        scale_out_price = None

        # Trailing Stop & Break-Even Tracking (mirrors live exit_manager.py)
        from exit_manager import compute_dynamic_trail_params
        profit_hurdle_dist, min_trail_buffer = compute_dynamic_trail_params(
            iv=str(interval),
            tf=str(interval),
            entry_price=entry_price,
            atr_dollars=atr_dollars,
            current_adx=adx_val,
            regime="Trending" if is_trending_state else "Ranging"
        )
        curr_sl = stop_loss
        be_activated = False
        trail_hurdle = profit_hurdle_dist

        for step in range(start_step, lookahead + 1):
            if i + step >= total_candles:
                break
            high = df.iloc[i + step]["high"]
            low = df.iloc[i + step]["low"]

            if ml_trend == "Bullish":
                if low <= curr_sl:
                    exit_price = curr_sl
                    exit_reason = "Trailing Stop" if be_activated else "Stop Loss"
                    candles_elapsed = step
                    break
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "Take Profit"
                    candles_elapsed = step
                    break
                # Scale-out check before full TP
                if not half_closed and high >= scale_out_target:
                    half_closed = True
                    scale_out_price = scale_out_target
                    be_activated = True
                    curr_sl = max(curr_sl, entry_price + (fee_rate * 2.0 * entry_price))
                # Ratchet break-even and trailing stop forward
                if high >= entry_price + trail_hurdle:
                    be_activated = True
                    new_trail_sl = max(entry_price + (fee_rate * 2.0 * entry_price), high - max(sl_dist, min_trail_buffer))
                    if new_trail_sl > curr_sl:
                        curr_sl = new_trail_sl
            else:
                if high >= curr_sl:
                    exit_price = curr_sl
                    exit_reason = "Trailing Stop" if be_activated else "Stop Loss"
                    candles_elapsed = step
                    break
                elif low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "Take Profit"
                    candles_elapsed = step
                    break
                # Scale-out check before full TP
                if not half_closed and low <= scale_out_target:
                    half_closed = True
                    scale_out_price = scale_out_target
                    be_activated = True
                    curr_sl = min(curr_sl, entry_price - (fee_rate * 2.0 * entry_price))
                # Ratchet break-even and trailing stop forward
                if low <= entry_price - trail_hurdle:
                    be_activated = True
                    new_trail_sl = min(entry_price - (fee_rate * 2.0 * entry_price), low + max(sl_dist, min_trail_buffer))
                    if new_trail_sl < curr_sl:
                        curr_sl = new_trail_sl

        fee_cost = (2.0 * fee_rate) if pessimistic_mode else (fee_rate + ((atr_norm * 0.05) if atr_norm is not None else 0.0005))
        if half_closed and scale_out_price is not None:
            gross_ret_so = (scale_out_price - entry_price) / entry_price if ml_trend == "Bullish" else (entry_price - scale_out_price) / entry_price
            net_ret_so = gross_ret_so - fee_cost
            gross_ret_rem = (exit_price - entry_price) / entry_price if ml_trend == "Bullish" else (entry_price - exit_price) / entry_price
            net_ret_rem = gross_ret_rem - fee_cost
            net_return = (scale_out_portion * net_ret_so) + ((1.0 - scale_out_portion) * net_ret_rem)
            gross_return = (scale_out_portion * gross_ret_so) + ((1.0 - scale_out_portion) * gross_ret_rem)
        else:
            if ml_trend == "Bullish":
                gross_return = (exit_price - entry_price) / entry_price
            else:
                gross_return = (entry_price - exit_price) / entry_price
            net_return = gross_return - fee_cost

        # Deduct 8h perpetual funding rate for holds >= 8 hours
        try:
            iv_num = int(str(interval).replace("m", "").replace("h", "0")) if str(interval).isdigit() else 60
        except Exception:
            iv_num = 60
        hours_held = (candles_elapsed * iv_num) / 60.0
        funding_cost = (hours_held / 8.0) * 0.0001 if hours_held >= 8.0 else 0.0
        net_return = net_return - funding_cost

        # Finding #25: Live capital-at-risk notional conversion:
        # Live computes: target_notional = (current_bal * f_clamped) / stop_loss_frac
        # where position_frac is f_clamped. Convert to notional equity fraction bounded by leverage cap.
        sl_dist_val = abs(entry_price - stop_loss)
        stop_loss_frac = max(0.002, sl_dist_val / max(1e-9, entry_price))
        notional_equity_frac = min(10.0, position_frac / stop_loss_frac)
        equity_trade_return = notional_equity_frac * net_return
        equity_compounded = equity_compounded * (1.0 + equity_trade_return)
        equity_simple += equity_trade_return

        peak_equity = max(peak_equity, equity_compounded)
        current_drawdown = (peak_equity - equity_compounded) / max(1e-9, peak_equity)
        max_drawdown = max(max_drawdown, current_drawdown)

        feat_snap = {}
        for c in PARTITION_FEATURES:
            if c in df.columns:
                try:
                    val = float(df.loc[i, c])
                    feat_snap[c] = val if not np.isnan(val) else None
                except Exception:
                    feat_snap[c] = None

        sym_name = str(getattr(df, "symbol", "BTCUSDT") if hasattr(df, "symbol") else df["symbol"].iloc[0] if "symbol" in df.columns else "BTCUSDT")

        trades.append({
            "entry_index": int(i),
            "timestamp": int(df.loc[i, "timestamp"]) if "timestamp" in df.columns else 0,
            "calibrated_confidence": float(calibrated_confidence),
            "ml_trend": str(ml_trend),
            "adx_val": float(adx_val),
            "atr_norm": float(atr_norm),
            "is_trending_state": bool(is_trending_state),
            "expected_pct_change": float(expected_pct_change),
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "tp_m": float(tp_m),
            "candles_elapsed": int(candles_elapsed),
            "exit_reason": str(exit_reason) + (" (Scale-Out)" if half_closed else ""),
            "exit_price": float(exit_price),
            "gross_return": float(gross_return),
            "net_return": float(net_return),
            "equity_return": float(equity_trade_return),
            "position_frac": float(position_frac),
            "notional_frac": float(notional_equity_frac),
            "half_closed": half_closed,
            "sl_frac": float(_sl_frac),
            "symbol": sym_name,
            "interval": str(interval),
            "feat": feat_snap
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

    # Finding #32 & #25: Sized returns matching Kelly notional_frac and equity_compounded
    returns = [t.get("equity_return", t["net_return"]) for t in trades]
    sl_fracs = [t.get("notional_frac", t.get("position_frac", 1.0)) * t.get("sl_frac", 0.01) for t in trades]
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
    
    # Also attach unsized stats for full transparency
    try:
        raw_returns = [t["net_return"] for t in trades]
        raw_sl_fracs = [t.get("sl_frac", 0.01) for t in trades]
        stats["unsized_stats"] = calculate_replay_statistics(raw_returns, initial_equity=100.0, risk_per_trade_pct=raw_sl_fracs, duration_days=duration_days, interval=str(INTERVAL))
    except Exception as ex_raw:
        log_event("WARNING", f"Unsized replay stats notice: {ex_raw}")
    
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


def format_pessimistic_win_rate(win_rate_p: float, t_count_p: int) -> Tuple[str, float, float]:
    """
    Computes Wilson score confidence interval and formats pessimistic win rate string.
    Appends [INSUFFICIENT_SAMPLE: n<100] when t_count_p < 100.
    Returns (pess_wr_str, ci_l, ci_u).
    """
    from pattern_miner import wilson_score_interval
    if t_count_p > 0:
        wins_p = int(round((win_rate_p / 100.0) * t_count_p))
        ci_l, ci_u = wilson_score_interval(wins_p, t_count_p)
        pess_wr_str = f"{win_rate_p:.1f}% [{ci_l*100:.1f}%, {ci_u*100:.1f}%] (n={t_count_p})"
        if t_count_p < 100:
            pess_wr_str += " [INSUFFICIENT_SAMPLE: n<100]"
        elif t_count_p < 784:
            pess_wr_str += " [n<784]"
    else:
        ci_l, ci_u = 0.0, 0.0
        pess_wr_str = "N/A"
    return pess_wr_str, ci_l, ci_u


def train_window_models(
    train_df: pd.DataFrame,
    p95: float = 0.55,
    max_conf: float = 0.75,
    min_confidence: float = 0.40,
    fee_rate: float = FEE_RATE,
    interval: str = "15",
    rule_feature: Optional[str] = None
):
    """
    Refits LightGBM/HistGradientBoosting models on walk-forward training window.
    Filters non-numeric, timestamp, and target columns.
    Raises RuntimeError on fit failure (Fail-Closed).
    """
    from functools import partial
    try:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        excluded_cols = {
            "timestamp", "open", "high", "low", "close", "volume",
            "open_btc", "high_btc", "low_btc", "close_btc", "volume_btc",
            "target_trend", "target_price", "target_price_change", "target_direction", "future_ret",
            "sample_weight", "datetime", "index"
        }
        feat_cols = [c for c in numeric_cols if c not in excluded_cols and not c.startswith("future_")]
        if not feat_cols or len(train_df) < 100:
            return None
        
        df_t = train_df.dropna(subset=["close"]).copy()
        if "target_trend" not in df_t.columns:
            ret = df_t["close"].pct_change(12).shift(-12).fillna(0.0)
            df_t["target_trend"] = np.where(ret > 0.005, 2, np.where(ret < -0.005, 0, 1))
            df_t["target_price"] = ret * df_t["close"]
            
        X_tr = df_t[feat_cols].fillna(0.0).values
        y_tr_t = df_t["target_trend"].astype(int).values
        y_tr_p = df_t["target_price"].astype(float).values
        
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        m_t = HistGradientBoostingClassifier(max_iter=30, random_state=42).fit(X_tr, y_tr_t)
        m_p = HistGradientBoostingRegressor(max_iter=30, random_state=42).fit(X_tr, y_tr_p)
        
        m_t.feature_names = feat_cols
        m_p.feature_names = feat_cols
        
        return partial(
            run_single_backtest,
            models_trending={"trend": m_t, "price": m_p, "feature_names": feat_cols},
            models_ranging={"trend": m_t, "price": m_p, "feature_names": feat_cols},
            p95=p95,
            max_conf=max_conf,
            min_confidence=min_confidence,
            fee_rate=fee_rate,
            interval=interval,
            rule_feature=rule_feature,
            pessimistic_mode=True,
            return_trades=True
        )
    except Exception as ex_tr:
        from logger import log_event
        log_event("ERROR", f"[WalkForward Refit Failure] Window model refit failed: {ex_tr}")
        raise RuntimeError(f"Walk-forward window model refit failed: {ex_tr}") from ex_tr


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
    all_scenario_trades = []
    for name, cfg in scenarios.items():
        # Execute pessimistic (realistic) run
        res_p = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=cfg["min_confidence"],
            use_regressor_fee_check=cfg["use_regressor_fee_check"],
            require_trend_alignment=cfg["require_trend_alignment"],
            fee_rate=FEE_RATE,
            interval=INTERVAL,
            pessimistic_mode=True,
            rule_feature=RULE_FEATURE,
            return_trades=True
        )
        if isinstance(res_p, dict):
            stat_t = res_p["metrics"]
            trades_p = res_p.get("trades", [])
        else:
            stat_t = res_p
            trades_p = []
        t_count_p, win_rate_p, pf_p, mdd_p, ret_p, exp_r_p = stat_t[:6]
        avg_ret_p = (ret_p / t_count_p) if t_count_p > 0 else 0.0
        for tr in trades_p:
            all_scenario_trades.append({
                "scenario": name,
                "symbol": tr.get("symbol", "BTCUSDT"),
                "interval": str(INTERVAL),
                **tr
            })

        # Execute optimistic (signal close) run for comparison
        res_o = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=cfg["min_confidence"],
            use_regressor_fee_check=cfg["use_regressor_fee_check"],
            require_trend_alignment=cfg["require_trend_alignment"],
            fee_rate=FEE_RATE,
            interval=INTERVAL,
            pessimistic_mode=False,
            rule_feature=RULE_FEATURE
        )
        t_count_o, win_rate_o, pf_o, mdd_o, ret_o, exp_r_o = res_o[:6]
        pess_wr_str, ci_l, ci_u = format_pessimistic_win_rate(win_rate_p, t_count_p)

        is_small_n = (0 < t_count_p < 100)
        results.append({
            "Scenario": name,
            "Trades": t_count_p,
            "Pessimistic Return": f"{ret_p:+.2f}%" if t_count_p > 0 else "0.00%",
            "Avg Net / Trade": f"{avg_ret_p:+.2f}%" if t_count_p > 0 else "0.00%",
            "Expectancy (R)": f"{exp_r_p:+.2f}R" if t_count_p > 0 else "0.00R",
            "Optimistic Return": f"{ret_o:+.2f}%" if t_count_o > 0 else "0.00%",
            "Pessimistic MDD": f"{mdd_p:.2f}%" if t_count_p > 0 else "N/A",
            "Pessimistic WinRate": pess_wr_str,
            "Statistical Conclusion": "SUPPRESSED (Sample size n < 100 does not support per-symbol conclusion)" if is_small_n else ("STATISTICALLY_VALID" if t_count_p >= 784 else "LOW_POWER (100 <= n < 784)"),
            "_ci_l": ci_l,
            "_ci_u": ci_u
        })

        # Export per-trade granular backtest context to JSONL
        export_backtest_trades(all_scenario_trades)

    # Finding #86: Overlapping Wilson CI detection & decision refusal
    baseline_ci = (results[0]["_ci_l"], results[0]["_ci_u"]) if results else None
    for r in results:
        r_l = r.pop("_ci_l", 0.0)
        r_u = r.pop("_ci_u", 1.0)
        if baseline_ci and r["Scenario"] != results[0]["Scenario"] and r["Trades"] > 0:
            # Check interval overlap: max(L1, L2) <= min(U1, U2)
            if max(baseline_ci[0], r_l) <= min(baseline_ci[1], r_u):
                r["Pessimistic WinRate"] += " [STATISTICALLY_INDISTINGUISHABLE]"

    # Print Comparison Table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("F-01 REALISM COMPARISON: OPTIMISTIC VS PESSIMISTIC FILL MODEL")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90 + "\n")

    # 7. Fee Sensitivity Analysis on Scenario D (Active trade population)
    print("\n[Step 5] Simulating fee sensitivity analysis on Scenario D (Pure Model Signals)...")
    from config import TAKER_FEE_PCT, MAKER_FEE_PCT
    fee_structures = {
        "1. Spot Taker Fee (0.20% roundtrip)": 0.0010,
        "2. Futures Taker Fee (0.11% roundtrip)": TAKER_FEE_PCT,
        "3. Futures Limit/Maker Fee (0.04% roundtrip)": MAKER_FEE_PCT
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
        if t_count > 0:
            w_cnt = int(round((win_rate / 100.0) * t_count))
            f_ci_l, f_ci_u = wilson_score_interval(w_cnt, t_count)
            f_wr_str = f"{win_rate:.1f}% [{f_ci_l*100:.1f}%, {f_ci_u*100:.1f}%] (n={t_count})"
        else:
            f_wr_str = "N/A"
        fee_results.append({
            "Fee Structure": structure_name,
            "Trades": t_count,
            "Win Rate": f_wr_str,
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
        "is_refitted": False,
        "provenance": {
            "git_commit": getattr(config, "BOT_VERSION", "v2.6"),
            "is_challenger": bool(USE_CHALLENGER),
            "output_target": OUTPUT_FILE
        },
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

        train_fn = partial(
            train_window_models,
            p95=p95,
            max_conf=max_conf,
            min_confidence=MIN_CONFIDENCE,
            fee_rate=FEE_RATE,
            interval=INTERVAL,
            rule_feature=RULE_FEATURE
        )

        wf_summary = run_walk_forward_backtest(df, trade_simulator_fn=sim_fn, train_fn=train_fn)
        if wf_summary.get("status") == "success":
            print("=" * 90)
            print(f"{'ROLLING WINDOW WALK-FORWARD VALIDATION' if wf_summary.get('all_windows_refitted') else 'ROLLING WINDOW REPLAY'} SUMMARY")
            print("=" * 90)
            print(f"Total Windows Evaluated        : {wf_summary.get('window_count')}")
            print(f"Evaluation Mode                : {wf_summary.get('evaluation_mode')}")
            mean_wr = float(wf_summary.get('mean_win_rate', 0.0))
            tot_w_trades = sum(int(w.get("trades", 0)) for w in wf_summary.get("windows", [])) if "windows" in wf_summary else 0
            if tot_w_trades > 0:
                from pattern_miner import wilson_score_interval
                w_wins = int(round((mean_wr / 100.0) * tot_w_trades))
                w_ci_l, w_ci_u = wilson_score_interval(w_wins, tot_w_trades)
                wr_disp = f"{mean_wr:.2f}% [{w_ci_l*100:.1f}%, {w_ci_u*100:.1f}%] (n={tot_w_trades})"
            else:
                wr_disp = f"{mean_wr:.2f}%"
            print(f"Mean Window Win Rate           : {wr_disp}")
            print(f"Mean Window Return             : {wf_summary.get('mean_return'):+.2f}%")
            print(f"Worst Window Drawdown          : {wf_summary.get('max_drawdown'):.2f}%")
            print("=" * 90 + "\n")
            if wf_summary.get("all_windows_refitted", False):
                export_data["walk_forward_validation"] = wf_summary
            else:
                export_data["rolling_window_replay"] = wf_summary
    except Exception as wf_err:
        print(f"[Walk-Forward Engine] Info: {wf_err}")

    total_trades_count = sum(int(r.get("Trades", 0)) for r in results)
    if total_trades_count == 0:
        log_event("ERROR", "[Backtest Critical Error] Zero trades were generated across all test scenarios. Refusing to overwrite backtest_results.json with a misleading zero-trade 0.00% result.")
        raise RuntimeError("Zero-trade backtest generated across all scenarios. Backtest failed to produce valid trading signals.")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"[Backtest] Results exported to {OUTPUT_FILE} successfully.")

if __name__ == "__main__":
    run_backtest()
