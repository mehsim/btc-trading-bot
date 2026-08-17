"""
Comprehensive Comparative Backtest Engine
Compares baseline (old uncalibrated architecture) vs Phases 1-4 new architecture
across 15m and 60m models on the multi-asset crypto universe.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_history, merge_derivatives_sentiment_features
from features import add_features, sanitize_feature_matrix
from ensemble import load_ensemble_classifier, get_model_feature_names
from tools.beta_calibrator import calibrate_probability
from trade_calculators import passes_economic_gate, calculate_required_p, REALIZED_RR_HAIRCUT

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
FEE_RATE = 0.0008  # 0.08% per leg (0.16% round-trip)


def load_dataset_for_interval(interval: int, pages: int = 5):
    print(f"\n[Data Loader] Ingesting {interval}m candle history across {len(SUPPORTED_SYMBOLS)} symbols...")
    dfs = []
    for s in SUPPORTED_SYMBOLS:
        df_s = get_history(symbol=s, interval=interval, limit=1000, pages=pages)
        if df_s is not None and len(df_s) > 200:
            df_s["symbol"] = s
            df_s["close_btc"] = df_s["close"]
            df_s = merge_derivatives_sentiment_features(df_s, symbol=s, interval=interval)
            df_s = add_features(df_s)
            dfs.append(df_s)
            print(f"  ✓ Loaded {s}: {len(df_s)} candles")
    if not dfs:
        return None
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)
    return df_all


def run_comparative_backtest(interval: str):
    iv = str(interval)
    iv_int = int(interval)
    print(f"\n{'='*70}")
    print(f"⚡ RUNNING COMPARATIVE BACKTEST: {iv}M CHAMPION MODEL")
    print(f"{'='*70}")

    df_all = load_dataset_for_interval(iv_int, pages=4)
    if df_all is None or len(df_all) < 500:
        print(f"❌ Insufficient data for {iv}m backtest.")
        return

    # Load active champion models and feature sets
    model_trending = load_ensemble_classifier(f"ensemble_trending_trend_{iv}")
    with open(f"selected_features_{iv}_trending.json", "r") as f:
        feats_t = json.load(f)

    # Load calibrator
    cal_file = f"calibrator_trending_{iv}.json"
    calibrator_data = None
    if os.path.exists(cal_file):
        with open(cal_file, "r") as f:
            calibrator_data = json.load(f)

    # Prepare feature matrix
    X_mat = df_all[feats_t].copy()
    X_clean = sanitize_feature_matrix(X_mat)
    
    # Generate ensemble probabilities
    probs = model_trending.predict_proba(X_clean.values)
    p_bears = probs[:, 0]
    p_neutrals = probs[:, 1]
    p_bulls = probs[:, 2]

    closes = df_all["close"].values
    highs = df_all["high"].values
    lows = df_all["low"].values
    opens = df_all["open"].values
    atr_norms = df_all["ATR_norm"].values
    timestamps = df_all["timestamp"].values
    symbols = df_all["symbol"].values
    adxs = df_all["ADX"].values

    # Determine barrier parameters
    opt_file = f"optimized_barriers_{iv}.json"
    tp_mult = 1.40 if iv == "15" else 1.4747
    sl_mult = 0.70 if iv == "15" else 0.6585
    lookahead = 12 if iv == "15" else 10
    if os.path.exists(opt_file):
        with open(opt_file, "r") as f:
            ob = json.load(f)
            tp_mult = float(ob.get("tp_mult_trending", tp_mult))
            sl_mult = float(ob.get("sl_mult", sl_mult))
            lookahead = int(ob.get("lookahead", lookahead))

    n_samples = len(df_all)
    time_span_days = max(1.0, (timestamps[-1] - timestamps[0]) / (1000.0 * 86400.0))

    # ==========================================
    # RUN 1: OLD / BASELINE ARCHITECTURE
    # ==========================================
    trades_old = []
    for i in range(n_samples - lookahead - 1):
        if adxs[i] < 24.0:  # ADX regime gate
            continue
            
        p_b = p_bulls[i]
        p_be = p_bears[i]
        
        # Naive raw confidence without economic gating
        if p_b > p_be and p_b >= 0.35:
            direction = "Bullish"
            raw_conf = p_b
        elif p_be > p_b and p_be >= 0.35:
            direction = "Bearish"
            raw_conf = p_be
        else:
            continue

        entry_p = opens[i + 1]
        atr_dist = atr_norms[i] * entry_p
        
        if direction == "Bullish":
            tp_p = entry_p + (tp_mult * atr_dist)
            sl_p = entry_p - (sl_mult * atr_dist)
        else:
            tp_p = entry_p - (tp_mult * atr_dist)
            sl_p = entry_p + (sl_mult * atr_dist)

        # Barrier simulation
        exit_p = None
        exit_reason = "TIMEOUT"
        for k in range(1, lookahead + 1):
            curr_h = highs[i + 1 + k]
            curr_l = lows[i + 1 + k]
            if direction == "Bullish":
                if curr_h >= tp_p and curr_l <= sl_p:
                    exit_p = sl_p  # pessimistic worst-case
                    exit_reason = "SL"
                    break
                elif curr_h >= tp_p:
                    exit_p = tp_p
                    exit_reason = "TP"
                    break
                elif curr_l <= sl_p:
                    exit_p = sl_p
                    exit_reason = "SL"
                    break
            else:
                if curr_l <= tp_p and curr_h >= sl_p:
                    exit_p = sl_p
                    exit_reason = "SL"
                    break
                elif curr_l <= tp_p:
                    exit_p = tp_p
                    exit_reason = "TP"
                    break
                elif curr_h >= sl_p:
                    exit_p = sl_p
                    exit_reason = "SL"
                    break
                    
        if exit_p is None:
            exit_p = closes[i + 1 + lookahead]

        # Calculate PnL in R-multiples and %
        sl_dist = abs(entry_p - sl_p)
        if direction == "Bullish":
            pnl_gross = (exit_p - entry_p) / entry_p
        else:
            pnl_gross = (entry_p - exit_p) / entry_p
            
        pnl_net = pnl_gross - (2.0 * FEE_RATE)
        r_multiple = (exit_p - entry_p) / sl_dist if direction == "Bullish" else (entry_p - exit_p) / sl_dist
        r_net = r_multiple - ((2.0 * FEE_RATE * entry_p) / sl_dist)

        trades_old.append({
            "direction": direction,
            "conf": raw_conf,
            "pnl_net": pnl_net,
            "r_net": r_net,
            "is_win": pnl_net > 0.0,
            "exit_reason": exit_reason
        })

    # ==========================================
    # RUN 2: PHASES 1-4 NEW ARCHITECTURE
    # ==========================================
    trades_new = []
    for i in range(n_samples - lookahead - 1):
        if adxs[i] < 24.0:
            continue
            
        p_b = p_bulls[i]
        p_be = p_bears[i]
        
        if p_b > p_be and p_b >= 0.36:
            direction = "Bullish"
            dir_conf = p_b / max(1e-5, p_b + p_be)
        elif p_be > p_b and p_be >= 0.36:
            direction = "Bearish"
            dir_conf = p_be / max(1e-5, p_b + p_be)
        else:
            continue

        entry_p = opens[i + 1]
        atr_dist = max(atr_norms[i] * entry_p, entry_p * (0.0040 if iv_int <= 15 else 0.0060))
        sl_dist = sl_mult * atr_dist
        tp_dist = tp_mult * atr_dist
        
        if direction == "Bullish":
            sl_p = entry_p - sl_dist
            tp_p = entry_p + tp_dist
        else:
            sl_p = entry_p + sl_dist
            tp_p = entry_p - tp_dist

        # Phase 2: Beta Calibration on Directional Confidence
        cal_conf = calibrate_probability(dir_conf, calibrator_data)

        # Phase 1: Realized R:R Economic Gate with Haircut & VIP / Maker-Taker cost fraction
        req_p = calculate_required_p(entry=entry_p, tp=tp_p, sl=sl_p, cost_frac=0.0008, realized_rr_haircut=REALIZED_RR_HAIRCUT)
        
        if cal_conf < req_p:
            # Gated / Filtered out by Realized Economic Gate
            continue

        # Barrier simulation
        exit_p = None
        exit_reason = "TIMEOUT"
        for k in range(1, lookahead + 1):
            curr_h = highs[i + 1 + k]
            curr_l = lows[i + 1 + k]
            if direction == "Bullish":
                if curr_h >= tp_p and curr_l <= sl_p:
                    exit_p = sl_p
                    exit_reason = "SL"
                    break
                elif curr_h >= tp_p:
                    exit_p = tp_p
                    exit_reason = "TP"
                    break
                elif curr_l <= sl_p:
                    exit_p = sl_p
                    exit_reason = "SL"
                    break
            else:
                if curr_l <= tp_p and curr_h >= sl_p:
                    exit_p = sl_p
                    exit_reason = "SL"
                    break
                elif curr_l <= tp_p:
                    exit_p = tp_p
                    exit_reason = "TP"
                    break
                elif curr_h >= sl_p:
                    exit_p = sl_p
                    exit_reason = "SL"
                    break
                    
        if exit_p is None:
            exit_p = closes[i + 1 + lookahead]

        sl_dist = abs(entry_p - sl_p)
        if direction == "Bullish":
            pnl_gross = (exit_p - entry_p) / entry_p
        else:
            pnl_gross = (entry_p - exit_p) / entry_p
            
        pnl_net = pnl_gross - (2.0 * FEE_RATE)
        r_multiple = (exit_p - entry_p) / sl_dist if direction == "Bullish" else (entry_p - exit_p) / sl_dist
        r_net = r_multiple - ((2.0 * FEE_RATE * entry_p) / sl_dist)

        trades_new.append({
            "direction": direction,
            "conf": cal_conf,
            "pnl_net": pnl_net,
            "r_net": r_net,
            "is_win": pnl_net > 0.0,
            "exit_reason": exit_reason
        })

    # Summary Statistics
    def compute_stats(t_list):
        if not t_list:
            return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "expectancy_r": 0.0, "total_return_pct": 0.0, "max_dd": 0.0, "trades_per_day": 0.0}
        n = len(t_list)
        wins = [t for t in t_list if t["is_win"]]
        losses = [t for t in t_list if not t["is_win"]]
        wr = (len(wins) / n) * 100.0
        gross_profit = sum(t["pnl_net"] for t in wins)
        gross_loss = abs(sum(t["pnl_net"] for t in losses))
        pf = (gross_profit / max(1e-6, gross_loss)) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        exp_r = np.mean([t["r_net"] for t in t_list])
        
        # Equity curve & Max Drawdown
        cum_pnl = np.cumsum([t["pnl_net"] for t in t_list])
        peak = np.maximum.accumulate(cum_pnl)
        dd = peak - cum_pnl
        max_dd = np.max(dd) * 100.0 if len(dd) > 0 else 0.0
        total_ret = cum_pnl[-1] * 100.0 if len(cum_pnl) > 0 else 0.0
        trades_per_day = n / time_span_days
        
        return {
            "trades": n,
            "win_rate": wr,
            "pf": pf,
            "expectancy_r": exp_r,
            "total_return_pct": total_ret,
            "max_dd": max_dd,
            "trades_per_day": trades_per_day
        }

    s_old = compute_stats(trades_old)
    s_new = compute_stats(trades_new)

    print(f"\n📊 {iv.upper()}M BACKTEST PERFORMANCE COMPARISON ({time_span_days:.1f} Days Evaluated)")
    print(f"{'-'*70}")
    print(f"{'METRIC':<26} | {'OLD (BASELINE)':<18} | {'NEW (PHASES 1-4)':<18}")
    print(f"{'-'*70}")
    print(f"{'Total Trades Taken':<26} | {s_old['trades']:<18} | {s_new['trades']:<18}")
    print(f"{'Portfolio Trades / Day':<26} | {s_old['trades_per_day']:<18.2f} | {s_new['trades_per_day']:<18.2f}")
    print(f"{'Win Rate %':<26} | {s_old['win_rate']:<17.1f}% | {s_new['win_rate']:<17.1f}%")
    print(f"{'Profit Factor (PF)':<26} | {s_old['pf']:<18.2f} | {s_new['pf']:<18.2f}")
    print(f"{'Net Expectancy (E[R])':<26} | {s_old['expectancy_r']:<+17.3f}R | {s_new['expectancy_r']:<+17.3f}R")
    print(f"{'Total Net Return %':<26} | {s_old['total_return_pct']:<+17.1f}% | {s_new['total_return_pct']:<+17.1f}%")
    print(f"{'Max Drawdown %':<26} | {s_old['max_dd']:<17.1f}% | {s_new['max_dd']:<17.1f}%")
    print(f"{'-'*70}")

if __name__ == "__main__":
    run_comparative_backtest("15")
    run_comparative_backtest("60")
