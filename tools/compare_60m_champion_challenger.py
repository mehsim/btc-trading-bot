"""
tools/compare_60m_champion_challenger.py
-----------------------------------------
Loads Champion 60m models (from backup) and Challenger 60m models (freshly trained),
evaluates both on 60m test dataset across 9 symbols, and produces a structured comparative report.
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
from ensemble import load_ensemble_classifier
from tools.beta_calibrator import calibrate_probability
from trade_calculators import passes_economic_gate, calculate_required_p, REALIZED_RR_HAIRCUT
from config import TIMEFRAME_CONFIG

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
FEE_RATE = 0.0008  # 0.08% per leg (0.16% round-trip)

def run_evaluation():
    print("=" * 70)
    print("🔬 60-MINUTE CHAMPION VS CHALLENGER COMPARATIVE EVALUATION")
    print("=" * 70)

    # 1. Inspect Feature Sets & Manifests
    with open("models_backup_champion_60m/ensemble_trending_trend_60_manifest.json", "r") as f:
        champ_manifest = json.load(f)
        champ_feats_trending = champ_manifest.get("feature_names", [])

    with open("ensemble_trending_trend_60_challenger_manifest.json", "r") as f:
        chall_manifest = json.load(f)
        chall_feats_trending = chall_manifest.get("feature_names", [])

    champ_feats_ranging = []
    if os.path.exists("models_backup_champion_60m/ensemble_ranging_trend_60_manifest.json"):
        with open("models_backup_champion_60m/ensemble_ranging_trend_60_manifest.json", "r") as f:
            champ_feats_ranging = json.load(f).get("feature_names", [])

    chall_feats_ranging = []
    if os.path.exists("ensemble_ranging_trend_60_challenger_manifest.json"):
        with open("ensemble_ranging_trend_60_challenger_manifest.json", "r") as f:
            chall_feats_ranging = json.load(f).get("feature_names", [])

    print(f"\n[Feature Selection Comparison (RFECV)]")
    print(f"  • Trending 60m: Champion = {len(champ_feats_trending)} features -> Challenger = {len(chall_feats_trending)} features")
    print(f"  • Ranging 60m:  Champion = {len(champ_feats_ranging)} features -> Challenger = {len(chall_feats_ranging)} features")

    # 2. Ingest Dataset
    print(f"\n[Data Loader] Ingesting 60m candle history across {len(SUPPORTED_SYMBOLS)} symbols...")
    dfs = []
    for s in SUPPORTED_SYMBOLS:
        df_s = get_history(symbol=s, interval=60, limit=1000, pages=4)
        if df_s is not None and len(df_s) > 200:
            df_s["symbol"] = s
            df_s["close_btc"] = df_s["close"]
            df_s = merge_derivatives_sentiment_features(df_s, symbol=s, interval=60)
            df_s = add_features(df_s)
            dfs.append(df_s)
            print(f"  ✓ Loaded {s}: {len(df_s)} candles")

    if not dfs:
        print("❌ Error: No data loaded.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)
    n_total = len(df_all)
    time_span_days = (df_all["timestamp"].max() - df_all["timestamp"].min()) / (1000 * 86400)
    print(f"\nTotal Multi-Asset Evaluation Dataset: {n_total} bars across {time_span_days:.1f} days.")

    # 3. Load Champion & Challenger Models
    print("\n[Model Loading]")
    # Champion
    champ_model = load_ensemble_classifier("models_backup_champion_60m/ensemble_trending_trend_60")
    print("  ✓ Champion 60m Trending Ensemble loaded")

    # Challenger
    chall_model = load_ensemble_classifier("ensemble_trending_trend_60_challenger")
    print("  ✓ Challenger 60m Trending Ensemble loaded")

    # Load Calibrators
    champ_cal = None
    if os.path.exists("models_backup_champion_60m/calibrator_trending_60.json"):
        with open("models_backup_champion_60m/calibrator_trending_60.json", "r") as f:
            champ_cal = json.load(f)

    chall_cal = None
    if os.path.exists("calibrator_trending_60_challenger.json"):
        with open("calibrator_trending_60_challenger.json", "r") as f:
            chall_cal = json.load(f)

    # 4. Generate Predictions
    print("\n[Inference Execution]")
    # Prepare X matrices
    df_sanitized = sanitize_feature_matrix(df_all)
    
    # Champion predictions
    X_champ = df_sanitized[[c for c in champ_feats_trending if c in df_sanitized.columns]]
    p_champ = champ_model.predict_proba(X_champ)

    # Challenger predictions
    X_chall = df_sanitized[[c for c in chall_feats_trending if c in df_sanitized.columns]]
    p_chall = chall_model.predict_proba(X_chall)

    # 5. Backtest Simulation Loop
    tf_cfg = TIMEFRAME_CONFIG.get("60", {})
    lookahead = int(tf_cfg.get("lookahead", 10))
    sl_mult = float(tf_cfg.get("sl_mult", 0.6585))
    tp_mult = float(tf_cfg.get("tp_mult_trending", 1.4747))
    min_adx = float(tf_cfg.get("min_adx", 24.0))
    conf_thresh = float(tf_cfg.get("base_confidence_threshold", 0.40))

    opens = df_all["open"].values
    highs = df_all["high"].values
    lows = df_all["low"].values
    closes = df_all["close"].values
    adxs = df_all["ADX"].values if "ADX" in df_all.columns else np.zeros(n_total)
    atr_norms = df_all["ATR_norm"].values if "ATR_norm" in df_all.columns else np.full(n_total, 0.01)

    def simulate_trades(probs, calibrator, name="Model"):
        trades = []
        n_samples = len(probs)
        for i in range(n_samples - lookahead - 1):
            if adxs[i] < min_adx:
                continue
            
            p_bear, p_neut, p_bull = probs[i][0], probs[i][1], probs[i][2]
            p_dir_bull = p_bull / max(1e-5, p_bull + p_bear)
            p_dir_bear = p_bear / max(1e-5, p_bull + p_bear)

            if p_bull > p_bear and p_dir_bull >= conf_thresh:
                direction = "Bullish"
                dir_conf = p_dir_bull
            elif p_bear > p_bull and p_dir_bear >= conf_thresh:
                direction = "Bearish"
                dir_conf = p_dir_bear
            else:
                continue

            entry_p = opens[i + 1]
            atr_floor = 0.0060
            atr_dist = max(atr_norms[i] * entry_p, entry_p * atr_floor)
            sl_dist = sl_mult * atr_dist
            tp_dist = tp_mult * atr_dist

            if direction == "Bullish":
                sl_p = entry_p - sl_dist
                tp_p = entry_p + tp_dist
            else:
                sl_p = entry_p + sl_dist
                tp_p = entry_p - tp_dist

            # Calibrate & Economic gate
            cal_conf = calibrate_probability(dir_conf, calibrator) if calibrator else dir_conf
            req_p = calculate_required_p(entry=entry_p, tp=tp_p, sl=sl_p, cost_frac=0.0008, realized_rr_haircut=REALIZED_RR_HAIRCUT)
            
            if cal_conf < req_p:
                continue

            # Barrier simulation
            exit_p = None
            is_win = False
            for k in range(1, lookahead + 1):
                idx = i + 1 + k
                curr_h = highs[idx]
                curr_l = lows[idx]

                if direction == "Bullish":
                    if curr_l <= sl_p:
                        exit_p = sl_p
                        is_win = False
                        break
                    elif curr_h >= tp_p:
                        exit_p = tp_p
                        is_win = True
                        break
                else:
                    if curr_h >= sl_p:
                        exit_p = sl_p
                        is_win = False
                        break
                    elif curr_l <= tp_p:
                        exit_p = tp_p
                        is_win = True
                        break

            if exit_p is None:
                exit_p = closes[i + 1 + lookahead]
                if direction == "Bullish":
                    is_win = exit_p > entry_p
                else:
                    is_win = exit_p < entry_p

            if direction == "Bullish":
                pnl_gross = (exit_p - entry_p) / entry_p
            else:
                pnl_gross = (entry_p - exit_p) / entry_p
            pnl_net = pnl_gross - (2.0 * FEE_RATE)
            r_net = (pnl_net * entry_p) / max(1e-6, sl_dist)

            trades.append({
                "direction": direction,
                "conf": cal_conf,
                "pnl_net": pnl_net,
                "r_net": r_net,
                "is_win": pnl_net > 0.0
            })

        n = len(trades)
        if n == 0:
            return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "expectancy_r": 0.0, "total_return_pct": 0.0, "max_dd": 0.0, "trades_per_day": 0.0}

        wins = [t["pnl_net"] for t in trades if t["is_win"]]
        losses = [abs(t["pnl_net"]) for t in trades if not t["is_win"]]
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = sum(losses) if losses else 1e-6
        pf = gross_profit / gross_loss if gross_loss > 0 else 99.0
        wr = (len(wins) / n) * 100.0
        exp_r = float(np.mean([t["r_net"] for t in trades]))
        cum_pnl = np.cumsum([t["pnl_net"] for t in trades])
        peak = np.maximum.accumulate(cum_pnl)
        dd = peak - cum_pnl
        max_dd = np.max(dd) * 100.0 if len(dd) > 0 else 0.0
        total_ret = cum_pnl[-1] * 100.0 if len(cum_pnl) > 0 else 0.0
        trades_per_day = n / max(1.0, time_span_days)

        return {
            "trades": n,
            "win_rate": wr,
            "pf": pf,
            "expectancy_r": exp_r,
            "total_return_pct": total_ret,
            "max_dd": max_dd,
            "trades_per_day": trades_per_day
        }

    stats_champ = simulate_trades(p_champ, champ_cal, "Champion")
    stats_chall = simulate_trades(p_chall, chall_cal, "Challenger")

    print(f"\n" + "=" * 75)
    print(f"📊 60-MINUTE CHAMPION VS CHALLENGER COMPARISON RESULTS")
    print(f"=" * 75)
    print(f"{'METRIC':<30} | {'CHAMPION (23 FEATS)':<18} | {'CHALLENGER (17 FEATS)':<18}")
    print(f"{'-'*75}")
    print(f"{'Feature Count (Trending)':<30} | {len(champ_feats_trending):<18} | {len(chall_feats_trending):<18}")
    print(f"{'Feature Count (Ranging)':<30} | {len(champ_feats_ranging):<18} | {len(chall_feats_ranging):<18}")
    print(f"{'Total Trades Executed':<30} | {stats_champ['trades']:<18} | {stats_chall['trades']:<18}")
    print(f"{'Portfolio Trades / Day':<30} | {stats_champ['trades_per_day']:<18.2f} | {stats_chall['trades_per_day']:<18.2f}")
    print(f"{'Win Rate %':<30} | {stats_champ['win_rate']:<17.1f}% | {stats_chall['win_rate']:<17.1f}%")
    print(f"{'Profit Factor (PF)':<30} | {stats_champ['pf']:<18.2f} | {stats_chall['pf']:<18.2f}")
    print(f"{'Net Expectancy (E[R])':<30} | {stats_champ['expectancy_r']:<+17.3f}R | {stats_chall['expectancy_r']:<+17.3f}R")
    print(f"{'Total Net Return %':<30} | {stats_champ['total_return_pct']:<+17.1f}% | {stats_chall['total_return_pct']:<+17.1f}%")
    print(f"{'Max Drawdown %':<30} | {stats_champ['max_dd']:<17.1f}% | {stats_chall['max_dd']:<17.1f}%")
    print(f"{'='*75}")

if __name__ == "__main__":
    run_evaluation()
