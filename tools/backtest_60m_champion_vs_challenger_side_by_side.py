"""
tools/backtest_60m_champion_vs_challenger_side_by_side.py
---------------------------------------------------------
Performs comprehensive side-by-side comparative backtesting of 60m Champion vs Challenger
across both Trending and Ranging market regimes on the multi-asset universe (36,000 hourly candles).
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import get_history, merge_derivatives_sentiment_features
from features import add_features, sanitize_feature_matrix
from ensemble import load_ensemble_classifier

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
ROUND_TRIP_FEE_PCT = 0.0008  # 0.08% fee + slippage (8 bps)

def load_data():
    print("1. Fetching historical candle datasets across 9 crypto pairs (4,000 candles each)...", flush=True)
    all_dfs = []
    for sym in SUPPORTED_SYMBOLS:
        df_sym = get_history(symbol=sym, interval=60, limit=1000, pages=4)
        if df_sym is None or df_sym.empty or len(df_sym) <= 100:
            continue
        df_sym["symbol"] = sym
        df_sym["close_btc"] = df_sym["close"]
        df_sym = merge_derivatives_sentiment_features(df_sym, symbol=sym, interval=60)
        df_sym = add_features(df_sym)
        all_dfs.append(df_sym)
        print(f"   • Loaded {sym}: {len(df_sym)} candles", flush=True)
    return all_dfs

def run_backtest_simulation():
    all_dfs = load_data()
    if not all_dfs:
        print("❌ Error: No datasets loaded.")
        return

    # Total days
    df_first = all_dfs[0]
    time_span_days = (df_first["timestamp"].max() - df_first["timestamp"].min()) / (1000 * 86400)
    total_bars = sum(len(d) for d in all_dfs)
    print(f"\nTotal Dataset: {total_bars} candles across {time_span_days:.1f} trading days.\n")

    # 1. Load Champion Models & Manifests
    with open("models_backup_champion_60m/ensemble_trending_trend_60_manifest.json") as f:
        champ_tr_feats = json.load(f)["feature_names"]
    champ_rn_feats = []
    if os.path.exists("models_backup_champion_60m/ensemble_ranging_trend_60_manifest.json"):
        with open("models_backup_champion_60m/ensemble_ranging_trend_60_manifest.json") as f:
            champ_rn_feats = json.load(f)["feature_names"]

    champ_tr_model = load_ensemble_classifier("models_backup_champion_60m/ensemble_trending_trend_60")
    champ_rn_model = None
    if os.path.exists("models_backup_champion_60m/ensemble_ranging_trend_60_xgb.json"):
        champ_rn_model = load_ensemble_classifier("models_backup_champion_60m/ensemble_ranging_trend_60")

    # 2. Load Challenger Models & Manifests
    with open("ensemble_trending_trend_60_challenger_manifest.json") as f:
        chall_tr_feats = json.load(f)["feature_names"]
    with open("ensemble_ranging_trend_60_challenger_manifest.json") as f:
        chall_rn_feats = json.load(f)["feature_names"]

    chall_tr_model = load_ensemble_classifier("ensemble_trending_trend_60_challenger")
    chall_rn_model = load_ensemble_classifier("ensemble_ranging_trend_60_challenger")

    print(f"Loaded Models:")
    print(f"  • Champion Trending:   {len(champ_tr_feats)} features | Ranging: {len(champ_rn_feats)} features")
    print(f"  • Challenger Trending: {len(chall_tr_feats)} features | Ranging: {len(chall_rn_feats)} features\n")

    # 3. Precompute Predictions for all symbols
    print("2. Generating Inference Probabilities on all candles...", flush=True)
    symbol_data = []
    for df in all_dfs:
        df_clean = sanitize_feature_matrix(df)
        
        # Champion inputs
        X_champ_tr = df_clean[[c for c in champ_tr_feats if c in df_clean.columns]]
        p_champ_tr = champ_tr_model.predict_proba(X_champ_tr)

        p_champ_rn = None
        if champ_rn_model and champ_rn_feats:
            X_champ_rn = df_clean[[c for c in champ_rn_feats if c in df_clean.columns]]
            p_champ_rn = champ_rn_model.predict_proba(X_champ_rn)

        # Challenger inputs
        X_chall_tr = df_clean[[c for c in chall_tr_feats if c in df_clean.columns]]
        p_chall_tr = chall_tr_model.predict_proba(X_chall_tr)

        X_chall_rn = df_clean[[c for c in chall_rn_feats if c in df_clean.columns]]
        p_chall_rn = chall_rn_model.predict_proba(X_chall_rn)

        symbol_data.append({
            "df": df,
            "p_champ_tr": p_champ_tr,
            "p_champ_rn": p_champ_rn,
            "p_chall_tr": p_chall_tr,
            "p_chall_rn": p_chall_rn,
        })

    # 4. Simulation Engine
    def backtest_model(p_type="champ", tp_mult=1.80, sl_mult=1.00, lookahead=12, conf_threshold=0.40, use_regime_routing=True):
        trades = []
        for item in symbol_data:
            df = item["df"]
            p_tr = item["p_champ_tr"] if p_type == "champ" else item["p_chall_tr"]
            p_rn = item["p_champ_rn"] if p_type == "champ" else item["p_chall_rn"]

            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values
            opens = df["open"].values
            adxs = df["ADX"].values if "ADX" in df.columns else np.full(len(df), 25.0)
            atr_norms = df["ATR_norm"].values if "ATR_norm" in df.columns else np.full(len(df), 0.01)

            n_bars = len(df)
            for i in range(50, n_bars - lookahead - 1):
                is_trending = adxs[i] >= 25.0
                if use_regime_routing:
                    probs = p_tr[i] if is_trending else (p_rn[i] if p_rn is not None else None)
                else:
                    probs = p_tr[i]

                if probs is None:
                    continue

                p_bear, p_neut, p_bull = float(probs[0]), float(probs[1]), float(probs[2]) if len(probs) >= 3 else (float(probs[0]), 0.0, float(probs[1]))
                
                # Direction resolution & confidence
                p_dir_bull = p_bull / max(1e-5, p_bull + p_bear)
                p_dir_bear = p_bear / max(1e-5, p_bull + p_bear)

                if p_bull > p_bear and (p_bull >= conf_threshold or p_dir_bull >= conf_threshold + 0.15):
                    direction = "Bullish"
                    dir_conf = p_dir_bull
                elif p_bear > p_bull and (p_bear >= conf_threshold or p_dir_bear >= conf_threshold + 0.15):
                    direction = "Bearish"
                    dir_conf = p_dir_bear
                else:
                    continue

                entry_price = opens[i + 1]
                atr_dist = max(atr_norms[i] * entry_price, entry_price * 0.0060)
                sl_dist = sl_mult * atr_dist
                tp_dist = tp_mult * atr_dist

                if direction == "Bullish":
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                else:
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist

                exit_price = None
                is_win = False

                for k in range(1, lookahead + 1):
                    idx = i + 1 + k
                    curr_h = highs[idx]
                    curr_l = lows[idx]

                    if direction == "Bullish":
                        if curr_l <= sl_price:
                            exit_price = sl_price
                            is_win = False
                            break
                        elif curr_h >= tp_price:
                            exit_price = tp_price
                            is_win = True
                            break
                    else:
                        if curr_h >= sl_price:
                            exit_price = sl_price
                            is_win = False
                            break
                        elif curr_l <= tp_price:
                            exit_price = tp_price
                            is_win = True
                            break

                if exit_price is None:
                    exit_price = closes[i + 1 + lookahead]
                    if direction == "Bullish":
                        is_win = exit_price > entry_price
                    else:
                        is_win = exit_price < entry_price

                pnl_gross = (exit_price - entry_price) / entry_price if direction == "Bullish" else (entry_price - exit_price) / entry_price
                pnl_net = pnl_gross - ROUND_TRIP_FEE_PCT
                r_net = (pnl_net * entry_price) / max(1e-6, sl_dist)

                trades.append({
                    "direction": direction,
                    "regime": "Trending" if is_trending else "Ranging",
                    "pnl_net": pnl_net,
                    "r_net": r_net,
                    "is_win": pnl_net > 0.0
                })

        n = len(trades)
        if n == 0:
            return {"trades": 0, "trades_per_day": 0.0, "win_rate": 0.0, "pf": 0.0, "expectancy_r": 0.0, "total_return_pct": 0.0, "max_dd": 0.0, "trending_trades": 0, "ranging_trades": 0}

        wins = [t["pnl_net"] for t in trades if t["is_win"]]
        losses = [abs(t["pnl_net"]) for t in trades if not t["is_win"]]
        gross_win = sum(wins) if wins else 0.0
        gross_loss = sum(losses) if losses else 1e-6
        pf = gross_win / gross_loss if gross_loss > 0 else 99.0
        wr = (len(wins) / n) * 100.0
        exp_r = float(np.mean([t["r_net"] for t in trades]))
        cum_pnl = np.cumsum([t["pnl_net"] for t in trades])
        peak = np.maximum.accumulate(cum_pnl)
        dd = peak - cum_pnl
        max_dd = np.max(dd) * 100.0 if len(dd) > 0 else 0.0
        total_ret = cum_pnl[-1] * 100.0 if len(cum_pnl) > 0 else 0.0
        trades_per_day = n / max(1.0, time_span_days)
        trending_trades = sum(1 for t in trades if t["regime"] == "Trending")
        ranging_trades = sum(1 for t in trades if t["regime"] == "Ranging")

        return {
            "trades": n,
            "trades_per_day": trades_per_day,
            "win_rate": wr,
            "pf": pf,
            "expectancy_r": exp_r,
            "total_return_pct": total_ret,
            "max_dd": max_dd,
            "trending_trades": trending_trades,
            "ranging_trades": ranging_trades
        }

    # 5. Execute Configurations Sweep
    configs_to_test = [
        {"name": "Config 1: Production Baseline (TP 1.8 ATR, SL 1.0 ATR, Conf >= 0.40, Lookahead 12)", "tp": 1.8, "sl": 1.0, "lh": 12, "conf": 0.40},
        {"name": "Config 2: High Conviction (TP 2.0 ATR, SL 1.2 ATR, Conf >= 0.45, Lookahead 16)", "tp": 2.0, "sl": 1.2, "lh": 16, "conf": 0.45},
        {"name": "Config 3: High Reward/Risk (TP 2.4 ATR, SL 1.0 ATR, Conf >= 0.40, Lookahead 16)", "tp": 2.4, "sl": 1.0, "lh": 16, "conf": 0.40},
        {"name": "Config 4: Tight Scalp (TP 1.5 ATR, SL 0.8 ATR, Conf >= 0.42, Lookahead 10)", "tp": 1.5, "sl": 0.8, "lh": 10, "conf": 0.42},
    ]

    print("=" * 88)
    print("🏆 60-MINUTE CHAMPION VS CHALLENGER SIDE-BY-SIDE BACKTEST MATRIX")
    print(f"Universe: 9 Symbols | Total Candles: {total_bars} | Period: {time_span_days:.1f} Days | Fee: 0.08%")
    print("=" * 88)

    for cfg in configs_to_test:
        c_stats = backtest_model("champ", tp_mult=cfg["tp"], sl_mult=cfg["sl"], lookahead=cfg["lh"], conf_threshold=cfg["conf"])
        ch_stats = backtest_model("chall", tp_mult=cfg["tp"], sl_mult=cfg["sl"], lookahead=cfg["lh"], conf_threshold=cfg["conf"])

        c_split = f"{c_stats['trending_trades']} / {c_stats['ranging_trades']}"
        ch_split = f"{ch_stats['trending_trades']} / {ch_stats['ranging_trades']}"
        c_wr = f"{c_stats['win_rate']:.2f}%"
        ch_wr = f"{ch_stats['win_rate']:.2f}%"
        c_pf = f"{c_stats['pf']:.2f}"
        ch_pf = f"{ch_stats['pf']:.2f}"
        c_exp = f"{c_stats['expectancy_r']:+.3f} R"
        ch_exp = f"{ch_stats['expectancy_r']:+.3f} R"
        c_ret = f"{c_stats['total_return_pct']:+.2f}%"
        ch_ret = f"{ch_stats['total_return_pct']:+.2f}%"
        c_dd = f"{c_stats['max_dd']:.2f}%"
        ch_dd = f"{ch_stats['max_dd']:.2f}%"

        print(f"\n▶ {cfg['name']}")
        print("-" * 88)
        print(f"{'PERFORMANCE METRIC':<28} | {'CHAMPION (23 FEATS)':<26} | {'CHALLENGER (17 FEATS)':<26}")
        print("-" * 88)
        print(f"{'Total Trades Taken':<28} | {c_stats['trades']:<26} | {ch_stats['trades']:<26}")
        print(f"{'Trades / Day (Portfolio)':<28} | {c_stats['trades_per_day']:<26.2f} | {ch_stats['trades_per_day']:<26.2f}")
        print(f"{'Trending / Ranging Split':<28} | {c_split:<26} | {ch_split:<26}")
        print(f"{'Win Rate %':<28} | {c_wr:<26} | {ch_wr:<26}")
        print(f"{'Profit Factor (PF)':<28} | {c_pf:<26} | {ch_pf:<26}")
        print(f"{'Net Expectancy (E[R])':<28} | {c_exp:<26} | {ch_exp:<26}")
        print(f"{'Total Net Return %':<28} | {c_ret:<26} | {ch_ret:<26}")
        print(f"{'Max Drawdown %':<28} | {c_dd:<26} | {ch_dd:<26}")
        print("-" * 88)

if __name__ == "__main__":
    run_backtest_simulation()
