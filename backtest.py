import os
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
parser.add_argument("--min-confidence", type=float, default=0.70, help="Minimum confidence threshold")
parser.add_argument("--pages", type=int, default=40, help="History pages count")
parser.add_argument("--pessimistic", action="store_true", default=True, help="Use pessimistic fill model (next-bar open + spread/slippage)")
parser.add_argument("--optimistic", action="store_true", default=False, help="Use optimistic fill model (signal close price)")

args, _ = parser.parse_known_args()
INTERVAL = args.interval
SYMBOL = args.symbol
FEE_RATE = args.fee_rate
MIN_CONFIDENCE = args.min_confidence
PAGES = args.pages
PESSIMISTIC_MODE = not args.optimistic

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


def run_single_backtest(df, models_trending, models_ranging, p95, max_conf, min_confidence=0.70, use_regressor_fee_check=True, require_trend_alignment=True, fee_rate=0.002, interval="60", pessimistic_mode=True):
    df = df.reset_index(drop=True)
    trades = []
    equity_compounded = 100.0
    equity_simple = 0.0
    peak_equity = 100.0
    max_drawdown = 0.0
    
    import json
    feat_trending = None
    feat_ranging = None
    selected_features_filename = f"selected_features_{interval}.json"
    if os.path.exists(selected_features_filename):
        with open(selected_features_filename, "r") as f:
            feat_dict = json.load(f)
            if isinstance(feat_dict, dict):
                feat_trending = feat_dict.get("trending")
                feat_ranging = feat_dict.get("ranging")
            elif isinstance(feat_dict, list):
                feat_trending = feat_dict
                feat_ranging = feat_dict
                
    if feat_trending is None:
        feat_trending = features
    if feat_ranging is None:
        feat_ranging = features

    X_matrix_trending = df[feat_trending].values
    X_matrix_ranging = df[feat_ranging].values

    i = 3
    total_candles = len(df)
    while i < total_candles - 1:
        adx_val = df.loc[i, "ADX"]
        close_price = df.loc[i, "close"]
        
        if adx_val >= 20.0:
            row_X = X_matrix_trending[i].reshape(1, -1)
            pred_pct = float(models_trending["price"].predict(row_X)[0])
            probs = models_trending["trend"].predict_proba(row_X)[0]
        else:
            row_X = X_matrix_ranging[i].reshape(1, -1)
            pred_pct = float(models_ranging["price"].predict(row_X)[0])
            probs = models_ranging["trend"].predict_proba(row_X)[0]

        pred_change = pred_pct * close_price

        winning_class = int(np.argmax(probs))
        prob_bearish = float(probs[0])
        prob_neutral = float(probs[1])
        prob_bullish = float(probs[2])
        dir_total = prob_bearish + prob_bullish
        
        # Apply Directional Conviction Normalization for 15M & 30M scalp timeframes
        if str(interval) in ["15", "30"] and dir_total > 1e-6:
            norm_bear = prob_bearish / max(1e-9, dir_total)
            norm_bull = prob_bullish / max(1e-9, dir_total)

            
            if norm_bear >= 0.70 and prob_bearish >= 0.12:
                ml_trend = "Bearish"
                ml_confidence = min(0.95, max(0.55, norm_bear * (1.0 - prob_neutral * 0.4)))
            elif norm_bull >= 0.70 and prob_bullish >= 0.12:
                ml_trend = "Bullish"
                ml_confidence = min(0.95, max(0.55, norm_bull * (1.0 - prob_neutral * 0.4)))
            else:
                ml_trend = "Neutral"
                ml_confidence = prob_neutral
        elif winning_class == 2:
            ml_trend = "Bullish"
            ml_confidence = prob_bullish
        elif winning_class == 0:
            ml_trend = "Bearish"
            ml_confidence = prob_bearish
        else:
            ml_trend = "Neutral"
            ml_confidence = prob_neutral

        # Skip Neutral trades
        if ml_trend == "Neutral":
            i += 1
            continue

        calibrated_confidence = calibrate_confidence(ml_confidence, p95, max_conf)
        expected_pct_change = (abs(pred_change) / max(1e-9, close_price)) * 100

        # 1. Confidence threshold
        if calibrated_confidence < min_confidence:
            i += 1
            continue

        # 2. Daily Trend Alignment
        if require_trend_alignment and "EMA_9_1d" in df.columns and "EMA_21_1d" in df.columns:
            trend_1d = "Bullish" if df.loc[i, "EMA_9_1d"] > df.loc[i, "EMA_21_1d"] else "Bearish"
            if ml_trend != trend_1d:
                i += 1
                continue

        # 3. 4h Trend Alignment
        if require_trend_alignment and "EMA_9_4h" in df.columns and "EMA_21_4h" in df.columns:
            trend_4h = "Bullish" if df.loc[i, "EMA_9_4h"] > df.loc[i, "EMA_21_4h"] else "Bearish"
            if ml_trend != trend_4h:
                i += 1
                continue

        # 4. 4h RSI
        if "RSI_4h" in df.columns:
            rsi_4h = df.loc[i, "RSI_4h"]
            if ml_trend == "Bullish" and rsi_4h >= 70.0:
                i += 1
                continue
            if ml_trend == "Bearish" and rsi_4h <= 30.0:
                i += 1
                continue

        # 5. 1h RSI
        rsi_1h = df.loc[i, "RSI"]
        if ml_trend == "Bullish" and rsi_1h >= 70.0:
            i += 1
            continue
        if ml_trend == "Bearish" and rsi_1h <= 30.0:
            i += 1
            continue

        # 6. Volume
        if not df.loc[i, "volume_pass"]:
            i += 1
            continue

        # 7. BB Edge
        bb_pct = df.loc[i, "BB_pct"]
        if ml_trend == "Bullish" and bb_pct >= 0.95:
            i += 1
            continue
        if ml_trend == "Bearish" and bb_pct <= 0.05:
            i += 1
            continue

        # 8. Counter Momentum
        c1_red = df.loc[i-1, "close"] < df.loc[i-1, "open"]
        c2_red = df.loc[i-2, "close"] < df.loc[i-2, "open"]
        c3_red = df.loc[i-3, "close"] < df.loc[i-3, "open"]
        if ml_trend == "Bullish" and c1_red and c2_red and c3_red:
            i += 1
            continue
        if ml_trend == "Bearish" and (not c1_red) and (not c2_red) and (not c3_red):
            i += 1
            continue

        # 9. Volatility Guard
        atr_norm = df.loc[i, "ATR_norm"]
        if not (df.loc[i, "p10_atr"] <= atr_norm <= df.loc[i, "p90_atr"]):
            i += 1
            continue

        # 10. ADX (Decoupled to allow routing to models_ranging)
        # if df.loc[i, "ADX"] < 20.0:
        #     i += 1
        #     continue

        # 11. Fee check
        if use_regressor_fee_check:
            if expected_pct_change < 0.25:
                i += 1
                continue
        else:
            # Volatility-based check: ATR must be >= 0.25%
            if atr_norm < 0.0025:
                i += 1
                continue

        # Trade execution
        if pessimistic_mode:
            # F-01 Fill Model: Next-bar open for market orders
            if i + 1 >= total_candles:
                break
            raw_entry = df.loc[i + 1, "open"]
            half_spread = 0.00015  # 0.015% half spread floor
            taker_fee = 0.0006     # 0.06% taker fee per leg
            vol_slippage = calculate_backtest_slippage(interval, atr_norm)
            
            if ml_trend == "Bullish":
                entry_price = raw_entry * (1.0 + half_spread + vol_slippage)
            else:
                entry_price = raw_entry * (1.0 - half_spread - vol_slippage)
        else:
            entry_price = close_price
            half_spread = 0.0
            taker_fee = fee_rate / 2.0
            vol_slippage = 0.0

        atr_dollars = atr_norm * entry_price
        
        # Regime-Adaptive Take-Profit & Stop-Loss Multipliers from TIMEFRAME_CONFIG
        cfg = TIMEFRAME_CONFIG.get(str(interval), {})
        sl_mult = cfg.get("sl_mult", 1.5)
        tp_multiplier = cfg.get("tp_mult_trending", 2.5) if adx_val >= 20.0 else cfg.get("tp_mult_ranging", 1.5)
            
        if ml_trend == "Bullish":
            stop_loss = entry_price - sl_mult * atr_dollars
            take_profit = entry_price + tp_multiplier * atr_dollars
        else:
            stop_loss = entry_price + sl_mult * atr_dollars
            take_profit = entry_price - tp_multiplier * atr_dollars

        # Look up to lookahead candles
        cfg = TIMEFRAME_CONFIG.get(str(INTERVAL), {"lookahead": 10})
        lookahead = cfg.get("lookahead", 10)
        
        start_step = 1 if pessimistic_mode else 1
        exit_price = df.loc[min(i + lookahead, total_candles - 1), "close"]
        exit_reason = "Timer Elapsed"
        candles_elapsed = lookahead

        for step in range(start_step, lookahead + 1):
            if i + step >= total_candles:
                candles_elapsed = step - 1
                break
            
            next_high = df.loc[i + step, "high"]
            next_low = df.loc[i + step, "low"]
            
            if ml_trend == "Bullish":
                sl_hit = (next_low <= stop_loss)
                tp_hit = (next_high >= take_profit)
                if sl_hit and tp_hit:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss Hit"
                    candles_elapsed = step
                    break
                elif sl_hit:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss Hit"
                    candles_elapsed = step
                    break
                elif tp_hit:
                    exit_price = take_profit
                    exit_reason = "Take Profit Hit"
                    candles_elapsed = step
                    break
            else:
                sl_hit = (next_high >= stop_loss)
                tp_hit = (next_low <= take_profit)
                if sl_hit and tp_hit:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss Hit"
                    candles_elapsed = step
                    break
                elif sl_hit:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss Hit"
                    candles_elapsed = step
                    break
                elif tp_hit:
                    exit_price = take_profit
                    exit_reason = "Take Profit Hit"
                    candles_elapsed = step
                    break

        if ml_trend == "Bullish":
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price
            
        if pessimistic_mode:
            # Exit leg taker fee + half spread
            exit_cost = taker_fee + half_spread
            net_return = gross_return - exit_cost
        else:
            net_return = gross_return - fee_rate - vol_slippage
        
        equity_compounded = equity_compounded * (1.0 + net_return)
        equity_simple += net_return
        
        peak_equity = max(peak_equity, equity_compounded)
        current_drawdown = (peak_equity - equity_compounded) / max(1e-9, peak_equity)
        max_drawdown = max(max_drawdown, current_drawdown)

        trades.append({
            "net_return": net_return
        })
        i += max(1, candles_elapsed)

    total_trades = len(trades)
    if total_trades == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    
    returns = [t["net_return"] for t in trades]
    win_rate = len([r for r in returns if r > 0]) / max(1, total_trades) * 100.0
    
    gross_profits = sum([r for r in returns if r > 0])
    gross_losses = abs(sum([r for r in returns if r <= 0]))
    profit_factor = gross_profits / max(1e-9, gross_losses) if gross_losses > 0 else float("inf")
    
    ending_return = equity_compounded - 100.0
    return total_trades, win_rate, profit_factor, max_drawdown * 100, ending_return

def run_backtest():
    print("=" * 60)
    print("BTC TRADING BOT - HISTORICAL BACKTESTING SIMULATOR")
    print("=" * 60)

    # 1. Load trained models
    print("Loading trained models...")
    try:
        from ensemble import load_ensemble_classifier, load_ensemble_regressor
        import json
        selected_features_filename = f"selected_features_{INTERVAL}.json"
        n_features = len(features)
        if os.path.exists(selected_features_filename):
            with open(selected_features_filename, "r") as f:
                f_data = json.load(f)
                if isinstance(f_data, dict):
                    n_features = len(f_data.get("trending", features))
                elif isinstance(f_data, list):
                    n_features = len(f_data)

        models_trending = {
            "trend": load_ensemble_classifier(f"ensemble_trending_trend_{INTERVAL}", n_features),
            "price": load_ensemble_regressor(f"ensemble_trending_price_{INTERVAL}", n_features)
        }
        models_ranging = {
            "trend": load_ensemble_classifier(f"ensemble_ranging_trend_{INTERVAL}", n_features),
            "price": load_ensemble_regressor(f"ensemble_ranging_price_{INTERVAL}", n_features)
        }
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}. Please run 'train.py' first.")
        sys.exit(1)

    # 2. Fetch Historical Data
    # Fetch 15 pages of 1-hour candles (~15,000 candles, which is ~625 days of spot history)
    pages = 15
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
        "A (Baseline - Regressor Fee Check, Conf >= 70%)": {
            "min_confidence": 0.70, "use_regressor_fee_check": True, "require_trend_alignment": True
        },
        "B (ATR Fee Check >= 0.25%, Conf >= 70%)": {
            "min_confidence": 0.70, "use_regressor_fee_check": False, "require_trend_alignment": True
        },
        "C (ATR Fee Check >= 0.25%, Conf >= 65%)": {
            "min_confidence": 0.65, "use_regressor_fee_check": False, "require_trend_alignment": True
        },
        "D (ATR Fee Check >= 0.25%, Conf >= 60%)": {
            "min_confidence": 0.60, "use_regressor_fee_check": False, "require_trend_alignment": True
        },
        "E (ATR Fee Check, Conf >= 65%, No Trend Align)": {
            "min_confidence": 0.65, "use_regressor_fee_check": False, "require_trend_alignment": False
        }
    }

    results = []
    for name, cfg in scenarios.items():
        # Execute pessimistic (realistic) run
        t_count_p, win_rate_p, pf_p, mdd_p, ret_p = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=cfg["min_confidence"],
            use_regressor_fee_check=cfg["use_regressor_fee_check"],
            require_trend_alignment=cfg["require_trend_alignment"],
            fee_rate=0.002,
            interval=INTERVAL,
            pessimistic_mode=True
        )

        # Execute optimistic (signal close) run for comparison
        t_count_o, win_rate_o, pf_o, mdd_o, ret_o = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=cfg["min_confidence"],
            use_regressor_fee_check=cfg["use_regressor_fee_check"],
            require_trend_alignment=cfg["require_trend_alignment"],
            fee_rate=0.002,
            interval=INTERVAL,
            pessimistic_mode=False
        )

        results.append({
            "Scenario": name,
            "Trades": t_count_p,
            "Pessimistic Return": f"{ret_p:+.2f}%" if t_count_p > 0 else "0.00%",
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

    # 7. Fee Sensitivity Analysis on Scenario B (Best filter balance)
    print("\n[Step 5] Simulating fee sensitivity analysis on Scenario B...")
    fee_structures = {
        "1. Spot Taker Fee (0.20% roundtrip)": 0.0020,
        "2. Futures Taker Fee (0.10% roundtrip)": 0.0010,
        "3. Futures Limit/Maker Fee (0.04% roundtrip)": 0.0004
    }
    
    fee_results = []
    for structure_name, rate in fee_structures.items():
        t_count, win_rate, pf, mdd, ret = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=0.70,
            use_regressor_fee_check=False,
            require_trend_alignment=True,
            fee_rate=rate
        )
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
        wf_summary = run_walk_forward_backtest(df)
        if wf_summary.get("status") == "success":
            print("=" * 90)
            print("WALK-FORWARD SLIDING WINDOW VALIDATION SUMMARY")
            print("=" * 90)
            print(f"Total Sliding Windows Evaluated: {wf_summary.get('window_count')}")
            print(f"Mean Out-of-Sample Win Rate    : {wf_summary.get('mean_win_rate'):.2f}%")
            print(f"Mean Out-of-Sample Return      : {wf_summary.get('mean_return'):+.2f}%")
            print(f"Worst Out-of-Sample Drawdown   : {wf_summary.get('max_drawdown'):.2f}%")
            print("=" * 90 + "\n")
            export_data["walk_forward_validation"] = wf_summary
    except Exception as wf_err:
        print(f"[Walk-Forward Engine] Info: {wf_err}")

    with open("backtest_results.json", "w") as f:
        json.dump(export_data, f, indent=2)
    print("[Backtest] Results exported to backtest_results.json successfully.")

if __name__ == "__main__":
    run_backtest()
