import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Add workspace path to python path
sys.path.append("/Users/mehsimkhurshid/Downloads/btc-trading-bot")
from data import get_history, merge_derivatives_sentiment_features
from main import (
    add_features,
    calibrate_confidence,
    calculate_historical_thresholds,
    SYMBOL,
    INTERVAL,
    features
)
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

def run_single_backtest(df, models_trending, models_ranging, p95, max_conf, min_confidence=0.70, use_regressor_fee_check=True, require_trend_alignment=True, fee_rate=0.002):
    trades = []
    equity_compounded = 100.0
    equity_simple = 0.0
    peak_equity = 100.0
    max_drawdown = 0.0
    
    import json
    selected_features_filename = f"selected_features_{INTERVAL}.json"
    if os.path.exists(selected_features_filename):
        with open(selected_features_filename, "r") as f:
            selected_features = json.load(f)
    else:
        selected_features = features
    X_matrix = df[selected_features].values

    i = 3
    total_candles = len(df)
    while i < total_candles - 1:
        row_X = X_matrix[i].reshape(1, -1)
        adx_val = df.loc[i, "ADX"]
        
        close_price = df.loc[i, "close"]
        if adx_val >= 20.0:
            pred_pct = float(models_trending["price"].predict(row_X)[0])
            pred_change = pred_pct * close_price
            prob_bullish = float(models_trending["trend"].predict_proba(row_X)[0][1])
        else:
            pred_pct = float(models_ranging["price"].predict(row_X)[0])
            pred_change = pred_pct * close_price
            prob_bullish = float(models_ranging["trend"].predict_proba(row_X)[0][1])

        if prob_bullish >= 0.50:
            ml_trend = "Bullish"
            ml_confidence = prob_bullish
        else:
            ml_trend = "Bearish"
            ml_confidence = 1.0 - prob_bullish

        calibrated_confidence = calibrate_confidence(ml_confidence, p95, max_conf)
        expected_pct_change = (abs(pred_change) / close_price) * 100

        # 1. Confidence threshold
        if calibrated_confidence < min_confidence:
            i += 1
            continue

        # 2. Daily Trend Alignment
        if require_trend_alignment:
            trend_1d = "Bullish" if df.loc[i, "EMA_9_1d"] > df.loc[i, "EMA_21_1d"] else "Bearish"
            if ml_trend != trend_1d:
                i += 1
                continue

        # 3. 4h Trend Alignment
        if require_trend_alignment:
            trend_4h = "Bullish" if df.loc[i, "EMA_9_4h"] > df.loc[i, "EMA_21_4h"] else "Bearish"
            if ml_trend != trend_4h:
                i += 1
                continue

        # 4. 4h RSI
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
        entry_price = close_price
        atr_dollars = atr_norm * entry_price
        
        # Regime-Adaptive Take-Profit Multiplier
        if adx_val >= 20.0:
            tp_multiplier = 1.50
        else:
            tp_multiplier = 1.00
            
        if ml_trend == "Bullish":
            stop_loss = entry_price - 0.75 * atr_dollars
            take_profit = entry_price + tp_multiplier * atr_dollars
        else:
            stop_loss = entry_price + 0.75 * atr_dollars
            take_profit = entry_price - tp_multiplier * atr_dollars

        next_high = df.loc[i+1, "high"]
        next_low = df.loc[i+1, "low"]
        next_close = df.loc[i+1, "close"]
        
        exit_price = next_close
        exit_reason = "Timer Elapsed"

        if ml_trend == "Bullish":
            sl_hit = (next_low <= stop_loss)
            tp_hit = (next_high >= take_profit)
            if sl_hit and tp_hit:
                exit_price = stop_loss
                exit_reason = "Stop Loss Hit"
            elif sl_hit:
                exit_price = stop_loss
                exit_reason = "Stop Loss Hit"
            elif tp_hit:
                exit_price = take_profit
                exit_reason = "Take Profit Hit"
        else:
            sl_hit = (next_high >= stop_loss)
            tp_hit = (next_low <= take_profit)
            if sl_hit and tp_hit:
                exit_price = stop_loss
                exit_reason = "Stop Loss Hit"
            elif sl_hit:
                exit_price = stop_loss
                exit_reason = "Stop Loss Hit"
            elif tp_hit:
                exit_price = take_profit
                exit_reason = "Take Profit Hit"

        if ml_trend == "Bullish":
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price
            
        net_return = gross_return - fee_rate
        
        equity_compounded = equity_compounded * (1.0 + net_return)
        equity_simple += net_return
        
        peak_equity = max(peak_equity, equity_compounded)
        current_drawdown = (peak_equity - equity_compounded) / peak_equity
        max_drawdown = max(max_drawdown, current_drawdown)

        trades.append({
            "net_return": net_return
        })
        i += 1

    total_trades = len(trades)
    if total_trades == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    
    returns = [t["net_return"] for t in trades]
    win_rate = len([r for r in returns if r > 0]) / total_trades * 100
    
    gross_profits = sum([r for r in returns if r > 0])
    gross_losses = abs(sum([r for r in returns if r <= 0]))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")
    
    ending_return = equity_compounded - 100.0
    return total_trades, win_rate, profit_factor, max_drawdown * 100, ending_return

def run_backtest():
    print("=" * 60)
    print("BTC TRADING BOT - HISTORICAL BACKTESTING SIMULATOR")
    print("=" * 60)

    # 1. Load trained models
    print("Loading trained models...")
    try:
        from xgboost import XGBClassifier, XGBRegressor
        models_trending = {
            "trend": XGBClassifier(),
            "price": XGBRegressor()
        }
        models_trending["trend"].load_model(f"ensemble_trending_trend_{INTERVAL}_xgb.json")
        models_trending["price"].load_model(f"ensemble_trending_price_{INTERVAL}_xgb.json")

        models_ranging = {
            "trend": XGBClassifier(),
            "price": XGBRegressor()
        }
        models_ranging["trend"].load_model(f"ensemble_ranging_trend_{INTERVAL}_xgb.json")
        models_ranging["price"].load_model(f"ensemble_ranging_price_{INTERVAL}_xgb.json")
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

    df["p10_atr"] = df["ATR_norm"].rolling(100).quantile(0.10)
    df["p90_atr"] = df["ATR_norm"].rolling(100).quantile(0.90)

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
    for name, params in scenarios.items():
        t_count, win_rate, pf, mdd, ret = run_single_backtest(
            df, models_trending, models_ranging, p95, max_conf,
            min_confidence=params["min_confidence"],
            use_regressor_fee_check=params["use_regressor_fee_check"],
            require_trend_alignment=params["require_trend_alignment"],
            fee_rate=0.002
        )
        results.append({
            "Scenario": name,
            "Trades": t_count,
            "Win Rate": f"{win_rate:.2f}%" if t_count > 0 else "N/A",
            "Profit Factor": f"{pf:.2f}" if t_count > 0 else "N/A",
            "Max Drawdown": f"{mdd:.2f}%" if t_count > 0 else "N/A",
            "Cumulative Return": f"{ret:+.2f}%" if t_count > 0 else "0.00%"
        })

    # Print Comparison Table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("SCENARIO COMPARISON (FEE RATE: 0.20% ROUNDTRIP)")
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

if __name__ == "__main__":
    run_backtest()
