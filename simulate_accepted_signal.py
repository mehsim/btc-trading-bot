import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

workspace_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace_dir)

from data import get_history
from core import (
    add_features, 
    calibrate_confidence,
    calculate_historical_thresholds,
    SYMBOL, 
    INTERVAL,
    features
)
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

def main():
    print("=" * 60)
    print("SEARCHING FOR AN APPROVED SIGNAL IN RECENT HISTORY")
    print("=" * 60)

    # Load models
    from main import models_by_interval, load_model_weights
    load_model_weights("60")

    models_trending = models_by_interval["60"]["trending"]
    models_ranging = models_by_interval["60"]["ranging"]

    # Fetch calibration limits
    p95, max_conf = calculate_historical_thresholds(models_trending["trend"], "60")

    # Fetch daily and 4h data for trend checks
    print("Fetching macro histories for trend verification...")
    df_1d = get_history(symbol=SYMBOL, interval="D", limit=150)
    df_4h = get_history(symbol=SYMBOL, interval="240", limit=150)

    # Fetch target 1h history (last 400 candles)
    df_1h = get_history(symbol=SYMBOL, interval=INTERVAL, limit=400)
    df_1h["close_btc"] = df_1h["close"]
    from data import merge_derivatives_sentiment_features
    df_1h = merge_derivatives_sentiment_features(df_1h, symbol=SYMBOL, interval=INTERVAL)
    df_1h = add_features(df_1h)

    found_signal = False

    # We iterate backwards from recent candles to older candles
    # Start at index 100 to allow lookback for lags
    for i in range(len(df_1h) - 1, 100, -1):
        candle = df_1h.iloc[i]
        timestamp = int(candle["timestamp"])
        candle_dt = datetime.fromtimestamp(timestamp / 1000)
        
        # Generate ML prediction
        selected_features_list = models_by_interval["60"].get("selected_features")
        if selected_features_list is not None:
            X_live = candle[selected_features_list].values.reshape(1, -1)
        else:
            X_live = candle[features].values.reshape(1, -1)
        
        # Dynamic Regime Routing based on ADX
        adx_regime = candle["ADX"]
        if adx_regime >= 20.0:
            active_model_price = models_trending["price"]
            active_model_trend = models_trending["trend"]
            regime_name = "Trending (ADX >= 20)"
        else:
            active_model_price = models_ranging["price"]
            active_model_trend = models_ranging["trend"]
            regime_name = "Ranging (ADX < 20)"

        pred_change = float(active_model_price.predict(X_live)[0])
        predicted_price = float(candle["close"]) + pred_change
        probs = active_model_trend.predict_proba(X_live)[0]
        if len(probs) >= 3:
            prob_bearish = float(probs[0])
            prob_neutral = float(probs[1])
            prob_bullish = float(probs[2])
            if prob_bullish >= prob_bearish:
                ml_trend = "Bullish"
                ml_confidence = prob_bullish
            else:
                ml_trend = "Bearish"
                ml_confidence = prob_bearish
        else:
            prob_bullish = float(probs[1]) if len(probs) > 1 else float(probs[0])
            if prob_bullish >= 0.50:
                ml_trend = "Bullish"
                ml_confidence = prob_bullish
            else:
                ml_trend = "Bearish"
                ml_confidence = 1.0 - prob_bullish


        calibrated_confidence = calibrate_confidence(ml_confidence, p95, max_conf)
        expected_pct_change = (abs(pred_change) / candle["close"]) * 100

        # We need a high confidence signal to begin with
        if calibrated_confidence < 0.70:
            continue

        # Now let's evaluate all technical confluence checks for this historical timestamp
        # Align 1d trend (strictly historical data up to this timestamp)
        df_1d_sub = df_1d[df_1d["timestamp"] <= timestamp].copy()
        if len(df_1d_sub) < 21:
            continue
        ema9_1d_val = EMAIndicator(df_1d_sub["close"], window=9).ema_indicator().iloc[-1]
        ema21_1d_val = EMAIndicator(df_1d_sub["close"], window=21).ema_indicator().iloc[-1]
        trend_1d_bullish = ema9_1d_val > ema21_1d_val
        
        # Align 4h trend (strictly historical data up to this timestamp)
        df_4h_sub = df_4h[df_4h["timestamp"] <= timestamp].copy()
        if len(df_4h_sub) < 21:
            continue
        ema9_4h_val = EMAIndicator(df_4h_sub["close"], window=9).ema_indicator().iloc[-1]
        ema21_4h_val = EMAIndicator(df_4h_sub["close"], window=21).ema_indicator().iloc[-1]
        rsi_4h_val = RSIIndicator(df_4h_sub["close"], window=14).rsi().iloc[-1]
        trend_4h_bullish = ema9_4h_val > ema21_4h_val

        # Check 1: 1d Trend
        if ml_trend == "Bullish":
            pass_1d = trend_1d_bullish
        else:
            pass_1d = not trend_1d_bullish

        # Check 2: 4h Trend
        if ml_trend == "Bullish":
            pass_4h_trend = trend_4h_bullish
        else:
            pass_4h_trend = not trend_4h_bullish

        # Check 3: 4h RSI
        if ml_trend == "Bullish":
            pass_4h_rsi = rsi_4h_val < 70.0
        else:
            pass_4h_rsi = rsi_4h_val > 30.0

        # Check 4: 1h RSI
        rsi_1h = candle["RSI"]
        if ml_trend == "Bullish":
            pass_1h_rsi = rsi_1h < 70.0
        else:
            pass_1h_rsi = rsi_1h > 30.0

        # Check 5: Volume
        df_prior = df_1h.iloc[:i]
        avg_vol_20 = df_prior["volume"].iloc[-20:].mean()
        pass_volume = candle["volume"] >= 0.8 * avg_vol_20

        # Check 6: BB Edge Guard
        bb_pct_val = candle["BB_pct"]
        if ml_trend == "Bullish":
            pass_bb = bb_pct_val < 0.95
        else:
            pass_bb = bb_pct_val > 0.05

        # Check 7: Counter Momentum
        c1 = df_1h.iloc[i-1]
        c2 = df_1h.iloc[i-2]
        c3 = df_1h.iloc[i-3]
        is_red = [c1["close"] < c1["open"], c2["close"] < c2["open"], c3["close"] < c3["open"]]
        is_green = [c1["close"] > c1["open"], c2["close"] > c2["open"], c3["close"] > c3["open"]]
        if ml_trend == "Bullish":
            pass_momentum = not all(is_red)
        else:
            pass_momentum = not all(is_green)

        # Check 8: Volatility Guard (expanding historical window up to current candle)
        atr_series = df_prior["ATR_norm"].dropna()
        p10 = float(np.percentile(atr_series, 10)) if len(atr_series) >= 20 else 0.001
        p90 = float(np.percentile(atr_series, 90)) if len(atr_series) >= 20 else 0.05
        current_atr = candle["ATR_norm"]
        pass_volatility = p10 <= current_atr <= p90

        # Check 9: ADX (Informational - routed dynamic regime)
        pass_adx = True

        # Check 10: Fee coverage
        pass_fee = candle["ATR_norm"] >= 0.0025

        # Checks 11, 12 & 13 (Orderbook, Funding Rate, and News are live checks - we mock them to PASS for historical simulation)
        pass_ob = True
        pass_funding = True
        pass_news = True

        all_pass = all([
            pass_1d, pass_4h_trend, pass_4h_rsi, pass_1h_rsi, pass_volume,
            pass_bb, pass_momentum, pass_volatility, pass_adx, pass_fee,
            pass_ob, pass_funding, pass_news
        ])

        if all_pass:
            found_signal = True
            atr_dollars = candle["ATR_norm"] * candle["close"]
            if adx_regime >= 20.0:
                tp_multiplier = 1.50
            else:
                tp_multiplier = 1.00

            if ml_trend == "Bullish":
                sl_price = candle["close"] - 0.75 * atr_dollars
                tp_price = candle["close"] + tp_multiplier * atr_dollars
            else:
                sl_price = candle["close"] + 0.75 * atr_dollars
                tp_price = candle["close"] - tp_multiplier * atr_dollars

            print("\n==================================================")
            print("HISTORICAL ACCEPTED SIGNAL FOUND IN RECENT DATA")
            print("--------------------------------------------------")
            print(f"Timestamp: {candle_dt.strftime('%Y-%m-%d %H:%M:%S')} (Index -{len(df_1h)-1-i}) | Symbol: {SYMBOL}")
            print(f"Regime Selected: {regime_name} | Signal: {ml_trend} | Calibrated Confidence: {calibrated_confidence * 100:.2f}%")
            print(f"Candle Close Price: {candle['close']:.2f} | Predicted Price: {predicted_price:.2f} (Expected: {pred_change:+.2f} [{expected_pct_change:.3f}%])")
            print("--------------------------------------------------")
            print("Technical Confluence Checks (All Passed):")
            print(f"  [PASS] 1. Daily Trend (1d EMA)  : {'Bullish' if trend_1d_bullish else 'Bearish'} (Matches {ml_trend})")
            print(f"  [PASS] 2. 4h Trend (4h EMA)     : {'Bullish' if trend_4h_bullish else 'Bearish'} (Matches {ml_trend})")
            print(f"  [PASS] 3. 4h RSI                : 4h RSI is {rsi_4h_val:.2f} ({'< 70' if ml_trend=='Bullish' else '> 30'}, Safe)")
            print(f"  [PASS] 4. 1h RSI                : 1h RSI is {rsi_1h:.2f} ({'< 70' if ml_trend=='Bullish' else '> 30'}, Safe)")
            print(f"  [PASS] 5. Volume Participation  : Vol: {candle['volume']:.1f} vs Avg20: {avg_vol_20:.1f} ({candle['volume']/avg_vol_20*100:.1f}%, Req >= 80%)")
            print(f"  [PASS] 6. BB Edge Guard         : BB Pct is {bb_pct_val:.3f}")
            print(f"  [PASS] 7. Counter Momentum      : Safe (No opposing candle momentum)")
            print(f"  [PASS] 8. Volatility Guard      : ATR Norm: {current_atr:.6f} (P10: {p10:.6f}, P90: {p90:.6f})")
            print(f"  [PASS] 9. ADX Regime            : ADX is {adx_regime:.2f}")
            print(f"  [PASS] 10. Fee Coverage          : ATR Volatility: {current_atr*100:.3f}% (Req >= 0.25%)")
            print(f"  [PASS] 11. Orderbook Imbalance   : Mocked PASS for history")
            print(f"  [PASS] 12. Funding Rate Guard    : Mocked PASS for history")
            print(f"  [PASS] 13. News Sentiment        : Mocked PASS for history")
            print("--------------------------------------------------")
            print("CONFLUENCE RESULT: APPROVED (All checks passed)")
            print("\nRisk Configuration (ATR-based SL/TP):")
            print(f"  - ATR Dollars: {atr_dollars:.2f}")
            print(f"  - Stop-Loss (SL): {sl_price:.2f} (Distance: {0.75*atr_dollars:.2f})")
            print(f"  - Take-Profit (TP): {tp_price:.2f} (Distance: {tp_multiplier*atr_dollars:.2f})")
            print("==================================================\n")
            break

    if not found_signal:
        print("\n[Simulation Result] No fully approved signal found in the last 300 historical candles.")
        print("Using demonstration output of a fully passed signal:")

        mock_atr_dollars = 270.0
        mock_price = df_1h["close"].iloc[-1]
        mock_sl = mock_price - 0.75 * mock_atr_dollars
        mock_tp = mock_price + 1.50 * mock_atr_dollars

        print("\n==================================================")
        print("DEMONSTRATION: ACCEPTED SIGNAL BREAKDOWN")
        print("--------------------------------------------------")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbol: {SYMBOL}")
        print(f"Regime Selected: Trending (ADX >= 20) | Signal: Bullish | Calibrated Confidence: 76.50%")
        print(f"Candle Close Price: {mock_price:.2f} | Predicted Price: {mock_price + 405.0:.2f} (Expected: +405.00 [+0.630%])")
        print("--------------------------------------------------")
        print("Technical Confluence Checks (All Passed):")
        print("  [PASS] 1. Daily Trend (1d EMA)  : Bullish (EMA9: 64150 > EMA21: 63800)")
        print("  [PASS] 2. 4h Trend (4h EMA)     : Bullish (EMA9: 64200 > EMA21: 64000)")
        print("  [PASS] 3. 4h RSI                : 4h RSI is 54.20 (< 70, Safe)")
        print("  [PASS] 4. 1h RSI                : 1h RSI is 58.12 (< 70, Safe)")
        print("  [PASS] 5. Volume Participation  : Vol: 850.0 vs Avg20: 700.0 (121.4%, Req >= 80%)")
        print("  [PASS] 6. BB Edge Guard         : BB Pct is 0.720 (< 0.95, Room to run)")
        print("  [PASS] 7. Counter Momentum      : Safe (No consecutive 3 red candles)")
        print("  [PASS] 8. Volatility Guard      : ATR Norm: 0.004200 (Safe zone)")
        print("  [PASS] 9. ADX Regime            : ADX is 24.50 (Req >= 20, Trending)")
        print("  [PASS] 10. Fee Coverage          : ATR Volatility: 0.420% (Req >= 0.25% to cover roundtrip Spot fees)")
        print("  [PASS] 11. Orderbook Imbalance   : Orderbook Imbalance is +0.15 (>= -0.20, Safe)")
        print("  [PASS] 12. Funding Rate Guard    : Funding rate is -0.0050% (MOCKED PASS)")
        print("  [PASS] 13. News Sentiment        : Model: Bullish vs News: Bullish")
        print("--------------------------------------------------")
        print("CONFLUENCE RESULT: APPROVED (All checks passed)")
        print("\nRisk Configuration (ATR-based SL/TP):")
        print(f"  - ATR Dollars: {mock_atr_dollars:.2f}")
        print(f"  - Stop-Loss (SL): {mock_sl:.2f} (Distance: {0.75*mock_atr_dollars:.2f})")
        print(f"  - Take-Profit (TP): {mock_tp:.2f} (Distance: {1.50*mock_atr_dollars:.2f})")
        print("==================================================\n")

if __name__ == "__main__":
    main()
