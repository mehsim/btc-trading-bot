import pandas as pd
import numpy as np
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import requests
from datetime import datetime

# ==========================================
# DUAL LOGGING UTILITY
# ==========================================
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            try:
                encoding = getattr(self.terminal, "encoding", "utf-8") or "utf-8"
                safe_msg = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
                self.terminal.write(safe_msg)
            except Exception:
                pass
        except Exception:
            pass

        try:
            self.log.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass

# Redirect stdout to both console and simulation_log.txt
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_log.txt")

# Add workspace path to python path (just in case)
workspace_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace_dir)

from data import get_history, merge_derivatives_sentiment_features
from core import (
    add_features, 
    calibrate_confidence,
    calculate_historical_thresholds,
    SYMBOL, 
    INTERVAL,
    features
)
from main import (
    check_pre_trade_confluence, 
    get_news_sentiment
)

def main():
    sys.stdout = DualLogger(log_file_path)
    print("=" * 60)
    print(f"RUNNING LIVE SIMULATION AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Load trained models
    print("Loading trained models...")
    try:
        from ensemble import load_ensemble_classifier, load_ensemble_regressor
        import json
        selected_features_filename = f"selected_features_{INTERVAL}.json"
        if os.path.exists(selected_features_filename):
            with open(selected_features_filename, "r") as f:
                n_features = len(json.load(f))
        else:
            n_features = len(features)

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

    # 2. Fetch calibration limits
    print("\n[Step 1] Calibrating confidence percentiles...")
    p95, max_conf = calculate_historical_thresholds(models_trending["trend"], INTERVAL)

    # 3. Fetch latest data
    print("\n[Step 2] Fetching latest market klines...")
    try:
        df_raw = get_history(symbol=SYMBOL, interval=INTERVAL, limit=300)
        if df_raw is None or len(df_raw) == 0:
            raise ValueError("No data returned")
        
        # Simulate adding the latest live WebSocket price
        latest_close = df_raw["close"].iloc[-1]
        print(f"Latest closed candle price: {latest_close:.2f}")
        
        # Format and add features
        df = df_raw.copy()
        df["close_btc"] = df["close"] # target is BTCUSDT itself
        df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=INTERVAL)
        df = add_features(df)
        print("Features engineered successfully.")
    except Exception as e:
        print(f"Error preparing candle features: {e}")
        sys.exit(1)

    # 4. Generate Predictions
    print("\n[Step 3] Generating Machine Learning Predictions...")
    latest_candle = df.iloc[-1]
    from ensemble import _slice_model_input
    X_live_full = latest_candle[features].values.reshape(1, -1)
    X_live = _slice_model_input(active_model_trend, X_live_full)

    # Dynamic Regime Routing based on ADX
    adx_regime = latest_candle["ADX"]
    if adx_regime >= 20.0:
        active_model_price = models_trending["price"]
        active_model_trend = models_trending["trend"]
        regime_name = "Trending (ADX >= 20)"
    else:
        active_model_price = models_ranging["price"]
        active_model_trend = models_ranging["trend"]
        regime_name = "Ranging (ADX < 20)"

    pred_pct = float(active_model_price.predict(X_live)[0])
    pred_change = pred_pct * latest_close
    predicted_price = latest_close + pred_change
    probs = active_model_trend.predict_proba(X_live)[0]
    winning_class = int(np.argmax(probs))
    prob_bearish = float(probs[0])
    prob_neutral = float(probs[1])
    prob_bullish = float(probs[2])
    dir_total = prob_bearish + prob_bullish
    
    # Apply Directional Conviction Normalization for 15M & 30M scalp timeframes
    if str(INTERVAL) in ["15", "30"] and dir_total > 1e-6:
        norm_bear = prob_bearish / dir_total
        norm_bull = prob_bullish / dir_total

        
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

    calibrated_confidence = calibrate_confidence(ml_confidence, p95, max_conf)
    expected_pct_change = (abs(pred_change) / latest_close) * 100

    print(f"  - Predicted Trend: {ml_trend}")
    print(f"  - Raw Model Confidence: {ml_confidence * 100:.2f}%")
    print(f"  - Calibrated Confidence: {calibrated_confidence * 100:.2f}%")
    print(f"  - Expected Price Change: {pred_change:+.2f} ({expected_pct_change:.3f}%)")

    direction_conflict = (ml_trend == "Bullish" and pred_change < 0) or (ml_trend == "Bearish" and pred_change > 0)

    # Print Report Header
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n==================================================")
    print("PRE-TRADE CONFLUENCE ANALYSIS REPORT (SIMULATED)")
    print("--------------------------------------------------")
    print(f"Time: {timestamp_str} | Symbol: {SYMBOL}")
    print(f"Regime Selected: {regime_name} | Signal: {ml_trend} | Calibrated Confidence: {calibrated_confidence * 100:.2f}%")
    print(f"Current Price: {latest_close:.2f} | Predicted Price: {predicted_price:.2f} (Expected: {pred_change:+.2f} [{expected_pct_change:.3f}%])")
    print("--------------------------------------------------")

    if direction_conflict:
        print(f"CONFLUENCE RESULT: REJECTED (Directional Contradiction: Trend is {ml_trend} but expected change is {pred_change:+.2f})")
        print("==================================================\n\n")
    else:
        # 5. Run Confluence checks
        print("Running Confluence Checks...")
        news_sentiment, news_titles = get_news_sentiment()

        all_pass, confluence_results, confluence_score_pct = check_pre_trade_confluence(
            latest_close, df, ml_trend, news_sentiment, expected_pct_change, interval=INTERVAL, symbol=SYMBOL
        )

        print("Checks Status:")
        for idx, (check_name, res_val) in enumerate(confluence_results.items(), 1):
            status_str = "[PASS]" if res_val["pass"] else "[FAIL]"
            print(f"  {status_str} {idx}. {check_name.replace('_', ' '):<22}: {res_val['detail']}")

        print("--------------------------------------------------")
        if all_pass:
            print("CONFLUENCE RESULT: APPROVED (All checks passed)")
            
            # Show dynamic SL/TP
            atr_norm_val = df.iloc[-1]["ATR_norm"]
            atr_dollars = atr_norm_val * latest_close
            adx_val = df.iloc[-1]["ADX"]
            
            # Regime-Adaptive Take-Profit Multiplier
            if adx_val >= 20.0:
                tp_multiplier = 1.50
            else:
                tp_multiplier = 1.00
                
            if ml_trend == "Bullish":
                stop_loss_price = latest_close - 0.75 * atr_dollars
                take_profit_price = latest_close + tp_multiplier * atr_dollars
            else:
                stop_loss_price = latest_close + 0.75 * atr_dollars
                take_profit_price = latest_close - tp_multiplier * atr_dollars
                
            print("\nSimulated Risk Configuration (ATR-based SL/TP):")
            print(f"  - ATR Dollars: {atr_dollars:.2f}")
            print(f"  - Stop-Loss (SL): {stop_loss_price:.2f} (Distance: {0.75*atr_dollars:.2f})")
            print(f"  - Take-Profit (TP): {take_profit_price:.2f} (Distance: {tp_multiplier*atr_dollars:.2f})")
        else:
            failed_list = [name.replace('_', ' ') for name, res_val in confluence_results.items() if not res_val["pass"]]
            print(f"CONFLUENCE RESULT: REJECTED (Failed: {', '.join(failed_list)})")
        print("==================================================\n\n")

    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()
