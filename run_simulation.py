import pandas as pd
import numpy as np
import sys
import os
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
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# Redirect stdout to both console and simulation_log.txt
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_log.txt")
sys.stdout = DualLogger(log_file_path)

# Add workspace path to python path (just in case)
workspace_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace_dir)

from data import get_history, merge_derivatives_sentiment_features
from main import (
    add_features, 
    check_pre_trade_confluence, 
    get_news_sentiment, 
    calibrate_confidence,
    calculate_historical_thresholds,
    SYMBOL, 
    INTERVAL,
    features
)

print("=" * 60)
print(f"RUNNING LIVE SIMULATION AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
import json
selected_features_filename = f"selected_features_{INTERVAL}.json"
if os.path.exists(selected_features_filename):
    with open(selected_features_filename, "r") as f:
        selected_features = json.load(f)
else:
    selected_features = features
X_live = latest_candle[selected_features].values.reshape(1, -1)

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
prob_bullish = float(active_model_trend.predict_proba(X_live)[0][1])

if prob_bullish >= 0.50:
    ml_trend = "Bullish"
    ml_confidence = prob_bullish
else:
    ml_trend = "Bearish"
    ml_confidence = 1.0 - prob_bullish

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

# Restore stdout and close file properly
sys.stdout = sys.stdout.terminal
