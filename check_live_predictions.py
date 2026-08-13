import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import load_ensemble_classifier

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

print("===================================================================================")
print("   LIVE MODEL PREDICTION ENGINE CHECK (15M & 60M TIMEFRAMES)")
print("===================================================================================")

for interval in [15, 60]:
    print(f"\n------------------ TIMEFRAME: {interval}M MODELS ------------------")
    
    try:
        model_ranging = load_ensemble_classifier(f"ensemble_ranging_trend_{interval}")
        model_trending = load_ensemble_classifier(f"ensemble_trending_trend_{interval}")
        
        with open(f"selected_features_{interval}_ranging.json") as f:
            feats_ranging = json.load(f)
        with open(f"selected_features_{interval}_trending.json") as f:
            feats_trending = json.load(f)
            
        print(f"[{interval}m Governance] Ranging Model & Trending Model Loaded Successfully.")
    except Exception as e:
        print(f"[{interval}m Governance ERROR] Failed to load models: {e}")
        continue

    print(f"{'Symbol':<9} | {'Regime':<8} | {'Prob Bear':<9} | {'Prob Neut':<9} | {'Prob Bull':<9} | {'Dir Signal':<10} | {'Confidence':<10}")
    print("-" * 80)

    for s in SUPPORTED_SYMBOLS:
        df_s = get_history(symbol=s, interval=interval, limit=300, pages=1)
        if df_s is None or len(df_s) < 215:
            print(f"{s:<9} | Insufficient data (got {len(df_s) if df_s is not None else 0} bars)")
            continue
            
        df_s["symbol"] = s
        df_s["close_btc"] = df_s["close"]
        df_s = merge_derivatives_sentiment_features(df_s, symbol=s, interval=interval)
        df_s = add_features(df_s)
        
        last_idx = len(df_s) - 1
        adx_val = df_s.loc[last_idx, "ADX"]
        is_trending = adx_val >= 25.0
        regime_str = "TRENDING" if is_trending else "RANGING"
        
        if is_trending:
            feats = feats_trending
            probs = model_trending.predict_proba(df_s.iloc[[last_idx]][feats].values)[0]
        else:
            feats = feats_ranging
            probs = model_ranging.predict_proba(df_s.iloc[[last_idx]][feats].values)[0]
            
        p_bear, p_neut, p_bull = probs[0], probs[1], probs[2]
        dir_total = p_bear + p_bull
        
        norm_bear = p_bear / max(1e-9, dir_total)
        norm_bull = p_bull / max(1e-9, dir_total)
        
        eval_thresh = 0.2862 if interval == 15 else 0.2918
        
        if norm_bull > norm_bear and norm_bull >= eval_thresh:
            signal = "BULLISH"
            conf = norm_bull
        elif norm_bear > norm_bull and norm_bear >= eval_thresh:
            signal = "BEARISH"
            conf = norm_bear
        else:
            signal = "NEUTRAL"
            conf = max(norm_bear, norm_bull)
            
        print(f"{s:<9} | {regime_str:<8} | {p_bear:9.4f} | {p_neut:9.4f} | {p_bull:9.4f} | {signal:<10} | {conf*100:6.2f}%")

print("\n===================================================================================")
