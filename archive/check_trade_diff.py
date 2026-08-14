import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import load_ensemble_classifier

df = get_history(symbol="BTCUSDT", interval=15, limit=1000, pages=3)
df["close_btc"] = df["close"]
df = merge_derivatives_sentiment_features(df, symbol="BTCUSDT", interval=15)
df = add_features(df)

model_ranging = load_ensemble_classifier("ensemble_ranging_trend_15")
with open("selected_features_15_ranging.json") as f:
    feats = json.load(f)

probs = model_ranging.predict_proba(df[feats].values)
pred_classes = probs.argmax(axis=1)

closes = df["close"].values
atr_norms = df["ATR_norm"].values

print("=== TRADE ENTRY VERIFICATION COMPARISON ===")
count = 0
for idx in range(100, 300):
    if pred_classes[idx] == 1 or atr_norms[idx] <= 0: continue
    p_entry = closes[idx]
    atr = atr_norms[idx] * p_entry
    is_bull = (pred_classes[idx] == 2)
    dir_str = "BULL" if is_bull else "BEAR"
    
    tp_old = p_entry + 0.98 * atr if is_bull else p_entry - 0.98 * atr
    sl_old = p_entry - 0.78 * atr if is_bull else p_entry + 0.78 * atr
    
    tp_new = p_entry + 2.50 * atr if is_bull else p_entry - 2.50 * atr
    sl_new = p_entry - 1.00 * atr if is_bull else p_entry + 1.00 * atr
    
    print(f"Trade #{count+1:02d} (Bar #{idx:03d}) [{dir_str}] Entry: ${p_entry:.2f} | ATR: ${atr:.2f}")
    print(f"   OLD (0.98/0.78): TP=${tp_old:.2f} (Target +${abs(tp_old-p_entry):.2f}) | SL=${sl_old:.2f} (Stop -${abs(sl_old-p_entry):.2f})")
    print(f"   NEW (2.50/1.00): TP=${tp_new:.2f} (Target +${abs(tp_new-p_entry):.2f}) | SL=${sl_new:.2f} (Stop -${abs(sl_new-p_entry):.2f})")
    print("-" * 65)
    count += 1
    if count >= 3: break
