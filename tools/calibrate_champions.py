import os
import sys
import json
import bisect
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.isotonic import IsotonicRegression
from ensemble import load_ensemble_classifier, _slice_model_input
from features import add_features
from data import get_history, merge_derivatives_sentiment_features
from train import add_triple_barrier_labels

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

def calibrate_champion_slot(regime: str, interval: str, economic_gate: float):
    prefix = f"ensemble_{regime}_trend_{interval}"
    manifest_path = f"{prefix}_manifest.json"
    
    if not os.path.exists(manifest_path):
        print(f"[{prefix}] Manifest {manifest_path} not found. Skipping.")
        return
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    features = manifest.get("feature_names", [])
    if not features:
        print(f"[{prefix}] No feature_names in manifest. Skipping.")
        return

    print(f"\n==================================================")
    print(f"Calibrating Champion: {prefix} ({len(features)} features)")
    print(f"==================================================")
    
    model = load_ensemble_classifier(prefix, n_features=len(features), feature_names=features)
    
    # Load dataset across symbols
    all_dfs = []
    for sym in SUPPORTED_SYMBOLS:
        try:
            df_sym = get_history(symbol=sym, interval=str(interval), limit=3000)
            df_btc = get_history(symbol="BTCUSDT", interval=str(interval), limit=3000) if sym != "BTCUSDT" else df_sym
            if df_sym is not None and len(df_sym) > 100:
                df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                df_merged = pd.merge(df_sym, df_btc_sub, on="timestamp", how="left")
                df_merged["close_btc"] = df_merged["close_btc"].ffill().bfill().fillna(df_merged["close"])
                df_merged = merge_derivatives_sentiment_features(df_merged, symbol=sym, interval=interval)
                df_feat = add_features(df_merged)
                df_labeled = add_triple_barrier_labels(df_feat, interval=interval)
                df_labeled["symbol"] = sym
                all_dfs.append(df_labeled)
        except Exception as e:
            print(f"  Warning loading {sym}: {e}")
            
    if not all_dfs:
        print(f"  Failed to load data for {prefix}")
        return
        
    df_combined = pd.concat(all_dfs, ignore_index=True).dropna(subset=["close", "target_trend"])
    
    # Filter regime by ADX if applicable
    if "ADX" in df_combined.columns:
        if regime == "trending":
            df_regime = df_combined[df_combined["ADX"] >= 25.0].copy()
        else:
            df_regime = df_combined[df_combined["ADX"] < 25.0].copy()
    else:
        df_regime = df_combined.copy()
        
    if len(df_regime) < 100:
        df_regime = df_combined.copy()
        
    available_cols = [col for col in features if col in df_regime.columns]
    X_mat = _slice_model_input(model, df_regime[available_cols])
    y_true = df_regime["target_trend"].values
    
    # Predict probabilities from Champion
    probs = model.predict_proba(X_mat)
    
    # Isotonic calibration on directional predictions (Bullish=2 vs Bearish=0)
    pred_classes = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    
    # Filter for directional predictions (exclude neutral=1)
    mask = (pred_classes != 1) & (y_true != 1)
    
    if np.sum(mask) < 50:
        mask = (pred_classes != 1)
        y_binary = (y_true == pred_classes).astype(float)
    else:
        y_binary = (y_true[mask] == pred_classes[mask]).astype(float)
        
    calibration_probs = max_probs[mask]
    calibration_labels = y_binary
    
    # Fit Isotonic Regression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(calibration_probs, calibration_labels)
    
    calibrator_data = {
        "X": ir.X_thresholds_.tolist(),
        "y": ir.y_thresholds_.tolist(),
        "fitting_sample_size": int(len(calibration_probs)),
        "scaling_method": "isotonic",
        "champion_sha": manifest.get("git_sha", "unknown")
    }
    
    cal_filename = f"calibrator_{regime}_{interval}.json"
    with open(cal_filename, "w") as f:
        json.dump(calibrator_data, f, indent=2)
        
    print(f"✅ Saved Champion Calibrator to {cal_filename} (N={len(calibration_probs)})")
    
    # Evaluate against gate
    X_thresh = calibrator_data["X"]
    y_thresh = calibrator_data["y"]
    max_cal = max(y_thresh)
    min_cal = min(y_thresh)
    status = "TRADEABLE" if max_cal >= economic_gate else "STILL BLOCKED"
    
    print(f"📊 {cal_filename} Status: {status}")
    print(f"   Calibrated Range: [{min_cal:.3f}, {max_cal:.3f}] | Economic Gate Needs: {economic_gate:.3f}")
    for r in (0.35, 0.40, 0.43, 0.45, 0.50, 0.55, 0.60):
        i = min(bisect.bisect_left(X_thresh, r), len(y_thresh) - 1)
        print(f"   Raw {r:.2f} -> Calibrated {y_thresh[i]:.3f}")

if __name__ == "__main__":
    calibrate_champion_slot("trending", "15", 0.422)
    calibrate_champion_slot("trending", "60", 0.358)
