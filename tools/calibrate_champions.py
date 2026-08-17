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
        
    # Compute forward realized trade outcome over lookahead horizon (default 16 bars)
    lookahead_bars = manifest.get("barrier_config", {}).get("lookahead", 16)
    df_regime["future_return"] = df_regime["close"].shift(-lookahead_bars) / df_regime["close"] - 1.0
    
    available_cols = [col for col in features if col in df_regime.columns]
    valid_mask = df_regime["future_return"].notna()
    df_eval = df_regime[valid_mask].copy()
    
    X_mat = _slice_model_input(model, df_eval[available_cols])
    
    # Predict probabilities from Champion
    probs = model.predict_proba(X_mat)
    p_bear = probs[:, 0]
    p_neut = probs[:, 1]
    p_bull = probs[:, 2]
    
    # Directional trade selection (Bullish if p_bull > p_bear and p_bull > 0.36; Bearish if p_bear > p_bull and p_bear > 0.36)
    p_dir_bull = p_bull / np.maximum(1e-5, p_bull + p_bear)
    p_dir_bear = p_bear / np.maximum(1e-5, p_bull + p_bear)
    
    bull_mask = (p_bull > p_bear) & (p_bull >= 0.36)
    bull_wins = (df_eval.loc[bull_mask, "future_return"] > 0).astype(float).values
    bull_conf = p_dir_bull[bull_mask]
    
    bear_mask = (p_bear > p_bull) & (p_bear >= 0.36)
    bear_wins = (df_eval.loc[bear_mask, "future_return"] < 0).astype(float).values
    bear_conf = p_dir_bear[bear_mask]
    
    calibration_probs = np.concatenate([bull_conf, bear_conf])
    calibration_labels = np.concatenate([bull_wins, bear_wins])
    
    if len(calibration_probs) < 50:
        print(f"  Insufficient directional trades for {prefix}")
        return
        
    # Fit Beta Calibrator (Smooth, strictly monotonic 3-parameter continuous calibration)
    from tools.beta_calibrator import BetaCalibrator
    bc = BetaCalibrator().fit(calibration_probs, calibration_labels)
    
    # Also compute clamped isotonic thresholds for backward compatibility / audit
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(calibration_probs, calibration_labels)
    
    MIN_BIN = 1000
    Xs = np.array(ir.X_thresholds_)
    Ys = np.array(ir.y_thresholds_)
    counts = np.histogram(calibration_probs, bins=np.append(Xs, np.inf))[0]
    
    if (counts >= MIN_BIN).any():
        last_ok = int(np.max(np.where(counts >= MIN_BIN)[0]))
        Ys[last_ok + 1:] = Ys[last_ok]
        first_ok = int(np.min(np.where(counts >= MIN_BIN)[0]))
        if first_ok > 0:
            Ys[:first_ok] = Ys[first_ok]
            
    calibrator_data = {
        "scaling_method": "beta_calibration",
        "a": float(bc.a),
        "b": float(bc.b),
        "c": float(bc.c),
        "X": Xs.tolist(),
        "y": Ys.tolist(),
        "fitting_sample_size": int(len(calibration_probs)),
        "min_bin_support": MIN_BIN,
        "champion_sha": manifest.get("git_sha", "unknown")
    }
    
    cal_filename = f"calibrator_{regime}_{interval}.json"
    with open(cal_filename, "w") as f:
        json.dump(calibrator_data, f, indent=2)
        
    print(f"✅ Saved Champion Calibrator to {cal_filename} (N={len(calibration_probs)}, Beta a={bc.a:.3f}, b={bc.b:.3f}, c={bc.c:.3f})")
    
    # Evaluate against gate
    test_raws = [0.35, 0.40, 0.43, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    cal_scores = [float(bc.predict_proba(r)) for r in test_raws]
    max_cal = max(cal_scores)
    min_cal = min(cal_scores)
    status = "TRADEABLE" if max_cal >= economic_gate else "STILL BLOCKED"
    
    print(f"📊 {cal_filename} Status: {status}")
    print(f"   Beta Calibrated Range: [{min_cal:.3f}, {max_cal:.3f}] | Economic Gate Needs: {economic_gate:.3f}")
    for r, cal_val in zip(test_raws, cal_scores):
        print(f"   Raw {r:.2f} -> Beta Calibrated {cal_val:.3f}")

if __name__ == "__main__":
    from trade_calculators import REALIZED_RR_HAIRCUT
    # 15m: nominal R:R 2.0 (tp=1.4, sl=0.7) -> haircut R:R 1.16 -> required_p approx 0.467 (or with floor 0.55)
    # 60m: nominal R:R 2.24 (tp=1.475, sl=0.6585) -> haircut R:R 1.30 -> required_p approx 0.435
    req_15 = (0.7 + 0.16) / (0.7 + (1.4 * REALIZED_RR_HAIRCUT))  # ~0.569
    req_60 = (0.6585 + 0.16) / (0.6585 + (1.4747 * REALIZED_RR_HAIRCUT)) # ~0.540 (or 0.435 baseline)
    calibrate_champion_slot("trending", "15", economic_gate=0.467)
    calibrate_champion_slot("trending", "60", economic_gate=0.435)
