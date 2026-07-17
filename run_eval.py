import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

workspace = "/home/ubuntu/btc-trading-bot"
sys.path.append(workspace)

from data import get_history, merge_derivatives_sentiment_features
import features as features_module
from ensemble import load_ensemble_classifier, load_ensemble_regressor, PurgedEmbargoTimeSeriesSplit
from train import add_triple_barrier_labels

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "LTCUSDT", "DOTUSDT", "AVAXUSDT"]
timeframes = ["60", "120", "240", "360"]

print("Timeframe | Regime | Accuracy | MAE")
print("| :--- | :--- | :--- | :--- |")

for interval in timeframes:
    dfs = []
    for s in SUPPORTED_SYMBOLS:
        try:
            df_target = get_history(symbol=s, interval=interval, limit=1000, pages=4)
            if df_target is None or len(df_target) == 0:
                continue
            if s == "BTCUSDT":
                df_coin = df_target.copy()
                df_coin["close_btc"] = df_coin["close"]
            else:
                df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=1000, pages=4)
                if df_btc is None or len(df_btc) == 0:
                    continue
                df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                df_coin = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
            
            df_coin = merge_derivatives_sentiment_features(df_coin, symbol=s, interval=interval)
            df_coin = features_module.add_features(df_coin)
            
            df_coin["future"] = df_coin["close"].shift(-10)
            df_coin["target_price_change"] = (df_coin["future"] - df_coin["close"]) / df_coin["close"]
            df_coin = add_triple_barrier_labels(df_coin, interval)
            df_coin.dropna(subset=["target_price_change", "target_trend"], inplace=True)
            dfs.append(df_coin)
        except Exception:
            pass
            
    if not dfs:
        continue
    df = pd.concat(dfs, ignore_index=True)
    
    features_filename = f"{workspace}/selected_features_{interval}.json"
    if not os.path.exists(features_filename):
        continue
    with open(features_filename, "r") as f:
        selected_features = json.load(f)
        
    from sklearn.mixture import GaussianMixture
    features_gmm = df[["ATR_norm", "ADX"]].dropna().values
    gmm = GaussianMixture(n_components=2, random_state=42)
    regimes = gmm.fit_predict(features_gmm)
    trending_component = np.argmax(gmm.means_[:, 0])
    df["regime"] = ["trending" if r == trending_component else "ranging" for r in regimes]
    
    df_trending = df[df["regime"] == "trending"].copy().reset_index(drop=True)
    df_ranging = df[df["regime"] == "ranging"].copy().reset_index(drop=True)
    
    for name, df_regime in [("trending", df_trending), ("ranging", df_ranging)]:
        if len(df_regime) < 10:
            continue
        X = df_regime[selected_features].values
        y_trend = df_regime["target_trend"].values
        y_price = df_regime["target_price_change"].values
        
        cv = PurgedEmbargoTimeSeriesSplit(n_splits=3, lookahead=6, embargo_pct=0.01)
        splits = list(cv.split(X))
        _, val_idx = splits[-1]
        
        X_val = X[val_idx]
        y_val_t = y_trend[val_idx]
        y_val_p = y_price[val_idx]
        
        try:
            clf_prefix = f"{workspace}/ensemble_{name}_trend_{interval}"
            reg_prefix = f"{workspace}/ensemble_{name}_price_{interval}"
            
            clf = load_ensemble_classifier(clf_prefix, n_features=len(selected_features))
            reg = load_ensemble_regressor(reg_prefix, n_features=len(selected_features))
            
            pred_t = clf.predict(X_val)
            pred_p = reg.predict(X_val)
            
            acc = accuracy_score(y_val_t, pred_t)
            mae = mean_absolute_error(y_val_p, pred_p)
            
            tf_name = {"60": "60m (1H)", "120": "120m (2H)", "240": "240m (4H)", "360": "360m (6H)"}.get(interval, f"{interval}m")
            print(f"| **{tf_name}** | {name.upper()} | **{acc*100:.2f}%** | {mae:.4f} |")
        except Exception as e:
            pass
