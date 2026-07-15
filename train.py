import os
import json
import pandas as pd
import numpy as np
import joblib
import optuna
from ensemble import (
    PurgedEmbargoTimeSeriesSplit, EnsembleClassifier, EnsembleRegressor,
    save_ensemble_classifier, save_ensemble_regressor
)

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from data import get_history, merge_derivatives_sentiment_features
import threading
import requests
import features as features_module
from datetime import datetime, timedelta

# Dynamic GPU training hardware auto-detection
GPU_XGB = False
GPU_LGB = False
GPU_CAT = False

def check_gpu_support():
    global GPU_XGB, GPU_LGB, GPU_CAT
    import numpy as np
    
    # Check XGBoost CUDA GPU
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        clf.fit(np.array([[1.0, 2.0]]), np.array([1]))
        GPU_XGB = True
        print("[GPU Detection] XGBoost CUDA GPU support detected.")
    except Exception:
        GPU_XGB = False
        
    # Check LightGBM GPU
    try:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(device='gpu', n_estimators=1, verbose=-1)
        clf.fit(np.array([[1.0, 2.0]]), np.array([1]))
        GPU_LGB = True
        print("[GPU Detection] LightGBM GPU support detected.")
    except Exception:
        GPU_LGB = False

    # Check CatBoost GPU
    try:
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(task_type='GPU', iterations=1, verbose=0)
        clf.fit(np.array([[1.0, 2.0]]), np.array([1]))
        GPU_CAT = True
        print("[GPU Detection] CatBoost GPU support detected.")
    except Exception:
        GPU_CAT = False

check_gpu_support()

def create_model(model_class, params):
    params = dict(params)
    name = model_class.__name__
    if "XGB" in name:
        if GPU_XGB:
            params['device'] = 'cuda'
            params['tree_method'] = 'hist'
    elif "LGBM" in name:
        if GPU_LGB:
            params['device'] = 'gpu'
    elif "CatBoost" in name:
        if GPU_CAT:
            params['task_type'] = 'GPU'
    return model_class(**params)

economic_calendar_cache = None
economic_calendar_lock = threading.Lock()

# Centralized timeframe parameters for training labels and live execution alignment
TIMEFRAME_CONFIG = {
    "60": {   # 1H Timeframe
        "lookahead": 10,
        "sl_mult": 0.8,
        "tp_mult_ranging": 1.5,
        "tp_mult_trending": 2.5
    },
    "120": {  # 2H Timeframe
        "lookahead": 12,
        "sl_mult": 0.75,
        "tp_mult_ranging": 1.4,
        "tp_mult_trending": 2.2
    },
    "240": {  # 4H Timeframe
        "lookahead": 12,
        "sl_mult": 0.7,
        "tp_mult_ranging": 1.3,
        "tp_mult_trending": 2.0
    },
    "360": {  # 6H Timeframe
        "lookahead": 16,
        "sl_mult": 0.65,
        "tp_mult_ranging": 1.2,
        "tp_mult_trending": 1.8
    }
}

# =========================
# CONFIGURATION
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "60"
PAGES = 40  # 40 pages of candles provides ~40,000 candles (larger balanced dataset size)
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

# Feature list matches train.py and main.py
features = [
    "RSI", "MACD_diff", "MFI", "ATR_norm",
    "close_to_EMA9", "close_to_EMA21", "close_to_EMA50", "close_to_EMA200", "EMA9_to_EMA21",
    "BB_pct", "BB_width", "return_5m", "volatility_10m", "volume_ratio",
    "high_low_ratio", "open_close_ratio", "RSI_diff", "MACD_diff_diff", "ROC_5", "ROC_10",
    "ADX", "ADX_pos", "ADX_neg", "close_to_VWAP",
    "btc_return_5m", "btc_return_5m_lag1", "btc_return_5m_lag2", "btc_return_5m_lag3",
    "RSI_24", "ROC_24", "volatility_24h",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "RSI_z", "ADX_z", "close_to_Kalman"
]
for lag in [1, 2, 3, 4, 5]:
    features.append(f"return_5m_lag{lag}")
for lag in [1, 2, 3]:
    features.append(f"volume_ratio_lag{lag}")
for lag in [1, 2]:
    features.append(f"RSI_lag{lag}")
    features.append(f"MACD_diff_lag{lag}")
    features.append(f"BB_pct_lag{lag}")

# New features from Option 2
features.extend(["open_interest", "funding_rate", "fear_greed"])
for lag in [1, 2]:
    features.append(f"open_interest_lag{lag}")
    features.append(f"funding_rate_lag{lag}")
    features.append(f"fear_greed_lag{lag}")

# New microstructure and derivatives momentum features
features.extend([
    "open_interest_pct_change", "funding_rate_diff", 
    "CVD_rolling_1h", "CVD_rolling_4h"
])
for lag in [1, 2]:
    features.append(f"open_interest_pct_change_lag{lag}")
    features.append(f"funding_rate_diff_lag{lag}")
    features.append(f"CVD_rolling_1h_lag{lag}")
    features.append(f"CVD_rolling_4h_lag{lag}")

# New Wick Volume features (absorption/liquidation proxies)
features.extend([
    "upper_wick_volume_ratio", "lower_wick_volume_ratio"
])
for lag in [1, 2]:
    features.append(f"upper_wick_volume_ratio_lag{lag}")
    features.append(f"lower_wick_volume_ratio_lag{lag}")

features.append("hours_to_news")

# New Correlation and OI momentum features
features.extend(["oi_change_1h", "oi_change_4h", "btc_close", "btc_volume", "btc_rsi"])
for lag in [1, 2]:
    features.append(f"oi_change_1h_lag{lag}")
    features.append(f"oi_change_4h_lag{lag}")
    features.append(f"btc_close_lag{lag}")
    features.append(f"btc_volume_lag{lag}")
    features.append(f"btc_rsi_lag{lag}")

# Advanced Microstructure features
features.extend(["roll_spread", "leverage_divergence", "oi_velocity", "funding_acceleration", "bid_ask_imbalance_ohlc"])
for lag in [1, 2]:
    features.append(f"roll_spread_lag{lag}")
    features.append(f"leverage_divergence_lag{lag}")
    features.append(f"oi_velocity_lag{lag}")
    features.append(f"funding_acceleration_lag{lag}")
    features.append(f"bid_ask_imbalance_ohlc_lag{lag}")
    features.append(f"close_to_Kalman_lag{lag}")

def fetch_economic_calendar_cached(start_ts_ms=None, end_ts_ms=None):
    global economic_calendar_cache
    with economic_calendar_lock:
        if economic_calendar_cache is not None:
            return economic_calendar_cache
            
        try:
            finnhub_token = os.environ.get("FINNHUB_TOKEN", "free")
            now = datetime.utcnow()
            if start_ts_ms:
                from_dt = datetime.utcfromtimestamp(start_ts_ms / 1000.0)
            else:
                from_dt = now - timedelta(days=60)
                
            if end_ts_ms:
                to_dt = datetime.utcfromtimestamp(end_ts_ms / 1000.0) + timedelta(days=2)
            else:
                to_dt = now + timedelta(days=7)
                
            from_str = from_dt.strftime("%Y-%m-%d")
            to_str = to_dt.strftime("%Y-%m-%d")
            
            print(f"[News/Sentiment] Fetching economic calendar from {from_str} to {to_str}...")
            resp = requests.get(
                "https://finnhub.io/api/v1/calendar/economic",
                params={"token": finnhub_token, "from": from_str, "to": to_str},
                timeout=10
            )
            if resp.status_code == 200:
                events = resp.json().get("economicCalendar", [])
                high_impact = ["CPI", "FOMC", "NFP", "Non-Farm", "Federal Reserve", "Interest Rate"]
                filtered_events = []
                for ev in events:
                    if any(kw.lower() in ev.get("event", "").lower() for kw in high_impact):
                        ev_time_str = ev.get("time", "")
                        try:
                            ev_time = datetime.strptime(ev_time_str, "%Y-%m-%d %H:%M:%S")
                            filtered_events.append(ev_time)
                        except Exception:
                            pass
                economic_calendar_cache = sorted(filtered_events)
                print(f"[News/Sentiment] Cached {len(economic_calendar_cache)} high-impact calendar events.")
                return economic_calendar_cache
        except Exception as e:
            print(f"[News/Sentiment] Error caching economic calendar: {e}")
        
        economic_calendar_cache = []
        return economic_calendar_cache

def add_features(df):
    return features_module.add_features(df, fetch_calendar_callback=fetch_economic_calendar_cached)

OPTIMIZED_BARRIERS = {}

def add_triple_barrier_labels(df, interval):
    atr = df["ATR_norm"] * df["close"]
    atr_vals = atr.values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    adxs = df["ADX"].values
    n_samples = len(df)
    labels = np.ones(n_samples, dtype=int) * 1  # 1: Neutral
    
    # Load optimized barriers if available globally, else default to timeframe config
    cfg = OPTIMIZED_BARRIERS if OPTIMIZED_BARRIERS else TIMEFRAME_CONFIG.get(str(interval), {
        "lookahead": 10,
        "sl_mult": 0.8,
        "tp_mult_ranging": 1.5,
        "tp_mult_trending": 2.5
    })
    
    lookahead = cfg.get("lookahead", 10)
    sl_mult = cfg.get("sl_mult", 0.8)
    tp_mult_trending = cfg.get("tp_mult_trending", 2.5)
    tp_mult_ranging = cfg.get("tp_mult_ranging", 1.5)
        
    for i in range(n_samples):
        p_t = closes[i]
        atr_t = atr_vals[i]
        adx_t = adxs[i]
        if atr_t <= 0:
            atr_t = p_t * 0.001
            
        tp_mult = tp_mult_trending if adx_t >= 20.0 else tp_mult_ranging
        
        # Symmetric threshold modeling
        upper_barrier = p_t + tp_mult * atr_t
        lower_barrier = p_t - tp_mult * atr_t
        
        upper_stop = p_t + sl_mult * atr_t
        lower_stop = p_t - sl_mult * atr_t
        
        for step in range(1, lookahead + 1):
            if i + step >= n_samples:
                break
            h = highs[i + step]
            l = lows[i + step]
            
            # Symmetric checking (hit target before stop)
            hit_bullish = h >= upper_barrier and l > lower_stop
            hit_bearish = l <= lower_barrier and h < upper_stop
            
            if hit_bullish and not hit_bearish:
                labels[i] = 2  # Bullish
                break
            elif hit_bearish and not hit_bullish:
                labels[i] = 0  # Bearish
                break
            elif l <= lower_stop: # Long hit SL
                labels[i] = 0
                break
            elif h >= upper_stop: # Short hit SL
                labels[i] = 2
                break
                
    df["target_trend"] = labels
    return df

def tune_triple_barrier_multipliers(df_coin, interval):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    selected_feats = ["ATR_norm", "ADX", "RSI", "MACD_diff", "volume_ratio"]
    df_clean = df_coin.dropna(subset=selected_feats).copy().reset_index(drop=True)
    if len(df_clean) < 100:
        return {}
    X = df_clean[selected_feats].values
    cv = PurgedEmbargoTimeSeriesSplit(n_splits=3, lookahead=10, embargo_pct=0.01)
    
    def objective(trial):
        tp_m_ranging = trial.suggest_float("tp_mult_ranging", 1.0, 3.0)
        tp_m_trending = trial.suggest_float("tp_mult_trending", 2.0, 5.0)
        sl_m = trial.suggest_float("sl_mult", 0.5, 2.0)
        
        atr_vals = (df_clean["ATR_norm"] * df_clean["close"]).values
        closes = df_clean["close"].values
        highs = df_clean["high"].values
        lows = df_clean["low"].values
        adxs = df_clean["ADX"].values
        n_samples = len(df_clean)
        labels = np.ones(n_samples, dtype=int) * 1
        
        for i in range(n_samples):
            p_t = closes[i]
            atr_t = atr_vals[i]
            adx_t = adxs[i]
            if atr_t <= 0: atr_t = p_t * 0.001
            tp_mult = tp_m_trending if adx_t >= 20.0 else tp_m_ranging
            upper_b = p_t + tp_mult * atr_t
            lower_b = p_t - tp_mult * atr_t
            upper_s = p_t + sl_m * atr_t
            lower_s = p_t - sl_m * atr_t
            
            for step in range(1, 11):
                if i + step >= n_samples: break
                h, l = highs[i + step], lows[i + step]
                hit_bull = h >= upper_b and l > lower_s
                hit_bear = l <= lower_b and h < upper_s
                if hit_bull and not hit_bear:
                    labels[i] = 2
                    break
                elif hit_bear and not hit_bull:
                    labels[i] = 0
                    break
                elif l <= lower_s:
                    labels[i] = 0
                    break
                elif h >= upper_s:
                    labels[i] = 2
                    break
        y = labels
        from xgboost import XGBClassifier
        scores = []
        try:
            for train_idx, val_idx in cv.split(X, y):
                if len(train_idx) < 10 or len(val_idx) < 10:
                    continue
                model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)
                model.fit(X[train_idx], y[train_idx])
                scores.append(accuracy_score(y[val_idx], model.predict(X[val_idx])))
            return np.mean(scores) if scores else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    best = study.best_params
    best["lookahead"] = 10
    print(f"[Optuna Barrier Tuning] Best Multipliers: TP Ranging={best['tp_mult_ranging']:.2f}, TP Trending={best['tp_mult_trending']:.2f}, SL={best['sl_mult']:.2f}")
    return best

def optimize_xgb_classifier(X_train, y_train, X_val, y_val, sample_weights, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if regime == "trending":
            max_depth_min, max_depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            max_depth_min, max_depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', max_depth_min, max_depth_max),
            'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1
        }
        model = create_model(XGBClassifier, params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)
    return study.best_params

def optimize_lgb_classifier(X_train, y_train, X_val, y_val, sample_weights, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if regime == "trending":
            max_depth_min, max_depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            max_depth_min, max_depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', max_depth_min, max_depth_max),
            'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'objective': 'multiclass',
            'num_class': 3,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        model = create_model(LGBMClassifier, params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)
    return study.best_params

def optimize_cat_classifier(X_train, y_train, X_val, y_val, sample_weights, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if regime == "trending":
            depth_min, depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            depth_min, depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'iterations': trial.suggest_int('iterations', 50, 150),
            'depth': trial.suggest_int('depth', depth_min, depth_max),
            'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'loss_function': 'MultiClass',
            'verbose': 0,
            'random_seed': 42
        }
        model = create_model(CatBoostClassifier, params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)
    return study.best_params

def optimize_xgb_regressor(X_train, y_train, X_val, y_val, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if regime == "trending":
            max_depth_min, max_depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            max_depth_min, max_depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', max_depth_min, max_depth_max),
            'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 42,
            'n_jobs': -1
        }
        model = create_model(XGBRegressor, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=60)
    return study.best_params

def optimize_lgb_regressor(X_train, y_train, X_val, y_val, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if regime == "trending":
            max_depth_min, max_depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            max_depth_min, max_depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', max_depth_min, max_depth_max),
            'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        model = create_model(LGBMRegressor, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=60)
    return study.best_params

def optimize_cat_regressor(X_train, y_train, X_val, y_val, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if regime == "trending":
            depth_min, depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            depth_min, depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'iterations': trial.suggest_int('iterations', 50, 150),
            'depth': trial.suggest_int('depth', depth_min, depth_max),
            'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'verbose': 0,
            'random_seed': 42
        }
        model = create_model(CatBoostRegressor, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=60)
    return study.best_params

def train_models(interval=INTERVAL, pages=PAGES):
    global OPTIMIZED_BARRIERS
    OPTIMIZED_BARRIERS = {}
    
    # Pre-tune Triple-Barrier Multipliers on BTCUSDT to maximize general accuracy
    try:
        print("\n--- Running Optuna pre-study to optimize Triple-Barrier Multipliers ---")
        df_tune = get_history(symbol="BTCUSDT", interval=interval, limit=1000, pages=min(pages, 4))
        if df_tune is not None and len(df_tune) > 100:
            df_tune["close_btc"] = df_tune["close"]
            df_tune = merge_derivatives_sentiment_features(df_tune, symbol="BTCUSDT", interval=interval)
            df_tune = add_features(df_tune)
            best_barriers = tune_triple_barrier_multipliers(df_tune, interval)
            if best_barriers:
                OPTIMIZED_BARRIERS = best_barriers
                # Save to JSON for live load
                with open(f"optimized_barriers_{interval}.json", "w") as f:
                    json.dump(best_barriers, f)
                print(f"Saved optimized barriers configuration to optimized_barriers_{interval}.json")
    except Exception as e:
        print(f"[Warning] Optuna multiplier pre-tuning failed, using defaults: {e}")

    # =========================
    # LOAD & PROCESS DATA FOR ALL SUPPORTED COINS
    # =========================
    dfs = []
    for s in SUPPORTED_SYMBOLS:
        try:
            print(f"--- Processing {s} for interval {interval}m ({pages * 1000} candles) ---")
            if s == "BTCUSDT":
                df_target = get_history(symbol=s, interval=interval, limit=1000, pages=pages)
                if df_target is None or len(df_target) == 0:
                    continue
                df_coin = df_target.copy()
                df_coin["close_btc"] = df_coin["close"]
            else:
                df_target = get_history(symbol=s, interval=interval, limit=1000, pages=pages)
                if df_target is None or len(df_target) == 0:
                    continue
                df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=1000, pages=pages)
                if df_btc is None or len(df_btc) == 0:
                    continue
                df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                df_coin = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
                
            if len(df_coin) > 0:
                print(f"Merging Open Interest, Funding Rate, and Fear & Greed for {s}...")
                df_coin = merge_derivatives_sentiment_features(df_coin, symbol=s, interval=interval)
                print(f"Engineering features for {s}...")
                df_coin = add_features(df_coin)
                
                # Check features are present
                cols_ok = True
                for feat in features:
                    if feat not in df_coin.columns:
                        print(f"Missing feature {feat} in {s} data. Skipping.")
                        cols_ok = False
                        break
                if cols_ok:
                    # Generate targets individually per coin to avoid cross-symbol data leak!
                    lookahead = 10
                    df_coin["future"] = df_coin["close"].shift(-lookahead)
                    df_coin["target_price_change"] = (df_coin["future"] - df_coin["close"]) / df_coin["close"]
                    df_coin = add_triple_barrier_labels(df_coin, interval)
                    df_coin.dropna(subset=["target_price_change", "target_trend"], inplace=True)
                    df_coin = df_coin.copy()
                    
                    dfs.append(df_coin)
                    print(f"Successfully processed {s}: {len(df_coin)} rows.")
        except Exception as e:
            print(f"Error processing {s} during training dataset creation: {e}")
            
    if not dfs:
        print("[Retraining Error] No symbol data could be processed. Aborting.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n=== Combined Training Dataset: {len(df)} total rows across {len(dfs)} symbols ===")

    # Inject live trade feedback samples if --live-feedback flag is set
    if globals().get("LIVE_FEEDBACK", False):
        live_df = load_live_trade_samples(interval)
        if live_df is not None and len(live_df) > 0:
            # Ensure all needed columns exist
            for col in ["target_price_change", "target_trend"]:
                if col not in live_df.columns:
                    live_df[col] = 0
            df = pd.concat([df, live_df], ignore_index=True)
            print(f"[Live Feedback] Training dataset expanded to {len(df)} rows.")

    # ==========================================
    # AUTOML FEATURE SELECTION (RFECV NOISE REDUCTION)
    # ==========================================
    from sklearn.feature_selection import RFECV
    from xgboost import XGBClassifier
    print("\nRunning advanced feature selection via RFECV with Purged CV...")
    X_prelim = df[features]
    y_prelim = df["target_trend"]
    
    # Use a small estimator and 3-fold Purged CV for rapid feature elimination
    cv_selector = PurgedEmbargoTimeSeriesSplit(n_splits=3, lookahead=6, embargo_pct=0.01)
    estimator = XGBClassifier(n_estimators=40, max_depth=3, random_state=42, n_jobs=-1)
    
    selector = RFECV(
        estimator=estimator,
        step=2,
        cv=cv_selector,
        scoring="accuracy",
        min_features_to_select=20,
        n_jobs=-1
    )
    
    print("Fitting RFECV model (this may take a few seconds)...")
    selector.fit(X_prelim, y_prelim)
    
    selected_features = [f for f, support in zip(features, selector.support_) if support]
    print(f"RFECV complete: Selected optimal subset of {len(selected_features)} features (out of {len(features)}):")
    for rank, f_name in enumerate(selected_features, 1):
        print(f"  {rank}. {f_name}")
        
    # Save selected features list to file
    features_filename = f"selected_features_{interval}.json"
    with open(features_filename, "w") as f:
        json.dump(selected_features, f)
    print(f"Saved selected features list to {features_filename}")

    # ==========================================
    # REGIME SPLITTING & REGIME MODEL TRAINING
    # ==========================================
    def train_regime_model(df_regime, name):
        print(f"\nTraining model set for regime: {name.upper()} (Candles: {len(df_regime)})")
        if len(df_regime) < 100:
            print(f"Skipping {name} due to insufficient data.")
            return

        X = df_regime[selected_features]
        y_trend = df_regime["target_trend"]
        y_price = df_regime["target_price_change"]

        # Purged and Embargoed Time-Series Cross Validation
        cv = PurgedEmbargoTimeSeriesSplit(n_splits=5, lookahead=6, embargo_pct=0.01)
        
        meta_features_list = []
        meta_labels_list = []
        
        primary_accuracies = []
        primary_maes = []
        
        calibration_probs = []
        calibration_labels = []
        
        first_fold = True
        best_params_xgb_t = None
        best_params_lgb_t = None
        best_params_cat_t = None
        
        best_params_xgb_p = None
        best_params_lgb_p = None
        best_params_cat_p = None
        
        from sklearn.utils.class_weight import compute_sample_weight
        
        print(f"  Running Purged & Embargoed Cross-Validation...")
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_trend)):
            X_train, y_train_t, y_train_p = X.iloc[train_idx], y_trend.iloc[train_idx], y_price.iloc[train_idx]
            X_val, y_val_t, y_val_p = X.iloc[val_idx], y_trend.iloc[val_idx], y_price.iloc[val_idx]
            
            sample_weight_train = compute_sample_weight(class_weight='balanced', y=y_train_t)
            decay_weights = np.linspace(0.3, 1.0, len(y_train_t))
            sample_weight_train = sample_weight_train * decay_weights
            
            if first_fold:
                print("  Optimizing hyperparameters on first fold...")
                best_params_xgb_t = optimize_xgb_classifier(X_train, y_train_t, X_val, y_val_t, sample_weight_train, regime=name)
                best_params_lgb_t = optimize_lgb_classifier(X_train, y_train_t, X_val, y_val_t, sample_weight_train, regime=name)
                best_params_cat_t = optimize_cat_classifier(X_train, y_train_t, X_val, y_val_t, sample_weight_train, regime=name)
                
                best_params_xgb_p = optimize_xgb_regressor(X_train, y_train_p, X_val, y_val_p, regime=name)
                best_params_lgb_p = optimize_lgb_regressor(X_train, y_train_p, X_val, y_val_p, regime=name)
                best_params_cat_p = optimize_cat_regressor(X_train, y_train_p, X_val, y_val_p, regime=name)
                first_fold = False
            
            # Instantiate models with best parameters
            xgb_t = create_model(XGBClassifier, best_params_xgb_t)
            lgb_t = create_model(LGBMClassifier, best_params_lgb_t)
            cat_t = create_model(CatBoostClassifier, best_params_cat_t)
            ensemble_t = EnsembleClassifier(xgb_t, lgb_t, cat_t)
            
            xgb_p = create_model(XGBRegressor, best_params_xgb_p)
            lgb_p = create_model(LGBMRegressor, best_params_lgb_p)
            cat_p = create_model(CatBoostRegressor, best_params_cat_p)
            ensemble_p = EnsembleRegressor(xgb_p, lgb_p, cat_p)
            
            # Train on this fold
            ensemble_t.fit(X_train, y_train_t, sample_weight=sample_weight_train)
            ensemble_p.fit(X_train, y_train_p)
            
            # Out of sample validation predictions
            pred_val_t = ensemble_t.predict(X_val)
            pred_val_p = ensemble_p.predict(X_val)
            
            # Metrics
            acc = accuracy_score(y_val_t, pred_val_t)
            mae = mean_absolute_error(y_val_p, pred_val_p)
            print(f"    - Chronological Walk-Forward Fold {fold+1} Accuracy: {acc*100:.2f}% | MAE: {mae:.5f}")
            primary_accuracies.append(acc)
            primary_maes.append(mae)
            
            # Out of sample prediction probabilities for calibration
            probs_val = ensemble_t.predict_proba(X_val)
            for j in range(len(X_val)):
                w_class = int(np.argmax(probs_val[j]))
                if w_class in [0, 2]: # Bearish or Bullish only
                    calibration_probs.append(float(probs_val[j][w_class]))
                    calibration_labels.append(1 if w_class == y_val_t.values[j] else 0)
            
            # Generate Meta-labels for this validation fold
            actual_val_t = y_val_t.values
            is_non_neutral = (pred_val_t != 1)
            is_correct = (pred_val_t == actual_val_t)
            
            meta_features_list.append(X_val[is_non_neutral])
            meta_labels_list.append(is_correct[is_non_neutral].astype(int))
            
        print(f"  Validation Out-of-Sample Accuracy (Ensemble Trend): {np.mean(primary_accuracies)*100:.2f}%")
        print(f"  Validation Out-of-Sample MAE (Ensemble Price): {np.mean(primary_maes):.4f}")
        
        # Meta-Classifier Dataset
        valid_dfs = [df_item for df_item in meta_features_list if not df_item.empty]
        valid_labels = [lbl_item for lbl_item in meta_labels_list if len(lbl_item) > 0]
        
        if len(valid_dfs) > 0 and len(valid_labels) > 0:
            meta_X = pd.concat(valid_dfs, ignore_index=True)
            meta_y = pd.Series(np.concatenate(valid_labels))
        else:
            # Fallback if no trades/non-neutral predictions occurred during validation
            meta_X = X.copy()
            meta_y = pd.Series(np.ones(len(X), dtype=int))
            
        if meta_y.nunique() < 2:
            # Inject a dummy opposite label at index 0 to ensure binary classification is possible
            if len(meta_y) > 0:
                current_val = meta_y.iloc[0]
                opposite_val = 0 if current_val == 1 else 1
                meta_y.iloc[0] = opposite_val
            else:
                meta_y = pd.Series([0, 1])
                meta_X = pd.concat([X.iloc[:1], X.iloc[:1]], ignore_index=True)
        
        print(f"  Meta-Classifier Training Samples: {len(meta_X)} (Positive rate: {meta_y.mean()*100:.2f}%)")
        
        # Train Meta-Classifier (XGBoost Binary Classifier)
        n_pos = sum(meta_y)
        n_neg = len(meta_y) - n_pos
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        
        meta_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        if len(meta_X) >= 20:
            meta_model.fit(meta_X, meta_y)
            print("  Meta-Classifier trained successfully.")
        else:
            meta_model.fit(X, np.ones(len(X)))
            print("  Warning: Insufficient samples for Meta-Classifier. Dummy classifier trained.")
            
        # Train Isotonic Regression Calibrator on validation predictions
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        
        # Fit calibrator
        if len(calibration_probs) > 10:
            ir.fit(calibration_probs, calibration_labels)
            calibrator_data = {
                "X": ir.X_thresholds_.tolist(),
                "y": ir.y_thresholds_.tolist()
            }
            calibrator_filename = f"calibrator_{name}_{interval}.json"
            with open(calibrator_filename, "w") as f:
                json.dump(calibrator_data, f)
            print(f"  [Calibrator] Saved Isotonic Regression calibrator to {calibrator_filename}")
        else:
            # Save default identity mapping if no predictions occurred
            calibrator_data = {"X": [0.0, 1.0], "y": [0.0, 1.0]}
            calibrator_filename = f"calibrator_{name}_{interval}.json"
            with open(calibrator_filename, "w") as f:
                json.dump(calibrator_data, f)
            print(f"  [Calibrator] Saved default calibrator to {calibrator_filename}")
            
        # Fit final primary models on complete regime dataset
        print(f"  Training final ensemble models on complete {name} dataset...")
        
        # Fallbacks for hyperparameter dictionary if cross-validation folds didn't run due to dataset constraints
        if best_params_xgb_t is None:
            best_params_xgb_t = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}
        if best_params_lgb_t is None:
            best_params_lgb_t = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}
        if best_params_cat_t is None:
            best_params_cat_t = {'iterations': 100, 'depth': 4, 'learning_rate': 0.05}
            
        if best_params_xgb_p is None:
            best_params_xgb_p = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}
        if best_params_lgb_p is None:
            best_params_lgb_p = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}
        if best_params_cat_p is None:
            best_params_cat_p = {'iterations': 100, 'depth': 4, 'learning_rate': 0.05}

        final_xgb_t = create_model(XGBClassifier, best_params_xgb_t)
        final_lgb_t = create_model(LGBMClassifier, best_params_lgb_t)
        final_cat_t = create_model(CatBoostClassifier, best_params_cat_t)
        final_ensemble_t = EnsembleClassifier(final_xgb_t, final_lgb_t, final_cat_t)
        
        final_xgb_p = create_model(XGBRegressor, best_params_xgb_p)
        final_lgb_p = create_model(LGBMRegressor, best_params_lgb_p)
        final_cat_p = create_model(CatBoostRegressor, best_params_cat_p)
        final_ensemble_p = EnsembleRegressor(final_xgb_p, final_lgb_p, final_cat_p)
        
        sample_weight_full = compute_sample_weight(class_weight='balanced', y=y_trend)
        decay_full = np.linspace(0.3, 1.0, len(y_trend))
        sample_weight_full = sample_weight_full * decay_full
        
        sample_weight_train_last = compute_sample_weight(class_weight='balanced', y=y_train_t)
        decay_train_last = np.linspace(0.3, 1.0, len(y_train_t))
        sample_weight_train_last = sample_weight_train_last * decay_train_last
        
        final_ensemble_t.fit(
            X, y_trend, sample_weight=sample_weight_full, 
            X_val=X_val, y_val=y_val_t, 
            X_train=X_train, y_train=y_train_t, 
            sample_weight_train=sample_weight_train_last
        )
        final_ensemble_p.fit(
            X, y_price, 
            X_val=X_val, y_val=y_val_p, 
            X_train=X_train, y_train=y_train_p
        )
        
        # Save models to disk using native text/JSON saving methods
        import os
        from ensemble import load_ensemble_classifier, load_ensemble_regressor
        
        c_prefix_t = f"ensemble_{name}_trend_{interval}"
        c_prefix_p = f"ensemble_{name}_price_{interval}"
        
        # Check if champion model exists
        champion_exists = os.path.exists(f"{c_prefix_t}_xgb.json")
        
        should_save = True
        if champion_exists:
            try:
                # Load existing champion
                champion_t = load_ensemble_classifier(c_prefix_t, n_features=X_val.shape[1])
                champion_p = load_ensemble_regressor(c_prefix_p, n_features=X_val.shape[1])
                
                # Evaluate champion on the last validation fold
                champ_pred_t = champion_t.predict(X_val)
                champ_pred_p = champion_p.predict(X_val)
                champ_acc = accuracy_score(y_val_t, champ_pred_t)
                champ_mae = mean_absolute_error(y_val_p, champ_pred_p)
                
                # Evaluate challenger on the last validation fold
                chal_pred_t = final_ensemble_t.predict(X_val)
                chal_pred_p = final_ensemble_p.predict(X_val)
                chal_acc = accuracy_score(y_val_t, chal_pred_t)
                chal_mae = mean_absolute_error(y_val_p, chal_pred_p)
                
                print(f"  [Champion-Challenger] Validation Comparison for {name.upper()}:")
                print(f"    - Classifier Accuracy: Champion = {champ_acc*100:.2f}% | Challenger = {chal_acc*100:.2f}%")
                print(f"    - Regressor MAE: Champion = {champ_mae:.4f} | Challenger = {chal_mae:.4f}")
                
                # Update if accuracy is strictly better, or equal accuracy with lower MAE
                if chal_acc > champ_acc:
                    should_save = True
                elif chal_acc == champ_acc and chal_mae < champ_mae:
                    should_save = True
                else:
                    should_save = False
            except Exception as eval_err:
                print(f"  [Champion-Challenger Warning] Error comparing models, defaulting to save: {eval_err}")
                should_save = True
        else:
            print(f"  [Champion-Challenger] No existing champion model for {name.upper()}. Saving challenger.")
            should_save = True
            
        if should_save:
            print(f"  [Champion-Challenger] Challenger approved. Overwriting active model files...")
            save_ensemble_classifier(final_ensemble_t, c_prefix_t)
            save_ensemble_regressor(final_ensemble_p, c_prefix_p)
            meta_model.save_model(f"meta_{name}_trend_{interval}.json")
            print(f"  Saved ensemble and meta-classifier models for regime: {name.upper()}")
        else:
            print(f"  [Champion-Challenger] Champion model retained (Challenger did not show improvement).")

    # Split dataset based on GMM Unsupervised Regime Classification
    from sklearn.mixture import GaussianMixture
    import numpy as np

    print("Fitting Gaussian Mixture Model for Unsupervised Regime Splitting...")
    features_gmm = df[["ATR_norm", "ADX"]].dropna().values
    gmm = GaussianMixture(n_components=2, random_state=42)
    regimes = gmm.fit_predict(features_gmm)

    # Save pre-trained GMM for inference mapping
    joblib.dump(gmm, f"gmm_regime_{interval}.pkl")
    print(f"Saved GMM model: gmm_regime_{interval}.pkl")

    trending_component = np.argmax(gmm.means_[:, 0])  # Index with highest mean ATR_norm

    df["regime"] = ["trending" if r == trending_component else "ranging" for r in regimes]
    df_trending = df[df["regime"] == "trending"].copy().reset_index(drop=True)
    df_ranging = df[df["regime"] == "ranging"].copy().reset_index(drop=True)

    train_regime_model(df_trending, "trending")
    train_regime_model(df_ranging, "ranging")

def load_live_trade_samples(interval, days=2, weight=3.0):
    """Load recent closed trades, re-fetch features at entry time, return as weighted DataFrame."""
    try:
        import time as _time
        history_file = "dashboard_history.json"
        if not os.path.exists(history_file):
            return None
        with open(history_file, "r") as f:
            data = json.load(f)
        trades = data.get("trade_history", [])
        cutoff = _time.time() - days * 86400
        trades = [t for t in trades if str(t.get("interval", "")) == str(interval) and float(t.get("exit_time", 0)) >= cutoff]
        if not trades:
            print(f"[Live Feedback] No trades in last {days} days for interval {interval}m.")
            return None
        
        sample_dfs = []
        for t in trades:
            symbol = t.get("symbol")
            exit_ts = float(t.get("exit_time", 0))
            pnl = float(t.get("pnl_usd", 0.0))
            direction = t.get("direction", "Bullish")
            # Fetch ~60 candles ending just before exit to capture entry features
            df_c = get_history(symbol=symbol, interval=interval, limit=60, pages=1)
            if df_c is None or len(df_c) < 20:
                continue
            # Keep only rows before exit time
            df_c = df_c[df_c["timestamp"] <= exit_ts * 1000].copy()
            if len(df_c) < 10:
                continue
            df_c = merge_derivatives_sentiment_features(df_c, symbol=symbol, interval=interval)
            df_c = add_features(df_c)
            df_c = df_c.dropna()
            if len(df_c) == 0:
                continue
            # Use the last available row (closest to entry)
            row = df_c.iloc[[-1]].copy()
            # Label: correct direction = 1, wrong = 0
            row["target_trend"] = 1 if pnl > 0 else 0
            row["target_price_change"] = 0.0
            row["sample_weight"] = weight
            sample_dfs.append(row)
        
        if not sample_dfs:
            return None
        result = pd.concat(sample_dfs, ignore_index=True)
        print(f"[Live Feedback] Injecting {len(result)} real trade samples (weight={weight}x) for interval {interval}m.")
        return result
    except Exception as e:
        print(f"[Live Feedback] Error loading live trade samples: {e}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train XGBoost models for BTC Trading Bot")
    parser.add_argument("--interval", type=str, default="60", choices=["60", "120", "240", "360", "all"], help="Timeframe interval to train")
    parser.add_argument("--pages", type=int, default=20, help="Number of data pages to fetch from Bybit")
    parser.add_argument("--live-feedback", action="store_true", help="Inject recent live trade outcomes as weighted samples")
    args = parser.parse_args()
    LIVE_FEEDBACK = args.live_feedback

    if args.interval == "all":
        for iv in ["60", "120", "240", "360"]:
            print(f"\n==============================================")
            print(f"TRAINING FOR INTERVAL: {iv}")
            print(f"==============================================")
            train_models(interval=iv, pages=args.pages)
    else:
        train_models(interval=args.interval, pages=args.pages)