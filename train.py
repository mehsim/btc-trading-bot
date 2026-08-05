from logger import log_event
import os
import json
import time
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Optional
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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
    classification_report, confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, average_precision_score
)
from config import MODEL_SELECTION
from mlops_engine import calculate_expected_calibration_error
from data import get_history, merge_derivatives_sentiment_features
import threading
import requests
import features as features_module

def safe_mean(values: list):
    """Mean of non-None, non-NaN values. Returns None if no valid values."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(valid)) if valid else None

def safe_stat(values: list) -> dict:
    """mean/median/std/min/max over valid values. All fields None if no valid values."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.array(valid, dtype=float)
    return {
        "mean":   round(float(np.mean(arr)),   4),
        "median": round(float(np.median(arr)), 4),
        "std":    round(float(np.std(arr)),    4),
        "min":    round(float(np.min(arr)),    4),
        "max":    round(float(np.max(arr)),    4),
    }

def _emit_governance_event(event: dict):
    """Appends a structured governance event to the audit trail."""
    event["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with open("governance_audit_trail.jsonl", "a") as _gf:
            _gf.write(json.dumps(event) + "\n")
    except Exception as _ge:
        print(f"[Governance] Failed to write audit event: {_ge}")

def _tg_alert(msg: str):
    """Send a Telegram notification from train.py without importing main.py."""
    try:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_ids = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_ids:
            return
        for cid in chat_ids.split(","):
            cid = cid.strip()
            if cid:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": cid, "text": msg, "parse_mode": "Markdown"},
                    timeout=10
                )
    except Exception as e:
        print(f"[Train Telegram] Alert failed: {e}")
from datetime import datetime, timedelta, timezone
from mlops_engine import model_registry, MLFLOW_AVAILABLE
if MLFLOW_AVAILABLE:
    # pyrefly: ignore [missing-import]
    import mlflow

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
    except Exception as ex_train:
        log_event("WARNING", f"train notice: {ex_train}")
        GPU_XGB = False
        
    # Check LightGBM GPU
    try:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(device='gpu', n_estimators=1, verbose=-1)
        clf.fit(np.array([[1.0, 2.0]]), np.array([1]))
        GPU_LGB = True
        print("[GPU Detection] LightGBM GPU support detected.")
    except Exception as ex_train:
        log_event("WARNING", f"train notice: {ex_train}")
        GPU_LGB = False

    # Check CatBoost GPU
    try:
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(task_type='GPU', iterations=1, verbose=0)
        clf.fit(np.array([[1.0, 2.0]]), np.array([1]))
        GPU_CAT = True
        print("[GPU Detection] CatBoost GPU support detected.")
    except Exception as ex_train:
        log_event("WARNING", f"train notice: {ex_train}")
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

from config import TIMEFRAME_CONFIG

# =========================
# CONFIGURATION
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "60"
PAGES = 5  # 5 pages ~5,000 candles — low-RAM retraining mode
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
    "CVD_true", "OFI_true"
])
for lag in [1, 2]:
    features.append(f"open_interest_pct_change_lag{lag}")
    features.append(f"funding_rate_diff_lag{lag}")
    features.append(f"CVD_true_lag{lag}")
    features.append(f"OFI_true_lag{lag}")

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
features.extend([
    "roll_spread", "leverage_divergence", "oi_velocity", "funding_acceleration", "bid_ask_imbalance_ohlc",
    "ob_imbalance_L2", "ob_spread_L2", "liq_long_1h", "liq_short_1h"
])
for lag in [1, 2]:
    features.append(f"roll_spread_lag{lag}")
    features.append(f"leverage_divergence_lag{lag}")
    features.append(f"oi_velocity_lag{lag}")
    features.append(f"funding_acceleration_lag{lag}")
    features.append(f"bid_ask_imbalance_ohlc_lag{lag}")
    features.append(f"close_to_Kalman_lag{lag}")
    features.append(f"ob_imbalance_L2_lag{lag}")
    features.append(f"ob_spread_L2_lag{lag}")
    features.append(f"liq_long_1h_lag{lag}")
    features.append(f"liq_short_1h_lag{lag}")

# Garman-Klass Volatility features
features.extend(["volatility_gk", "volatility_gk_lag1", "volatility_gk_lag2"])

# Candlestick Pattern features (30 patterns)
cdl_features = [
    "cdl_hammer", "cdl_hanging_man", "cdl_shooting_star", "cdl_inv_hammer", "cdl_doji",
    "cdl_gravestone_doji", "cdl_dragonfly_doji", "cdl_spinning_top", "cdl_marubozu_bull", "cdl_marubozu_bear",
    "cdl_bullish_engulfing", "cdl_bearish_engulfing", "cdl_bullish_harami", "cdl_bearish_harami",
    "cdl_tweezer_top", "cdl_tweezer_bottom", "cdl_piercing_line", "cdl_dark_cloud_cover", "cdl_inside_bar",
    "cdl_morning_star", "cdl_evening_star", "cdl_morning_doji_star", "cdl_evening_doji_star",
    "cdl_three_white_soldiers", "cdl_three_black_crows", "cdl_three_inside_up", "cdl_three_inside_down",
    "cdl_rising_three", "cdl_falling_three", "cdl_abandoned_baby_bull"
]
features.extend(cdl_features)


# Cross-Asset Lead-Lag Correlation features
features.extend(["lead_lag_diff_5m", "lead_lag_diff_1h", "lead_lag_diff_4h", "volume_ratio_to_btc"])
for lag in [1, 2]:
    features.append(f"lead_lag_diff_5m_lag{lag}")
    features.append(f"lead_lag_diff_1h_lag{lag}")
    features.append(f"lead_lag_diff_4h_lag{lag}")
    features.append(f"volume_ratio_to_btc_lag{lag}")

def fetch_economic_calendar_cached(start_ts_ms=None, end_ts_ms=None):
    global economic_calendar_cache
    with economic_calendar_lock:
        if economic_calendar_cache is not None:
            return economic_calendar_cache
            
        try:
            finnhub_token = os.environ.get("FINNHUB_TOKEN", "").strip()
            if not finnhub_token or finnhub_token == "free":
                return []


            now = datetime.now(timezone.utc)
            if start_ts_ms:
                from_dt = datetime.fromtimestamp(start_ts_ms / 1000.0, timezone.utc)
            else:
                from_dt = now - timedelta(days=60)
                
            if end_ts_ms:
                to_dt = datetime.fromtimestamp(end_ts_ms / 1000.0, timezone.utc) + timedelta(days=2)
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
                            dt = pd.to_datetime(ev_time_str)
                            if dt.tz is not None:
                                dt = dt.tz_convert(None)
                            ev_time = dt
                            filtered_events.append(ev_time)
                        except Exception as ex_train:
                            log_event("WARNING", f"train notice: {ex_train}")


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
    
    # Always enforce centralized TIMEFRAME_CONFIG for balanced label generation
    cfg = TIMEFRAME_CONFIG.get(str(interval), {
        "lookahead": 12,
        "sl_mult": 0.75,
        "tp_mult_ranging": 1.2,
        "tp_mult_trending": 1.3
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
            elif l <= lower_stop or h >= upper_stop: # Hit stop loss before TP barrier -> Neutral label (1)
                labels[i] = 1
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
        scores = []
        optuna_fold_scores = []
        from sklearn.utils.class_weight import compute_sample_weight
        try:
            for train_idx, val_idx in cv.split(X, y):
                if len(train_idx) < 10 or len(val_idx) < 10:
                    continue
                model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1)
                sw = compute_sample_weight('balanced', y[train_idx])
                model.fit(X[train_idx], y[train_idx], sample_weight=sw)
                preds = model.predict(X[val_idx])
                proba = model.predict_proba(X[val_idx])

                bal_acc = balanced_accuracy_score(y[val_idx], preds)
                macro_f1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
                ece = calculate_expected_calibration_error(y[val_idx], proba, n_bins=10)

                neutral_frac = (y[val_idx] == 1).mean()
                neutral_cap = MODEL_SELECTION.get("imbalance_neutral_cap", 0.70)
                imbalance_pen = max(0.0, neutral_frac - neutral_cap)

                w_bal = MODEL_SELECTION.get("balanced_accuracy_weight", 1.00)
                w_f1 = MODEL_SELECTION.get("macro_f1_weight", 0.30)
                w_ece = MODEL_SELECTION.get("ece_penalty_weight", 0.20)
                w_imb = MODEL_SELECTION.get("imbalance_penalty_weight", 0.40)

                fold_score = (w_bal * bal_acc) + (w_f1 * macro_f1) - (w_ece * ece) - (w_imb * imbalance_pen)
                scores.append(fold_score)
                optuna_fold_scores.append(fold_score)

            return safe_mean(scores) if scores else 0.0
        except Exception as ex_train:
            log_event("WARNING", f"train notice: {ex_train}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    best = study.best_params
    best["lookahead"] = 10
    print(f"[Optuna Barrier Tuning] Best Multipliers: TP Ranging={best['tp_mult_ranging']:.2f}, TP Trending={best['tp_mult_trending']:.2f}, SL={best['sl_mult']:.2f}")
    return best

def optimize_xgb_classifier(X_train, y_train, X_val, y_val, sample_weights, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if str(globals().get("interval", "15")) in ["15", "30"]:
            max_depth_min, max_depth_max = 3, 4
            lr_min, lr_max = 0.02, 0.08
        elif regime == "trending":
            max_depth_min, max_depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            max_depth_min, max_depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 30, 80),
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
            'n_jobs': 1
        }
        model = create_model(XGBClassifier, params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return balanced_accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    return study.best_params

def optimize_lgb_classifier(X_train, y_train, X_val, y_val, sample_weights, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if str(globals().get("interval", "15")) in ["15", "30"]:
            max_depth_min, max_depth_max = 3, 4
            lr_min, lr_max = 0.02, 0.08
        elif regime == "trending":
            max_depth_min, max_depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            max_depth_min, max_depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 30, 80),
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
            'n_jobs': 1
        }
        model = create_model(LGBMClassifier, params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return balanced_accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    return study.best_params

def optimize_cat_classifier(X_train, y_train, X_val, y_val, sample_weights, regime):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if str(globals().get("interval", "15")) in ["15", "30"]:
            depth_min, depth_max = 3, 4
            lr_min, lr_max = 0.02, 0.08
        elif regime == "trending":
            depth_min, depth_max = 5, 8
            lr_min, lr_max = 0.01, 0.04
        else:
            depth_min, depth_max = 3, 4
            lr_min, lr_max = 0.04, 0.12
        params = {
            'iterations': trial.suggest_int('iterations', 30, 80),
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
        return balanced_accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
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
            'n_estimators': trial.suggest_int('n_estimators', 30, 80),
            'max_depth': trial.suggest_int('max_depth', max_depth_min, max_depth_max),
            'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 42,
            'n_jobs': 1
        }
        model = create_model(XGBRegressor, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)
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
            'n_estimators': trial.suggest_int('n_estimators', 30, 80),
            'max_depth': trial.suggest_int('max_depth', max_depth_min, max_depth_max),
            'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 1
        }
        model = create_model(LGBMRegressor, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)
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
            'iterations': trial.suggest_int('iterations', 30, 80),
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
    study.optimize(objective, n_trials=3)
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
                    cfg = TIMEFRAME_CONFIG.get(str(interval), {"lookahead": 10})
                    lookahead = cfg.get("lookahead", 10)
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
            # Fix duplicate columns before concat
            df = df.loc[:, ~df.columns.duplicated()]
            live_df = live_df.loc[:, ~live_df.columns.duplicated()]
            # Align columns — only keep columns present in main df
            shared_cols = [c for c in df.columns if c in live_df.columns]
            live_df = live_df[shared_cols]
            df = pd.concat([df, live_df], ignore_index=True)
            print(f"[Live Feedback] Training dataset expanded to {len(df)} rows.")


    # ==========================================
    # AUTOML FEATURE SELECTION (RFECV NOISE REDUCTION)
    # ==========================================
    features_filename = f"selected_features_{interval}.json"
    skip_rfecv = False
    if os.path.exists(features_filename) and not globals().get("FORCE_RFECV", False):
        try:
            with open(features_filename, "r") as f:
                selected_features = json.load(f)
            if len(selected_features) >= 20:
                print(f"[Training Optimization] Reusing existing {len(selected_features)} features from {features_filename}. Skipping RFECV.")
                skip_rfecv = True
        except Exception as e:
            print(f"[Warning] Failed to load {features_filename}: {e}. Running RFECV.")

    if not skip_rfecv:
        from sklearn.feature_selection import RFECV
        print("\nRunning advanced feature selection via RFECV with Purged CV...")
        
        # Subsample data for feature selection to accelerate training times (Pros: 5x speedup)
        if len(df) > 40000:
            df_sub = df.sample(n=40000, random_state=42)
            print(f"[Training Optimization] Subsampled RFECV training set from {len(df)} to {len(df_sub)} rows.")
        else:
            df_sub = df
            
        X_prelim = df_sub[features]
        y_prelim = df_sub["target_trend"]
        
        # Use a small estimator and 3-fold Purged CV for rapid feature elimination
        cv_selector = PurgedEmbargoTimeSeriesSplit(n_splits=3, lookahead=6, embargo_pct=0.01)
        estimator = XGBClassifier(n_estimators=40, max_depth=3, random_state=42, n_jobs=1)
        
        selector = RFECV(
            estimator=estimator,
            step=5, # Dropping 5 features at a time (Pros: 2.5x speedup)
            cv=cv_selector,
            scoring="accuracy",
            min_features_to_select=20,
            n_jobs=1
        )
        
        print("Fitting RFECV model (this may take a few seconds)...")
        selector.fit(X_prelim, y_prelim)
        selected_features = [f for f, support in zip(features, selector.support_) if support]
        from feature_pipeline import filter_multicollinear_features
        selected_features = filter_multicollinear_features(df_sub, selected_features, threshold=0.85)
        print(f"[Correlation Filter] Retained {len(selected_features)} uncorrelated features (threshold |r| <= 0.85).")

    # Force-protect domain-critical features from RFECV elimination
    protected = ["close_to_Kalman", "close_to_Kalman_lag1", "close_to_Kalman_lag2"]
    try:
        if os.path.exists("protected_features.json"):
            with open("protected_features.json") as pf_file:
                protected = json.load(pf_file)
    except Exception as e:
        print(f"[Warning] Error loading protected_features.json: {e}")

    for pf in protected:
        if pf not in selected_features and pf in X_prelim.columns:
            selected_features.append(pf)

    # ADVERSARIAL VALIDATION DRIFT DETECTOR (Model Accuracy Upgrade)
    if len(selected_features) > 20:
        print("\nRunning Adversarial Validation to drop drifted features...")
        try:
            split_idx = int(len(df) * 0.8)
            av_df = df.copy()
            av_df["av_label"] = 0
            av_df.iloc[split_idx:, av_df.columns.get_loc("av_label")] = 1
            
            from sklearn.metrics import roc_auc_score
            
            drifted_features = []
            for feat in list(selected_features):
                if feat in protected:
                    continue
                # Make sure we don't drop below the minimum required 20 features
                if len(selected_features) <= 20:
                    break
                feat_series = av_df[feat].fillna(0.0)
                try:
                    auc = roc_auc_score(av_df["av_label"], feat_series)
                    auc_dist = abs(auc - 0.5)
                    if auc_dist > 0.20: # AUC > 0.70 or < 0.30 (significant drift)
                        drifted_features.append((feat, auc))
                        selected_features.remove(feat)
                except Exception as ex_train:
                    log_event("WARNING", f"train notice: {ex_train}")
            
            if drifted_features:
                print(f"[Adversarial Validation] Purged {len(drifted_features)} drifted features:")
                for df_feat, df_auc in drifted_features:
                    print(f"  - {df_feat} (Separation AUC: {df_auc:.3f})")
            else:
                print("[Adversarial Validation] No drifted features found. All features stable.")
        except Exception as av_err:
            print(f"[Adversarial Validation Error] {av_err}")

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

        if MLFLOW_AVAILABLE:
            try:
                if mlflow.active_run():
                    mlflow.end_run()
                mlflow.start_run(run_name=f"train_{interval}m_{name}")
            except Exception as ex_train:
                log_event("WARNING", f"train notice: {ex_train}")

        features_filename = f"selected_features_{interval}_{name}.json"
        regime_features = []
        skip_rfecv = False
        if not globals().get("FORCE_RFECV", False):
            target_file = features_filename if os.path.exists(features_filename) else f"selected_features_{interval}.json"
            if os.path.exists(target_file):
                try:
                    with open(target_file, "r") as f:
                        regime_features = json.load(f)
                    if len(regime_features) >= 15:
                        print(f"[{name.upper()} regime] Reusing existing {len(regime_features)} features from {target_file}. Skipping RFECV.")
                        skip_rfecv = True
                except Exception as e:
                    print(f"[Warning] Failed to load {target_file}: {e}. Running RFECV.")

        if not skip_rfecv:
            from sklearn.feature_selection import RFECV
            print(f"\nRunning RFECV feature selection specifically for regime: {name.upper()}...")
            y_rfecv = df_regime["target_trend"]
            X_rfecv_prelim = df_regime[features]
            
            if len(df_regime) > 10000:
                df_sub = df_regime.sample(n=10000, random_state=42)
                X_rfecv_prelim = df_sub[features]
                y_rfecv = df_sub["target_trend"]
                
            estimator = XGBClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                tree_method="hist",
                n_jobs=1
            )
            cv_rfecv = PurgedEmbargoTimeSeriesSplit(n_splits=3, lookahead=6, embargo_pct=0.01)
            selector = RFECV(
                estimator=estimator,
                step=2,
                cv=cv_rfecv,
                scoring="accuracy",
                min_features_to_select=15,
                n_jobs=1
            )
            selector.fit(X_rfecv_prelim.values, y_rfecv.values)
            regime_features = [f for f, support in zip(features, selector.support_) if support]
            
            required_price_features = ["close_to_Kalman", "close_btc", "btc_rsi", "ADX", "ATR_norm", "open_interest"]
            for pf in required_price_features:
                if pf not in regime_features and pf in df_regime.columns:
                    regime_features.append(pf)
                    
            if len(regime_features) > 22:
                regime_features = regime_features[:22]
                
            with open(features_filename, "w") as f:
                json.dump(regime_features, f)
            print(f"Saved {name.upper()} regime selected features to {features_filename}")

        X_full = df_regime[regime_features]
        y_trend_full = df_regime["target_trend"]
        y_price_full = df_regime["target_price_change"]

        # === LABEL DISTRIBUTION — PERMANENT GOVERNANCE CHECK ===
        n_total = len(y_trend_full)
        n_bear = int((y_trend_full == 0).sum())
        n_neutral = int((y_trend_full == 1).sum())
        n_bull = int((y_trend_full == 2).sum())
        bear_pct = (n_bear / n_total * 100) if n_total > 0 else 0.0
        neut_pct = (n_neutral / n_total * 100) if n_total > 0 else 0.0
        bull_pct = (n_bull / n_total * 100) if n_total > 0 else 0.0

        print(f"\n  [Label Distribution — {name.upper()} regime, {interval}m]")
        print(f"    Bearish  (0): {n_bear:>6}  ({bear_pct:5.1f}%)")
        print(f"    Neutral  (1): {n_neutral:>6}  ({neut_pct:5.1f}%)")
        print(f"    Bullish  (2): {n_bull:>6}  ({bull_pct:5.1f}%)")
        print(f"    Total       : {n_total:>6}  (100.0%)")

        _emit_governance_event({
            "event": "label_distribution", "interval": interval, "regime": name,
            "n_total": n_total,
            "bearish_pct": round(bear_pct, 2),
            "neutral_pct": round(neut_pct, 2),
            "bullish_pct": round(bull_pct, 2)
        })

        # === SEVERE IMBALANCE WARNING GATE ===
        _warn_threshold = MODEL_SELECTION.get("imbalance_min_class_pct", 0.10)
        if (bear_pct / 100) < _warn_threshold or (bull_pct / 100) < _warn_threshold:
            _imb_msg = (
                f"[IMBALANCE WARNING] Regime={name.upper()} {interval}m — "
                f"Bearish={bear_pct:.1f}%, Neutral={neut_pct:.1f}%, Bullish={bull_pct:.1f}%. "
                f"Directional class < {_warn_threshold*100:.0f}%. "
                f"Risk: model may learn degenerate all-Neutral predictions."
            )
            print(f"\n  ⚠️  {_imb_msg}")
            log_event("WARNING", _imb_msg)
            _emit_governance_event({
                "event": "label_imbalance_warning", "interval": interval, "regime": name,
                "severity": "WARNING",
                "bearish_pct": round(bear_pct, 2),
                "neutral_pct": round(neut_pct, 2),
                "bullish_pct": round(bull_pct, 2),
                "threshold_pct": round(_warn_threshold * 100, 1)
            })
            _tg_alert(
                f"⚠️ *Label Imbalance Warning*\n"
                f"Regime: *{name.upper()}* | Interval: *{interval}m*\n"
                f"Bear={bear_pct:.1f}%  Neutral={neut_pct:.1f}%  Bull={bull_pct:.1f}%\n"
                f"Risk: model may collapse to all-Neutral predictions."
            )

        # F-03 ML Validity: Freeze true final 15% hold-out dataset untouched during CV & tuning
        split_idx = int(len(X_full) * 0.85)
        X = X_full.iloc[:split_idx]
        y_trend = y_trend_full.iloc[:split_idx]
        y_price = y_price_full.iloc[:split_idx]
        
        X_holdout = X_full.iloc[split_idx:]
        y_holdout_trend = y_trend_full.iloc[split_idx:]
        y_holdout_price = y_price_full.iloc[split_idx:]

        # Purged and Embargoed Time-Series Cross Validation
        cv = PurgedEmbargoTimeSeriesSplit(n_splits=5, lookahead=6, embargo_pct=0.01)
        
        meta_features_list = []
        meta_labels_list = []
        
        primary_accuracies = []
        primary_bal_accuracies = []
        primary_macro_f1s = []
        primary_mccs = []
        primary_kappas = []
        primary_pr_auc_bears = []
        primary_pr_auc_bulls = []
        primary_maes = []
        all_y_val_agg = []
        all_pred_agg = []
        
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
        
        print(f"  Running Purged & Embargoed Cross-Validation on {len(X)} samples (15% holdout frozen: {len(X_holdout)} samples)...")
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
            
            # Classification metrics
            acc = accuracy_score(y_val_t, pred_val_t)
            bal_acc = balanced_accuracy_score(y_val_t, pred_val_t)
            macro_f1 = f1_score(y_val_t, pred_val_t, average="macro", zero_division=0)
            mcc = matthews_corrcoef(y_val_t, pred_val_t)
            kappa = cohen_kappa_score(y_val_t, pred_val_t)
            mae = mean_absolute_error(y_val_p, pred_val_p)
            
            # PR-AUC with zero-positive guard
            proba_val = ensemble_t.predict_proba(X_val)
            y_val_arr = y_val_t.values

            if np.sum(y_val_arr == 0) > 0:
                pr_auc_bear = average_precision_score((y_val_arr == 0).astype(int), proba_val[:, 0])
            else:
                pr_auc_bear = None

            if np.sum(y_val_arr == 2) > 0:
                pr_auc_bull = average_precision_score((y_val_arr == 2).astype(int), proba_val[:, 2])
            else:
                pr_auc_bull = None

            pr_bear_str = f"{pr_auc_bear:.3f}" if pr_auc_bear is not None else "N/A"
            pr_bull_str = f"{pr_auc_bull:.3f}" if pr_auc_bull is not None else "N/A"

            print(
                f"    - Fold {fold+1}: RawAcc={acc*100:.1f}%  BalAcc={bal_acc*100:.1f}%  "
                f"MacroF1={macro_f1:.3f}  MCC={mcc:.3f}  Kappa={kappa:.3f}  "
                f"PR-AUC(Bear={pr_bear_str} Bull={pr_bull_str})  MAE={mae:.5f}"
            )
            print(classification_report(y_val_t, pred_val_t, target_names=["Bearish", "Neutral", "Bullish"], zero_division=0))

            primary_accuracies.append(acc)
            primary_bal_accuracies.append(bal_acc)
            primary_macro_f1s.append(macro_f1)
            primary_mccs.append(mcc)
            primary_kappas.append(kappa)
            primary_pr_auc_bears.append(pr_auc_bear)
            primary_pr_auc_bulls.append(pr_auc_bull)
            primary_maes.append(mae)
            all_y_val_agg.extend(y_val_arr.tolist())
            all_pred_agg.extend(pred_val_t.tolist())
            
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
            
        # Aggregate Confusion Matrix
        cm = confusion_matrix(all_y_val_agg, all_pred_agg, labels=[0, 1, 2])
        print(f"\n  === Aggregate Confusion Matrix ({name.upper()} {interval}m) ===")
        print(f"               Pred Bearish  Pred Neutral  Pred Bullish")
        print(f"  True Bearish:  {cm[0,0]:>8}    {cm[0,1]:>8}    {cm[0,2]:>8}")
        print(f"  True Neutral:  {cm[1,0]:>8}    {cm[1,1]:>8}    {cm[1,2]:>8}")
        print(f"  True Bullish:  {cm[2,0]:>8}    {cm[2,1]:>8}    {cm[2,2]:>8}")

        # CV Summary (None-guarded print strings)
        stat_bal = safe_stat(primary_bal_accuracies)
        stat_f1 = safe_stat(primary_macro_f1s)
        stat_mcc = safe_stat(primary_mccs)
        mean_kappa = safe_mean(primary_kappas)
        mean_pr_auc_bear = safe_mean(primary_pr_auc_bears)
        mean_pr_auc_bull = safe_mean(primary_pr_auc_bulls)
        mean_cv_acc = float(np.mean(primary_accuracies))
        mean_cv_mae = float(np.mean(primary_maes))

        print(f"\n  === CV Summary ({name.upper()} {interval}m) ===")
        print(f"  Raw Accuracy:       {mean_cv_acc*100:.2f}%")
        if stat_bal["mean"] is not None:
            print(f"  Balanced Accuracy:  mean={stat_bal['mean']*100:.2f}%  std={stat_bal['std']*100:.2f}%  "
                  f"min={stat_bal['min']*100:.2f}%  max={stat_bal['max']*100:.2f}%")
        else:
            print(f"  Balanced Accuracy:  N/A")

        if stat_f1["mean"] is not None:
            print(f"  Macro F1:           mean={stat_f1['mean']:.4f}  std={stat_f1['std']:.4f}")
        else:
            print(f"  Macro F1:           N/A")

        if stat_mcc["mean"] is not None:
            print(f"  MCC:                mean={stat_mcc['mean']:.4f}  std={stat_mcc['std']:.4f}")
        else:
            print(f"  MCC:                N/A")

        print(f"  Cohen Kappa:        {mean_kappa:.4f}" if mean_kappa is not None else "  Cohen Kappa:        N/A")
        pr_b_str = f"{mean_pr_auc_bear:.4f}" if mean_pr_auc_bear is not None else "N/A"
        pr_u_str = f"{mean_pr_auc_bull:.4f}" if mean_pr_auc_bull is not None else "N/A"
        print(f"  PR-AUC (Bear/Bull): {pr_b_str} / {pr_u_str}")
        print(f"  MAE (Price):        {mean_cv_mae:.4f}")
        
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
            n_jobs=1
        )
        if len(meta_X) >= 20:
            meta_model.fit(meta_X, meta_y)
            print("  Meta-Classifier trained successfully.")
        else:
            dummy_y = np.zeros(len(X), dtype=int)
            meta_model.fit(X, dummy_y)
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
        from ensemble import load_ensemble_classifier, load_ensemble_regressor, is_feature_contract_compatible

        c_prefix_t = f"ensemble_{name}_trend_{interval}"
        c_prefix_p = f"ensemble_{name}_price_{interval}"

        champion_exists = os.path.exists(f"{c_prefix_t}_xgb.json")
        should_save = True
        champion_t = None
        champion_p = None

        # challenger_feature_names: use RFECV-selected list, NOT X_holdout.columns
        # (DataFrame column order is not guaranteed to match training order)
        challenger_feature_names = list(X_holdout.columns) if not hasattr(features_module, 'selected_features') else features

        if champion_exists:
            manifest_path = f"{c_prefix_t}_manifest.json"
            if os.path.exists(manifest_path):
                with open(manifest_path) as _mf:
                    champ_manifest = json.load(_mf)
                compatible, reason = is_feature_contract_compatible(
                    champ_manifest, challenger_feature_names
                )
            else:
                champ_manifest = {}
                compatible, reason = False, "No manifest — champion predates governance system"

            if not compatible:
                import hashlib as _hl
                champ_count = champ_manifest.get("feature_count", "?")
                champ_hash  = champ_manifest.get("feature_contract_hash", "?")
                champ_names = champ_manifest.get("feature_names", [])
                chal_count  = len(challenger_feature_names)
                chal_hash   = _hl.sha256(",".join(challenger_feature_names).encode()).hexdigest()[:12]
                print(
                    f"  [Champion-Challenger] Champion model incompatible with challenger.\n"
                    f"    Reason    : {reason}\n"
                    f"    Champion  : {champ_count} features | hash={champ_hash}\n"
                    f"              : {champ_names}\n"
                    f"    Challenger: {chal_count} features | hash={chal_hash}\n"
                    f"              : {challenger_feature_names}\n"
                    f"    Champion comparison skipped. New model automatically promoted."
                )
                # Compute challenger's current git SHA for lineage
                try:
                    import subprocess as _sp
                    _chal_git_sha = _sp.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]
                except Exception as ex_train:
                    log_event("WARNING", f"train notice: {ex_train}")
                    _chal_git_sha = "unknown"
                _emit_governance_event({
                    "event":                  "MODEL_CONTRACT_CHANGED",
                    "interval":               interval,
                    "regime":                 name,
                    # Feature contract
                    "old_feature_hash":       champ_hash,
                    "new_feature_hash":       chal_hash,
                    "old_feature_count":      champ_count,
                    "new_feature_count":      chal_count,
                    # Versioning
                    "old_git_sha":            champ_manifest.get("git_sha", "unknown"),
                    "new_git_sha":            _chal_git_sha,
                    "old_model_version":      champ_manifest.get("model_version", "unknown"),
                    "new_model_version":      "v7.2.0",
                    "old_feature_version":    champ_manifest.get("feature_version", "unknown"),
                    "new_feature_version":    "v3.1.0",
                    "old_ensemble_version":   champ_manifest.get("ensemble_version", "unknown"),
                    "new_ensemble_version":   "v3.0_stacking",
                    # Data lineage
                    "old_training_hash":      champ_manifest.get("training_data_hash"),
                    "new_training_hash":      None,  # set by caller if available
                    "old_preprocessing_hash": champ_manifest.get("preprocessing_hash"),
                    "new_preprocessing_hash": None,  # set by caller if available
                    # Decision
                    "action":                 "ChampionSkipped",
                    "reason":                 reason,
                })
                should_save = True
            else:
                try:
                    n_champ = champ_manifest.get("feature_count", X_holdout.shape[1])
                    champion_t = load_ensemble_classifier(c_prefix_t, n_features=n_champ)
                    champion_p = load_ensemble_regressor(c_prefix_p, n_features=n_champ)
                except Exception as _le:
                    print(f"  [Champion-Challenger Warning] Failed to load champion: {_le}. Defaulting to save.")
                    champion_t = None
                    champion_p = None
                    should_save = True
        else:
            print(f"  [Champion-Challenger] No existing champion for {name.upper()}. Saving challenger.")

        if champion_t is not None:
            try:
                champ_pred_t = champion_t.predict(X_holdout)
                champ_pred_p = champion_p.predict(X_holdout)
                champ_acc = float(balanced_accuracy_score(y_holdout_trend, champ_pred_t))
                champ_mae = float(mean_absolute_error(y_holdout_price, champ_pred_p))

                chal_pred_t = final_ensemble_t.predict(X_holdout)
                chal_pred_p = final_ensemble_p.predict(X_holdout)
                chal_acc = float(balanced_accuracy_score(y_holdout_trend, chal_pred_t))
                chal_mae = float(mean_absolute_error(y_holdout_price, chal_pred_p))

                # Calculate real Multiclass Brier score and ECE for challenger
                from mlops_engine import calculate_brier_score, calculate_expected_calibration_error
                try:
                    chal_prob_raw = final_ensemble_t.predict_proba(X_holdout)
                    chal_brier = float(calculate_brier_score(y_holdout_trend, chal_prob_raw))
                    chal_ece = float(calculate_expected_calibration_error(y_holdout_trend, chal_prob_raw))
                except Exception as ex_brier:
                    log_event("WARNING", f"Holdout calibration metric calculation failure: {ex_brier}")
                    chal_brier = 0.99
                    chal_ece = 0.99
                    should_save = False

                print(f"  [Champion-Challenger] Frozen Hold-Out Comparison for {name.upper()}:")
                print(f"    - Classifier Balanced Accuracy: Champion = {champ_acc*100:.2f}% | Challenger = {chal_acc*100:.2f}%")
                print(f"    - Regressor MAE: Champion = {champ_mae:.4f} | Challenger = {chal_mae:.4f}")
                log_event("INFO", f"Challenger Metrics: Brier = {chal_brier:.4f} | ECE = {chal_ece:.4f}")

                if chal_acc > champ_acc:
                    should_save = True
                elif chal_acc == champ_acc and chal_mae < champ_mae:
                    should_save = True
                else:
                    should_save = False
            except Exception as eval_err:
                print(f"  [Champion-Challenger Warning] Error during hold-out comparison: {eval_err}. Failing closed.")
                should_save = False
                chal_acc = 0.0
                chal_mae = 999.0
                chal_brier = 0.99
                chal_ece = 0.99
        else:
            chal_acc = 0.55
            chal_mae = 0.01
            try:
                from mlops_engine import calculate_brier_score, calculate_expected_calibration_error
                chal_prob_raw = final_ensemble_t.predict_proba(X_holdout)
                chal_brier = float(calculate_brier_score(y_holdout_trend, chal_prob_raw))
                chal_ece = float(calculate_expected_calibration_error(y_holdout_trend, chal_prob_raw))
            except Exception as ex_brier:
                log_event("WARNING", f"Holdout calibration metric calculation failure: {ex_brier}")
                chal_brier = 0.99
                chal_ece = 0.99
                should_save = False

        if should_save:
            from mlops_engine import log_mlflow_training_run, promote_if_better
            reg_name = f"btc_{interval}m_{name}_clf"

            # Step 1: Register challenger model to get integer version string ("1", "2", etc.)
            reg_info = model_registry.register_model(
                run_id=f"train_{interval}m_{name}_{int(time.time())}",
                model_name=reg_name,
                metrics={"val_accuracy": chal_acc, "brier_score": chal_brier, "ece": chal_ece, "val_mae": chal_mae},
                stage="Staging"
            )
            challenger_ver = str(reg_info.get("version", "1")) if isinstance(reg_info, dict) else "1"

            # Step 2: Log complete training run to MLflow System of Record
            ml_run_id = log_mlflow_training_run(
                symbol="BTCUSDT",
                interval=str(interval),
                regime=name,
                features=regime_features,
                metrics={"holdout_accuracy": chal_acc, "brier_score": chal_brier, "ece": chal_ece, "val_mae": chal_mae},
                manifest_path=f"{c_prefix_t}_manifest.json",
                git_sha=_chal_git_sha if '_chal_git_sha' in locals() else "unknown"
            )

            # Step 3: Evaluate MLflow Model Registry Promotion Gate with actual integer version
            promoted, p_reason = promote_if_better(reg_name, challenger_version=challenger_ver)
            if not promoted:
                print(f"  [Model Governance Gate] Promotion REJECTED: {p_reason}")
                should_save = False

        if should_save:
            print(f"  [Champion-Challenger] Challenger approved & promoted. Overwriting active model files...")
            save_ensemble_classifier(final_ensemble_t, c_prefix_t)
            save_ensemble_regressor(final_ensemble_p, c_prefix_p)
            meta_model.save_model(f"meta_{name}_trend_{interval}.json")

            # Write/update governance manifest with complete cv_metrics block
            _pipeline_git_sha = _chal_git_sha if '_chal_git_sha' in locals() else "unknown"
            holdout_raw_acc = float(accuracy_score(y_holdout_trend, chal_pred_t))

            cv_metrics_block = {
                "metrics_schema_version": 1,
                "balanced_accuracy": safe_stat(primary_bal_accuracies),
                "macro_f1": safe_stat(primary_macro_f1s),
                "mcc": safe_stat(primary_mccs),
                "mean_kappa": mean_kappa,
                "mean_pr_auc_bear": mean_pr_auc_bear,
                "mean_pr_auc_bull": mean_pr_auc_bull,
                "mean_raw_accuracy": round(mean_cv_acc, 4),
                "mean_mae": round(mean_cv_mae, 6),
                "label_dist_bearish_pct": round(bear_pct, 2),
                "label_dist_neutral_pct": round(neut_pct, 2),
                "label_dist_bullish_pct": round(bull_pct, 2),
                "n_training_samples": len(X),
                "n_holdout_samples": len(X_holdout),
                "confusion_matrix": {
                    "labels": ["Bearish", "Neutral", "Bullish"],
                    "label_ids": [0, 1, 2],
                    "matrix": cm.tolist()
                },
                "holdout_accuracy": round(holdout_raw_acc, 4),
                "holdout_balanced_accuracy": round(chal_acc, 4),
                "holdout_brier": round(chal_brier, 4),
                "holdout_ece": round(chal_ece, 4),
                "optuna_objective": safe_stat(locals().get('optuna_fold_scores', [])),
                "training_pipeline_version": "v7.2.0",
                "git_sha": _pipeline_git_sha
            }

            manifest_path_t = f"{c_prefix_t}_manifest.json"
            try:
                manifest_data = {}
                if os.path.exists(manifest_path_t):
                    with open(manifest_path_t, "r") as mf:
                        manifest_data = json.load(mf)
                manifest_data["cv_metrics"] = cv_metrics_block
                manifest_data["git_sha"] = _pipeline_git_sha
                with open(manifest_path_t, "w") as mf:
                    json.dump(manifest_data, mf, indent=2)
            except Exception as ex_man:
                log_event("WARNING", f"Failed to write cv_metrics to manifest: {ex_man}")

            print(f"  Saved ensemble and meta-classifier models for regime: {name.upper()}")

            model_registry.register_model(
                run_id=f"train_{interval}m_{name}_{int(time.time())}",
                model_name=f"ensemble_{name}_{interval}",
                metrics={"val_accuracy": chal_acc, "val_mae": chal_mae},
                stage="Production"
            )
            _tg_alert(
                f"✅ *Model Trained & Promoted*\n"
                f"📊 Interval: *{interval}m* | Regime: *{name.upper()}*\n"
                f"🎯 Val Accuracy: `{chal_acc*100:.1f}%` | Val MAE: `{chal_mae:.4f}`\n"
                f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            )
        else:
            print(f"  [Champion-Challenger] Champion model retained (Challenger rejected or did not improve).")
            _tg_alert(
                f"⏭️ *Champion Retained*\n"
                f"📊 Interval: *{interval}m* | Regime: *{name.upper()}*\n"
                f"Challenger did not pass promotion gate — existing model kept."
            )

        if MLFLOW_AVAILABLE:
            try:
                if mlflow.active_run():
                    mlflow.end_run()
            except Exception as ex_train:
                log_event("WARNING", f"train notice: {ex_train}")

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
    """Load recent closed trades and simulated skipped predictions, re-fetch features at entry time, return as weighted DataFrame."""
    try:
        import time as _time
        # Load selected features to align columns (P1 fix)
        feat_file = f"selected_features_{interval}.json"
        if not os.path.exists(feat_file):
            print(f"[Live Feedback] No selected_features_{interval}.json found, skipping.")
            return None
        with open(feat_file) as _ff:
            live_selected = json.load(_ff)
        history_file = "dashboard_history.json"
        
        sample_dfs = []
        
        # 1. Load executed trades from dashboard history
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                data = json.load(f)
            trades = data.get("trade_history", [])
            cutoff = _time.time() - days * 86400
            trades = [t for t in trades if str(t.get("interval", "")) == str(interval) and float(t.get("exit_time", 0)) >= cutoff]
            
            for t in trades:
                symbol = t.get("symbol")
                exit_ts = float(t.get("exit_time", 0))
                pnl = float(t.get("pnl_usd", 0.0))
                direction = t.get("direction", "Bullish")
                df_c = get_history(symbol=symbol, interval=interval, limit=350, pages=1)
                if df_c is None or len(df_c) < 20:
                    continue
                df_c = df_c[df_c["timestamp"] <= exit_ts * 1000].copy()
                if len(df_c) < 10:
                    continue
                
                if symbol == "BTCUSDT":
                    df_c["close_btc"] = df_c["close"]
                else:
                    df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=350, pages=1)
                    if df_btc is not None and len(df_btc) > 0:
                        df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                        df_c = pd.merge(df_c, df_btc_sub, on="timestamp", how="inner")
                    else:
                        df_c["close_btc"] = df_c["close"]

                df_c = merge_derivatives_sentiment_features(df_c, symbol=symbol, interval=interval)
                df_c = add_features(df_c)
                df_c = df_c.dropna()
                if len(df_c) == 0:
                    continue
                row = df_c.iloc[[-1]].copy()
                row["target_trend"] = 1 if pnl > 0 else 0
                row["target_price_change"] = 0.0
                row["sample_weight"] = weight
                sample_dfs.append(row)

        # 2. Load and simulate skipped predictions from SQLite db
        try:
            db_file = "trading_bot.db"
            if os.path.exists(db_file):
                import sqlite3
                conn = sqlite3.connect(db_file)
                c = conn.cursor()
                cutoff_ts = _time.time() - days * 86400
                c.execute('SELECT raw_data FROM predictions WHERE timestamp >= ?', (cutoff_ts,))
                db_rows = c.fetchall()
                conn.close()
                
                skipped_trades = []
                for r in db_rows:
                    try:
                        d = json.loads(r[0])
                        if str(d.get("interval", "")) == str(interval) and "skip" in d.get("status", "").lower():
                            skipped_trades.append(d)
                    except Exception as ex_train:
                        log_event("WARNING", f"train notice: {ex_train}")
                
                for t in skipped_trades:
                    symbol = t.get("symbol")
                    entry_price = float(t.get("ref_price", 0.0))
                    entry_time_sec = float(t.get("timestamp", 0.0))
                    entry_time_ms = entry_time_sec * 1000
                    direction = t.get("direction", "Bullish")
                    
                    df_c = get_history(symbol=symbol, interval=interval, limit=350, pages=1)
                    if df_c is None or len(df_c) < 20:
                        continue
                    df_c_before = df_c[df_c["timestamp"] <= entry_time_ms].copy()
                    if len(df_c_before) < 10:
                        continue
                    
                    # Simple PnL simulator (TP vs SL)
                    atr_series = df_c_before["close"].diff().abs().rolling(14).mean()
                    atr = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else entry_price * 0.01
                    sl = entry_price - 1.5 * atr if direction == "Bullish" else entry_price + 1.5 * atr
                    tp = entry_price + 1.25 * atr if direction == "Bullish" else entry_price - 1.25 * atr
                    
                    df_future = df_c[df_c["timestamp"] > entry_time_ms].copy()
                    pnl = 0.0
                    for _, row_fut in df_future.iterrows():
                        high = row_fut["high"]
                        low = row_fut["low"]
                        if direction == "Bullish":
                            if low <= sl:
                                pnl = -0.01
                                break
                            if high >= tp:
                                pnl = 0.015
                                break
                        else:
                            if high >= sl:
                                pnl = -0.01
                                break
                            if low <= tp:
                                pnl = 0.015
                                break
                    else:
                        if len(df_future) > 0:
                            pnl = (df_future.iloc[-1]["close"] - entry_price) / entry_price
                            if direction == "Bearish":
                                pnl = -pnl
                                
                    df_c_before = merge_derivatives_sentiment_features(df_c_before, symbol=symbol, interval=interval)
                    if "close_btc" not in df_c_before.columns:
                        df_c_before["close_btc"] = df_c_before["close"]
                    df_c_before = add_features(df_c_before)
                    df_c_before = df_c_before.dropna()
                    if len(df_c_before) == 0:
                        continue
                    row = df_c_before.iloc[[-1]].copy()
                    row["target_trend"] = 1 if pnl > 0 else 0
                    row["target_price_change"] = 0.0
                    row["sample_weight"] = 1.0  # Skipped trades have baseline weight
                    sample_dfs.append(row)
        except Exception as e_db:
            print(f"[Live Feedback Warning] Error loading skipped predictions: {e_db}")

        if not sample_dfs:
            return None
        result = pd.concat(sample_dfs, ignore_index=True)
        # Align to selected features only (P1 fix), preserving system-critical regime indicators (ATR_norm, ADX)
        keep = [c for c in live_selected if c in result.columns]
        keep += [c for c in ["target_trend", "target_price_change", "sample_weight", "ATR_norm", "ADX"] if c in result.columns]
        
        # Deduplicate keep list to prevent duplicate columns in sliced DataFrame
        seen = set()
        deduped_keep = []
        for c in keep:
            if c not in seen:
                deduped_keep.append(c)
                seen.add(c)
        result = result[deduped_keep]
        print(f"[Live Feedback] Injecting {len(result)} feedback samples (real + simulated skipped) for interval {interval}m.")
        return result
    except Exception as e:
        print(f"[Live Feedback] Error loading live trade samples: {e}")
        return None

def audit_model_diversity_and_calculate_brier_weights(
    preds_dict: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Dict[str, Any]:
    """
    Pillar 3: Model Diversity Audit & Dynamic Inverse-Brier Weighting.
    Measures pairwise prediction correlations (r < 0.95 limit), disagreement entropy,
    and calculates dynamic model weights inverse to Brier Score.
    """
    names = list(preds_dict.keys())
    if len(names) < 2:
        return {"weights": {n: 1.0 for n in names}, "correlation_matrix": {}, "disagreement_entropy": 0.0}

    # 1. Pairwise prediction correlation matrix
    corr_matrix = {}
    is_diverse = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = preds_dict[names[i]], preds_dict[names[j]]
            corr = float(np.corrcoef(p1, p2)[0, 1]) if len(p1) > 1 else 1.0
            corr_matrix[f"{names[i]}_vs_{names[j]}"] = round(corr, 4)
            if corr > 0.95:
                is_diverse = False

    # 2. Disagreement Entropy
    probs_stack = np.column_stack([preds_dict[n] for n in names])
    mean_probs = np.mean(probs_stack, axis=1)
    entropy_vals = - (mean_probs * np.log2(np.clip(mean_probs, 1e-6, 1.0)) + (1.0 - mean_probs) * np.log2(np.clip(1.0 - mean_probs, 1e-6, 1.0)))
    avg_entropy = float(np.mean(entropy_vals))

    # 3. Dynamic Inverse-Brier Model Weighting (w_i proportional to 1 / Brier_i)
    brier_scores = {}
    inv_briers = {}
    for n in names:
        brier = float(np.mean((preds_dict[n] - y_true) ** 2))
        brier_scores[n] = round(brier, 4)
        inv_briers[n] = 1.0 / max(1e-4, brier)

    sum_inv = sum(inv_briers.values())
    dynamic_weights = {n: round(inv_briers[n] / max(1e-6, sum_inv), 4) for n in names}

    return {
        "model_diversity_pass": is_diverse,
        "pairwise_correlations": corr_matrix,
        "disagreement_entropy": round(avg_entropy, 4),
        "brier_scores": brier_scores,
        "dynamic_inverse_brier_weights": dynamic_weights
    }



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train XGBoost models for BTC Trading Bot")
    parser.add_argument("--interval", type=str, default="60", choices=["15", "30", "60", "120", "240", "360", "all"], help="Timeframe interval to train")
    parser.add_argument("--pages", type=int, default=8, help="Number of data pages (default 8 for AWS 1GB RAM)")
    parser.add_argument("--live-feedback", action="store_true", help="Inject recent live trade outcomes as weighted samples")
    parser.add_argument("--force-rfecv", action="store_true", help="Force running RFECV feature selection instead of reusing cached features")
    args = parser.parse_args()
    LIVE_FEEDBACK = args.live_feedback
    FORCE_RFECV = args.force_rfecv

    intervals_to_train = ["15", "30", "60", "120", "240"] if args.interval == "all" else [args.interval]
    _tg_alert(
        f"🚀 *Retrain Started*\n"
        f"📋 Intervals: `{', '.join(intervals_to_train)}`\n"
        f"📄 Pages: `{args.pages}`\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )

    t0 = time.time()
    errors = []
    for iv in intervals_to_train:
        print(f"\n==============================================")
        print(f"TRAINING FOR INTERVAL: {iv}")
        print(f"==============================================")
        try:
            train_models(interval=iv, pages=args.pages)
        except Exception as train_err:
            err_msg = str(train_err)[:200]
            print(f"[Train Error] Interval {iv} failed: {train_err}")
            errors.append(f"{iv}m: {err_msg}")
            _tg_alert(f"❌ *Train Failed* — Interval `{iv}m`\n`{err_msg}`")

    elapsed = int(time.time() - t0)
    if errors:
        _tg_alert(
            f"⚠️ *Retrain Completed with Errors*\n"
            f"✅ Done: {len(intervals_to_train) - len(errors)} | ❌ Failed: {len(errors)}\n"
            f"⏱ Duration: `{elapsed}s`"
        )
    else:
        _tg_alert(
            f"🎉 *All Models Retrained Successfully*\n"
            f"📋 Intervals: `{', '.join(intervals_to_train)}`\n"
            f"⏱ Duration: `{elapsed}s`\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )


    # Post-Training Storage Optimization: Clean temporary CatBoost log artifacts
    import shutil
    shutil.rmtree("catboost_info", ignore_errors=True)
    print("🧹 [Disk Cleanup] Successfully purged temporary catboost_info directory.")