import os
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
from datetime import datetime, timedelta

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
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "NEARUSDT", "LINKUSDT", "LTCUSDT", "DOGEUSDT", "SUIUSDT", "APTUSDT", "DOTUSDT"]

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

def add_news_proximity_feature(df):
    if df.empty:
        df["hours_to_news"] = 72.0
        return df
        
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    events = fetch_economic_calendar_cached(start_ts, end_ts)
    
    if not events:
        df["hours_to_news"] = 72.0
        return df
        
    import bisect
    df_dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    hours_to_news_list = []
    events_utc = [pd.Timestamp(ev).tz_localize("UTC") for ev in events]
    
    for current_time in df_dt:
        idx = bisect.bisect_right(events_utc, current_time)
        if idx < len(events_utc):
            next_event = events_utc[idx]
            diff_hours = (next_event - current_time).total_seconds() / 3600.0
            hours_to_news_list.append(min(72.0, max(0.0, diff_hours)))
        else:
            hours_to_news_list.append(72.0)
            
    df["hours_to_news"] = hours_to_news_list
    return df

try:
    from numba import jit
    @jit(nopython=True, cache=True)
    def _kalman_loop(prices, state_estimate, error_covariance, process_variance, measurement_variance):
        n = len(prices)
        for i in range(1, n):
            pred_state = state_estimate[i-1]
            pred_error = error_covariance[i-1] + process_variance
            kalman_gain = pred_error / (pred_error + measurement_variance)
            state_estimate[i] = pred_state + kalman_gain * (prices[i] - pred_state)
            error_covariance[i] = (1.0 - kalman_gain) * pred_error
        return state_estimate
except ImportError:
    def _kalman_loop(prices, state_estimate, error_covariance, process_variance, measurement_variance):
        n = len(prices)
        for i in range(1, n):
            pred_state = state_estimate[i-1]
            pred_error = error_covariance[i-1] + process_variance
            kalman_gain = pred_error / (pred_error + measurement_variance)
            state_estimate[i] = pred_state + kalman_gain * (prices[i] - pred_state)
            error_covariance[i] = (1.0 - kalman_gain) * pred_error
        return state_estimate

def calculate_kalman_feature(prices):
    n = len(prices)
    if n == 0:
        return np.zeros(0)
    state_estimate = np.zeros(n)
    error_covariance = np.zeros(n)
    state_estimate[0] = prices[0]
    error_covariance[0] = 1.0
    
    process_variance = 1e-4
    measurement_variance = 1e-2
    
    state_estimate = _kalman_loop(prices, state_estimate, error_covariance, process_variance, measurement_variance)
    return (prices / (state_estimate + 1e-8)) - 1.0

def add_features(df):
    df = df.copy()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
    
    macd = MACD(df["close"])
    df["MACD_diff"] = macd.macd_diff() / (df["close"] + 1e-8)
    
    df["EMA_9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["EMA_21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    df["EMA_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["EMA_200"] = EMAIndicator(df["close"], window=200).ema_indicator()
    
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()
    
    df["MFI"] = MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14).money_flow_index()
    
    # ATR Volatility normalized
    atr_ind = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["ATR_norm"] = atr_ind.average_true_range() / (df["close"] + 1e-8)
    
    # Normalized scale-invariant features
    df["close_to_EMA9"] = df["close"] / df["EMA_9"] - 1.0
    df["close_to_EMA21"] = df["close"] / df["EMA_21"] - 1.0
    df["close_to_EMA50"] = df["close"] / df["EMA_50"] - 1.0
    df["close_to_EMA200"] = df["close"] / df["EMA_200"] - 1.0
    df["EMA9_to_EMA21"] = df["EMA_9"] / df["EMA_21"] - 1.0
    df["BB_pct"] = (df["close"] - df["BB_low"]) / (df["BB_high"] - df["BB_low"] + 1e-8)
    df["BB_width"] = (df["BB_high"] - df["BB_low"]) / df["BB_mid"]
    
    df["return_5m"] = df["close"].pct_change(1)
    df["volatility_10m"] = df["return_5m"].rolling(10).std()
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-8)
    
    # Additional engineered features
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
    df["open_close_ratio"] = (df["close"] - df["open"]) / df["open"]
    df["RSI_diff"] = df["RSI"].diff()
    df["MACD_diff_diff"] = df["MACD_diff"].diff()
    
    # ADX Indicator
    adx_ind = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX"] = adx_ind.adx()
    df["ADX_pos"] = adx_ind.adx_pos()
    df["ADX_neg"] = adx_ind.adx_neg()

    # Rolling z-score normalization for RSI and ADX (200-candle window)
    for col in ["RSI", "ADX"]:
        rolling_mean = df[col].rolling(200, min_periods=20).mean()
        rolling_std = df[col].rolling(200, min_periods=20).std().replace(0, 1)
        df[f"{col}_z"] = (df[col] - rolling_mean) / rolling_std

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    rolling_pv = (typical_price * df["volume"]).rolling(window=168, min_periods=1).sum()
    rolling_v = df["volume"].rolling(window=168, min_periods=1).sum()
    df["VWAP"] = rolling_pv / (rolling_v + 1e-8)
    df["close_to_VWAP"] = df["close"] / df["VWAP"] - 1.0
    
    # Momentum (Rate of Change)
    df["ROC_5"] = df["close"].pct_change(5)
    df["ROC_10"] = df["close"].pct_change(10)
    
    # BTC correlation return and lags
    df["btc_return_5m"] = df["close_btc"].pct_change(1)
    for lag in [1, 2, 3]:
        df[f"btc_return_5m_lag{lag}"] = df["btc_return_5m"].shift(lag)
        
    # Autoregressive target coin lags
    for lag in [1, 2, 3, 4, 5]:
        df[f"return_5m_lag{lag}"] = df["return_5m"].shift(lag)
    for lag in [1, 2, 3]:
        df[f"volume_ratio_lag{lag}"] = df["volume_ratio"].shift(lag)
    for lag in [1, 2]:
        df[f"RSI_lag{lag}"] = df["RSI"].shift(lag)
        df[f"MACD_diff_lag{lag}"] = df["MACD_diff"].shift(lag)
        df[f"BB_pct_lag{lag}"] = df["BB_pct"].shift(lag)
        
    # Macro-technical indicators
    df["RSI_24"] = RSIIndicator(df["close"], window=24).rsi()
    df["ROC_24"] = df["close"].pct_change(24)
    df["volatility_24h"] = df["return_5m"].rolling(24).std()
    
    # Derivatives & sentiment lags
    for lag in [1, 2]:
        df[f"open_interest_lag{lag}"] = df["open_interest"].shift(lag)
        df[f"funding_rate_lag{lag}"] = df["funding_rate"].shift(lag)
        df[f"fear_greed_lag{lag}"] = df["fear_greed"].shift(lag)
        
    # Derivatives momentum
    df["open_interest_pct_change"] = df["open_interest"].pct_change(1).fillna(0.0)
    df["funding_rate_diff"] = df["funding_rate"].diff(1).fillna(0.0)
    
    # High-fidelity Delta Volume / CVD Proxy
    high_low_range = df["high"] - df["low"] + 1e-8
    delta_volume = df["volume"] * (2 * (df["close"] - df["low"]) / high_low_range - 1.0)
    df["CVD_rolling_1h"] = delta_volume.rolling(window=4, min_periods=1).sum()
    df["CVD_rolling_4h"] = delta_volume.rolling(window=16, min_periods=1).sum()
    
    # Wick Volume (Liquidation & Stop-Loss Sweep Proxies)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    
    upper_wick_vol = df["volume"] * (upper_wick / high_low_range)
    lower_wick_vol = df["volume"] * (lower_wick / high_low_range)
    
    df["upper_wick_volume_ratio"] = upper_wick_vol / (upper_wick_vol.rolling(20).mean() + 1e-8)
    df["lower_wick_volume_ratio"] = lower_wick_vol / (lower_wick_vol.rolling(20).mean() + 1e-8)
    
    # Ensure source correlation features exist
    for col in ["oi_change_1h", "oi_change_4h", "btc_close", "btc_volume", "btc_rsi"]:
        if col not in df.columns:
            if "rsi" in col:
                df[col] = 50.0
            elif "close" in col:
                df[col] = df["close"]
            elif "volume" in col:
                df[col] = df["volume"]
            else:
                df[col] = 0.0

    # Lag new features
    for lag in [1, 2]:
        df[f"open_interest_pct_change_lag{lag}"] = df["open_interest_pct_change"].shift(lag)
        df[f"funding_rate_diff_lag{lag}"] = df["funding_rate_diff"].shift(lag)
        df[f"CVD_rolling_1h_lag{lag}"] = df["CVD_rolling_1h"].shift(lag)
        df[f"CVD_rolling_4h_lag{lag}"] = df["CVD_rolling_4h"].shift(lag)
        df[f"upper_wick_volume_ratio_lag{lag}"] = df["upper_wick_volume_ratio"].shift(lag)
        df[f"lower_wick_volume_ratio_lag{lag}"] = df["lower_wick_volume_ratio"].shift(lag)
        
        # Lag correlation features
        df[f"oi_change_1h_lag{lag}"] = df["oi_change_1h"].shift(lag).fillna(0.0)
        df[f"oi_change_4h_lag{lag}"] = df["oi_change_4h"].shift(lag).fillna(0.0)
        df[f"btc_close_lag{lag}"] = df["btc_close"].shift(lag).ffill().bfill().fillna(0.0)
        df[f"btc_volume_lag{lag}"] = df["btc_volume"].shift(lag).ffill().bfill().fillna(0.0)
        df[f"btc_rsi_lag{lag}"] = df["btc_rsi"].shift(lag).ffill().bfill().fillna(50.0)
        
    # Cyclical time features
    datetime_series = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["hour_sin"] = np.sin(2 * np.pi * datetime_series.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * datetime_series.dt.hour / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * datetime_series.dt.dayofweek / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * datetime_series.dt.dayofweek / 7.0)

    # 1D Kalman Filter trend feature
    df["close_to_Kalman"] = calculate_kalman_feature(df["close"].values)

    df = add_news_proximity_feature(df)
    df.dropna(inplace=True)
    return df

def add_triple_barrier_labels(df, interval):
    atr = df["ATR_norm"] * df["close"]
    atr_vals = atr.values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    adxs = df["ADX"].values
    n_samples = len(df)
    labels = np.ones(n_samples, dtype=int) * 1  # 1: Neutral
    
    cfg = TIMEFRAME_CONFIG.get(str(interval), {
        "lookahead": 10,
        "sl_mult": 0.8,
        "tp_mult_ranging": 1.5,
        "tp_mult_trending": 2.5
    })
    
    lookahead = cfg["lookahead"]
    sl_mult = cfg["sl_mult"]
    tp_mult_trending = cfg["tp_mult_trending"]
    tp_mult_ranging = cfg["tp_mult_ranging"]
        
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
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': 1
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
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
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'objective': 'multiclass',
            'num_class': 3,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
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
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'loss_function': 'MultiClass',
            'verbose': 0,
            'random_seed': 42
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
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
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'n_jobs': 1
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
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
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 1
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
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
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'verbose': 0,
            'random_seed': 42
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    return study.best_params

def train_models(interval=INTERVAL, pages=PAGES):
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
                    
                    dfs.append(df_coin)
                    print(f"Successfully processed {s}: {len(df_coin)} rows.")
        except Exception as e:
            print(f"Error processing {s} during training dataset creation: {e}")
            
    if not dfs:
        print("[Retraining Error] No symbol data could be processed. Aborting.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n=== Combined Training Dataset: {len(df)} total rows across {len(dfs)} symbols ===")

    # ==========================================
    # AUTOML FEATURE SELECTION (NOISE REDUCTION)
    # ==========================================
    import json
    from xgboost import XGBClassifier
    print("Running preliminary feature selection via XGBoost...")
    X_prelim = df[features]
    y_prelim = df["target_trend"]
    prelim_model = XGBClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=1)
    prelim_model.fit(X_prelim, y_prelim)
    importances = prelim_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Keep top 35 features
    top_n = min(35, len(features))
    selected_features = [features[idx] for idx in indices[:top_n]]
    print(f"Feature selection complete: Selected top {top_n} features out of {len(features)}:")
    for rank, f_name in enumerate(selected_features, 1):
        print(f"  {rank}. {f_name} (importance: {importances[features.index(f_name)]:.4f})")
        
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
            xgb_t = XGBClassifier(**best_params_xgb_t)
            lgb_t = LGBMClassifier(**best_params_lgb_t)
            cat_t = CatBoostClassifier(**best_params_cat_t)
            ensemble_t = EnsembleClassifier(xgb_t, lgb_t, cat_t)
            
            xgb_p = XGBRegressor(**best_params_xgb_p)
            lgb_p = LGBMRegressor(**best_params_lgb_p)
            cat_p = CatBoostRegressor(**best_params_cat_p)
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
            n_jobs=1
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
            import json
            with open(calibrator_filename, "w") as f:
                json.dump(calibrator_data, f)
            print(f"  [Calibrator] Saved Isotonic Regression calibrator to {calibrator_filename}")
        else:
            # Save default identity mapping if no predictions occurred
            calibrator_data = {"X": [0.0, 1.0], "y": [0.0, 1.0]}
            calibrator_filename = f"calibrator_{name}_{interval}.json"
            import json
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

        final_xgb_t = XGBClassifier(**best_params_xgb_t)
        final_lgb_t = LGBMClassifier(**best_params_lgb_t)
        final_cat_t = CatBoostClassifier(**best_params_cat_t)
        final_ensemble_t = EnsembleClassifier(final_xgb_t, final_lgb_t, final_cat_t)
        
        final_xgb_p = XGBRegressor(**best_params_xgb_p)
        final_lgb_p = LGBMRegressor(**best_params_lgb_p)
        final_cat_p = CatBoostRegressor(**best_params_cat_p)
        final_ensemble_p = EnsembleRegressor(final_xgb_p, final_lgb_p, final_cat_p)
        
        sample_weight_full = compute_sample_weight(class_weight='balanced', y=y_trend)
        final_ensemble_t.fit(X, y_trend, sample_weight=sample_weight_full)
        final_ensemble_p.fit(X, y_price)
        
        # Save models to disk using native text/JSON saving methods
        save_ensemble_classifier(final_ensemble_t, f"ensemble_{name}_trend_{interval}")
        save_ensemble_regressor(final_ensemble_p, f"ensemble_{name}_price_{interval}")
        meta_model.save_model(f"meta_{name}_trend_{interval}.json")
        
        print(f"  Saved ensemble and meta-classifier models for regime: {name.upper()}")

    # Split dataset based on ADX (Regime Detection)
    df_trending = df[df["ADX"] >= 20.0].copy().reset_index(drop=True)
    df_ranging = df[df["ADX"] < 20.0].copy().reset_index(drop=True)

    train_regime_model(df_trending, "trending")
    train_regime_model(df_ranging, "ranging")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train XGBoost models for BTC Trading Bot")
    parser.add_argument("--interval", type=str, default="60", choices=["60", "120", "240", "360", "all"], help="Timeframe interval to train")
    parser.add_argument("--pages", type=int, default=20, help="Number of data pages to fetch from Bybit")
    args = parser.parse_args()

    if args.interval == "all":
        for iv in ["60", "120", "240", "360"]:
            print(f"\n==============================================")
            print(f"TRAINING FOR INTERVAL: {iv}")
            print(f"==============================================")
            train_models(interval=iv, pages=args.pages)
    else:
        train_models(interval=args.interval, pages=args.pages)