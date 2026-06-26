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

# =========================
# CONFIGURATION
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "60"
PAGES = 20  # 20 pages of candles provides ~20,000 candles (balanced dataset size)

# Feature list matches train.py and main.py
features = [
    "RSI", "MACD_diff", "MFI", "ATR_norm",
    "close_to_EMA9", "close_to_EMA21", "close_to_EMA50", "close_to_EMA200", "EMA9_to_EMA21",
    "BB_pct", "BB_width", "return_5m", "volatility_10m", "volume_ratio",
    "high_low_ratio", "open_close_ratio", "RSI_diff", "MACD_diff_diff", "ROC_5", "ROC_10",
    "ADX", "ADX_pos", "ADX_neg", "close_to_VWAP",
    "btc_return_5m", "btc_return_5m_lag1", "btc_return_5m_lag2", "btc_return_5m_lag3",
    "RSI_24", "ROC_24", "volatility_24h"
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
    
    # Signed volume & Cumulative Volume Delta (CVD) proxies
    signed_vol = df["volume"] * np.sign(df["close"] - df["open"])
    df["CVD_rolling_1h"] = signed_vol.rolling(window=4, min_periods=1).sum()
    df["CVD_rolling_4h"] = signed_vol.rolling(window=16, min_periods=1).sum()
    
    # Lag new features
    for lag in [1, 2]:
        df[f"open_interest_pct_change_lag{lag}"] = df["open_interest_pct_change"].shift(lag)
        df[f"funding_rate_diff_lag{lag}"] = df["funding_rate_diff"].shift(lag)
        df[f"CVD_rolling_1h_lag{lag}"] = df["CVD_rolling_1h"].shift(lag)
        df[f"CVD_rolling_4h_lag{lag}"] = df["CVD_rolling_4h"].shift(lag)
        
    df.dropna(inplace=True)
    return df

def add_triple_barrier_labels(df, interval):
    # Dynamic SL & TP targets based on ATR: TP = 1.5 * ATR, SL = 1.0 * ATR
    atr = df["ATR_norm"] * df["close"]
    
    lookahead = 6
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr_vals = atr.values
    
    n_samples = len(df)
    labels = np.ones(n_samples, dtype=int) * 1  # 1: Neutral/Expiration
    
    for i in range(n_samples):
        p_t = closes[i]
        atr_t = atr_vals[i]
        
        # Fallback if ATR is 0
        if atr_t <= 0:
            atr_t = p_t * 0.001
            
        upper_barrier = p_t + 1.5 * atr_t
        lower_barrier = p_t - 1.0 * atr_t
        
        for step in range(1, lookahead + 1):
            if i + step >= n_samples:
                break
            
            h = highs[i + step]
            l = lows[i + step]
            
            hit_upper = h >= upper_barrier
            hit_lower = l <= lower_barrier
            
            if hit_upper and hit_lower:
                c = closes[i + step]
                if c >= upper_barrier:
                    labels[i] = 2  # Bullish
                elif c <= lower_barrier:
                    labels[i] = 0  # Bearish
                else:
                    labels[i] = 1
                break
            elif hit_upper:
                labels[i] = 2  # Bullish
                break
            elif hit_lower:
                labels[i] = 0  # Bearish
                break
                
    df["target_trend"] = labels
    return df

def optimize_xgb_classifier(X_train, y_train, X_val, y_val, sample_weights):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
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
    study.optimize(objective, n_trials=15)
    return study.best_params

def optimize_lgb_classifier(X_train, y_train, X_val, y_val, sample_weights):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
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
    study.optimize(objective, n_trials=15)
    return study.best_params

def optimize_cat_classifier(X_train, y_train, X_val, y_val, sample_weights):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 150),
            'depth': trial.suggest_int('depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'loss_function': 'MultiClass',
            'verbose': 0,
            'random_seed': 42
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    return study.best_params

def optimize_xgb_regressor(X_train, y_train, X_val, y_val):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'random_state': 42,
            'n_jobs': 1
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    return study.best_params

def optimize_lgb_regressor(X_train, y_train, X_val, y_val):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 1
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    return study.best_params

def optimize_cat_regressor(X_train, y_train, X_val, y_val):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 150),
            'depth': trial.suggest_int('depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'verbose': 0,
            'random_seed': 42
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    return study.best_params

def train_models(interval=INTERVAL, pages=PAGES):
    # =========================
    # LOAD DATA
    # =========================
    if SYMBOL == "BTCUSDT":
        print(f"Fetching {pages} pages of {interval}-minute {SYMBOL} data ({pages * 1000} candles)...")
        df_target = get_history(symbol=SYMBOL, interval=interval, limit=1000, pages=pages)
        print(f"Loaded {len(df_target)} {SYMBOL} candles.")
        df = df_target.copy()
        df["close_btc"] = df["close"]
    else:
        print(f"Fetching {pages} pages of {interval}-minute {SYMBOL} data ({pages * 1000} candles)...")
        df_target = get_history(symbol=SYMBOL, interval=interval, limit=1000, pages=pages)
        print(f"Loaded {len(df_target)} {SYMBOL} candles.")
        
        print(f"Fetching {pages} pages of {interval}-minute BTCUSDT data ({pages * 1000} candles) for correlation...")
        df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=1000, pages=pages)
        print(f"Loaded {len(df_btc)} BTCUSDT candles.")
        
        # Inner merge to align timestamps
        df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
        df = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
        print(f"Aligned dataset has {len(df)} candles.")

    # Merge Open Interest, Funding Rate, and Fear & Greed index historical data
    print("Merging Open Interest, Funding Rate, and Fear & Greed historical data...")
    df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=interval)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    print("Engineering features...")
    df = add_features(df)
    
    # =========================
    # TARGET (TRIPLE BARRIER + REGRESSION PRICE CHANGE)
    # =========================
    df["future"] = df["close"].shift(-1)
    df["target_price_change"] = (df["future"] - df["close"]) / df["close"]
    df = add_triple_barrier_labels(df, interval)
    df.dropna(subset=["target_price_change", "target_trend"], inplace=True)

    # ==========================================
    # REGIME SPLITTING & REGIME MODEL TRAINING
    # ==========================================
    def train_regime_model(df_regime, name):
        print(f"\nTraining model set for regime: {name.upper()} (Candles: {len(df_regime)})")
        if len(df_regime) < 100:
            print(f"Skipping {name} due to insufficient data.")
            return

        X = df_regime[features]
        y_trend = df_regime["target_trend"]
        y_price = df_regime["target_price_change"]

        # Purged and Embargoed Time-Series Cross Validation
        cv = PurgedEmbargoTimeSeriesSplit(n_splits=5, lookahead=6, embargo_pct=0.01)
        
        meta_features_list = []
        meta_labels_list = []
        
        primary_accuracies = []
        primary_maes = []
        
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
                best_params_xgb_t = optimize_xgb_classifier(X_train, y_train_t, X_val, y_val_t, sample_weight_train)
                best_params_lgb_t = optimize_lgb_classifier(X_train, y_train_t, X_val, y_val_t, sample_weight_train)
                best_params_cat_t = optimize_cat_classifier(X_train, y_train_t, X_val, y_val_t, sample_weight_train)
                
                best_params_xgb_p = optimize_xgb_regressor(X_train, y_train_p, X_val, y_val_p)
                best_params_lgb_p = optimize_lgb_regressor(X_train, y_train_p, X_val, y_val_p)
                best_params_cat_p = optimize_cat_regressor(X_train, y_train_p, X_val, y_val_p)
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
            
            # Generate Meta-labels for this validation fold
            actual_val_t = y_val_t.values
            is_non_neutral = (pred_val_t != 1)
            is_correct = (pred_val_t == actual_val_t)
            
            meta_features_list.append(X_val[is_non_neutral])
            meta_labels_list.append(is_correct[is_non_neutral].astype(int))
            
        print(f"  Validation Out-of-Sample Accuracy (Ensemble Trend): {np.mean(primary_accuracies)*100:.2f}%")
        print(f"  Validation Out-of-Sample MAE (Ensemble Price): {np.mean(primary_maes):.4f}")
        
        # Meta-Classifier Dataset
        meta_X = pd.concat(meta_features_list, ignore_index=True)
        meta_y = pd.Series(np.concatenate(meta_labels_list))
        
        print(f"  Meta-Classifier Training Samples: {len(meta_X)} (Positive rate: {meta_y.mean()*100:.2f}%)")
        
        # Train Meta-Classifier (XGBoost Binary Classifier)
        meta_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=1
        )
        if len(meta_X) >= 20:
            meta_model.fit(meta_X, meta_y)
            print("  Meta-Classifier trained successfully.")
        else:
            meta_model.fit(X, np.ones(len(X)))
            print("  Warning: Insufficient samples for Meta-Classifier. Dummy classifier trained.")
            
        # Fit final primary models on complete regime dataset
        print(f"  Training final ensemble models on complete {name} dataset...")
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
    parser.add_argument("--interval", type=str, default="60", choices=["5", "15", "60", "all"], help="Timeframe interval to train")
    parser.add_argument("--pages", type=int, default=20, help="Number of data pages to fetch from Bybit")
    args = parser.parse_args()

    if args.interval == "all":
        for iv in ["5", "15", "60"]:
            print(f"\n==============================================")
            print(f"TRAINING FOR INTERVAL: {iv}")
            print(f"==============================================")
            train_models(interval=iv, pages=args.pages)
    else:
        train_models(interval=args.interval, pages=args.pages)