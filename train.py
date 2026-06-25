import pandas as pd
import numpy as np
import joblib

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from data import get_history

# =========================
# CONFIGURATION
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "60"
PAGES = 40  # 40 pages of 1-hour candles provides ~40,000 candles (limit on Bybit spot history)

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
        
    df.dropna(inplace=True)
    return df

def train_models():
    # =========================
    # LOAD DATA
    # =========================
    if SYMBOL == "BTCUSDT":
        print(f"Fetching {PAGES} pages of {INTERVAL}-minute {SYMBOL} data ({PAGES * 1000} candles)...")
        df_target = get_history(symbol=SYMBOL, interval=INTERVAL, limit=1000, pages=PAGES)
        print(f"Loaded {len(df_target)} {SYMBOL} candles.")
        df = df_target.copy()
        df["close_btc"] = df["close"]
    else:
        print(f"Fetching {PAGES} pages of {INTERVAL}-minute {SYMBOL} data ({PAGES * 1000} candles)...")
        df_target = get_history(symbol=SYMBOL, interval=INTERVAL, limit=1000, pages=PAGES)
        print(f"Loaded {len(df_target)} {SYMBOL} candles.")
        
        print(f"Fetching {PAGES} pages of {INTERVAL}-minute BTCUSDT data ({PAGES * 1000} candles) for correlation...")
        df_btc = get_history(symbol="BTCUSDT", interval=INTERVAL, limit=1000, pages=PAGES)
        print(f"Loaded {len(df_btc)} BTCUSDT candles.")
        
        # Inner merge to align timestamps
        df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
        df = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
        print(f"Aligned dataset has {len(df)} candles.")

    # =========================
    # FEATURE ENGINEERING
    # =========================
    print("Engineering features...")
    df = add_features(df)
    
    # =========================
    # TARGET (NEXT CANDLE AHEAD)
    # =========================
    df["future"] = df["close"].shift(-1)
    df["target_trend"] = (df["future"] > df["close"]).astype(int)
    df["target_price_change"] = (df["future"] - df["close"]) / df["close"]
    df.dropna(inplace=True)

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

        # Models (Tuned to prevent overfitting)
        model_trend = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=5.0,
            eval_metric="logloss",
            random_state=42
        )

        model_price = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=42
        )

        # Out-of-sample Validation Split (Temporal validation)
        X_train, X_test, y_train_t, y_test_t = train_test_split(X, y_trend, test_size=0.2, random_state=42, shuffle=False)
        X_train, X_test, y_train_p, y_test_p = train_test_split(X, y_price, test_size=0.2, random_state=42, shuffle=False)

        print(f"  Validating {name} models...")
        model_trend.fit(X_train, y_train_t)
        y_pred_t = model_trend.predict(X_test)
        acc = accuracy_score(y_test_t, y_pred_t)

        model_price.fit(X_train, y_train_p)
        y_pred_p = model_price.predict(X_test)
        mae = mean_absolute_error(y_test_p, y_pred_p)

        print(f"  Validation Out-of-Sample Accuracy (Trend): {acc*100:.2f}%")
        print(f"  Validation Out-of-Sample MAE (Price Change): {mae:.2f}")

        print(f"  Training final {name} models on complete regime dataset...")
        model_trend.fit(X, y_trend)
        model_price.fit(X, y_price)

        model_trend.save_model(f"xgb_{name}_trend.json")
        model_price.save_model(f"xgb_{name}_price.json")
        print(f"  Models trained and saved to xgb_{name}_trend.json and xgb_{name}_price.json successfully.")

    # Split dataset based on ADX (Regime Detection)
    df_trending = df[df["ADX"] >= 20.0].copy().reset_index(drop=True)
    df_ranging = df[df["ADX"] < 20.0].copy().reset_index(drop=True)

    train_regime_model(df_trending, "trending")
    train_regime_model(df_ranging, "ranging")

if __name__ == "__main__":
    train_models()