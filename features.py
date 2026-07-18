import pandas as pd
import numpy as np
import bisect
from ta.momentum import RSIIndicator
from ta.volume import MFIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

try:
    from numba import jit
    @jit(nopython=True, cache=True)
    def _kalman_loop(prices, state_estimate, error_covariance, process_variance_arr, measurement_variance):
        n = len(prices)
        for i in range(1, n):
            pred_state = state_estimate[i-1]
            pred_error = error_covariance[i-1] + process_variance_arr[i]
            kalman_gain = pred_error / (pred_error + measurement_variance)
            state_estimate[i] = pred_state + kalman_gain * (prices[i] - pred_state)
            error_covariance[i] = (1.0 - kalman_gain) * pred_error
        return state_estimate
except ImportError:
    def _kalman_loop(prices, state_estimate, error_covariance, process_variance_arr, measurement_variance):
        n = len(prices)
        for i in range(1, n):
            pred_state = state_estimate[i-1]
            pred_error = error_covariance[i-1] + process_variance_arr[i]
            kalman_gain = pred_error / (pred_error + measurement_variance)
            state_estimate[i] = pred_state + kalman_gain * (prices[i] - pred_state)
            error_covariance[i] = (1.0 - kalman_gain) * pred_error
        return state_estimate

def calculate_kalman_feature(prices, atr_norm=None):
    n = len(prices)
    if n == 0:
        return np.zeros(0)
    state_estimate = np.zeros(n)
    error_covariance = np.zeros(n)
    state_estimate[0] = prices[0]
    error_covariance[0] = 1.0
    
    if atr_norm is not None and len(atr_norm) == n:
        # Scale process variance dynamically based on normalized volatility (clamped 1e-5 to 1e-3)
        process_variance_arr = np.clip(np.asarray(atr_norm, dtype=float) * 0.05, 1e-5, 1e-3)
    else:
        process_variance_arr = np.full(n, 1e-4)
        
    measurement_variance = 1e-2
    state_estimate = _kalman_loop(prices, state_estimate, error_covariance, process_variance_arr, measurement_variance)
    return (prices / (state_estimate + 1e-8)) - 1.0

def calculate_garman_klass_vol(df, window=14):
    log_hl = np.log(df["high"] / (df["low"] + 1e-8))
    log_co = np.log(df["close"] / (df["open"] + 1e-8))
    gk = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    return np.sqrt(np.maximum(0.0, gk.rolling(window).mean()))

def add_news_proximity_feature(df, fetch_calendar_callback=None):
    if df.empty:
        df["hours_to_news"] = 72.0
        return df
        
    if fetch_calendar_callback is None:
        df["hours_to_news"] = 72.0
        return df
        
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    events = fetch_calendar_callback(start_ts, end_ts)
    
    if not events:
        df["hours_to_news"] = 72.0
        return df
        
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

def add_features(df, fetch_calendar_callback=None):
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
        
    # Cross-Asset Lead-Lag Correlation Features
    df["return_5m"] = df["close"].pct_change(1)
    df["return_1h"] = df["close"].pct_change(12)
    df["btc_return_1h"] = df["close_btc"].pct_change(12)
    df["return_4h"] = df["close"].pct_change(48)
    df["btc_return_4h"] = df["close_btc"].pct_change(48)
    
    df["lead_lag_diff_5m"] = (df["return_5m"] - df["btc_return_5m"]).fillna(0.0)
    df["lead_lag_diff_1h"] = (df["return_1h"] - df["btc_return_1h"]).fillna(0.0)
    df["lead_lag_diff_4h"] = (df["return_4h"] - df["btc_return_4h"]).fillna(0.0)
    df["volume_ratio_to_btc"] = (df["volume"] / (df["btc_volume"] + 1e-8)).fillna(0.0)
    
    for lag in [1, 2]:
        df[f"lead_lag_diff_5m_lag{lag}"] = df["lead_lag_diff_5m"].shift(lag).fillna(0.0)
        df[f"lead_lag_diff_1h_lag{lag}"] = df["lead_lag_diff_1h"].shift(lag).fillna(0.0)
        df[f"lead_lag_diff_4h_lag{lag}"] = df["lead_lag_diff_4h"].shift(lag).fillna(0.0)
        df[f"volume_ratio_to_btc_lag{lag}"] = df["volume_ratio_to_btc"].shift(lag).fillna(0.0)
        
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
    df["open_interest_pct_change"] = df["open_interest"].pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["funding_rate_diff"] = df["funding_rate"].diff(1).fillna(0.0)
    
    # High-fidelity Delta Volume / CVD Proxy (OHLC-based fallback)
    high_low_range = df["high"] - df["low"] + 1e-8
    delta_volume = df["volume"] * (2 * (df["close"] - df["low"]) / high_low_range - 1.0)
    df["CVD_rolling_1h"] = delta_volume.rolling(window=4, min_periods=1).sum()
    df["CVD_rolling_4h"] = delta_volume.rolling(window=16, min_periods=1).sum()

    # Attempt to merge TRUE historical CVD/OFI/L2/Liquidations from SQLite (WebSocket-aggregated)
    try:
        import sqlite3, os
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kline_cache.db")
        sym = df.attrs.get("symbol", "BTCUSDT") if hasattr(df, "attrs") else "BTCUSDT"
        with sqlite3.connect(db_path, timeout=10) as conn:
            of_df = pd.read_sql_query(
                "SELECT timestamp, cvd, ofi, ob_imbalance_L2, ob_spread_L2, liq_long_1h, liq_short_1h FROM historical_order_flow WHERE symbol=? ORDER BY timestamp ASC",
                conn, params=(sym,)
            )
        if not of_df.empty:
            of_df["timestamp"] = (of_df["timestamp"] * 1000).astype("int64")
            of_df = of_df.rename(columns={"timestamp": "datetime"})
            # Align on minute-level timestamps by merging on nearest candle
            df["_ts"] = df.index.astype("int64") if hasattr(df.index, "astype") else df.index
            of_df = of_df.set_index("datetime").reindex(df["_ts"], method="nearest", tolerance=60000)
            df["CVD_true"] = of_df["cvd"].values
            df["OFI_true"] = of_df["ofi"].values
            df["ob_imbalance_L2"] = of_df["ob_imbalance_L2"].values
            df["ob_spread_L2"] = of_df["ob_spread_L2"].values
            df["liq_long_1h"] = of_df["liq_long_1h"].values
            df["liq_short_1h"] = of_df["liq_short_1h"].values
            # Fill NaN with proxy values where real data not yet available
            df["CVD_true"] = df["CVD_true"].fillna(df["CVD_rolling_1h"])
            df["OFI_true"] = df["OFI_true"].fillna(0.0)
            df["ob_imbalance_L2"] = df["ob_imbalance_L2"].fillna(0.0)
            df["ob_spread_L2"] = df["ob_spread_L2"].fillna(0.0)
            df["liq_long_1h"] = df["liq_long_1h"].fillna(0.0)
            df["liq_short_1h"] = df["liq_short_1h"].fillna(0.0)
            df.drop(columns=["_ts"], inplace=True, errors="ignore")
        else:
            raise ValueError("Empty order flow table")
    except Exception:
        df["CVD_true"] = df["CVD_rolling_1h"]
        df["OFI_true"] = 0.0
        df["ob_imbalance_L2"] = 0.0
        df["ob_spread_L2"] = 0.0
        df["liq_long_1h"] = 0.0
        df["liq_short_1h"] = 0.0
    
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

    # 1D Kalman Filter trend feature (Adaptive process variance)
    df["close_to_Kalman"] = calculate_kalman_feature(df["close"].values, df["ATR_norm"].values)
    for lag in [1, 2]:
        df[f"close_to_Kalman_lag{lag}"] = df["close_to_Kalman"].shift(lag)

    # Garman-Klass Volatility
    df["volatility_gk"] = calculate_garman_klass_vol(df, window=14)
    # Volatility Term Structure (gk_5 / gk_60)
    gk_5 = calculate_garman_klass_vol(df, window=5)
    gk_60 = calculate_garman_klass_vol(df, window=60)
    df["volatility_vts"] = gk_5 / (gk_60 + 1e-8)
    for lag in [1, 2]:
        df[f"volatility_gk_lag{lag}"] = df["volatility_gk"].shift(lag)
        df[f"volatility_vts_lag{lag}"] = df["volatility_vts"].shift(lag)

    # Advanced Microstructure Features
    dp_ret = df["close"].pct_change(1).fillna(0.0)
    autocov = dp_ret.rolling(24).cov(dp_ret.shift(1)).fillna(0.0)
    df["roll_spread"] = 2.0 * np.sqrt(np.maximum(0.0, -autocov))
    df["leverage_divergence"] = df["open_interest_pct_change"] - dp_ret
    df["oi_velocity"] = df["open_interest_pct_change"].diff(1).fillna(0.0)
    df["funding_acceleration"] = df["funding_rate_diff"].diff(1).fillna(0.0)
    df["bid_ask_imbalance_ohlc"] = (df["close"] - df["low"]) / (high_low_range) - 0.5
    
    # Lag advanced microstructure features
    for lag in [1, 2]:
        df[f"roll_spread_lag{lag}"] = df["roll_spread"].shift(lag)
        df[f"leverage_divergence_lag{lag}"] = df["leverage_divergence"].shift(lag)
        df[f"oi_velocity_lag{lag}"] = df["oi_velocity"].shift(lag)
        df[f"funding_acceleration_lag{lag}"] = df["funding_acceleration"].shift(lag)
        df[f"bid_ask_imbalance_ohlc_lag{lag}"] = df["bid_ask_imbalance_ohlc"].shift(lag)

    df = add_news_proximity_feature(df, fetch_calendar_callback)
    df.dropna(inplace=True)
    df = df.copy()  # Defragment memory layout to silence PerformanceWarnings
    return df
