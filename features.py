import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from logger import log_event

FEATURE_PIPELINE_VERSION = "v3.1.0"

import numpy as np
import bisect
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volume import MFIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
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

    @jit(nopython=True, cache=True)
    def _garman_klass_jit(high, low, close, open_p, window):
        n = len(high)
        res = np.zeros(n)
        const_factor = 2.0 * 0.6931471805599453 - 1.0
        for i in range(window - 1, n):
            sub_h = high[i - window + 1:i + 1]
            sub_l = low[i - window + 1:i + 1]
            sub_c = close[i - window + 1:i + 1]
            sub_o = open_p[i - window + 1:i + 1]
            log_hl = np.log(sub_h / (sub_l + 1e-8))
            log_co = np.log(sub_c / (sub_o + 1e-8))
            gk = 0.5 * log_hl**2 - const_factor * log_co**2
            res[i] = np.sqrt(np.maximum(0.0, np.mean(gk)))
        return res
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

    def _garman_klass_jit(high, low, close, open_p, window):
        n = len(high)
        res = np.zeros(n)
        const_factor = 2.0 * np.log(2.0) - 1.0
        for i in range(window - 1, n):
            sub_h = high[i - window + 1:i + 1]
            sub_l = low[i - window + 1:i + 1]
            sub_c = close[i - window + 1:i + 1]
            sub_o = open_p[i - window + 1:i + 1]
            log_hl = np.log(sub_h / (sub_l + 1e-8))
            log_co = np.log(sub_c / (sub_o + 1e-8))
            gk = 0.5 * log_hl**2 - const_factor * log_co**2
            res[i] = np.sqrt(np.maximum(0.0, np.mean(gk)))
        return res

def calculate_kalman_feature(prices, atr_norm=None):
    n = len(prices)
    if n == 0:
        return np.zeros(0)
    state_estimate = np.zeros(n)
    error_covariance = np.zeros(n)
    state_estimate[0] = prices[0]
    error_covariance[0] = 1.0
    
    if atr_norm is not None and len(atr_norm) == n:
        process_variance_arr = np.clip(np.asarray(atr_norm, dtype=float) * 0.05, 1e-5, 1e-3)
    else:
        process_variance_arr = np.full(n, 1e-4)
        
    measurement_variance = 1e-2
    state_estimate = _kalman_loop(prices, state_estimate, error_covariance, process_variance_arr, measurement_variance)
    return (prices / (state_estimate + 1e-8)) - 1.0

def calculate_garman_klass_vol(df, window=14):
    high = np.asarray(df["high"].values, dtype=float)
    low = np.asarray(df["low"].values, dtype=float)
    close = np.asarray(df["close"].values, dtype=float)
    open_p = np.asarray(df["open"].values, dtype=float)
    return pd.Series(_garman_klass_jit(high, low, close, open_p, window), index=df.index)

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
    events_utc = []
    for ev in events:
        ts = pd.Timestamp(ev)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        events_utc.append(ts)

    
    for current_time in df_dt:
        idx = bisect.bisect_left(events_utc, current_time)
        if idx < len(events_utc):
            next_event = events_utc[idx]
            diff_hours = (next_event - current_time).total_seconds() / 3600.0
            hours_to_news_list.append(min(72.0, max(0.0, diff_hours)))
        else:
            hours_to_news_list.append(72.0)
            
    df["hours_to_news"] = hours_to_news_list
    return df

MIN_FEATURE_HISTORY = 200 + 14 + 1
_FEATURES_CACHE = {}

def add_features(df, fetch_calendar_callback=None, symbol=None, interval=None):
    if df is not None and len(df) < MIN_FEATURE_HISTORY:
        raise ValueError(f"add_features requires >= {MIN_FEATURE_HISTORY} bars, got {len(df)}")
        
    cache_key = None
    if df is not None and len(df) > 0 and "timestamp" in df.columns and "close" in df.columns:
        try:
            sym_val = str(symbol) if symbol is not None else (str(df["symbol"].iloc[-1]) if "symbol" in df.columns else str(getattr(df, "attrs", {}).get("symbol", "BTCUSDT")))
            if interval is not None:
                iv_val = str(interval)
            elif "interval" in df.columns:
                iv_val = str(df["interval"].iloc[-1])
            elif getattr(df, "attrs", {}).get("interval"):
                iv_val = str(df.attrs["interval"])
            elif len(df) >= 2:
                dt_mins = round(abs(float(df["timestamp"].iloc[-1]) - float(df["timestamp"].iloc[-2])) / 60000.0)
                iv_val = str(dt_mins) if dt_mins > 0 else "15"
            else:
                iv_val = "15"

            last_ts = float(df["timestamp"].iloc[-1])
            last_close = float(df["close"].iloc[-1])
            cache_key = (sym_val, iv_val, len(df), last_ts, last_close)
            if cache_key in _FEATURES_CACHE:
                return _FEATURES_CACHE[cache_key].copy()
        except Exception:
            cache_key = None

    df = df.copy()
    
    # Ensure source derivative & sentiment columns exist with proper defaults if not merged
    if "open_interest" not in df.columns:
        df["open_interest"] = 0.0
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    if "fear_greed" not in df.columns:
        df["fear_greed"] = 50.0
    if "oi_change_1h" not in df.columns:
        df["oi_change_1h"] = 0.0
    if "oi_change_4h" not in df.columns:
        df["oi_change_4h"] = 0.0
        
    # Symbol Microstructure Features
    sym_name = str(df["symbol"].iloc[-1]) if "symbol" in df.columns else "BTCUSDT"
    df["is_majors"] = 1.0 if sym_name.upper() in ("BTCUSDT", "ETHUSDT") else 0.0
    
    atr_tmp = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range() / (df["close"] + 1e-8)
    df["sym_atr_rank"] = atr_tmp.rolling(100, min_periods=10).apply(lambda x: (pd.Series(x).rank(pct=True).iloc[-1]), raw=False).fillna(0.50)
    df["sym_vol_rank"] = df["volume"].rolling(100, min_periods=10).apply(lambda x: (pd.Series(x).rank(pct=True).iloc[-1]), raw=False).fillna(0.50)
    
    rets_tmp = df["close"].pct_change().fillna(0.0)
    btc_rets = df["close_btc"].pct_change().fillna(0.0) if "close_btc" in df.columns else rets_tmp
    cov = rets_tmp.rolling(30, min_periods=10).cov(btc_rets)
    var = btc_rets.rolling(30, min_periods=10).var()
    df["sym_beta_btc"] = (cov / var.replace(0, np.nan)).fillna(1.0)

    if "close_btc" in df.columns:
        _r  = df["close"].pct_change()
        _rb = df["close_btc"].pct_change()
        df["eth_btc_ratio"]  = df["close"] / df["close_btc"].replace(0, np.nan)
        df["corr_to_btc_60"] = _r.rolling(60, min_periods=20).corr(_rb)
        df["beta_to_btc_30"] = (_r.rolling(30, min_periods=10).cov(_rb) /
                                _rb.rolling(30, min_periods=10).var().replace(0, np.nan))
        df["btc_lead_1"]     = _rb.shift(1)
        df["btc_lead_2"]     = _rb.shift(2)
        df["rel_strength"]   = _r.rolling(20, min_periods=5).mean() - _rb.rolling(20, min_periods=5).mean()


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
    
    # Choppiness Index (0-100) for regime detection (fully vectorized across time)
    tr_chop = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
    atr_chop_sum = tr_chop.rolling(14).sum()
    high_chop_max = df["high"].rolling(14).max()
    low_chop_min = df["low"].rolling(14).min()
    denom_chop = (high_chop_max - low_chop_min).replace(0, np.nan)
    df["choppiness_index"] = (100.0 * np.log10(atr_chop_sum / denom_chop) / np.log10(14)).fillna(50.0)
    df["CHOP"] = df["choppiness_index"]
    
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
    
    # Higher-Timeframe (HTF) Macro Trend Integration
    try:
        base_iv_min = int(df["interval"].iloc[-1]) if "interval" in df.columns else 15
    except Exception:
        base_iv_min = 15
    m_4h = max(1, 240 // max(1, base_iv_min))
    m_1d = max(1, 1440 // max(1, base_iv_min))

    def _safe_ema(series, target_w):
        if len(series) > target_w and target_w >= 2:
            return EMAIndicator(series, window=target_w).ema_indicator()
        return pd.Series(np.nan, index=series.index)

    ema_144 = _safe_ema(df["close"], 9 * m_4h)
    ema_336 = _safe_ema(df["close"], 21 * m_4h)
    ema_800 = _safe_ema(df["close"], 50 * m_4h)
    ema_500 = _safe_ema(df["close"], min(200, 50 * m_1d))

    df["htf_4h_ema_alignment"] = (ema_144 / (ema_336 + 1e-8) - 1.0).fillna(0.0)
    df["htf_4h_trend_dir"] = np.where(ema_144 > ema_336, 1.0, np.where(ema_144 < ema_336, -1.0, 0.0))
    df["htf_4h_close_to_ema200"] = (df["close"] / (ema_800 + 1e-8) - 1.0).fillna(0.0)
    df["htf_1d_close_to_ema200"] = (df["close"] / (ema_500 + 1e-8) - 1.0).fillna(0.0)
    df["htf_trend_momentum_consensus"] = (
        np.sign(df["close_to_EMA9"].fillna(0.0)) +
        np.sign(df["close_to_EMA21"].fillna(0.0)) +
        np.sign(df["close_to_EMA200"].fillna(0.0)) +
        df["htf_4h_trend_dir"]
    ) / 4.0
    
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

    # 3. Leading Microstructure & Derivatives Dynamics (Phase 3)
    # A. Intra-bar CVD (Cumulative Volume Delta) momentum
    signed_vol = np.where(df["high"] > df["low"], ((df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)) * df["volume"], 0.0)
    df["cvd_momentum_14"] = pd.Series(signed_vol, index=df.index).rolling(14, min_periods=3).sum() / (df["volume"].rolling(14, min_periods=3).sum() + 1e-8)
    
    # B. Open Interest acceleration interacting with price returns
    df["oi_acceleration"] = (df["oi_change_1h"] - df["oi_change_4h"]) * df["close"].pct_change(1).fillna(0.0)
    
    # C. Funding Rate Z-score (200-candle rolling window)
    funding_mean = df["funding_rate"].rolling(200, min_periods=20).mean()
    funding_std = df["funding_rate"].rolling(200, min_periods=20).std().replace(0, np.nan).fillna(1e-5)
    df["funding_zscore"] = ((df["funding_rate"] - funding_mean) / funding_std).clip(-3.0, 3.0)
    
    # D. Orderbook Imbalance Proxy from Wick Volume Absorption
    upper_wick = df["high"] - np.maximum(df["open"], df["close"])
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    df["orderbook_imbalance"] = ((lower_wick - upper_wick) / (df["high"] - df["low"] + 1e-8)).clip(-1.0, 1.0)

    # Rolling z-score normalization for RSI and ADX (200-candle window)
    for col in ["RSI", "ADX"]:
        rolling_mean = df[col].rolling(200, min_periods=200).mean()
        rolling_std = df[col].rolling(200, min_periods=200).std().replace(0, np.nan).fillna(1.0)
        df[f"{col}_z"] = ((df[col] - rolling_mean) / rolling_std).fillna(0.0)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    rolling_pv = (typical_price * df["volume"]).rolling(window=168, min_periods=1).sum()
    rolling_v = df["volume"].rolling(window=168, min_periods=1).sum()
    df["VWAP"] = rolling_pv / (rolling_v + 1e-8)
    df["close_to_VWAP"] = df["close"] / df["VWAP"] - 1.0

    # Mean-Reversion Oscillators (Crucial for Ranging Regime)
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = (stoch.stoch() / 100.0).fillna(0.5)
    df["stoch_d"] = (stoch.stoch_signal() / 100.0).fillna(0.5)
    
    williams = WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14)
    df["williams_r"] = ((williams.williams_r() + 100.0) / 100.0).fillna(0.5)
    
    cci_ind = CCIIndicator(df["high"], df["low"], df["close"], window=20)
    df["cci"] = (cci_ind.cci() / 100.0).fillna(0.0)
    
    bb_std = (df["BB_high"] - df["BB_low"]) / 4.0
    df["bb_zscore"] = ((df["close"] - df["BB_mid"]) / (bb_std + 1e-8)).fillna(0.0)
    
    # Momentum (Rate of Change)
    df["ROC_5"] = df["close"].pct_change(5)
    df["ROC_10"] = df["close"].pct_change(10)
    
    # BTC correlation return and lags
    btc_close_series = df["btc_close"] if "btc_close" in df.columns else (df["close_btc"] if "close_btc" in df.columns else df["close"])
    btc_vol_series = df["btc_volume"] if "btc_volume" in df.columns else df["volume"]
    
    df["btc_return_5m"] = btc_close_series.pct_change(1)
    for lag in [1, 2, 3]:
        df[f"btc_return_5m_lag{lag}"] = df["btc_return_5m"].shift(lag)
        
    # Cross-Asset Lead-Lag Correlation Features
    df["return_1h"] = df["close"].pct_change(12)
    df["btc_return_1h"] = btc_close_series.pct_change(12)
    df["return_4h"] = df["close"].pct_change(48)
    df["btc_return_4h"] = btc_close_series.pct_change(48)
    
    df["lead_lag_diff_5m"] = (df["return_5m"] - df["btc_return_5m"]).fillna(0.0)
    df["lead_lag_diff_1h"] = (df["return_1h"] - df["btc_return_1h"]).fillna(0.0)
    df["lead_lag_diff_4h"] = (df["return_4h"] - df["btc_return_4h"]).fillna(0.0)
    df["volume_ratio_to_btc"] = (df["volume"] / (btc_vol_series + 1e-8)).fillna(0.0)
    
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
        sym = symbol or (str(df["symbol"].iloc[-1]) if ("symbol" in df.columns and len(df) > 0) else (df.attrs.get("symbol", "BTCUSDT") if hasattr(df, "attrs") else "BTCUSDT"))
        with sqlite3.connect(db_path, timeout=10) as conn:
            of_df = pd.read_sql_query(
                "SELECT timestamp, cvd, ofi, ob_imbalance_L2, ob_spread_L2, liq_long_1h, liq_short_1h FROM historical_order_flow WHERE symbol=? ORDER BY timestamp ASC",
                conn, params=(sym,)
            )
        if not of_df.empty:
            first_ts = float(of_df["timestamp"].iloc[0])
            if first_ts < 1e11:
                of_df["timestamp"] = (of_df["timestamp"] * 1000).astype("int64")
            else:
                of_df["timestamp"] = of_df["timestamp"].astype("int64")
            of_df = of_df.rename(columns={"timestamp": "datetime"})

            # Align without look-ahead using pd.merge_asof (direction="backward")
            ts_series = df["timestamp"] if "timestamp" in df.columns else df.index.astype("int64")
            df["_ts"] = ts_series
            of_df_sorted = of_df.sort_values("datetime")
            df_sorted = df[["_ts"]].copy().sort_values("_ts")
            merged_of = pd.merge_asof(df_sorted, of_df_sorted, left_on="_ts", right_on="datetime", direction="backward")
            merged_of.set_index("_ts", inplace=True)
            
            real_cvd_map = df["_ts"].map(merged_of["cvd"])
            df["cvd_is_real"] = real_cvd_map.notna().astype(int)
            df["CVD_true"] = real_cvd_map
            df["OFI_true"] = df["_ts"].map(merged_of["ofi"])
            df["ob_imbalance_L2"] = df["_ts"].map(merged_of["ob_imbalance_L2"])
            df["ob_spread_L2"] = df["_ts"].map(merged_of["ob_spread_L2"])
            df["liq_long_1h"] = df["_ts"].map(merged_of["liq_long_1h"])
            df["liq_short_1h"] = df["_ts"].map(merged_of["liq_short_1h"])
            # Fill NaN with proxy / NaN where real data not yet available
            df["CVD_true"] = df["CVD_true"].fillna(df["CVD_rolling_1h"])
            df["ob_imbalance_L2"] = df["ob_imbalance_L2"].where(df["cvd_is_real"] == 1, np.nan)
            df["ob_spread_L2"]    = df["ob_spread_L2"].where(df["cvd_is_real"] == 1, np.nan)
            df["OFI_true"]        = df["OFI_true"].where(df["cvd_is_real"] == 1, np.nan)
            df["liq_long_1h"]     = df["liq_long_1h"].fillna(0.0)
            df["liq_short_1h"]    = df["liq_short_1h"].fillna(0.0)
            df.drop(columns=["_ts"], inplace=True, errors="ignore")
        else:
            raise ValueError("Empty order flow table")
    except Exception:
        df["cvd_is_real"] = 0
        df["CVD_true"] = df["CVD_rolling_1h"]
        df["OFI_true"] = np.nan
        df["ob_imbalance_L2"] = np.nan
        df["ob_spread_L2"] = np.nan
        df["liq_long_1h"] = 0.0
        df["liq_short_1h"] = 0.0

    _vol_ma = df["volume"].rolling(60, min_periods=20).mean().replace(0, np.nan)
    df["CVD_norm"] = (df["CVD_true"] / _vol_ma).fillna(0.0)
    df["OFI_norm"] = (df["OFI_true"] / _vol_ma)
    
    # Wick Volume (Liquidation & Stop-Loss Sweep Proxies)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    
    high_low_range_safe = np.where(high_low_range > 1e-8, high_low_range, 1e-8)
    upper_wick_vol = df["volume"] * (upper_wick / high_low_range_safe)
    lower_wick_vol = df["volume"] * (lower_wick / high_low_range_safe)
    
    df["upper_wick_volume_ratio"] = upper_wick_vol / (upper_wick_vol.rolling(20).mean() + 1e-8)
    df["lower_wick_volume_ratio"] = lower_wick_vol / (lower_wick_vol.rolling(20).mean() + 1e-8)
    
    # Ensure source correlation features exist without DataFrame fragmentation
    missing_cols = {}
    for col in ["oi_change_1h", "oi_change_4h", "btc_close", "btc_volume", "btc_rsi", "hours_to_news"]:
        if col not in df.columns:
            if "hours_to_news" in col:
                missing_cols[col] = 72.0
            elif "rsi" in col:
                missing_cols[col] = 50.0
            elif "close" in col:
                missing_cols[col] = df["close"]
            elif "volume" in col:
                missing_cols[col] = df["volume"]
            else:
                missing_cols[col] = 0.0
    if missing_cols:
        df = df.assign(**missing_cols)
                
    # Batch all lag and cyclical features into a single dictionary to eliminate DataFrame fragmentation
    new_lag_cols = {}
    for lag in [1, 2]:
        new_lag_cols[f"open_interest_pct_change_lag{lag}"] = df["open_interest_pct_change"].shift(lag)
        new_lag_cols[f"funding_rate_diff_lag{lag}"] = df["funding_rate_diff"].shift(lag)
        new_lag_cols[f"CVD_rolling_1h_lag{lag}"] = df["CVD_rolling_1h"].shift(lag)
        new_lag_cols[f"CVD_rolling_4h_lag{lag}"] = df["CVD_rolling_4h"].shift(lag)
        new_lag_cols[f"upper_wick_volume_ratio_lag{lag}"] = df["upper_wick_volume_ratio"].shift(lag)
        new_lag_cols[f"lower_wick_volume_ratio_lag{lag}"] = df["lower_wick_volume_ratio"].shift(lag)
        new_lag_cols[f"oi_change_1h_lag{lag}"] = df["oi_change_1h"].shift(lag).fillna(0.0)
        new_lag_cols[f"oi_change_4h_lag{lag}"] = df["oi_change_4h"].shift(lag).fillna(0.0)
        new_lag_cols[f"btc_close_lag{lag}"] = df["btc_close"].shift(lag).ffill().bfill().fillna(0.0)
        new_lag_cols[f"btc_volume_lag{lag}"] = df["btc_volume"].shift(lag).ffill().bfill().fillna(0.0)
        new_lag_cols[f"btc_rsi_lag{lag}"] = df["btc_rsi"].shift(lag).ffill().bfill().fillna(50.0)
        new_lag_cols[f"htf_4h_ema_alignment_lag{lag}"] = df["htf_4h_ema_alignment"].shift(lag).fillna(0.0)
        new_lag_cols[f"htf_4h_trend_dir_lag{lag}"] = df["htf_4h_trend_dir"].shift(lag).fillna(0.0)
        new_lag_cols[f"htf_4h_close_to_ema200_lag{lag}"] = df["htf_4h_close_to_ema200"].shift(lag).fillna(0.0)
        new_lag_cols[f"htf_1d_close_to_ema200_lag{lag}"] = df["htf_1d_close_to_ema200"].shift(lag).fillna(0.0)
        new_lag_cols[f"htf_trend_momentum_consensus_lag{lag}"] = df["htf_trend_momentum_consensus"].shift(lag).fillna(0.0)

    # Cyclical time features
    first_ts = float(df["timestamp"].dropna().iloc[0]) if ("timestamp" in df.columns and len(df["timestamp"].dropna()) > 0) else 0.0
    unit_val = "ms" if first_ts > 1e11 else "s"
    datetime_series = pd.to_datetime(df["timestamp"], unit=unit_val, utc=True)
    new_lag_cols["hour_sin"] = np.sin(2 * np.pi * datetime_series.dt.hour / 24.0)
    new_lag_cols["hour_cos"] = np.cos(2 * np.pi * datetime_series.dt.hour / 24.0)
    new_lag_cols["day_of_week_sin"] = np.sin(2 * np.pi * datetime_series.dt.dayofweek / 7.0)
    new_lag_cols["day_of_week_cos"] = np.cos(2 * np.pi * datetime_series.dt.dayofweek / 7.0)

    # 1D Kalman Filter trend feature
    kalman_series = calculate_kalman_feature(df["close"].values, df["ATR_norm"].values)
    new_lag_cols["close_to_Kalman"] = kalman_series
    for lag in [1, 2]:
        new_lag_cols[f"close_to_Kalman_lag{lag}"] = pd.Series(kalman_series, index=df.index).shift(lag)

    # Garman-Klass Volatility
    vol_gk = calculate_garman_klass_vol(df, window=14)
    gk_5 = calculate_garman_klass_vol(df, window=5)
    gk_60 = calculate_garman_klass_vol(df, window=60)
    vol_vts = gk_5 / (gk_60 + 1e-8)
    
    new_lag_cols["volatility_gk"] = vol_gk
    new_lag_cols["volatility_vts"] = vol_vts
    for lag in [1, 2]:
        new_lag_cols[f"volatility_gk_lag{lag}"] = vol_gk.shift(lag)
        new_lag_cols[f"volatility_vts_lag{lag}"] = vol_vts.shift(lag)

    # Advanced Microstructure Features (VWAP & VWAP Deviation)
    cum_vol = df["volume"].cumsum() + 1e-8
    cum_pv = (df["close"] * df["volume"]).cumsum()
    vwap_series = cum_pv / cum_vol
    vwap_dev_series = (df["close"] - vwap_series) / (vwap_series + 1e-8)
    
    dp_ret = df["close"].pct_change(1).fillna(0.0)
    autocov = dp_ret.rolling(24).cov(dp_ret.shift(1)).fillna(0.0)
    roll_spread_series = 2.0 * np.sqrt(np.maximum(0.0, -autocov))
    lev_div_series = df["open_interest_pct_change"] - dp_ret
    oi_vel_series = df["open_interest_pct_change"].diff(1).fillna(0.0)
    funding_acc_series = df["funding_rate_diff"].diff(1).fillna(0.0)
    bid_ask_imb_series = (df["close"] - df["low"]) / (high_low_range) - 0.5

    new_lag_cols["VWAP"] = vwap_series
    new_lag_cols["vwap_deviation"] = vwap_dev_series
    new_lag_cols["roll_spread"] = roll_spread_series
    new_lag_cols["leverage_divergence"] = lev_div_series
    new_lag_cols["oi_velocity"] = oi_vel_series
    new_lag_cols["funding_acceleration"] = funding_acc_series
    new_lag_cols["bid_ask_imbalance_ohlc"] = bid_ask_imb_series

    # Lag advanced microstructure features
    for lag in [1, 2]:
        new_lag_cols[f"vwap_deviation_lag{lag}"] = vwap_dev_series.shift(lag)
        new_lag_cols[f"roll_spread_lag{lag}"] = roll_spread_series.shift(lag)
        new_lag_cols[f"leverage_divergence_lag{lag}"] = lev_div_series.shift(lag)
        new_lag_cols[f"oi_velocity_lag{lag}"] = oi_vel_series.shift(lag)
        new_lag_cols[f"funding_acceleration_lag{lag}"] = funding_acc_series.shift(lag)
        new_lag_cols[f"bid_ask_imbalance_ohlc_lag{lag}"] = bid_ask_imb_series.shift(lag)
        if "CVD_true" in df.columns:
            new_lag_cols[f"CVD_true_lag{lag}"] = df["CVD_true"].shift(lag)
        if "CVD_norm" in df.columns:
            new_lag_cols[f"CVD_norm_lag{lag}"] = df["CVD_norm"].shift(lag)
        if "OFI_true" in df.columns:
            new_lag_cols[f"OFI_true_lag{lag}"] = df["OFI_true"].shift(lag)
        if "OFI_norm" in df.columns:
            new_lag_cols[f"OFI_norm_lag{lag}"] = df["OFI_norm"].shift(lag)
        if "cvd_is_real" in df.columns:
            new_lag_cols[f"cvd_is_real_lag{lag}"] = df["cvd_is_real"].shift(lag)
        if "ob_imbalance_L2" in df.columns:
            new_lag_cols[f"ob_imbalance_L2_lag{lag}"] = df["ob_imbalance_L2"].shift(lag)
        if "ob_spread_L2" in df.columns:
            new_lag_cols[f"ob_spread_L2_lag{lag}"] = df["ob_spread_L2"].shift(lag)
        if "liq_long_1h" in df.columns:
            new_lag_cols[f"liq_long_1h_lag{lag}"] = df["liq_long_1h"].shift(lag)
        if "liq_short_1h" in df.columns:
            new_lag_cols[f"liq_short_1h_lag{lag}"] = df["liq_short_1h"].shift(lag)

    df = pd.concat([df, pd.DataFrame(new_lag_cols, index=df.index)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = add_candlestick_patterns(df)
    try:
        from high_alpha_feature_engine import high_alpha_feature_engine
        df = high_alpha_feature_engine.compute_all_high_alpha_features(df)
    except Exception as e:
        print(f"[Features Warning] Failed computing high alpha features: {e}")
    df = df.ffill()
    df = sanitize_feature_matrix(df)
    if cache_key is not None:
        if len(_FEATURES_CACHE) > 50:
            _FEATURES_CACHE.clear()
        _FEATURES_CACHE[cache_key] = df.copy()
    return df

def add_candlestick_patterns(df):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-8)
    u_wick = h - np.maximum(o, c)
    l_wick = np.minimum(o, c) - l
    is_green = c > o
    is_red = c < o

    res = {}
    res["cdl_hammer"] = np.where((l_wick >= 2 * body) & (u_wick <= 0.1 * rng) & (body / rng > 0.05), 1, 0)
    res["cdl_hanging_man"] = np.where((l_wick >= 2 * body) & (u_wick <= 0.1 * rng) & (df["close"] > df["EMA_21"]), -1, 0)
    res["cdl_shooting_star"] = np.where((u_wick >= 2 * body) & (l_wick <= 0.1 * rng) & (body / rng > 0.05), -1, 0)
    res["cdl_inv_hammer"] = np.where((u_wick >= 2 * body) & (l_wick <= 0.1 * rng) & (df["close"] < df["EMA_21"]), 1, 0)
    res["cdl_doji"] = np.where(body / rng <= 0.1, 1, 0)
    res["cdl_gravestone_doji"] = np.where((body / rng <= 0.1) & (u_wick >= 0.7 * rng), -1, 0)
    res["cdl_dragonfly_doji"] = np.where((body / rng <= 0.1) & (l_wick >= 0.7 * rng), 1, 0)
    res["cdl_spinning_top"] = np.where((body / rng <= 0.3) & (u_wick >= 0.3 * rng) & (l_wick >= 0.3 * rng), 1, 0)
    res["cdl_marubozu_bull"] = np.where(is_green & (u_wick <= 0.02 * rng) & (l_wick <= 0.02 * rng), 1, 0)
    res["cdl_marubozu_bear"] = np.where(is_red & (u_wick <= 0.02 * rng) & (l_wick <= 0.02 * rng), -1, 0)

    def _shift(arr, n):
        s = pd.Series(arr).shift(n)
        if s.dtype == bool or np.issubdtype(s.dtype, np.bool_) or isinstance(arr.iloc[0] if hasattr(arr, 'iloc') else arr[0], (bool, np.bool_)):
            return s.astype('boolean').fillna(False).to_numpy(dtype=bool)
        return s.fillna(0.0).to_numpy(dtype=float)

    # 2-candle patterns
    prev_o, prev_c = _shift(o, 1), _shift(c, 1)
    prev_h, prev_l = _shift(h, 1), _shift(l, 1)
    prev_red, prev_green = prev_c < prev_o, prev_c > prev_o
    res["cdl_bullish_engulfing"] = np.where(prev_red & is_green & (c >= prev_o) & (o <= prev_c), 1, 0)
    res["cdl_bearish_engulfing"] = np.where(prev_green & is_red & (c <= prev_o) & (o >= prev_c), -1, 0)
    res["cdl_bullish_harami"] = np.where(prev_red & is_green & (o >= prev_c) & (c <= prev_o), 1, 0)
    res["cdl_bearish_harami"] = np.where(prev_green & is_red & (o <= prev_c) & (c >= prev_o), -1, 0)
    res["cdl_tweezer_top"] = np.where((np.abs(h - prev_h) / h <= 0.001) & is_red & prev_green, -1, 0)
    res["cdl_tweezer_bottom"] = np.where((np.abs(l - prev_l) / l <= 0.001) & is_green & prev_red, 1, 0)
    res["cdl_piercing_line"] = np.where(prev_red & is_green & (o < prev_l) & (c >= prev_c + 0.5 * np.abs(prev_c - prev_o)), 1, 0)
    res["cdl_dark_cloud_cover"] = np.where(prev_green & is_red & (o > prev_h) & (c <= prev_c - 0.5 * np.abs(prev_c - prev_o)), -1, 0)
    res["cdl_inside_bar"] = np.where((h <= prev_h) & (l >= prev_l), 1, 0)

    # 3-candle patterns
    p2_o, p2_c = _shift(o, 2), _shift(c, 2)
    p2_red, p2_green = p2_c < p2_o, p2_c > p2_o
    mid_small = np.abs(prev_c - prev_o) / np.maximum(prev_h - prev_l, 1e-8) <= 0.3
    res["cdl_morning_star"] = np.where(p2_red & mid_small & is_green & (c >= p2_c + 0.5 * np.abs(p2_c - p2_o)), 1, 0)
    res["cdl_evening_star"] = np.where(p2_green & mid_small & is_red & (c <= p2_c - 0.5 * np.abs(p2_c - p2_o)), -1, 0)
    prev_doji = _shift(res["cdl_doji"], 1)
    res["cdl_morning_doji_star"] = np.where(p2_red & (prev_doji == 1) & is_green, 1, 0)
    res["cdl_evening_doji_star"] = np.where(p2_green & (prev_doji == 1) & is_red, -1, 0)

    c1_g, c2_g, c3_g = is_green, _shift(is_green, 1), _shift(is_green, 2)
    c1_r, c2_r, c3_r = is_red, _shift(is_red, 1), _shift(is_red, 2)
    res["cdl_three_white_soldiers"] = np.where(c1_g & c2_g & c3_g & (c > prev_c) & (prev_c > p2_c), 1, 0)
    res["cdl_three_black_crows"] = np.where(c1_r & c2_r & c3_r & (c < prev_c) & (prev_c < p2_c), -1, 0)
    prev_harami_bull = _shift(res["cdl_bullish_harami"], 1)
    prev_harami_bear = _shift(res["cdl_bearish_harami"], 1)
    res["cdl_three_inside_up"] = np.where(p2_red & (prev_harami_bull == 1) & is_green & (c > p2_o), 1, 0)
    res["cdl_three_inside_down"] = np.where(p2_green & (prev_harami_bear == 1) & is_red & (c < p2_o), -1, 0)

    # 5-candle patterns
    p4_green, p4_red = _shift(is_green, 4), _shift(is_red, 4)
    r1, r2, r3 = _shift(is_red, 1), _shift(is_red, 2), _shift(is_red, 3)
    g1, g2, g3 = _shift(is_green, 1), _shift(is_green, 2), _shift(is_green, 3)
    res["cdl_rising_three"] = np.where(p4_green & is_green & r1 & r2 & r3 & (c > _shift(h, 4)), 1, 0)
    res["cdl_falling_three"] = np.where(p4_red & is_red & g1 & g2 & g3 & (c < _shift(l, 4)), -1, 0)
    res["cdl_abandoned_baby_bull"] = np.where(p2_red & (prev_doji == 1) & is_green & (prev_h < _shift(l, 2)) & (l > prev_h), 1, 0)

    # Set initial 5 rows to 0 due to rolling
    pattern_df = pd.DataFrame(res, index=df.index)
    pattern_df.iloc[:5] = 0
    df_out = pd.concat([df, pattern_df], axis=1)
    if df_out.columns.duplicated().any():
        df_out = df_out.loc[:, ~df_out.columns.duplicated()].copy()
    return df_out


def sanitize_feature_matrix(X):
    """
    Canonical feature matrix sanitization shared identically across training and serving.
    Replaces NaN, Inf, -Inf, and None with context-appropriate neutral baselines:
      - 1.0 for self-normalized ratio features (e.g. volume_ratio, atr_ratio, upper_wick_volume_ratio)
      - 50.0 for RSI
      - 72.0 for hours_to_news
      - 0.0 for z-scores, returns, and deltas
    """
    if X is None:
        return X
    import math
    ratio_cols = ("volume_ratio", "vol_ratio", "oi_ratio", "mfe_ratio", "atr_ratio", "wick_volume_ratio")
    try:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            for c in num_cols:
                c_low = str(c).lower()
                if "hours_to_news" in c_low:
                    fill_val = 72.0
                elif "rsi" in c_low and not any(z in c_low for z in ("_z", "zscore", "z_score")):
                    fill_val = 50.0
                elif any(rc in c_low for rc in ratio_cols) and "to_btc" not in c_low:
                    fill_val = 1.0
                else:
                    fill_val = 0.0
                df[c] = df[c].fillna(fill_val).astype(float)
            return df
        elif isinstance(X, pd.Series):
            if pd.api.types.is_numeric_dtype(X):
                s = X.copy().replace([np.inf, -np.inf], np.nan)
                s_name = str(s.name or "").lower()
                if "hours_to_news" in s_name:
                    fill_val = 72.0
                elif "rsi" in s_name and not any(z in s_name for z in ("_z", "zscore", "z_score")):
                    fill_val = 50.0
                elif any(rc in s_name for rc in ratio_cols) and "to_btc" not in s_name:
                    fill_val = 1.0
                else:
                    fill_val = 0.0
                return s.fillna(fill_val).astype(float)
            return X
        elif isinstance(X, dict):
            clean_dict = {}
            for k, v in X.items():
                k_lower = str(k).lower()
                if "hours_to_news" in k_lower:
                    fill_val = 72.0
                elif "rsi" in k_lower:
                    fill_val = 50.0
                elif any(rc in k_lower for rc in ratio_cols) and "to_btc" not in k_lower:
                    fill_val = 1.0
                else:
                    fill_val = 0.0
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    clean_dict[k] = fill_val
                else:
                    try:
                        clean_dict[k] = float(v)
                    except Exception:
                        clean_dict[k] = fill_val
            return clean_dict
        elif isinstance(X, np.ndarray):
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X_clean
    except Exception as ex:
        log_event("WARNING", f"Feature sanitization notice: {ex}")
        try:
            arr = np.asarray(X, dtype=float)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as ex_arr:
            log_event("WARNING", f"Feature array conversion notice: {ex_arr}")
            return X
    return X

