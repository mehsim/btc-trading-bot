import numpy as np
import pandas as pd
from sentiment_decay import sentiment_decay_engine

def handle_feature_outliers(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Rule 18: Adaptive Outlier Threshold using Median Absolute Deviation (MAD) & IQR."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ["timestamp", "open", "high", "low", "close", "volume"]:
            continue
        q25 = df_clean[col].rolling(window, min_periods=1).quantile(0.25)
        q75 = df_clean[col].rolling(window, min_periods=1).quantile(0.75)
        iqr = np.maximum(1e-5, q75 - q25)
        median = df_clean[col].rolling(window, min_periods=1).median()
        
        mask = np.abs(df_clean[col] - median) > (1.5 * iqr)
        df_clean.loc[mask, col] = median[mask]
    return df_clean

def simple_kalman_imputation(series: pd.Series) -> pd.Series:
    """Rule 19: 1D Kalman Filter state estimation for missing indicator steps."""
    s = series.copy()
    if not s.isna().any():
        return s
    
    x_hat = s.dropna().iloc[0] if s.dropna().shape[0] > 0 else 0.0
    P = 1.0
    Q = 1e-4  # process variance
    R = 1e-2  # measurement variance

    result = []
    for val in s.values:
        if np.isnan(val):
            P = P + Q
            result.append(x_hat)
        else:
            K = P / (P + R)
            x_hat = x_hat + K * (val - x_hat)
            P = (1 - K) * P
            result.append(x_hat)

    return pd.Series(result, index=s.index)

def intelligent_data_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Rules 19 & 20: Kalman Filter Imputation and Half-Life Sentiment Decay."""
    df_imp = df.copy()
    tech_cols = [c for c in df_imp.columns if c not in ["timestamp", "open", "high", "low", "close", "volume", "news_sentiment", "funding_rate"]]
    
    # 1. Technical indicators: 1D Kalman Filter prediction
    for c in tech_cols:
        if df_imp[c].isna().any():
            df_imp[c] = simple_kalman_imputation(df_imp[c])
    
    # 2. Rule 20: Dynamic Sentiment/Funding Decay using sentiment_decay_engine
    decay_factor = sentiment_decay_engine.get_decay_factor()
    for ext_col in ["news_sentiment", "funding_rate"]:
        if ext_col in df_imp.columns:
            s = df_imp[ext_col].copy()
            valid_mask = s.notna()
            if valid_mask.any():
                ffill_s = s.ffill()
                cum_group = valid_mask.cumsum()
                gap_count = s.groupby(cum_group).cumcount()
                decay_weights = decay_factor ** gap_count
                df_imp[ext_col] = (ffill_s * decay_weights).fillna(0.0)
            else:
                df_imp[ext_col] = 0.0
            
    return df_imp.bfill().fillna(0.0)

def filter_multicollinear_features(df: pd.DataFrame, feature_list: list, vif_threshold: float = 10.0) -> list:
    """Rule 21: Iterative Variance Inflation Factor (VIF > 10.0 dropped) & correlation filtering."""
    valid_feats = [f for f in feature_list if f in df.columns]
    if len(valid_feats) <= 1:
        return valid_feats

    # Step 1: Pairwise absolute correlation filtering (>0.85)
    df_sub = df[valid_feats].dropna().astype(float)
    if len(df_sub) < 10:
        return valid_feats

    corr_matrix = df_sub.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set(column for column in upper_tri.columns if any(upper_tri[column] > 0.85))
    remaining = [f for f in valid_feats if f not in to_drop]

    # Step 2: Iterative VIF feature removal (VIF > vif_threshold)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_mat = df_sub[remaining].values
        keep = list(remaining)
        while len(keep) > 1:
            vifs = [variance_inflation_factor(X_mat, i) for i in range(len(keep))]
            max_vif = max(vifs)
            if max_vif > vif_threshold and not np.isnan(max_vif) and not np.isinf(max_vif):
                max_idx = vifs.index(max_vif)
                keep.pop(max_idx)
                X_mat = np.delete(X_mat, max_idx, axis=1)
            else:
                break
        return keep
    except Exception:
        return remaining

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df_int = df.copy()
    if "ATR_norm" in df_int.columns and "ADX" in df_int.columns:
        df_int["ATR_x_trend"] = df_int["ATR_norm"] * df_int["ADX"]
    if "funding_rate" in df_int.columns and "momentum" in df_int.columns:
        df_int["funding_x_momentum"] = df_int["funding_rate"] * df_int["momentum"]
    if "CVD" in df_int.columns and "volume" in df_int.columns:
        vol_z = (df_int["volume"] - df_int["volume"].rolling(20, min_periods=1).mean()) / (df_int["volume"].rolling(20, min_periods=1).std() + 1e-8)
        df_int["CVD_x_volz"] = df_int["CVD"] * vol_z
    return df_int

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pillar 2: Advanced Microstructure Feature Pipeline.
    Computes Realized Volatility (15m, 1h, 4h), Vol-of-Vol, Orderbook Convexity,
    Queue Imbalance, and Liquidity Sweep Detection.
    """
    df_feat = df.copy()
    close_s = df_feat["close"] if "close" in df_feat.columns else pd.Series(dtype=float)
    if not close_s.empty:
        log_ret = np.log(close_s / close_s.shift(1)).fillna(0.0)
        df_feat["realized_vol_15m"] = log_ret.rolling(3, min_periods=2).std().fillna(0.0) * np.sqrt(96)
        df_feat["realized_vol_1h"] = log_ret.rolling(4, min_periods=2).std().fillna(0.0) * np.sqrt(96)
        df_feat["realized_vol_4h"] = log_ret.rolling(16, min_periods=2).std().fillna(0.0) * np.sqrt(96)
        df_feat["vol_of_vol_24h"] = df_feat["realized_vol_1h"].rolling(24, min_periods=2).std().fillna(0.0)

    if "high" in df_feat.columns and "low" in df_feat.columns:
        range_s = df_feat["high"] - df_feat["low"]
        avg_range = range_s.rolling(20, min_periods=1).mean() + 1e-8
        df_feat["orderbook_convexity"] = range_s / avg_range
        high_roll = df_feat["high"].rolling(10, min_periods=1).max()
        low_roll = df_feat["low"].rolling(10, min_periods=1).min()
        df_feat["liquidity_sweep_flag"] = ((df_feat["high"] >= high_roll) | (df_feat["low"] <= low_roll)).astype(float)

    if "volume" in df_feat.columns:
        vol_mean = df_feat["volume"].rolling(20, min_periods=1).mean() + 1e-8
        df_feat["queue_imbalance"] = (df_feat["volume"] - vol_mean) / vol_mean

    return df_feat

def calculate_adaptive_triple_barrier_labels(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    interval: str = "15",
    regime: str = "Trending"
) -> pd.Series:
    """
    Pillar 2: Adaptive Triple Barrier Labeling.
    Evaluates High/Low price barriers dynamically resolved from TIMEFRAME_CONFIG by timeframe interval and market regime, emitting {0: Bearish, 1: Neutral, 2: Bullish} labels.
    """
    from config import TIMEFRAME_CONFIG
    cfg = TIMEFRAME_CONFIG.get(str(interval), {
        "lookahead": 12,
        "sl_mult": 0.8,
        "tp_mult_ranging": 1.35,
        "tp_mult_trending": 1.85
    })

    lookahead = cfg.get("lookahead", 10)
    sl_mult = cfg.get("sl_mult", 0.8)
    regime_upper = str(regime).upper()
    if "TRENDING" in regime_upper:
        tp_mult = cfg.get("tp_mult_trending", 1.85)
    else:
        tp_mult = cfg.get("tp_mult_ranging", 1.35)

    closes = df["close"].values
    highs = df["high"].values if "high" in df.columns else closes
    lows = df["low"].values if "low" in df.columns else closes
    atr_vals = df["ATR"].values if "ATR" in df.columns else (closes * 0.01)
    
    n_samples = len(df)
    labels = np.ones(n_samples, dtype=int) * 1  # Default 1: Neutral

    for i in range(n_samples):
        p_t = closes[i]
        atr_t = atr_vals[i] if atr_vals[i] > 0 else p_t * 0.001
        
        upper_b = p_t + tp_mult * atr_t
        lower_b = p_t - tp_mult * atr_t
        upper_s = p_t + sl_mult * atr_t
        lower_s = p_t - sl_mult * atr_t
        
        for step in range(1, lookahead + 1):
            if i + step >= n_samples:
                break
            h, l = highs[i + step], lows[i + step]
            hit_bullish = h >= upper_b and l > lower_s
            hit_bearish = l <= lower_b and h < upper_s
            
            if hit_bullish and not hit_bearish:
                labels[i] = 2  # Bullish
                break
            elif hit_bearish and not hit_bullish:
                labels[i] = 0  # Bearish
                break
            elif l <= lower_s or h >= upper_s:
                labels[i] = 1  # Neutral (hit SL before TP)
                break

    return pd.Series(labels, index=df.index)

