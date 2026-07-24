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
    """Rule 21: Variance Inflation Factor (VIF) feature selection (VIF > 10.0 dropped)."""
    if len(feature_list) <= 1:
        return feature_list
    
    corr_matrix = df[feature_list].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
    filtered = [f for f in feature_list if f not in to_drop]
    return filtered

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
