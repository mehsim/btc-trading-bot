import numpy as np
import pandas as pd

def handle_feature_outliers(df: pd.DataFrame, z_threshold: float = 4.0, window: int = 30) -> pd.DataFrame:
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ["timestamp", "open", "high", "low", "close", "volume"]:
            continue
        rolling_median = df_clean[col].rolling(window, min_periods=1).median()
        rolling_std = df_clean[col].rolling(window, min_periods=1).std().fillna(1e-5)
        z_scores = (df_clean[col] - rolling_median) / (rolling_std + 1e-8)
        
        mask = (z_scores > z_threshold) | (z_scores < -z_threshold)
        df_clean.loc[mask, col] = rolling_median[mask]
    return df_clean

def intelligent_data_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df_imp = df.copy()
    # 1. Technical indicators: ffill max 5, then median
    tech_cols = [c for c in df_imp.columns if c not in ["timestamp", "open", "high", "low", "close", "volume", "news_sentiment", "funding_rate"]]
    df_imp[tech_cols] = df_imp[tech_cols].ffill(limit=5)
    df_imp[tech_cols] = df_imp[tech_cols].fillna(df_imp[tech_cols].median())
    
    # 2. Sentiment/Funding: decay 0.95 multiplier per missing period
    for ext_col in ["news_sentiment", "funding_rate"]:
        if ext_col in df_imp.columns:
            s = df_imp[ext_col].copy()
            valid_mask = s.notna()
            if valid_mask.any():
                ffill_s = s.ffill()
                cum_group = valid_mask.cumsum()
                gap_count = s.groupby(cum_group).cumcount()
                decay_factor = 0.95 ** gap_count
                df_imp[ext_col] = (ffill_s * decay_factor).fillna(0.0)
            else:
                df_imp[ext_col] = 0.0
            
    return df_imp.bfill().fillna(0.0)

def filter_multicollinear_features(df: pd.DataFrame, feature_list: list, threshold: float = 0.85) -> list:
    if len(feature_list) <= 1:
        return feature_list
    corr_matrix = df[feature_list].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
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
