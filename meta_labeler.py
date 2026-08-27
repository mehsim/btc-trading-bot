"""
meta_labeler.py
---------------
López de Prado Second-Stage Meta-Labeling Engine.
Trains a binary meta-classifier on realized trade outcomes (138+ closed trades)
to filter primary signals before order placement ("should I take this signal?").
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from logger import log_event

_META_MODEL_CACHE = {}


MIN_META_TRAIN_TRADES = 200
MIN_META_CV_AUC = 0.60


def train_meta_labeler() -> tuple[bool, str]:
    """
    Fits second-stage MetaLabeler model on completed trades dataset from trading_bot.db.
    Target y = 1 if change_pct > 0 (profitable trade), else 0.
    Enforces fail-open safety guards: MIN_META_TRAIN_TRADES >= 200 & CV AUC >= 0.60.
    """
    db_path = "trading_bot.db"
    if not os.path.exists(db_path):
        _META_MODEL_CACHE["pass_all"] = True
        return True, "Database file not found (fail-open engaged)"

    try:
        conn = sqlite3.connect(db_path)
        df_trades = pd.read_sql_query(
            "SELECT symbol, interval, direction, change_pct, pnl_usd, confidence FROM completed_trades ORDER BY exit_time DESC",
            conn
        )
        conn.close()

        if len(df_trades) < MIN_META_TRAIN_TRADES:
            log_event("INFO", f"[MetaLabeler Fail-Open] Completed trade count ({len(df_trades)}) below threshold ({MIN_META_TRAIN_TRADES}). Passing all signals.")
            _META_MODEL_CACHE["pass_all"] = True
            return True, f"Trade count {len(df_trades)} < {MIN_META_TRAIN_TRADES} (fail-open engaged)"

        df_trades["target"] = (df_trades["change_pct"] > 0.0).astype(int)
        df_trades["is_majors"] = df_trades["symbol"].apply(lambda s: 1.0 if str(s).upper() in ("BTCUSDT", "ETHUSDT") else 0.0)
        df_trades["is_long"] = df_trades["direction"].astype(str).str.upper().isin(["BUY", "BULLISH", "LONG", "1"]).astype(float)
        df_trades["tf_norm"] = df_trades["interval"].astype(str).str.replace("m", "").str.replace("h", "").apply(lambda v: float(v) if v.isdigit() else 60.0) / 240.0
        df_trades["conf_val"] = pd.to_numeric(df_trades["confidence"], errors="coerce").fillna(0.50)

        feature_cols = ["is_majors", "is_long", "tf_norm", "conf_val"]
        X = df_trades[feature_cols].values
        y = df_trades["target"].values

        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=1)

        auc_scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
        mean_auc = float(np.mean(auc_scores))

        if mean_auc < MIN_META_CV_AUC:
            log_event("WARNING", f"[MetaLabeler Fail-Open] Meta-model CV AUC ({mean_auc:.4f}) below statistical significance floor ({MIN_META_CV_AUC}). Passing all signals.")
            _META_MODEL_CACHE["pass_all"] = True
            return True, f"CV AUC {mean_auc:.4f} < {MIN_META_CV_AUC} (fail-open engaged)"

        model.fit(X, y)

        _META_MODEL_CACHE["model"] = model
        _META_MODEL_CACHE["n_samples"] = len(df_trades)
        _META_MODEL_CACHE["win_rate"] = float(y.mean())
        _META_MODEL_CACHE["pass_all"] = False

        log_event("INFO", f"[MetaLabeler] Trained on {len(y)} trades | AUC: {mean_auc:.4f} | Win Rate: {y.mean()*100:.1f}%")
        return True, f"MetaLabeler trained on {len(y)} trades (AUC: {mean_auc:.4f})"

    except Exception as ex_meta:
        log_event("WARNING", f"[MetaLabeler Exception] {ex_meta}")
        _META_MODEL_CACHE["pass_all"] = True
        return True, str(ex_meta)


def evaluate_meta_filter(symbol: str, interval: str, direction: str, confidence: float = 0.50, min_prob: float = 0.30) -> tuple[bool, float]:
    """
    Evaluates whether second-stage MetaLabeler approves taking the primary signal.
    Returns (approved: bool, meta_prob: float).
    """
    if _META_MODEL_CACHE.get("pass_all", False):
        return True, 1.0

    if "model" not in _META_MODEL_CACHE:
        train_meta_labeler()
        if _META_MODEL_CACHE.get("pass_all", False) or "model" not in _META_MODEL_CACHE:
            return True, 1.0  # Fail-open baseline

    try:
        model = _META_MODEL_CACHE["model"]
        is_majors = 1.0 if str(symbol).upper() in ("BTCUSDT", "ETHUSDT") else 0.0
        is_long = 1.0 if str(direction).upper() in ("BUY", "BULLISH", "LONG") else 0.0
        iv_clean = str(interval).replace("m", "").replace("h", "")
        tf_norm = float(iv_clean) / 240.0 if iv_clean.isdigit() else 0.25
        conf_val = float(confidence) if confidence is not None else 0.50

        x_vec = np.array([[is_majors, is_long, tf_norm, conf_val]])
        prob_success = float(model.predict_proba(x_vec)[0, 1])

        approved = prob_success >= min_prob
        log_event("INFO", f"[MetaLabeler Evaluated] {symbol} {interval}m {direction} (conf={conf_val*100:.1f}%) -> MetaProb: {prob_success*100:.1f}% | Approved: {approved}")
        return approved, prob_success

    except Exception as ex_eval:
        log_event("WARNING", f"[MetaLabeler Eval Error] {ex_eval}")
        return True, 1.0
