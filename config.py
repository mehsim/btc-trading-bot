"""
Centralized Configuration System for BTC Trading Bot
Consolidates strategy thresholds, leverage caps, risk parameters, and memory bounds.
"""

# F-09 Isolated Governance Risk Limits
from risk_limits import (
    HARD_TIMEFRAME_MAX_LEVERAGE_CAPS as TIMEFRAME_MAX_LEVERAGE_CAPS,
    HARD_MAX_WALLET_MARGIN_UTILIZATION_PCT as MAX_WALLET_MARGIN_UTILIZATION_PCT,
    HARD_MAX_SYMBOL_EXPOSURE_PCT as MAX_SYMBOL_EXPOSURE_PCT,
    assert_risk_governance_invariants
)

# System & Memory Bounds
MAX_TRADE_HISTORY_MEMORY = 1000
MAX_PREDICTION_HISTORY_MEMORY = 1000
MAX_LOG_ENTRIES_MEMORY = 200

# Institutional Model Governance & Promotion Policy
SUPPORTED_MANIFEST_SCHEMA_VERSION = 3
MODEL_GOVERNANCE = {
    "version": "v2.4",
    "max_ece": 0.08,
    "max_brier": 0.22,
    "min_sharpe_delta": 0.10,
    "min_samples": 20
}

# Optuna Barrier-Tuning Objective Weights
# score = w_bal*BalAcc + w_f1*MacroF1 - w_ece*ECE - w_imb*ImbalancePenalty
MODEL_SELECTION = {
    "balanced_accuracy_weight":  1.00,
    "macro_f1_weight":           0.30,
    "ece_penalty_weight":        0.20,   # actual ECE from mlops_engine — no proxy
    "imbalance_penalty_weight":  0.40,
    "imbalance_neutral_cap":     0.70,   # penalty activates above this Neutral fraction
    "imbalance_min_class_pct":   0.10,   # warning threshold for any directional class
}

# Order Execution Bounds
MIN_ORDER_VALUE_USDT = 5.1
MAX_SCALED_RISK_CAP_RATIO = 1.10  # 110% hard cap on approved risk when order size is scaled up

INTERVAL_MAX_POSITION_PCT = {
    "5": 0.05,
    "15": 0.08,
    "30": 0.10,
    "60": 0.20,
    "120": 0.20,
    "240": 0.25,
    "360": 0.25
}

# Volatility & Market Regime Thresholds
TARGET_VOLATILITY_ATR = 0.005
PARAMETRIC_VAR_CONFIDENCE_LEVEL = 0.95
PARAMETRIC_VAR_DAILY_EQUITY_CAP_PCT = 0.05 # 5% daily equity VaR limit
MONTE_CARLO_MAX_STRESS_LOSS_PCT = 0.25     # 25% max stress loss equity budget
MONTE_CARLO_SHOCK_PCT = -0.30              # 30% market index shock parameter

# v2 & v3 Institutional Quant Engine Feature Flags (disabled / shadow / active)
EXIT_QUALITY_MODE = "shadow"          # "disabled", "shadow", "active"
EXPECTANCY_GATE_MODE = "shadow"       # "disabled", "shadow", "active"
SHADOW_EVALUATION_MODE = "active"     # "disabled", "shadow", "active"

ENABLE_REGIME_HYSTERESIS = True
ENABLE_UNCERTAINTY_TP_SCALING = True
ENABLE_DYNAMIC_STRUCTURAL_BUFFERS = True
ENABLE_REGIME_ADAPTIVE_RR_GATES = True
ENABLE_PARTIAL_TP_SYSTEM = True
ENABLE_SMART_BREAKEVEN = True
ENABLE_DYNAMIC_TRAILING_STOP = True
ENABLE_EXPLAINABLE_TRADE_LOG = True

import os

def _get_tf_env(key: str, default: float) -> float:
    try:
        val = os.environ.get(key)
        return float(val) if val is not None else default
    except Exception:
        return default

# Centralized Single Source of Truth for Timeframe Parameters (Recommendation #3)
# Allows dynamic overrides via .env (e.g. TF_15M_SL_MULT=1.25) or Optuna tuning
TIMEFRAME_CONFIG = {
    "15": {   # 15M Timeframe - Hardened Institutional Scalp
        "lookahead": int(_get_tf_env("TF_15M_LOOKAHEAD", 12)),
        "sl_mult": _get_tf_env("TF_15M_SL_MULT", 1.25),
        "base_confidence_threshold": _get_tf_env("TF_15M_CONF_THRESH", 0.68),
        "tp_mult_ranging": _get_tf_env("TF_15M_TP_RANGING", 1.35),
        "tp_mult_trending": _get_tf_env("TF_15M_TP_TRENDING", 1.65)
    },
    "30": {   # 30M Timeframe - Short Swing
        "lookahead": int(_get_tf_env("TF_30M_LOOKAHEAD", 12)),
        "sl_mult": _get_tf_env("TF_30M_SL_MULT", 0.80),
        "tp_mult_ranging": _get_tf_env("TF_30M_TP_RANGING", 1.45),
        "tp_mult_trending": _get_tf_env("TF_30M_TP_TRENDING", 1.75)
    },
    "60": {   # 1H Timeframe - Swing
        "lookahead": int(_get_tf_env("TF_60M_LOOKAHEAD", 10)),
        "sl_mult": _get_tf_env("TF_60M_SL_MULT", 1.20),
        "tp_mult_ranging": _get_tf_env("TF_60M_TP_RANGING", 2.16),
        "tp_mult_trending": _get_tf_env("TF_60M_TP_TRENDING", 2.70)
    },
    "120": {  # 2H Timeframe - Extended Swing
        "lookahead": int(_get_tf_env("TF_120M_LOOKAHEAD", 12)),
        "sl_mult": _get_tf_env("TF_120M_SL_MULT", 1.20),
        "tp_mult_ranging": _get_tf_env("TF_120M_TP_RANGING", 2.16),
        "tp_mult_trending": _get_tf_env("TF_120M_TP_TRENDING", 2.70)
    },
    "240": {  # 4H Timeframe - Macro Swing
        "lookahead": int(_get_tf_env("TF_240M_LOOKAHEAD", 12)),
        "sl_mult": _get_tf_env("TF_240M_SL_MULT", 1.50),
        "tp_mult_ranging": _get_tf_env("TF_240M_TP_RANGING", 2.70),
        "tp_mult_trending": _get_tf_env("TF_240M_TP_TRENDING", 3.30)
    },
    "360": {  # 6H Timeframe - Macro Trend
        "lookahead": int(_get_tf_env("TF_360M_LOOKAHEAD", 16)),
        "sl_mult": _get_tf_env("TF_360M_SL_MULT", 1.50),
        "tp_mult_ranging": _get_tf_env("TF_360M_TP_RANGING", 2.70),
        "tp_mult_trending": _get_tf_env("TF_360M_TP_TRENDING", 3.30)
    }
}

# Learnable EQS Weights Dictionary (Bayesian Optimization Ready)
EQS_WEIGHTS = {
    "structure": 20.0,
    "liquidity": 15.0,
    "expected_move": 20.0,
    "spread": 15.0,
    "funding": 10.0,
    "volatility": 10.0,
    "regime": 10.0
}

# v2 & v3 Quant Thresholds & Bounds
STRONG_TREND_ADX_ENTER = 32.0
STRONG_TREND_ADX_EXIT = 28.0
MIN_EXIT_QUALITY_SCORE = 75.0
MIN_VOLATILITY_ATR_NORM = 0.0035  # 0.35% minimum ATR volatility gate
MIN_STRATEGY_HEALTH_SCORE = 50.0  # Halt threshold below 50 SHS


