"""
Centralized Configuration System for BTC Trading Bot
Consolidates strategy thresholds, leverage caps, risk parameters, and memory bounds.
"""

import os
import json

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

# Supported Assets
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

# Institutional Model Governance & Promotion Policy
SUPPORTED_MANIFEST_SCHEMA_VERSION = 3
MODEL_GOVERNANCE_VERSION = 2
MODEL_GOVERNANCE = {
    "version": "v2.5",
    "max_ece": 0.08,
    "max_brier": 0.50,
    "min_oos_sharpe": 0.50,
    "min_sharpe_delta": 0.10,
    "max_cv_bal_acc_std": 0.05,
    "max_cv_fold_range": 0.12,
    "enforce_stability_gate": True,
    "min_samples": 20,
    "min_mcc": 0.05,                   # C-1 Predictive Floor: MCC < 0.05 is at statistical chance for multi-hour bars
    "min_balanced_accuracy": 0.36,     # C-1 Predictive Floor: 3-class random chance is 0.333
    "min_holdout_mcc": 0.02,                    # out-of-sample floor — applies with or without a champion
    "min_holdout_balanced_accuracy": 0.34,      # 3-class chance is 0.3333
    "mcc_regression_tolerance": 0.010   # R-2: Absorbs run-to-run noise during challenger evaluation
}

# Timeframe-Adaptive Predictive Floors (Sub-hourly microstructure vs Multi-hour bars)
TIMEFRAME_MIN_MCC = {"15": 0.015, "30": 0.035, "60": 0.040, "default": 0.050}
TIMEFRAME_MIN_BAL_ACC = {"15": 0.345, "30": 0.350, "60": 0.348, "default": 0.360}
TIMEFRAME_MIN_HOLDOUT_MCC = {"15": 0.010, "30": 0.015, "60": 0.015, "default": 0.020}
TIMEFRAME_MIN_HOLDOUT_BAL_ACC = {"15": 0.335, "30": 0.338, "60": 0.338, "default": 0.340}

# Dynamic Regime-Based Model Routing (Trending vs Ranging)
# When False (Default): Routes to validated universal/trending models across all market regimes.
# When True: Routes strictly to regime-specific models (trending vs ranging) and abstains fail-closed if unservable.
ENABLE_DYNAMIC_REGIME_ROUTING = True
DYNAMIC_REGIME_ROUTING_INTERVALS = {"15", "30", "60", "120", "240"}  # Dynamic GMM regime routing enabled for all active timeframes
_persisted_denylist = set()
if os.path.exists("governance_denylist.json"):
    try:
        with open("governance_denylist.json", "r") as _gdf:
            _pdata = json.load(_gdf)
            if isinstance(_pdata, list):
                _persisted_denylist = set(_pdata)
            elif isinstance(_pdata, dict):
                _persisted_denylist = set(_pdata.keys())
    except Exception:
        pass

MODEL_SLOT_DENYLIST = {
    # 15m trending & ranging lifted 2026-08-20: retrained with verified barrier contract (23 feats, Delta PF +0.15)
    "trending_120",   # holdout MCC 0.0000, balacc 0.3333 — degenerate out-of-sample
    # trending_240 lifted 2026-08-18: retrained with RFECV (CV MCC +0.0582, Holdout MCC +0.0331)
    "ranging_240",    # holdout MCC 0.0000, balacc 0.3333 — degenerate out-of-sample
    # trending_30 & ranging_30 lifted 2026-08-18: retrained with RFECV (MCC +0.0638 / +0.0831)
    # trending_60 & ranging_60 lifted: verified 34 features with matching barrier contracts
}.union(_persisted_denylist)

# Architectural Remediation Configurations (F-1, F-2, F-7, B-1, B-9)
SPRT_MIN_DETECTABLE_D = 0.20                  # Pre-registered minimum detectable effect size for SPRT sequential testing
MCC_LEVERAGE_QUALIFICATION_THRESHOLD = 0.15  # F-1: Models with MCC < 0.15 are clamped to conservative leverage
CONSERVATIVE_LEVERAGE_CAPS = {"BTCUSDT": 5.0, "ETHUSDT": 5.0, "default": 3.0}
MIN_SL_PCT_CONFIG = {"15": 0.006, "30": 0.008, "60": 0.010, "120": 0.012, "240": 0.015, "360": 0.015, "default": 0.008}
MIN_TARGET_ATR_MULT = {"15": 1.50, "60": 1.50, "default": 1.50}  # Enforces tp_mult >= 1.50 ATR

def resolve_min_sl_pct(symbol: str = "BTCUSDT", interval: str = "60") -> float:
    iv_clean = str(interval).lower().strip().replace("m", "")
    if iv_clean in ("1h", "60"): iv_str = "60"
    elif iv_clean in ("2h", "120"): iv_str = "120"
    elif iv_clean in ("4h", "240"): iv_str = "240"
    elif iv_clean in ("6h", "360"): iv_str = "360"
    elif iv_clean in ("15m", "15"): iv_str = "15"
    elif iv_clean in ("30m", "30"): iv_str = "30"
    elif iv_clean in ("5m", "5"): iv_str = "5"
    else: iv_str = iv_clean
    val = MIN_SL_PCT_CONFIG.get(iv_str, MIN_SL_PCT_CONFIG.get("default", 0.008))
    return float(val) * 100.0
HORIZON_REACHABILITY_FACTOR = 1.5  # Canonical horizon reachability scale factor (lookahead^0.5 * ATR * factor)
MIN_LEVERAGE_RAMP_START = 1.5       # B-1: Leverage starts at 1.5x to eliminate deadband below validation floor
NEUTRAL_PENALTY_COEFFICIENT = 0.0  # B-9: Neutral penalty coefficient (0.0 allows pure isotonic probability calibration)

# R-1: Model Quality Sizing Policy (Capital allocation scales by measured predictive content)
QUALITY_SIZING = {
    "enabled": True,
    "reference_mcc": 0.15,  # Fleet benchmark defining 1.0x scaling point
    "floor": 0.35           # Minimum scaling multiplier floor
}

# R-3: Institutional Kill Criteria Policy (Evaluated at >= 250 trades per timeframe)
KILL_CRITERIA = {
    "min_closed_trades": 250,
    "win_rate_stop_delta": 0.08,    # Realised win rate > 8 points below claimed calibrated confidence -> STOP
    "win_rate_review_delta": 0.04,  # Realised win rate 4-8 points below claimed -> REVIEW
    "expectancy_stop_floor": 0.0    # Realised expectancy after costs < 0 -> STOP
}

# R-5: Retraining Governance Policy (Prevents retraining on noise variance)
RETRAIN_POLICY = {
    "min_days_between_retrains": 7,
    "require_mhi_trigger": True,
    "noise_band_mcc": 0.012
}

CVAR_PARAMETRIC_FALLBACK_RATIO = 1.25
PSI_INSUFFICIENT_CYCLES_THRESHOLD = 5
MIN_LIQUIDITY_MULTIPLIER = 0.20
MIN_EVAL_THRESHOLD_FLOOR = 0.25  # Minimum decision threshold floor based on risk preference & calibration uncertainty
MAX_THRESHOLD_UPLIFT = 0.08     # Maximum allowed threshold uplift above economic base threshold
ALMGREN_CHRISS_CALIBRATION_INTERVAL_DAYS = 30
ALMGREN_CHRISS_MIN_FILLS = 1000
HISTORICAL_STRESS_QUANTILE = 0.001
MIN_STRESS_HISTORICAL_BARS = 5000
MIN_KELLY_SAMPLE_SIZE = 30

# M-1: Train/Holdout ratio & minimum effective independent holdout sample assertion
HOLDOUT_FRACTION: float = 0.15
MIN_EFFECTIVE_HOLDOUT_SAMPLES: int = 80  # min independent observations N / lookahead

# M-4: Consistent cross-validation fold count across hyperparameter search, feature selection, and main CV
CV_N_SPLITS: int = 5

# C-1 / H-1: Model Health Index (MHI) Governance Policy & Retraining Triggers
MHI_POLICY: dict = {
    "retrain_threshold": 60.0,
    "severe_psi": 0.25,
    "max_ece": 0.08,
    "wr_drop_factor": 0.30,
    "wr_drop_min": 0.10,
    "wr_drop_max": 0.25,
    "max_psi_penalty": 40.0,
    "max_ece_penalty": 30.0,
    "max_wr_penalty": 30.0,
}

# H-2 / M-2: Continuous MHI Sizing Multiplier Floor & Circuit Breaker Hysteresis
MHI_MULT_FLOOR: float = 0.50
CIRCUIT_BREAKER_RESUME_RATIO: float = 0.75

# M-3: Kelly payoff lower bound Monte Carlo bootstrap samples
KELLY_BOOTSTRAP_SAMPLES: int = 1000

# M-1: Default indicator period window across ATR, ADX, Choppiness
DEFAULT_INDICATOR_WINDOW: int = 14

# H-1: Confluence Scoring Governance Policy & Continuous Ramp Scaling
CONFLUENCE_POLICY: dict = {
    "strong_move_pct": 0.015,
    "max_spread_pct": 0.0010,
    "max_funding_rate": 0.0003,
    "min_atr_norm": 0.0035,
    "structure_weight": 20.0,
    "liquidity_weight": 15.0,
    "move_weight": 20.0,
    "spread_weight": 15.0,
    "funding_weight": 10.0,
    "volatility_weight": 10.0,
    "regime_weight": 10.0,
}

# H-2: Ensemble Uncertainty Governance Policy & Cutoffs (Calibrated for 3-class probability distribution)
UNCERTAINTY_POLICY: dict = {
    "disagreement_weight": 0.7,
    "margin_weight": 0.3,
    "margin_reference": 0.25,
    "margin_cutoff": 0.02,
    "disagreement_cutoff": 0.15,
}

# H-1: Exit Quality Score Policy & Continuous Ramp Scaling
EXIT_SCORE_POLICY: dict = {
    "norm_r_weight": 0.30,
    "regime_weight": 0.20,
    "volatility_weight": 0.20,
    "decay_weight": 0.15,
    "opportunity_weight": 0.15,
    "opp_hurdle": 1.5,
}

# H-2: ATR Timeout Soft Adjustment Bounds & Sensitivity
ATR_TIMEOUT_SENSITIVITY: float = 5.0
ATR_TIMEOUT_MAX_ADJ: float = 2.0

# M-1: Timeframe & Regime-Adaptive Time Decay Policy
TIME_DECAY_POLICY: dict = {
    "trending": {"15": 0.15, "30": 0.12, "60": 0.10, "120": 0.08, "240": 0.05, "default": 0.10},
    "ranging":  {"15": 0.30, "30": 0.28, "60": 0.25, "120": 0.20, "240": 0.15, "default": 0.25},
    "unknown":  {"15": 0.45, "30": 0.42, "60": 0.40, "120": 0.35, "240": 0.30, "default": 0.40},
}

# M-2: Time Decay Floor Governance Policy
TIME_DECAY_FLOOR: float = 0.20
TIME_DECAY_FLOOR_EXCEEDED: float = 0.00

# M-3: Portfolio Utility Capital Allocation Policy
PORTFOLIO_UTILITY_POLICY: dict = {
    "min_utility_threshold": 0.20,
    "max_stagnant_candles": 15,
    "scaleout_utility_threshold": 0.50,
    "loss_utility_cutoff": -2.0,
    "portfolio_heat_max": 0.80,
    "runner_boost_multiplier": 1.20,
}



# H-1: Per-timeframe Kelly rolling window (max trades, last N calendar days).
# Window expressed as (max_trades, lookback_days); effective window = trades within lookback_days, min 30.
KELLY_WINDOW_CONFIG: dict = {
    "15":  {"max_trades": 100, "lookback_days": 7},
    "30":  {"max_trades": 100, "lookback_days": 14},
    "60":  {"max_trades": 80,  "lookback_days": 30},
    "120": {"max_trades": 60,  "lookback_days": 60},
    "240": {"max_trades": 40,  "lookback_days": 90},
    "360": {"max_trades": 30,  "lookback_days": 120},
}
KELLY_WINDOW_DEFAULT = {"max_trades": 100, "lookback_days": 30}

# H-2: Per-timeframe exponential decay half-life (days) for sample weighting.
# w_i = 0.5 ** (age_days / HALF_LIFE_DAYS)
DECAY_HALF_LIFE_CONFIG: dict = {
    "15":  7,
    "30":  14,
    "60":  30,
    "120": 45,
    "240": 90,
    "360": 120,
}
DECAY_HALF_LIFE_DEFAULT = 30

# Optuna Barrier-Tuning Objective Weights
# score = w_bal*BalAcc + w_f1*MacroF1 - w_ece*ECE - w_imb*ImbalancePenalty
MODEL_SELECTION = {
    "balanced_accuracy_weight":  1.00,
    "macro_f1_weight":           0.30,
    "ece_penalty_weight":        0.20,   # actual ECE from mlops_engine — no proxy
    "imbalance_penalty_weight":  0.40,
    "imbalance_neutral_cap":     0.50,   # penalty activates above this Neutral fraction
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
import json


def _get_tf_env(key: str, default: float) -> float:
    try:
        val = os.environ.get(key)
        return float(val) if val is not None else default
    except Exception:
        return default

# Centralized Single Source of Truth for Timeframe Parameters (Recommendation #3)
# Allows dynamic overrides via .env (e.g. TF_15M_SL_MULT=1.25) or Optuna tuning
TIMEFRAME_CONFIG = {
    "15": {   # 15M Timeframe - High-Conviction Scalp (Targets ~4-5 trades/day across portfolio)
        "lookahead": int(_get_tf_env("TF_15M_LOOKAHEAD", 12)),
        "sl_mult": _get_tf_env("TF_15M_SL_MULT", 0.85),
        "base_confidence_threshold": _get_tf_env("TF_15M_CONF_THRESH", 0.38),
        "min_adx": _get_tf_env("TF_15M_MIN_ADX", 16.0),
        "tp_mult_ranging": _get_tf_env("TF_15M_TP_RANGING", 1.40),
        "tp_mult_trending": _get_tf_env("TF_15M_TP_TRENDING", 1.85)
    },
    "30": {   # 30M Timeframe - Short Swing
        "lookahead": int(_get_tf_env("TF_30M_LOOKAHEAD", 12)),
        "sl_mult": _get_tf_env("TF_30M_SL_MULT", 1.20),
        "base_confidence_threshold": _get_tf_env("TF_30M_CONF_THRESH", 0.40),
        "tp_mult_ranging": _get_tf_env("TF_30M_TP_RANGING", 1.45),
        "tp_mult_trending": _get_tf_env("TF_30M_TP_TRENDING", 1.75)
    },
    "60": {   # 1H Timeframe - High-Conviction Swing (Targets ~3-4 trades/day across portfolio)
        "lookahead": int(_get_tf_env("TF_60M_LOOKAHEAD", 10)),
        "sl_mult": _get_tf_env("TF_60M_SL_MULT", 0.6585006543095501),
        "base_confidence_threshold": _get_tf_env("TF_60M_CONF_THRESH", 0.40),
        "min_adx": _get_tf_env("TF_60M_MIN_ADX", 24.0),
        "tp_mult_ranging": _get_tf_env("TF_60M_TP_RANGING", 1.258257285199672),
        "tp_mult_trending": _get_tf_env("TF_60M_TP_TRENDING", 1.4746788008303522)
    },
    "120": {  # 2H Timeframe - Extended Swing
        "lookahead": int(_get_tf_env("TF_120M_LOOKAHEAD", 12)),
        "sl_mult": _get_tf_env("TF_120M_SL_MULT", 0.5016884584379878),
        "tp_mult_ranging": _get_tf_env("TF_120M_TP_RANGING", 0.9996827578790153),
        "tp_mult_trending": _get_tf_env("TF_120M_TP_TRENDING", 1.1191795909907887)
    },
    "240": {  # 4H Timeframe - High-Conviction Macro Swing (Targets ~1-2 trades/day across portfolio)
        "lookahead": int(_get_tf_env("TF_240M_LOOKAHEAD", 12)),
        "sl_mult": _get_tf_env("TF_240M_SL_MULT", 0.7693),
        "base_confidence_threshold": _get_tf_env("TF_240M_CONF_THRESH", 0.58),
        "min_adx": _get_tf_env("TF_240M_MIN_ADX", 28.0),
        "tp_mult_ranging": _get_tf_env("TF_240M_TP_RANGING", 1.3370),
        "tp_mult_trending": _get_tf_env("TF_240M_TP_TRENDING", 1.80)
    },
    "360": {  # 6H Timeframe - Macro Trend
        "lookahead": int(_get_tf_env("TF_360M_LOOKAHEAD", 16)),
        "sl_mult": _get_tf_env("TF_360M_SL_MULT", 1.50),
        "tp_mult_ranging": _get_tf_env("TF_360M_TP_RANGING", 2.70),
        "tp_mult_trending": _get_tf_env("TF_360M_TP_TRENDING", 3.30)
    }
}

# Auto-sync TIMEFRAME_CONFIG with optimized_barriers_{tf}.json if present
for _tf in list(TIMEFRAME_CONFIG.keys()):
    _opt_path = f"optimized_barriers_{_tf}.json"
    if os.path.exists(_opt_path):
        try:
            with open(_opt_path, "r") as _f:
                _opt_data = json.load(_f)
            if isinstance(_opt_data, dict):
                for _k in ["lookahead", "sl_mult", "tp_mult_ranging", "tp_mult_trending"]:
                    if _k in _opt_data:
                        if _k == "lookahead":
                            TIMEFRAME_CONFIG[_tf][_k] = int(_opt_data[_k])
                        else:
                            TIMEFRAME_CONFIG[_tf][_k] = float(_opt_data[_k])
        except Exception:
            pass

DYNAMIC_CONFIDENCE_THRESHOLDS = {
    _tf: float(_cfg.get("base_confidence_threshold", 0.55))
    for _tf, _cfg in TIMEFRAME_CONFIG.items()
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

# v2 & v3 Quant Thresholds & Bounds (Per-Interval Regime ADX Hysteresis)
REGIME_ADX_ENTER_BY_INTERVAL = {
    "15": 28.0,
    "30": 28.0,
    "60": 28.0,
    "120": 28.0,
    "240": 28.0,
    "360": 28.0,
}

REGIME_ADX_EXIT_BY_INTERVAL = {
    "15": 24.0,
    "30": 24.0,
    "60": 24.0,
    "120": 24.0,
    "240": 24.0,
    "360": 24.0,
}

# Global defaults for backwards compatibility
STRONG_TREND_ADX_ENTER = 32.0
STRONG_TREND_ADX_EXIT = 28.0
MIN_EXIT_QUALITY_SCORE = 75.0
MIN_VOLATILITY_ATR_NORM = 0.0035  # 0.35% minimum ATR volatility gate
MIN_STRATEGY_HEALTH_SCORE = 50.0  # Halt threshold below 50 SHS

# C-1: Model Health Index (MHI) Continuous Kelly Policy with Hysteresis
MHI_KELLY_POLICY = {
    "halt_below": 50.0,
    "resume_above": 53.0,
    "full_at": 85.0,
    "max_kelly": 0.25,
}

# H-1 & H-2: Volatility Sizing Profile Continuous Interpolation across all timeframes
VOLATILITY_SIZING_PROFILE = {
    "15":  [(0.000, 0.5), (0.002, 0.5), (0.003, 1.0), (0.008, 1.0), (0.011, 0.7), (0.015, 0.5)],
    "30":  [(0.000, 0.5), (0.003, 0.5), (0.005, 1.0), (0.012, 1.0), (0.015, 0.7), (0.020, 0.5)],
    "60":  [(0.000, 0.5), (0.004, 0.5), (0.007, 1.0), (0.015, 1.0), (0.020, 0.7), (0.025, 0.5)],
    "120": [(0.000, 0.5), (0.005, 0.5), (0.008, 1.0), (0.018, 1.0), (0.025, 0.7), (0.030, 0.5)],
    "240": [(0.000, 0.5), (0.006, 0.5), (0.010, 1.0), (0.020, 1.0), (0.030, 0.7), (0.035, 0.5)],
    "default": [(0.000, 0.5), (0.004, 0.5), (0.007, 1.0), (0.015, 1.0), (0.020, 0.7), (0.025, 0.5)],
}

MIN_ACCEPTABLE_FILL_PCT: float = 0.60
RESIDUAL_ACTION: str = "CLOSE"

# M-1: Dynamic Margin Utilization Policy
MARGIN_UTILIZATION_POLICY = {
    "warning_pct": 50.0,
    "halt_pct": 70.0,
    "emergency_pct": 85.0,
    "hard_cap_pct": 90.0,
}

# M-2: Pre-Trade Risk Checklist Policy
PRE_TRADE_POLICY = {
    "max_leverage": 25.0,
    "min_leverage": 1.0,
    "min_position_usd": 0.20,
    "min_notional_usd": 1.0,
    "max_var_pct": 5.0,
    "max_heat_pct": 300.0,
    "max_stress_loss_pct": 25.0,
    "max_correlation": 0.70,
}

# M-3: Portfolio Correlation Lookback Window by Timeframe
CORRELATION_WINDOW_CONFIG = {
    "15": 20,
    "30": 20,
    "60": 24,
    "120": 30,
    "240": 40,
    "default": 20,
}

# H-1: Timeframe-Adaptive Structural Resistance Lookback
STRUCTURE_LOOKBACK_CONFIG = {
    "15": 16,
    "30": 20,
    "60": 24,
    "120": 30,
    "240": 40,
    "default": 20,
}

# Fee Constants
TAKER_FEE_PCT = 0.00055
MAKER_FEE_PCT = 0.00020

# Scale-Out Configuration
SCALE_OUT_CONFIG = {
    "atr_trigger_mult": 1.0,
    "position_portion": 0.50,
}

# Fibonacci Step-Lock Levels (progress threshold -> locked profit portion)
FIBONACCI_STEP_LOCKS = {
    0.618: 0.55,
    0.50: 0.40,
    0.382: 0.25
}

# Micro-Trading Run Risk Bounds & Circuit Breakers
MIN_POSITION_BALANCE_FRAC = 0.02
MAX_POSITION_BALANCE_FRAC = 0.05
CVAR_TAIL_PERCENTILE = 0.05
CVAR_FALLBACK = 0.03
DAILY_LOSS_BUDGET_FRAC = 0.05
MAX_CONCURRENT_POSITIONS = int(os.environ.get("MAX_CONCURRENT_POSITIONS", "3"))
MAX_LIVE_TRADES_CAP = int(os.environ.get("MAX_LIVE_TRADES_CAP", "60"))
MAX_LIVE_LOSS_CAP = float(os.environ.get("MAX_LIVE_LOSS_CAP", "15.0"))



