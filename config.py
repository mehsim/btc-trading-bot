"""
Centralized Configuration System for BTC Trading Bot
Consolidates strategy thresholds, leverage caps, risk parameters, and memory bounds.
"""

# System & Memory Bounds
MAX_TRADE_HISTORY_MEMORY = 1000
MAX_PREDICTION_HISTORY_MEMORY = 1000
MAX_LOG_ENTRIES_MEMORY = 200

# Order Execution Bounds
MIN_ORDER_VALUE_USDT = 5.1
MAX_SCALED_RISK_CAP_RATIO = 1.10  # 110% hard cap on approved risk when order size is scaled up
MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.90 # 90% wallet balance margin limit

# Leverage & Timeframe Caps
TIMEFRAME_MAX_LEVERAGE_CAPS = {
    "5": 10.0,
    "15": 10.0,
    "30": 10.0,
    "60": 5.0,
    "120": 5.0,
    "240": 3.0,
    "360": 3.0
}

INTERVAL_MAX_POSITION_PCT = {
    "5": 0.05,
    "15": 0.08,
    "30": 0.10,
    "60": 0.20,
    "120": 0.20,
    "240": 0.25,
    "360": 0.25
}

MAX_SYMBOL_EXPOSURE_PCT = 0.20  # Max 20% total balance in one symbol across all intervals

# Volatility & Market Regime Thresholds
TARGET_VOLATILITY_ATR = 0.005
PARAMETRIC_VAR_CONFIDENCE_LEVEL = 0.95
PARAMETRIC_VAR_DAILY_EQUITY_CAP_PCT = 0.05 # 5% daily equity VaR limit

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


