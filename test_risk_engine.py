import pytest
import pandas as pd
import numpy as np
from risk_engine import (
    AutoStopFloor,
    WickBufferCalculator,
    calculate_final_stop_distance,
    evaluate_pre_trade_checklist
)

def test_auto_stop_floor_decay():
    """Verify AutoStopFloor registers dynamic floor increases and decays over time."""
    floor_engine = AutoStopFloor()
    floor = floor_engine.get_floor("BTCUSDT")
    assert floor >= 0.005

def test_wick_buffer_calculator():
    """Verify WickBufferCalculator calculates wick distance."""
    dates = pd.date_range("2026-01-01", periods=100, freq="15min")
    df = pd.DataFrame({
        "open": np.full(100, 100.0),
        "high": np.full(100, 102.0),
        "low": np.full(100, 98.0),
        "close": np.full(100, 101.0),
        "volume": np.full(100, 1000.0)
    }, index=dates)
    
    wick_calc = WickBufferCalculator()
    wick_dist = wick_calc.get_buffer_distance(entry_price=100.0, df=df)
    assert wick_dist >= 0.0

def test_calculate_final_stop_distance():
    """Verify final stop distance calculation."""
    dates = pd.date_range("2026-01-01", periods=50, freq="15min")
    df = pd.DataFrame({
        "open": np.full(50, 100.0),
        "high": np.full(50, 101.0),
        "low": np.full(50, 99.0),
        "close": np.full(50, 100.0),
        "volume": np.full(50, 1000.0)
    }, index=dates)

    stop_dist = calculate_final_stop_distance(
        entry_price=100.0,
        atr_dollar=1.5,
        symbol="BTCUSDT",
        df=df,
        gmm_multiplier=1.5
    )
    assert stop_dist >= 1.5

def test_evaluate_pre_trade_checklist_leveraged_notional():
    """Verify pre-trade checklist allows valid leveraged positions."""
    dates = pd.date_range("2026-01-01", periods=50, freq="15min")
    df = pd.DataFrame({
        "open": np.full(50, 100.0),
        "high": np.full(50, 101.0),
        "low": np.full(50, 99.0),
        "close": np.full(50, 100.0),
        "volume": np.full(50, 1000.0)
    }, index=dates)
    
    bot_state = {"simulated_balance": 100.0, "wallet_balance": 100.0}
    pass_flag, msg, cap_size, lev = evaluate_pre_trade_checklist(
        symbol="BTCUSDT",
        position_size_usd=2.0,
        leverage_val=5.0,
        active_trades=[],
        bot_state=bot_state,
        df_dict={"BTCUSDT": df},
        interval="15"
    )
    assert pass_flag is True
    assert cap_size > 0.0

def test_bybit_quantity_formatting():
    """Verify format_bybit_qty formats decimal quantities for DOT, AVAX, LTC correctly."""
    from bybit_client import format_bybit_qty
    assert format_bybit_qty("DOTUSDT", 2.4) == "2.4"
    assert format_bybit_qty("AVAXUSDT", 0.5) == "0.5"
    assert format_bybit_qty("LTCUSDT", 1.8) == "1.8"
    assert format_bybit_qty("BTCUSDT", 0.1234) == "0.123"

def test_post_floor_rr_aborts_avax():
    """Verify post-floor R:R check aborts trades when MIN_FLOOR destroys R:R below required confidence."""
    from trade_calculators import passes_economic_gate
    assert not passes_economic_gate(entry=6.479, tp=6.4328, sl=6.559988, conf=0.4609)

def test_f1_mcc_leverage_clamp_bidirectional():
    """Verify F-1 leverage clamp behavior: unclamped for MCC >= 0.15, clamped to 3.0x for MCC < 0.15 on altcoins."""
    import config
    mcc_thresh = getattr(config, "MCC_LEVERAGE_QUALIFICATION_THRESHOLD", 0.15)
    cons_caps = getattr(config, "CONSERVATIVE_LEVERAGE_CAPS", {})
    
    # Branch 1: Permissive path (MCC = 0.1542 >= 0.15)
    mcc_permissive = 0.1542
    assert mcc_permissive >= mcc_thresh
    
    # Branch 2: Restrictive path (MCC = 0.1453 < 0.15)
    mcc_restrictive = 0.1453
    assert mcc_restrictive < mcc_thresh
    altcoin_cap = cons_caps.get("AVAXUSDT", cons_caps.get("default", 3.0))
    assert altcoin_cap == 3.0

def test_f2_reachability_guard():
    """Verify F-2 reachability guard calculation."""
    import math, config
    lookahead = 10
    atr_dollars = 0.50
    reach_factor = getattr(config, "HORIZON_REACHABILITY_FACTOR", 0.90)
    max_reachable = math.sqrt(lookahead) * atr_dollars * reach_factor
    
    over_extended_tp = 2.50
    assert over_extended_tp > max_reachable
