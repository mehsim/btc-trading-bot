import pytest
from exit_manager import compute_dynamic_trail_params, evaluate_trailing_and_break_even

def test_dynamic_trail_params_timeframe_scaling():
    """Verify that shorter timeframes have tighter hurdles than longer timeframes."""
    entry_price = 100.0
    atr_dollars = 2.0
    adx = 25.0
    
    hurdle_15, buf_15 = compute_dynamic_trail_params("15", "15m", entry_price, atr_dollars, adx, "Trending")
    hurdle_60, buf_60 = compute_dynamic_trail_params("60", "1h", entry_price, atr_dollars, adx, "Trending")
    hurdle_240, buf_240 = compute_dynamic_trail_params("240", "4h", entry_price, atr_dollars, adx, "Trending")
    
    assert hurdle_15 < hurdle_60 < hurdle_240
    assert hurdle_15 == 0.60 * atr_dollars
    assert hurdle_60 == 0.90 * atr_dollars
    assert hurdle_240 == 1.20 * atr_dollars

def test_dynamic_trail_params_regime_and_adx_adaptation():
    """Verify that strong trends expand the hurdle to allow riding momentum."""
    entry_price = 100.0
    atr_dollars = 2.0
    
    # Strong Trend (ADX 35) vs Choppy Range (ADX 15)
    hurdle_trend, buf_trend = compute_dynamic_trail_params("60", "1h", entry_price, atr_dollars, 35.0, "Trending")
    hurdle_range, buf_range = compute_dynamic_trail_params("60", "1h", entry_price, atr_dollars, 15.0, "Ranging")
    
    assert hurdle_trend > hurdle_range
    # Buffer should be safer (wider) in chop to prevent noise triggers
    assert buf_range >= buf_trend

def test_evaluate_trailing_bullish_no_early_tightening():
    """Verify that stop loss is NOT prematurely tightened when price is below profit hurdle."""
    entry_price = 1.4800
    atr_dollars = 0.0400
    initial_sl = 1.4200
    
    trade = {
        "symbol": "XRPUSDT",
        "entry_price": entry_price,
        "highest_price": entry_price,
        "stop_loss": initial_sl,
        "atr_dollars": atr_dollars,
        "leverage": 1.0,
        "adx": 22.0,
        "entry_regime": "Trending",
        "break_even_triggered": False
    }
    
    # Price moves slightly up (0.01 ATR) - should NOT trigger trailing
    current_price = 1.4805
    res = evaluate_trailing_and_break_even(
        active_symbol="XRPUSDT",
        iv="240",
        tf="4h",
        direction="Bullish",
        entry_price=entry_price,
        current_price=current_price,
        highest_price=entry_price,
        lowest_price=entry_price,
        stop_loss=initial_sl,
        break_even_triggered=False,
        atr_dollars=atr_dollars,
        position_size_usd=5.0,
        active_trade=trade,
        required_be_dist=0.03,
        trailing_multiplier=1.0,
        update_sl_fn=lambda sym, sl, tr: True,
        trade_mode="simulation"
    )
    
    # SL must stay at original protective initial stop
    assert res["stop_loss"] == initial_sl
    assert res["break_even_triggered"] is False

def test_evaluate_trailing_bullish_triggers_after_hurdle():
    """Verify that stop loss trails properly once profit hurdle is cleared."""
    entry_price = 1.4800
    atr_dollars = 0.0400
    initial_sl = 1.4200
    
    trade = {
        "symbol": "XRPUSDT",
        "entry_price": entry_price,
        "highest_price": entry_price,
        "stop_loss": initial_sl,
        "atr_dollars": atr_dollars,
        "leverage": 1.0,
        "adx": 22.0,
        "entry_regime": "Trending",
        "break_even_triggered": False
    }
    
    # Price surges to 1.5500 (+1.75 ATR profit, clearing 4H 1.2x ATR hurdle)
    current_price = 1.5500
    res = evaluate_trailing_and_break_even(
        active_symbol="XRPUSDT",
        iv="240",
        tf="4h",
        direction="Bullish",
        entry_price=entry_price,
        current_price=current_price,
        highest_price=entry_price,
        lowest_price=entry_price,
        stop_loss=initial_sl,
        break_even_triggered=False,
        atr_dollars=atr_dollars,
        position_size_usd=5.0,
        active_trade=trade,
        required_be_dist=0.03,
        trailing_multiplier=1.0,
        update_sl_fn=lambda sym, sl, tr: True,
        trade_mode="simulation"
    )
    
    # SL must have moved into profit (> entry_price) and locked gains
    assert res["stop_loss"] > entry_price
    assert res["stop_loss"] <= current_price - 0.002

def test_evaluate_trailing_bearish_triggers_after_hurdle():
    """Verify that bearish trailing stop works symmetrically."""
    entry_price = 100.0
    atr_dollars = 2.0
    initial_sl = 104.0
    
    trade = {
        "symbol": "SOLUSDT",
        "entry_price": entry_price,
        "lowest_price": entry_price,
        "stop_loss": initial_sl,
        "atr_dollars": atr_dollars,
        "leverage": 1.0,
        "adx": 25.0,
        "entry_regime": "Trending",
        "break_even_triggered": False
    }
    
    # Price drops to 96.0 (-2.0 ATR profit, clearing 1H 0.9x ATR hurdle)
    current_price = 96.0
    res = evaluate_trailing_and_break_even(
        active_symbol="SOLUSDT",
        iv="60",
        tf="1h",
        direction="Bearish",
        entry_price=entry_price,
        current_price=current_price,
        highest_price=entry_price,
        lowest_price=entry_price,
        stop_loss=initial_sl,
        break_even_triggered=False,
        atr_dollars=atr_dollars,
        position_size_usd=5.0,
        active_trade=trade,
        required_be_dist=1.5,
        trailing_multiplier=1.0,
        update_sl_fn=lambda sym, sl, tr: True,
        trade_mode="simulation"
    )
    
    # SL must have trailed down into profit (< entry_price)
    assert res["stop_loss"] < entry_price
    assert res["stop_loss"] >= current_price + 0.1
