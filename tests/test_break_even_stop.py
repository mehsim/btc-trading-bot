import pytest
from trade_calculators import calculate_break_even_stop

def test_long_break_even_sl_above_entry():
    entry_price = 100.0
    current_price = 102.0
    sl = calculate_break_even_stop("Bullish", entry_price, current_price)
    assert sl > entry_price, f"Long break-even SL {sl} must be strictly greater than entry {entry_price}"
    assert sl == 100.175, f"Expected 100.175, got {sl}"

def test_short_break_even_sl_below_entry():
    entry_price = 100.0
    current_price = 98.0
    sl = calculate_break_even_stop("Bearish", entry_price, current_price)
    assert sl < entry_price, f"Short break-even SL {sl} must be strictly less than entry {entry_price}"
    assert sl == 99.825, f"Expected 99.825, got {sl}"

def test_direction_variants():
    assert calculate_break_even_stop("Long", 100.0) > 100.0
    assert calculate_break_even_stop("BUY", 100.0) > 100.0
    assert calculate_break_even_stop("Short", 100.0) < 100.0
    assert calculate_break_even_stop("SELL", 100.0) < 100.0

def test_immediate_market_trigger_guard_long():
    entry_price = 100.0
    current_price = 100.10  # Less than normal break-even cost buffer (100.175)
    sl = calculate_break_even_stop("Bullish", entry_price, current_price)
    assert sl < current_price, f"Long SL {sl} must be below current market price {current_price} to prevent immediate trigger"
    assert sl > entry_price, f"Long SL {sl} must still be above entry {entry_price}"

def test_immediate_market_trigger_guard_short():
    entry_price = 100.0
    current_price = 99.90  # Price moved down slightly, but less than 99.825
    sl = calculate_break_even_stop("Bearish", entry_price, current_price)
    assert sl > current_price, f"Short SL {sl} must be above current market price {current_price} to prevent immediate trigger"
    assert sl < entry_price, f"Short SL {sl} must still be below entry {entry_price}"
