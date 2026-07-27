import pytest
from input_validator import sanitize_symbol, validate_price, validate_quantity

def test_sanitize_symbol():
    """Verify symbol sanitization cleans messy inputs."""
    assert sanitize_symbol("btcusdt") == "BTCUSDT"
    assert sanitize_symbol("  sol / usdt  ") == "SOLUSDT"
    assert sanitize_symbol("eth") == "ETHUSDT"
    assert sanitize_symbol("ADAUSDT") == "ADAUSDT"
    assert sanitize_symbol(None) == "BTCUSDT"

def test_validate_price():
    """Verify validate_price handles valid, negative, and invalid prices."""
    assert validate_price(50000.5) == 50000.5
    assert validate_price(-100.0, default_price=0.0) == 0.0
    assert validate_price("invalid", default_price=10.0) == 10.0

def test_validate_quantity():
    """Verify validate_quantity bounds invalid quantities."""
    assert validate_quantity(1.5) == 1.5
    assert validate_quantity(0.0, min_qty=0.001) == 0.001
    assert validate_quantity(-5.0, min_qty=0.001) == 0.001
    assert validate_quantity("invalid", min_qty=0.001) == 0.001
