from execution_validator import ExecutionValidator


def test_live_price_stop_loss_breach_rejection():
    """
    Verify that if the live market price drops at or below the stop loss for a Long
    (or rises at/above for a Short), ExecutionValidator immediately rejects the order.
    """
    validator = ExecutionValidator()

    # Long Scenario: Candle closed at 62,000, SL at 61,500, TP at 63,000
    # Over computation delay, live price dropped to 61,400 (underwater)
    is_valid, msg = validator.validate_order(
        symbol="BTCUSDT",
        direction="Bullish",
        entry_price=62000.0,
        stop_loss_price=61500.0,
        take_profit_price=63000.0,
        position_size_usd=10.0,
        live_price=61400.0,
        top_book_depth_usd=50000.0,
        portfolio_heat=0.10
    )
    assert is_valid is False
    assert "already at or below Long Stop Loss" in msg

    # Short Scenario: Candle closed at 62,000, SL at 62,500, TP at 61,000
    # Over computation delay, live price spiked to 62,600
    is_valid_short, msg_short = validator.validate_order(
        symbol="BTCUSDT",
        direction="Bearish",
        entry_price=62000.0,
        stop_loss_price=62500.0,
        take_profit_price=61000.0,
        position_size_usd=10.0,
        live_price=62600.0,
        top_book_depth_usd=50000.0,
        portfolio_heat=0.10
    )
    assert is_valid_short is False
    assert "already at or above Short Stop Loss" in msg_short


def test_portfolio_heat_budget_rejection():
    """
    Verify that if portfolio heat meets or exceeds the max heat budget,
    ExecutionValidator rejects the order.
    """
    validator = ExecutionValidator()

    is_valid, msg = validator.validate_order(
        symbol="BTCUSDT",
        direction="Bullish",
        entry_price=62000.0,
        stop_loss_price=61500.0,
        take_profit_price=63000.0,
        position_size_usd=10.0,
        live_price=62050.0,
        top_book_depth_usd=50000.0,
        portfolio_heat=0.85,
        max_portfolio_heat=0.80
    )
    assert is_valid is False
    assert "reaches max budget" in msg


def test_live_price_valid_execution():
    """
    Verify that when live price is healthy and heat is within budget, order is APPROVED.
    """
    validator = ExecutionValidator()

    is_valid, msg = validator.validate_order(
        symbol="BTCUSDT",
        direction="Bullish",
        entry_price=62000.0,
        stop_loss_price=61500.0,
        take_profit_price=63000.0,
        position_size_usd=10.0,
        live_price=62050.0,
        top_book_depth_usd=100000.0,
        portfolio_heat=0.25,
        max_portfolio_heat=0.80
    )
    assert is_valid is True
    assert msg == "VALID"
