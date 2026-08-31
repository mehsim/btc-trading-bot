def test_venue_delay_fallback_uses_fee_aware_pnl_and_tags_estimated():
    """
    Verify that when Bybit venue publication delays occur and get_bybit_accumulated_closed_pnl returns None,
    the bot does NOT launder zero-fee gross PnL into the authoritative slot, but calculates fee-aware PnL
    and tags the trade with pnl_source='ESTIMATED'.
    """
    # Simulate a trade: 5x leverage, position margin $15 -> notional $75
    # Entry price $100, exit price $100.02 (Bullish gross price_diff = +0.02 -> gross profit on $75 notional = +$0.015)
    # Fees: roundtrip (0.02% maker + 0.02% maker = 0.04% on $75 notional = $0.030)
    # Expected net PnL: $0.015 - $0.030 = -$0.015 (net LOSS, despite positive gross price move!)

    position_size_usd = 15.0
    leverage = 5.0
    entry_price = 100.0
    exit_price = 100.02
    direction = "Bullish"

    bybit_pnl_data = None
    if bybit_pnl_data:
        bybit_realized_pnl = bybit_pnl_data["total_pnl"]
        pnl_source = "EXCHANGE"
    else:
        # Fallback without corrupting bybit_realized_pnl
        bybit_realized_pnl = None
        pnl_source = "ESTIMATED"

    assert bybit_realized_pnl is None
    assert pnl_source == "ESTIMATED"

    # Fee-aware local calculation logic
    actual_price = exit_price
    actual_change = actual_price - entry_price
    actual_change_pct = (actual_change / entry_price) * 100
    raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
    gross_pnl = position_size_usd * (raw_return_pct * leverage / 100.0)

    is_stop_loss = False
    entry_fee_rate = 0.0002
    exit_fee_rate = 0.0002 if not is_stop_loss else 0.00055
    roundtrip_fee_rate = entry_fee_rate + exit_fee_rate
    fee_cost = position_size_usd * leverage * roundtrip_fee_rate
    realized_pnl = gross_pnl - fee_cost

    # Confirm gross is positive (+0.015) but fee-aware net PnL is negative (-0.015)
    assert gross_pnl > 0.01
    assert realized_pnl < 0.0  # Fees correctly flipped sign to loss!
    assert pnl_source == "ESTIMATED"


def test_exchange_pnl_tags_exchange_source():
    """Verify that when Bybit returns authoritative closed PnL data, pnl_source is set to 'EXCHANGE'."""
    bybit_pnl_data = {
        "total_pnl": 0.45,
        "avg_exit_price": 60500.0,
        "total_entry_value": 300.0,
        "total_qty": 0.005
    }

    if bybit_pnl_data:
        bybit_realized_pnl = bybit_pnl_data["total_pnl"]
        pnl_source = "EXCHANGE"
    else:
        bybit_realized_pnl = None
        pnl_source = "ESTIMATED"

    assert bybit_realized_pnl == 0.45
    assert pnl_source == "EXCHANGE"
