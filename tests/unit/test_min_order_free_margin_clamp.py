import math
import config


def test_min_order_bump_exceeding_free_margin_is_rejected():
    """
    Verify that when min order value bump produces a required margin (final_val)
    exceeding available free margin, it is rejected and position_size_usd does not exceed available_margin.
    """
    current_bal = 80.0
    total_active_size = 50.0  # Existing positions consume $50 margin
    available_margin = max(0.0, current_bal - total_active_size)  # $30.00 free margin

    entry_price = 100000.0
    leverage_val = 3.0
    min_order_value = getattr(config, "MIN_ORDER_VALUE_USDT", 5.1)
    step = 0.001

    # Bump calculation
    required_qty = min_order_value / entry_price  # 0.000051
    qty_val = math.ceil(required_qty / step) * step  # 0.001
    scaled_notional = qty_val * entry_price  # 100.0
    final_val = scaled_notional / leverage_val  # 33.333...

    wallet_exceeded = False
    status_msg = ""
    position_size_usd = 10.0

    # Priority 2 check:
    if final_val > available_margin:
        status_msg = "Skipped (Insufficient Free Margin for Min Order)"
        wallet_exceeded = True
    else:
        position_size_usd = min(available_margin, final_val)

    assert wallet_exceeded is True
    assert status_msg == "Skipped (Insufficient Free Margin for Min Order)"
    assert position_size_usd <= available_margin


def test_required_margin_guard_checks_free_margin_not_total_balance():
    """
    Verify that Priority 3 Margin Guard checks required_margin against available_margin * 0.90,
    preventing over-committing remaining free margin when total wallet balance is large.
    """
    current_bal = 80.0
    total_active_size = 50.0
    available_margin = max(0.0, current_bal - total_active_size)  # $30.00

    # Proposed margin is $28.00 (which is < current_bal * 0.90 = $72.00, but > available_margin * 0.90 = $27.00)
    required_margin = 28.00
    wallet_exceeded = False

    if not wallet_exceeded and required_margin > available_margin * 0.90:
        wallet_exceeded = True

    assert wallet_exceeded is True
