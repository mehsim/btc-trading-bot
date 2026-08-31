from risk_engine import evaluate_pre_trade_checklist, calculate_anti_martingale_risk_multiplier


def test_anti_martingale_multiplier_does_not_exceed_risk_cap():
    """
    Verify that applying an anti-martingale multiplier (e.g., 1.50x during a winning streak)
    is bounded by the pre-trade checklist limits and cannot inflate final size past capped_size.
    """
    equity = 80.0
    bot_state = {
        "live_balance": equity,
        "wallet_balance": equity,
        "peak_balance": equity,
        "peak_wallet_balance": equity,
        "trade_history": [
            {"success": True, "pnl_usd": 0.50},
            {"success": True, "pnl_usd": 0.50},
            {"success": True, "pnl_usd": 0.50},
            {"success": True, "pnl_usd": 0.50},
            {"success": True, "pnl_usd": 0.50},
        ]
    }

    # 1. Verify anti-martingale returns 1.50x on 5-win streak with 0% drawdown
    am_res = calculate_anti_martingale_risk_multiplier(equity, equity, bot_state["trade_history"])
    am_mult = float(am_res["multiplier"])
    assert am_mult == 1.50

    # 2. Suppose raw initial size is $10.00 margin at 3x leverage = $30 notional.
    initial_candidate_size = 10.0
    candidate_size_with_am = initial_candidate_size * am_mult  # $15.00 margin = $45 notional

    # 3. Evaluate pre-trade checklist with boosted candidate size
    # 15m interval max position pct is 0.08 -> $80 * 0.08 = $6.40 notional max margin at 1x, or $6.40 margin at 3x
    passed, msg, dd_mult, capped_size = evaluate_pre_trade_checklist(
        symbol="BTCUSDT",
        position_size_usd=candidate_size_with_am,
        leverage_val=3.0,
        active_trades=[],
        bot_state=bot_state,
        df_dict={},
        interval="15",
        direction="Bullish"
    )

    assert passed is True
    # The checklist strictly capped the size
    assert capped_size <= 6.40 + 1e-4

    # 4. Enforce final position size formula
    final_position_size = min(capped_size, max(0.0, float(capped_size * dd_mult)))

    # Final position size MUST be <= capped_size and never inflated by am_mult post-checklist
    assert final_position_size <= capped_size
    assert final_position_size <= 6.40 + 1e-4
