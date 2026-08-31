import pytest
from collections.abc import Mapping, MutableMapping
from state_manager import StateManager, state_manager
from risk_engine import evaluate_pre_trade_checklist


def test_state_manager_is_mutable_mapping():
    """Verify StateManager implements collections.abc.MutableMapping."""
    assert isinstance(state_manager, MutableMapping)
    assert isinstance(state_manager, Mapping)

    sm = StateManager()
    assert isinstance(sm, MutableMapping)
    assert isinstance(sm, Mapping)

    # Test dictionary-like operations
    sm["test_key"] = 123
    assert sm["test_key"] == 123
    assert sm.get("test_key") == 123
    assert "test_key" in sm
    assert len(sm) >= 1
    del sm["test_key"]
    assert "test_key" not in sm


def test_evaluate_pre_trade_checklist_preserves_state_manager():
    """Verify evaluate_pre_trade_checklist accurately consumes StateManager singleton."""
    sm = StateManager()
    sm["live_balance"] = 500.0
    sm["peak_balance"] = 500.0
    sm["circuit_breaker_active"] = False

    approved, reason, dd_mult, _ = evaluate_pre_trade_checklist(
        symbol="BTCUSDT",
        interval="60",
        direction="Bullish",
        position_size_usd=10.0,
        leverage_val=2.0,
        bot_state=sm,
        active_trades=[],
        df_dict={}
    )
    # Sizing and checklist must use live_balance ($500) rather than failing over to hardcoded $80
    assert dd_mult == 1.0


def test_circuit_breaker_active_veto_with_state_manager():
    """Verify circuit_breaker_active veto fires when StateManager is passed."""
    sm = StateManager()
    sm["live_balance"] = 500.0
    sm["circuit_breaker_active"] = True

    approved, reason, _, _ = evaluate_pre_trade_checklist(
        symbol="BTCUSDT",
        interval="60",
        direction="Bullish",
        position_size_usd=10.0,
        leverage_val=2.0,
        bot_state=sm,
        active_trades=[],
        df_dict={}
    )
    assert approved is False
    assert "Daily Drawdown Circuit Breaker is active" in reason


def test_drawdown_halt_with_state_manager():
    """Verify 20%+ drawdown from peak in StateManager halts trading."""
    sm = StateManager()
    sm["live_balance"] = 60.0    # 70% drawdown from $200
    sm["peak_balance"] = 200.0
    sm["circuit_breaker_active"] = False

    approved, reason, dd_mult, _ = evaluate_pre_trade_checklist(
        symbol="BTCUSDT",
        interval="60",
        direction="Bullish",
        position_size_usd=5.0,
        leverage_val=2.0,
        bot_state=sm,
        active_trades=[],
        df_dict={}
    )
    assert approved is False
    assert "Circuit breaker active (>=20% Drawdown)" in reason
    assert dd_mult == 0.0


def test_missing_or_zero_equity_fails_closed():
    """Verify missing or zero equity fails closed."""
    sm = StateManager()
    sm["live_balance"] = 0.0
    sm["wallet_balance"] = 0.0
    sm["simulated_balance"] = 0.0
    sm["balance"] = 0.0

    approved, reason, _, _ = evaluate_pre_trade_checklist(
        symbol="BTCUSDT",
        interval="60",
        direction="Bullish",
        position_size_usd=5.0,
        leverage_val=2.0,
        bot_state=sm,
        active_trades=[],
        df_dict={}
    )
    assert approved is False
    assert "Fail-Closed" in reason
