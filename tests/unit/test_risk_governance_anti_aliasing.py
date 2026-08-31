import pytest
import config
from risk_limits import (
    HARD_TIMEFRAME_MAX_LEVERAGE_CAPS,
    HARD_MAX_WALLET_MARGIN_UTILIZATION_PCT,
    HARD_MAX_SYMBOL_EXPOSURE_PCT,
    HARD_MAX_DRAWDOWN_HALT_PCT,
    HARD_MAX_RISK_PER_TRADE_PCT,
    assert_risk_governance_invariants
)
from risk_engine import calculate_drawdown_multiplier, JointRiskBudgetAllocator


def test_anti_aliasing_identity_and_independence():
    """Ensure config.TIMEFRAME_MAX_LEVERAGE_CAPS is an independent dict and not aliased to HARD_TIMEFRAME_MAX_LEVERAGE_CAPS."""
    assert config.TIMEFRAME_MAX_LEVERAGE_CAPS is not HARD_TIMEFRAME_MAX_LEVERAGE_CAPS

    # Startup assertion must pass on default config
    assert assert_risk_governance_invariants(config) is True


def test_aliasing_attempt_raises_permission_error():
    """Attempting to alias config.TIMEFRAME_MAX_LEVERAGE_CAPS directly to HARD_TIMEFRAME_MAX_LEVERAGE_CAPS must be rejected."""
    class AliasedConfig:
        TIMEFRAME_MAX_LEVERAGE_CAPS = HARD_TIMEFRAME_MAX_LEVERAGE_CAPS
        MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.90
        MAX_SYMBOL_EXPOSURE_PCT = 0.20
        MAX_DRAWDOWN_HALT_PCT = 0.20
        MAX_RISK_PER_TRADE_PCT = 0.03

    with pytest.raises(PermissionError, match="must be an independent config dictionary"):
        assert_risk_governance_invariants(AliasedConfig)


def test_leverage_breach_detected():
    """Exceeding any timeframe leverage cap raises PermissionError."""
    class BreachedLeverageConfig:
        TIMEFRAME_MAX_LEVERAGE_CAPS = {
            "5": 10.0,
            "15": 10.0,
            "30": 10.0,
            "60": 5.0,
            "120": 5.0,
            "240": 20.0,  # Hard cap is 3.0x
            "360": 3.0
        }
        MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.90
        MAX_SYMBOL_EXPOSURE_PCT = 0.20
        MAX_DRAWDOWN_HALT_PCT = 0.20
        MAX_RISK_PER_TRADE_PCT = 0.03

    with pytest.raises(PermissionError, match="Leverage cap for timeframe 240m"):
        assert_risk_governance_invariants(BreachedLeverageConfig)


def test_wallet_margin_breach_detected():
    """Exceeding wallet margin utilization raises PermissionError."""
    class BreachedWalletConfig:
        TIMEFRAME_MAX_LEVERAGE_CAPS = dict(HARD_TIMEFRAME_MAX_LEVERAGE_CAPS)
        MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.98  # Hard cap is 0.90
        MAX_SYMBOL_EXPOSURE_PCT = 0.20
        MAX_DRAWDOWN_HALT_PCT = 0.20
        MAX_RISK_PER_TRADE_PCT = 0.03

    with pytest.raises(PermissionError, match="Wallet margin utilization limit"):
        assert_risk_governance_invariants(BreachedWalletConfig)


def test_symbol_exposure_breach_detected():
    """Exceeding symbol exposure raises PermissionError."""
    class BreachedSymbolConfig:
        TIMEFRAME_MAX_LEVERAGE_CAPS = dict(HARD_TIMEFRAME_MAX_LEVERAGE_CAPS)
        MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.90
        MAX_SYMBOL_EXPOSURE_PCT = 0.50  # Hard cap is 0.20
        MAX_DRAWDOWN_HALT_PCT = 0.20
        MAX_RISK_PER_TRADE_PCT = 0.03

    with pytest.raises(PermissionError, match="Symbol exposure limit"):
        assert_risk_governance_invariants(BreachedSymbolConfig)


def test_drawdown_halt_breach_detected():
    """Exceeding max drawdown halt raises PermissionError."""
    class BreachedDrawdownConfig:
        TIMEFRAME_MAX_LEVERAGE_CAPS = dict(HARD_TIMEFRAME_MAX_LEVERAGE_CAPS)
        MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.90
        MAX_SYMBOL_EXPOSURE_PCT = 0.20
        MAX_DRAWDOWN_HALT_PCT = 0.40  # Hard cap is 0.20
        MAX_RISK_PER_TRADE_PCT = 0.03

    with pytest.raises(PermissionError, match="Drawdown halt threshold"):
        assert_risk_governance_invariants(BreachedDrawdownConfig)


def test_risk_per_trade_breach_detected():
    """Exceeding max risk per trade raises PermissionError."""
    class BreachedRiskConfig:
        TIMEFRAME_MAX_LEVERAGE_CAPS = dict(HARD_TIMEFRAME_MAX_LEVERAGE_CAPS)
        MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.90
        MAX_SYMBOL_EXPOSURE_PCT = 0.20
        MAX_DRAWDOWN_HALT_PCT = 0.20
        MAX_RISK_PER_TRADE_PCT = 0.15  # Hard cap is 0.03

    with pytest.raises(PermissionError, match="Max per-trade risk limit"):
        assert_risk_governance_invariants(BreachedRiskConfig)


def test_max_position_balance_frac_breach_detected():
    """Setting MAX_POSITION_BALANCE_FRAC > HARD_MAX_RISK_PER_TRADE_PCT raises PermissionError."""
    class BreachedPosFracConfig:
        TIMEFRAME_MAX_LEVERAGE_CAPS = dict(HARD_TIMEFRAME_MAX_LEVERAGE_CAPS)
        MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.90
        MAX_SYMBOL_EXPOSURE_PCT = 0.20
        MAX_DRAWDOWN_HALT_PCT = 0.20
        MAX_RISK_PER_TRADE_PCT = 0.03
        MAX_POSITION_BALANCE_FRAC = 0.05  # Breaches 0.03 hard limit

    with pytest.raises(PermissionError, match="MAX_POSITION_BALANCE_FRAC"):
        assert_risk_governance_invariants(BreachedPosFracConfig)


def test_risk_engine_imports_symbol_exposure_from_config():
    """Verify risk_engine.py uses config.MAX_SYMBOL_EXPOSURE_PCT."""
    import risk_engine
    assert risk_engine.MAX_SYMBOL_EXPOSURE_PCT == config.MAX_SYMBOL_EXPOSURE_PCT


def test_runtime_drawdown_and_trade_risk_enforcement():
    """Verify runtime calculate_drawdown_multiplier and sizing honor hard bounds."""
    # At or above HARD_MAX_DRAWDOWN_HALT_PCT (20%), multiplier must be 0.0 (halt)
    assert calculate_drawdown_multiplier(80.0, 100.0) == 0.0
    assert calculate_drawdown_multiplier(70.0, 100.0) == 0.0
    assert calculate_drawdown_multiplier(95.0, 100.0) > 0.0

    # Sizing capital_at_risk cannot exceed HARD_MAX_RISK_PER_TRADE_PCT * total_equity
    allocator = JointRiskBudgetAllocator()
    total_eq = 1000.0
    res = allocator.allocate_risk_budget(
        symbol="BTCUSDT",
        entry_price=50000.0,
        atr_dollars=1000.0,
        atr_norm=0.02,
        calibrated_confidence=0.99,  # High confidence
        direction="BUY",
        total_equity=total_eq,
        portfolio_heat=0.05,
        mhi_score=95.0,
        top_book_depth_usd=100000.0
    )
    max_allowed_risk = total_eq * HARD_MAX_RISK_PER_TRADE_PCT
    assert res["capital_at_risk"] <= max_allowed_risk + 1e-4
