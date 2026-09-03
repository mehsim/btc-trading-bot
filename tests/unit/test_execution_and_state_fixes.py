import pytest
from state_manager import default_json_serializer, StateManager
from trade_calculators import get_realized_rr_haircut, estimate_empirical_realized_rr
from production_regime_engine import production_regime_engine


def test_default_json_serializer_fallback_str():
    """Verify default_json_serializer falls back to str(obj) instead of raising or returning None."""
    class CustomObject:
        def __str__(self):
            return "custom_str_repr"
    obj = CustomObject()
    res = default_json_serializer(obj)
    assert res == "custom_str_repr"


def test_state_manager_bool_truthiness():
    """Verify StateManager truthiness is True without executing Redis KEYS scans."""
    sm = StateManager()
    assert bool(sm) is True


def test_realized_rr_haircut_capped_at_point_six():
    """Verify get_realized_rr_haircut cannot exceed 0.60 even when empirical R:R is high."""
    closed_trades = (
        [{"interval": "15", "regime": "trending", "pnl_usd": 300.0}] * 10 +
        [{"interval": "15", "regime": "trending", "pnl_usd": -50.0}] * 10
    )
    # empirical_rr = 300 / 50 = 6.0, nominal_rr = 1.5 -> ratio = 4.0 -> capped at 0.60
    haircut = get_realized_rr_haircut(
        interval="15", regime="trending", nominal_rr=1.5, closed_trades=closed_trades
    )
    assert haircut == 0.60


def test_estimate_empirical_realized_rr_pnl_usd_and_regime():
    """Verify estimate_empirical_realized_rr recognizes pnl_usd and normalizes regime string."""
    closed_trades = (
        [{"interval": "15", "regime": "Trending_Bullish", "pnl_usd": 200.0}] * 5 +
        [{"interval": "15", "regime": "STRONG_TREND", "pnl_usd": 200.0}] * 5 +
        [{"interval": "15", "regime": "TRENDING", "pnl_usd": -100.0}] * 10
    )
    rr = estimate_empirical_realized_rr(closed_trades=closed_trades, interval="15", regime="trending")
    assert rr is not None
    assert round(rr, 2) == 2.0


def test_high_vol_ranging_normalizes_to_ranging():
    """Verify High Vol, Ranging normalizes to RANGING rather than CHOPPY."""
    reg = production_regime_engine._normalize_regime("High Vol, Ranging")
    assert reg == "RANGING"


def test_is_calibrator_viable_rejects_identity_map_and_flat_isotonic():
    """Verify is_calibrator_viable rejects identity fallback a=1,b=1,c=0 and flat isotonic maps."""
    from tools.beta_calibrator import is_calibrator_viable

    # Identity fallback map
    identity_cal = {
        "scaling_method": "beta_calibration",
        "a": 1.0,
        "b": 1.0,
        "c": 0.0,
        "fitting_sample_size": 15
    }
    assert is_calibrator_viable(identity_cal) is False

    # Flat constant isotonic map
    flat_iso = {
        "scaling_method": "isotonic",
        "X": [0.1, 0.5, 0.9],
        "y": [0.45, 0.45, 0.45]
    }
    assert is_calibrator_viable(flat_iso) is False

    # Dynamic valid isotonic map
    valid_iso = {
        "scaling_method": "isotonic",
        "X": [0.1, 0.5, 0.9],
        "y": [0.30, 0.55, 0.75]
    }
    assert is_calibrator_viable(valid_iso, min_required_p_star=0.60) is True


def test_risk_limits_blocks_reexported_hard_symbol():
    """Verify assert_risk_governance_invariants raises PermissionError if config re-exports hard symbols."""
    import types
    from risk_limits import assert_risk_governance_invariants

    fake_config = types.ModuleType("fake_config")
    fake_config.TIMEFRAME_MAX_LEVERAGE_CAPS = {"15": 5.0, "30": 5.0, "60": 5.0, "120": 5.0, "240": 3.0, "360": 3.0, "5": 10.0}
    fake_config.MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.85
    fake_config.MAX_SYMBOL_EXPOSURE_PCT = 0.15
    fake_config.MAX_DRAWDOWN_HALT_PCT = 0.18
    fake_config.MAX_RISK_PER_TRADE_PCT = 0.025
    fake_config.HARD_MAX_RISK_PER_TRADE_PCT = 0.03

    with pytest.raises(PermissionError, match="must not import or alias 'HARD_MAX_RISK_PER_TRADE_PCT'"):
        assert_risk_governance_invariants(fake_config)
