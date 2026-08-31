import pytest
from trade_calculators import (
    REALIZED_RR_HAIRCUT,
    DEFAULT_EMPIRICAL_REALIZED_RR_HAIRCUT,
    estimate_empirical_realized_rr,
    get_realized_rr_haircut,
    calculate_required_p,
    passes_economic_gate
)


def test_empirical_realized_rr_haircut_baseline():
    """Verify REALIZED_RR_HAIRCUT baseline is set to empirical 0.28 instead of falsified 0.80."""
    assert REALIZED_RR_HAIRCUT == 0.28
    assert DEFAULT_EMPIRICAL_REALIZED_RR_HAIRCUT == 0.28


def test_estimate_empirical_realized_rr_from_trades():
    """Verify estimate_empirical_realized_rr calculates mean win / mean loss accurately."""
    # 10 wins averaging +200, 10 losses averaging -100 -> empirical RR = 2.0
    mock_trades = [{"pnl": 200.0, "interval": "60", "regime": "trending"} for _ in range(10)] + \
                  [{"pnl": -100.0, "interval": "60", "regime": "trending"} for _ in range(10)]

    rr = estimate_empirical_realized_rr(closed_trades=mock_trades, min_samples=20)
    assert rr is not None
    assert round(rr, 2) == 2.00


def test_get_realized_rr_haircut_dynamic_vs_default():
    """Verify get_realized_rr_haircut dynamically scales with empirical data or falls back to 0.28."""
    # When no trades are available, falls back to conservative default 0.28
    haircut_default = get_realized_rr_haircut(interval="60", regime="trending", nominal_rr=2.24, closed_trades=[])
    assert haircut_default == 0.28

    # When empirical trades yield empirical RR = 0.632 and nominal RR is 2.24
    mock_trades = [{"pnl": 63.2, "interval": "60", "regime": "trending"} for _ in range(10)] + \
                  [{"pnl": -100.0, "interval": "60", "regime": "trending"} for _ in range(10)]
    haircut_emp = get_realized_rr_haircut(interval="60", regime="trending", nominal_rr=2.24, closed_trades=mock_trades)
    # empirical_rr = 0.632, nominal_rr = 2.24 -> haircut = 0.632 / 2.24 = 0.282
    assert 0.27 <= haircut_emp <= 0.29


def test_economic_gate_rejects_sub_breakeven_signal():
    """Verify calculate_required_p requires ~61% break-even with 0.28 haircut on 2.24 R:R geometry."""
    entry = 50000.0
    tp = 51474.7   # +2.95% (nominal tp_dist / sl_dist = 2.24)
    sl = 49341.5   # -1.32%

    req_p = calculate_required_p(entry, tp, sl, cost_frac=0.0006, realized_rr_haircut=0.28)
    # With empirical 0.28 haircut, required break-even probability is > 60%
    assert req_p > 0.60

    # A calibrated confidence of 0.40 (which falsely passed under 0.80 haircut) MUST now fail the gate
    assert passes_economic_gate(entry, tp, sl, conf=0.40, cost_frac=0.0006, realized_rr_haircut=0.28) is False
    assert passes_economic_gate(entry, tp, sl, conf=0.65, cost_frac=0.0006, realized_rr_haircut=0.28) is True
