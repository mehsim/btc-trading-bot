import pytest
import time
from unittest.mock import patch, MagicMock
import pandas as pd
import main


def test_signal_ttl_expired_aborts_execution():
    """Verify that _execute_bybit_trade_async_inner aborts immediately when signal TTL is exceeded."""
    with patch("main.place_bybit_limit_order") as mock_limit, \
         patch("main.place_bybit_taker_ioc_order") as mock_ioc, \
         patch("main.send_telegram_alert") as mock_alert, \
         patch("main.log_event") as mock_log:

        # Decision made 200 seconds ago on a 15m timeframe (TTL is min(120, max(30, 15*60*0.1)) = 90s)
        stale_decision_ts = time.time() - 200.0

        main._execute_bybit_trade_async_inner(
            symbol="BTCUSDT",
            iv="15",
            tf="15m",
            ml_trend="Bullish",
            leverage_val=3.0,
            qty_str="0.01",
            raw_qty=0.01,
            entry_price=60000.0,
            stop_loss_price=59500.0,
            take_profit_price=61000.0,
            position_size_usd=200.0,
            kelly_fraction=0.1,
            calibrated_confidence=0.65,
            ml_confidence=0.65,
            dynamic_conf_threshold=0.55,
            latest_completed_ts=int(time.time() * 1000),
            latest_candle={"close": 60000.0, "ATR_norm": 0.01},
            pred_change=500.0,
            predicted_price=60500.0,
            atr_dollars=600.0,
            tp_multiplier_adjusted=1.67,
            sl_multiplier_adjusted=0.83,
            df_completed=pd.DataFrame({"ATR_norm": [0.01] * 35}),
            trade_uuid="test_ttl_uuid",
            duration_seconds=900,
            active_trade_key="active_trade_15m",
            decision_ts=stale_decision_ts
        )

        # No order should be placed
        assert not mock_limit.called
        assert not mock_ioc.called
        assert mock_alert.called
        assert any("Signal TTL" in str(c) for c in mock_log.call_args_list)


def test_adverse_price_drift_aborts_bullish_order():
    """Verify that _execute_bybit_trade_async_inner aborts when live market has dropped significantly below entry price."""
    with patch("main.get_all_bybit_positions", return_value=[]), \
         patch("main.get_bybit_bid_ask", return_value=(59800.0, 59810.0, 59805.0)), \
         patch("main.place_bybit_limit_order") as mock_limit, \
         patch("main.place_bybit_taker_ioc_order") as mock_ioc, \
         patch("main.send_telegram_alert") as mock_alert, \
         patch("main.log_event") as mock_log:

        # Entry price was 60000.0 at candle close, ATR is $600. Max adverse drift is 0.25*600 = $150.
        # Live mid is 59805.0 (drift = $195 > $150).
        fresh_decision_ts = time.time()

        main._execute_bybit_trade_async_inner(
            symbol="BTCUSDT",
            iv="15",
            tf="15m",
            ml_trend="Bullish",
            leverage_val=3.0,
            qty_str="0.01",
            raw_qty=0.01,
            entry_price=60000.0,
            stop_loss_price=59500.0,
            take_profit_price=61000.0,
            position_size_usd=200.0,
            kelly_fraction=0.1,
            calibrated_confidence=0.85,
            ml_confidence=0.85,
            dynamic_conf_threshold=0.55,
            latest_completed_ts=int(time.time() * 1000),
            latest_candle={"close": 60000.0, "ATR_norm": 0.01},
            pred_change=500.0,
            predicted_price=60500.0,
            atr_dollars=600.0,
            tp_multiplier_adjusted=1.67,
            sl_multiplier_adjusted=0.83,
            df_completed=pd.DataFrame({"ATR_norm": [0.01] * 35}),
            trade_uuid="test_drift_bullish_uuid",
            duration_seconds=900,
            active_trade_key="active_trade_15m",
            decision_ts=fresh_decision_ts
        )

        assert not mock_limit.called
        assert not mock_ioc.called
        assert mock_alert.called
        assert any("Adverse Drift" in str(c) for c in mock_log.call_args_list)


def test_adverse_price_drift_aborts_bearish_order():
    """Verify that _execute_bybit_trade_async_inner aborts when live market has rallied significantly above entry price for Bearish setup."""
    with patch("main.get_all_bybit_positions", return_value=[]), \
         patch("main.get_bybit_bid_ask", return_value=(60200.0, 60210.0, 60205.0)), \
         patch("main.place_bybit_limit_order") as mock_limit, \
         patch("main.place_bybit_taker_ioc_order") as mock_ioc, \
         patch("main.send_telegram_alert") as mock_alert, \
         patch("main.log_event") as mock_log:

        # Entry price was 60000.0 at candle close, ATR is $600. Max adverse drift is 0.25*600 = $150.
        # Live mid is 60205.0 (drift = $205 > $150).
        fresh_decision_ts = time.time()

        main._execute_bybit_trade_async_inner(
            symbol="BTCUSDT",
            iv="15",
            tf="15m",
            ml_trend="Bearish",
            leverage_val=3.0,
            qty_str="0.01",
            raw_qty=0.01,
            entry_price=60000.0,
            stop_loss_price=60500.0,
            take_profit_price=59000.0,
            position_size_usd=200.0,
            kelly_fraction=0.1,
            calibrated_confidence=0.85,
            ml_confidence=0.85,
            dynamic_conf_threshold=0.55,
            latest_completed_ts=int(time.time() * 1000),
            latest_candle={"close": 60000.0, "ATR_norm": 0.01},
            pred_change=-500.0,
            predicted_price=59500.0,
            atr_dollars=600.0,
            tp_multiplier_adjusted=1.67,
            sl_multiplier_adjusted=0.83,
            df_completed=pd.DataFrame({"ATR_norm": [0.01] * 35}),
            trade_uuid="test_drift_bearish_uuid",
            duration_seconds=900,
            active_trade_key="active_trade_15m",
            decision_ts=fresh_decision_ts
        )

        assert not mock_limit.called
        assert not mock_ioc.called
        assert mock_alert.called
        assert any("Adverse Drift" in str(c) for c in mock_log.call_args_list)


def test_candle_freshness_check_logic():
    """Verify candle freshness evaluation calculates age relative to candle close with production formula."""
    from main import compute_max_allowed_candle_age

    # Verify production thresholds across intervals
    assert compute_max_allowed_candle_age("15") == 300.0
    assert compute_max_allowed_candle_age("60") == 900.0
    assert compute_max_allowed_candle_age("240") == 900.0

    now_ms = 1700000000000.0
    iv = "240"
    interval_ms = 240 * 60 * 1000  # 4 hours = 14,400,000 ms
    
    # Stale 240m candle that closed 2 hours ago (7,200,000 ms ago)
    stale_completed_ts = now_ms - interval_ms - 7200000.0
    candle_close_ms = stale_completed_ts + interval_ms
    candle_age_sec = (now_ms - candle_close_ms) / 1000.0
    # Direct invocation of is_candle_fresh helper
    from main import is_candle_fresh, CANDLE_CLOCK_SKEW_TOLERANCE_SEC
    assert CANDLE_CLOCK_SKEW_TOLERANCE_SEC == -30.0

    fresh_bool, age, max_age = is_candle_fresh(stale_completed_ts, iv, now_ms=now_ms)
    assert fresh_bool is False
    assert age == 7200.0
    assert max_age == 900.0

    # Fresh 240m candle that closed 45 seconds ago
    fresh_completed_ts = now_ms - interval_ms - 45000.0
    fresh_bool, age, max_age = is_candle_fresh(fresh_completed_ts, iv, now_ms=now_ms)
    assert fresh_bool is True
    assert age == 45.0

    # Future candle outside -30s tolerance (clock desync)
    future_completed_ts = now_ms - interval_ms + 45000.0
    fresh_bool, age, max_age = is_candle_fresh(future_completed_ts, iv, now_ms=now_ms)
    assert fresh_bool is False
    assert age == -45.0
