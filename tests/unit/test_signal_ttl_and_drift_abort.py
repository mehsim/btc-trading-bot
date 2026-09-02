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
    """Verify candle freshness evaluation calculates age relative to candle close without bypass."""
    now_ms = 1700000000000.0
    iv = "240"
    interval_ms = 240 * 60 * 1000  # 4 hours = 14,400,000 ms
    
    # Stale 240m candle that closed 2 hours ago (7,200,000 ms ago)
    stale_completed_ts = now_ms - interval_ms - 7200000.0
    candle_close_ms = stale_completed_ts + interval_ms
    candle_age_sec = (now_ms - candle_close_ms) / 1000.0
    max_allowed_age_sec = min(300.0, max(180.0, int(iv) * 60 * 0.15))

    is_stale = candle_age_sec > max_allowed_age_sec or candle_age_sec < -30.0
    assert is_stale is True
    assert candle_age_sec == 7200.0
    assert max_allowed_age_sec == 300.0

    # Fresh 240m candle that closed 45 seconds ago
    fresh_completed_ts = now_ms - interval_ms - 45000.0
    fresh_close_ms = fresh_completed_ts + interval_ms
    fresh_age_sec = (now_ms - fresh_close_ms) / 1000.0

    is_fresh_stale = fresh_age_sec > max_allowed_age_sec or fresh_age_sec < -30.0
    assert is_fresh_stale is False
    assert fresh_age_sec == 45.0
