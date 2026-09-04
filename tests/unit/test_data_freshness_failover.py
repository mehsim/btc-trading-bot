import time
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data import get_history

def test_get_history_freshness_attributes():
    """Verify get_history attaches fetch_ok, last_bar_age_sec, and latest_ts to final_df.attrs."""
    now_ms = time.time() * 1000.0
    mock_candles = [
        [now_ms - 1800000, 50000, 50100, 49900, 50050, 10.0, 500000.0],
        [now_ms - 900000, 50050, 50200, 50000, 50150, 12.0, 600000.0],
        [now_ms, 50150, 50300, 50100, 50250, 15.0, 750000.0]
    ]
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": {"list": mock_candles}}
    
    with patch("data.bybit_public_get", return_value=mock_resp):
        df = get_history(symbol="TEST_FRESHNESS", interval="15", limit=10, pages=1)
        assert df is not None
        assert len(df) > 0
        assert "fetch_ok" in df.attrs
        assert "last_bar_age_sec" in df.attrs
        assert "latest_ts" in df.attrs
        assert df.attrs["fetch_ok"] is True
        assert df.attrs["last_bar_age_sec"] <= 300.0

def test_get_history_warm_cache_failover_to_binance():
    """Verify that when Bybit incremental fetch fails on a warm cache, Binance fallback is triggered."""
    now_ms = time.time() * 1000.0
    # Old cached candles (5 hours old)
    cached_candles = [
        [now_ms - 18000000, 50000, 50100, 49900, 50050, 10.0, 500000.0],
        [now_ms - 14400000, 50050, 50200, 50000, 50150, 12.0, 600000.0],
    ]
    df_cache = pd.DataFrame(cached_candles, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]).astype(float)
    
    # Binance returns fresh candles
    binance_candles = [
        [now_ms - 3600000, 50000, 50100, 49900, 50050, 10.0, 0, 500000.0],
        [now_ms, 50050, 50200, 50000, 50150, 12.0, 0, 600000.0],
    ]
    
    bybit_fail_resp = MagicMock()
    bybit_fail_resp.status_code = 403
    
    binance_ok_resp = MagicMock()
    binance_ok_resp.status_code = 200
    binance_ok_resp.json.return_value = binance_candles
    
    with patch("data.safe_get_sqlite_conn") as mock_db, \
         patch("data.bybit_public_get", return_value=bybit_fail_resp), \
         patch("requests.get", return_value=binance_ok_resp) as mock_requests_get:
        
        # Mock database returning old cache
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = cached_candles
        mock_db.return_value.cursor.return_value = mock_cursor
        
        df = get_history(symbol="BTCUSDT", interval="60", limit=10, pages=1)
        assert df is not None
        assert len(df) > 0
        # Check that Binance fallback was invoked
        assert mock_requests_get.called
        assert df.attrs["fetch_ok"] is True
        assert df["timestamp"].max() >= now_ms - 1000.0
