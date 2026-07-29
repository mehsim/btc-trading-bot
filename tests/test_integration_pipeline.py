"""
tests/test_integration_pipeline.py
-----------------------------------
Integration tests covering automated database disaster recovery backups,
structured JSON logging, and pre-flight trade structure validation.
"""

import os
import pytest
import database
import logger
from trade_calculators import validate_trade_structure


def test_database_online_backup(tmp_path):
    dest_backup = os.path.join(tmp_path, "test_backup.db")
    success = database.backup_database(dest_path=dest_backup)
    assert success is True
    assert os.path.exists(dest_backup)
    assert os.path.getsize(dest_backup) > 0


def test_structured_logger(caplog):
    with caplog.at_level("INFO"):
        logger.log_event("info", "Test integration log event", correlation_id="test1234")
        assert "Test integration log event" in caplog.text


def test_trade_structure_validation():
    is_valid, struct, logs = validate_trade_structure(
        entry_price=100.0,
        stop_price=98.0,
        tp_price=105.0,
        atr_dollars=1.0,
        leverage=5.0,
        interval="15m",
        symbol="BTCUSDT",
        direction="Bullish"
    )
    assert is_valid is True
    assert struct["stop_price"] <= 99.25 or struct["leverage"] <= 10.0
