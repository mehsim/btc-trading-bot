import pytest
from unittest.mock import patch
from drift_detector import CUSUMDriftDetector


def test_cusum_state_persistence_and_restoration(tmp_path):
    """
    Verify that CUSUMDriftDetector correctly persists S_high and processed_trade_ids
    to the database (using set_setting) and that a newly instantiated detector restores them.
    """
    db_store = {}

    def mock_set_setting(key, value):
        db_store[key] = str(value)
        return True

    def mock_get_setting(key, default=None):
        return db_store.get(key, default)

    with patch("database.set_setting", side_effect=mock_set_setting), \
         patch("database.get_setting", side_effect=mock_get_setting):

        # 1. Initialize detector #1 and ingest data to build state
        detector1 = CUSUMDriftDetector(threshold_H=5.0)
        detector1.update(actual_outcome=0, predicted_confidence=0.75, trade_id="trade_uuid_001")
        detector1.update(actual_outcome=0, predicted_confidence=0.80, trade_id="trade_uuid_002")

        saved_s_high = detector1.S_high
        saved_trade_ids = set(detector1.processed_trade_ids)

        assert saved_s_high > 0.0, "Expected S_high to accumulate on loss outcomes"
        assert "trade_uuid_001" in saved_trade_ids
        assert "trade_uuid_002" in saved_trade_ids

        # Verify items were written into DB store under expected keys
        assert "cusum_s_high" in db_store
        assert "cusum_processed_trade_ids" in db_store

        # 2. Instantiate detector #2 (simulating service restart)
        detector2 = CUSUMDriftDetector(threshold_H=5.0)

        # Assert detector #2 restored state from DB store
        assert detector2.S_high == pytest.approx(saved_s_high), f"Expected restored S_high {saved_s_high}, got {detector2.S_high}"
        assert detector2.processed_trade_ids == saved_trade_ids, f"Expected restored trade IDs {saved_trade_ids}, got {detector2.processed_trade_ids}"

        # 3. Verify idempotency survives restart: re-submitting trade_uuid_001 on detector #2 does not alter S_high
        prev_s_high = detector2.S_high
        detector2.update(actual_outcome=0, predicted_confidence=0.75, trade_id="trade_uuid_001")
        assert detector2.S_high == prev_s_high, "Re-ingesting trade_uuid_001 should be idempotent"
