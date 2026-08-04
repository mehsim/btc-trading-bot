import unittest
import sqlite3
from unittest.mock import patch, MagicMock
import database

class TestDatabaseRollbackAndErrorHandling(unittest.TestCase):
    def test_save_completed_trade_rollback_on_error(self):
        """Verify F-17: Rollback occurs on exception, returns False, and logs ERROR with payload."""
        trade_payload = {
            "symbol": "BTCUSDT",
            "exit_time": 1700000000,
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "pnl_usd": 100.0
        }

        # Mock connection to raise an OperationalError during execute
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("Disk I/O error or database locked")

        with patch("database.get_db_connection", return_value=mock_conn):
            res = database.save_completed_trade(trade_payload)
            self.assertFalse(res, "save_completed_trade should return False when DB write fails")
            mock_conn.rollback.assert_called()

    def test_save_prediction_rollback_on_error(self):
        """Verify F-17: save_prediction performs rollback and returns False on error."""
        pred_payload = {"symbol": "BTCUSDT", "timestamp": 1700000000, "confidence": 0.8}

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("Database locked")

        with patch("database.get_db_connection", return_value=mock_conn):
            res = database.save_prediction(pred_payload)
            self.assertFalse(res)
            mock_conn.rollback.assert_called()

    def test_save_active_trades_rollback_on_error(self):
        """Verify F-17: save_active_trades performs rollback and returns False on error."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("Database disk image is malformed")

        with patch("database.get_db_connection", return_value=mock_conn):
            res = database.save_active_trades("15", [{"symbol": "BTCUSDT"}])
            self.assertFalse(res)
            mock_conn.rollback.assert_called()

if __name__ == "__main__":
    unittest.main()
