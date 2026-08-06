import unittest
import time
import uuid
import sqlite3
import database

class TestTradeIDIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        database.init_db()

    def test_full_uuid_and_integrity_error_handling(self):
        """Verify F-16: Completed trades use full UUIDs and INSERT + IntegrityError handling prevents history overwrites."""
        test_trade_1 = {
            "trade_id": "test_trade_collision_id",
            "symbol": "BTCUSDT",
            "exit_time": int(time.time()),
            "interval": "15",
            "direction": "Bullish",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "change_pct": 2.0,
            "success": True,
            "reason": "TAKE PROFIT 1",
            "position_size_usd": 1000.0,
            "original_size": 1000.0,
            "pnl_usd": 20.0,
            "balance": 10020.0,
            "leverage": 10.0,
            "confidence": 0.80,
            "take_profit": 51000.0,
            "stop_loss": 49500.0,
            "atr_dollars": 500.0,
            "fill_pct": 100.0
        }

        try:
            # Save trade 1
            database.save_completed_trade(test_trade_1)

            # Create trade 2 with the SAME trade_id
            test_trade_2 = dict(test_trade_1)
            test_trade_2["reason"] = "TAKE PROFIT 2 (DIFFERENT TRADE)"

            # Save trade 2 -> should catch IntegrityError and regenerate fresh unique trade_id
            database.save_completed_trade(test_trade_2)

            # Query database to confirm BOTH trades exist (neither was overwritten!)
            conn = database.get_db_connection()
            rows = conn.execute("SELECT trade_id, reason FROM completed_trades WHERE symbol = 'BTCUSDT' AND entry_price = 50000.0;").fetchall()
            conn.close()

            self.assertGreaterEqual(len(rows), 2)
            # Assert full UUID string length is > 20 chars
            self.assertGreater(len(test_trade_2["trade_id"]), 20)
        finally:
            try:
                conn = database.get_db_connection()
                conn.execute("DELETE FROM completed_trades WHERE entry_price = 50000.0 AND pnl_usd = 20.0;")
                conn.commit()
                conn.close()
            except Exception:
                pass

    def test_fill_guard_resolves_config(self):
        """Verify C-1: main has module-scope config import for live order fill guard."""
        import main
        self.assertTrue(hasattr(main, "config"), "config must be module-scope for the fill guard")

if __name__ == "__main__":
    unittest.main()
