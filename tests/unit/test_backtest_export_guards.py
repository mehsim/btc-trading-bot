import os
import json
import pytest
from backtest import export_backtest_trades


def test_zero_trades_refuses_to_overwrite_backtest_trades_jsonl(tmp_path, monkeypatch):
    """Verify that an empty trade list does not truncate existing backtest_trades.jsonl."""
    monkeypatch.chdir(tmp_path)

    # Create a pre-existing non-empty backtest_trades.jsonl
    trades_file = tmp_path / "backtest_trades.jsonl"
    trades_file.write_text('{"trade_id": "test_1", "pnl": 100.0}\n')
    initial_size = os.path.getsize(trades_file)
    assert initial_size > 0

    all_scenario_trades = []

    # Call the actual backtest export function
    res = export_backtest_trades(all_scenario_trades, output_file=str(trades_file))
    assert res is False

    # File must retain its original contents and size
    assert os.path.getsize(trades_file) == initial_size
    assert "test_1" in trades_file.read_text()


def test_valid_trades_atomic_export(tmp_path, monkeypatch):
    """Verify that valid trades are exported atomically to output file and archive."""
    monkeypatch.chdir(tmp_path)
    trades_file = tmp_path / "backtest_trades.jsonl"
    archive_dir = tmp_path / "backtest_runs"

    trades = [{"trade_id": "trade_1", "pnl": 50.0}, {"trade_id": "trade_2", "pnl": -20.0}]
    res = export_backtest_trades(trades, output_file=str(trades_file), archive_dir=str(archive_dir))
    assert res is True
    assert trades_file.exists()
    content = trades_file.read_text()
    assert "trade_1" in content
    assert "trade_2" in content
    assert len(list(archive_dir.glob("*.jsonl"))) == 1
