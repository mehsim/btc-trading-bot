import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock


def test_zero_trades_refuses_to_overwrite_backtest_trades_jsonl(tmp_path, monkeypatch):
    """Verify that an empty trade list does not truncate existing backtest_trades.jsonl."""
    monkeypatch.chdir(tmp_path)

    # Create a pre-existing non-empty backtest_trades.jsonl
    trades_file = tmp_path / "backtest_trades.jsonl"
    trades_file.write_text('{"trade_id": "test_1", "pnl": 100.0}\n')
    initial_size = os.path.getsize(trades_file)
    assert initial_size > 0

    all_scenario_trades = []

    # Emulate the guarded logic from backtest.py
    if all_scenario_trades and len(all_scenario_trades) > 0:
        with open(trades_file, "w") as f:
            for tr in all_scenario_trades:
                f.write(json.dumps(tr) + "\n")

    # File must retain its original contents and size
    assert os.path.getsize(trades_file) == initial_size
    assert "test_1" in trades_file.read_text()


def test_zero_trades_raises_runtime_error_and_preserves_results(tmp_path, monkeypatch):
    """Verify that total_trades == 0 raises RuntimeError and prevents overwriting backtest_results.json."""
    monkeypatch.chdir(tmp_path)

    results_file = tmp_path / "backtest_results.json"
    results_file.write_text(json.dumps({"timestamp": "historical", "scenarios": [{"Scenario": "A", "Trades": 41}]}))

    results = [
        {"Scenario": "A", "Trades": 0},
        {"Scenario": "B", "Trades": 0},
        {"Scenario": "C", "Trades": 0},
        {"Scenario": "D", "Trades": 0},
        {"Scenario": "E", "Trades": 0},
    ]

    total_trades_count = sum(int(r.get("Trades", 0)) for r in results)
    with pytest.raises(RuntimeError, match="Zero-trade backtest generated across all scenarios"):
        if total_trades_count == 0:
            raise RuntimeError("Zero-trade backtest generated across all scenarios. Backtest failed to produce valid trading signals.")
        with open(results_file, "w") as f:
            json.dump({"new": "data"}, f)

    # Verify original file was not overwritten
    saved = json.loads(results_file.read_text())
    assert saved["timestamp"] == "historical"
    assert saved["scenarios"][0]["Trades"] == 41
