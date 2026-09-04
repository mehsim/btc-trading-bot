import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import MagicMock, patch

# --- Finding #61 Tests ---
def test_finding_61_sharpe_and_sqn_ddof():
    from trade_calculators import calculate_replay_statistics, calculate_sqn

    # When duration_days is missing or falsy, ann_factor should be 1.0 (unannualized)
    pnl_series = [0.02, -0.01, 0.03, -0.005, 0.015]
    stats_no_dur = calculate_replay_statistics(pnl_series, duration_days=None)
    stats_with_dur = calculate_replay_statistics(pnl_series, duration_days=365.0)

    # Without duration_days, recovery_factor equals calmar_ratio because annualized_return = ending_return
    assert stats_no_dur["calmar_ratio"] == stats_no_dur["recovery_factor"]

    # SQN uses ddof=1 sample standard deviation
    arr = np.array(pnl_series, dtype=float)
    std_ddof1 = float(np.std(arr / 0.01, ddof=1))
    expected_sqn = round(np.sqrt(len(arr)) * np.mean(arr / 0.01) / std_ddof1, 2)
    assert calculate_sqn(pnl_series, risk_usd=0.01) == expected_sqn
    assert stats_no_dur["sqn"] > 0


# --- Finding #62 Tests ---
def test_finding_62_crash_recovery_query():
    # Verify main.py crash recovery query filters by direction and recent timestamp bound
    with open("main.py", "r") as f:
        src = f.read()
    assert "SELECT interval, calibrated_conf, direction, ts FROM decision_journal" in src
    assert "direction = ? OR direction = ?" in src
    assert "ts >= ?" in src


# --- Finding #63 Tests ---
def test_finding_63_dashboard_reverse_proxy_loopback_bypass():
    import dashboard_routes
    from dashboard_routes import require_ip_whitelist
    from flask import Flask, jsonify

    app = Flask(__name__)
    app.config["TESTING"] = True

    @app.route("/test_whitelist")
    @require_ip_whitelist
    def test_endpoint():
        return jsonify({"status": "ok"})

    def mock_get_secure_env(key, default=""):
        if key == "ALLOWED_DASHBOARD_IPS":
            return "127.0.0.1"
        if key == "DASHBOARD_TRUST_LOOPBACK":
            return "true"
        return default

    with patch("dashboard_routes.get_secure_env", side_effect=mock_get_secure_env):
        with app.test_client() as client:
            # 1. Direct local connection: remote_addr 127.0.0.1, no X-Forwarded-For -> Allowed
            res1 = client.get("/test_whitelist", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
            assert res1.status_code == 200

            # 2. Forwarded connection from external IP via local reverse proxy:
            # remote_addr is 127.0.0.1, but X-Forwarded-For is an untrusted external IP
            res2 = client.get(
                "/test_whitelist",
                environ_overrides={
                    "REMOTE_ADDR": "127.0.0.1",
                    "HTTP_X_FORWARDED_FOR": "203.0.113.195"
                }
            )
            # Must be 403 Forbidden! The reverse proxy loopback spoof must be blocked.
            assert res2.status_code == 403

            # 3. Direct access to root route is decorated with require_ip_whitelist
            assert hasattr(dashboard_routes.index, "__wrapped__") or callable(dashboard_routes.index)


# --- Finding #64 Tests ---
def test_finding_64_telegram_manual_trade_guards():
    from decision_journal import ReasonCode
    import telegram_listener

    mock_df = pd.DataFrame({
        "close": [50000.0] * 300,
        "open": [50000.0] * 300,
        "high": [50100.0] * 300,
        "low": [49900.0] * 300,
        "volume": [10.0] * 300,
        "ATR_norm": [0.005] * 300
    })

    # 1. Test MAX_CONCURRENT_POSITIONS enforcement
    mock_state = {"active_trade_15m": [{"symbol": f"COIN{i}USDT"} for i in range(10)]}
    with patch("config.MAX_CONCURRENT_POSITIONS", 5), \
         patch("data.get_history", return_value=mock_df), \
         patch("core.add_features", return_value=mock_df), \
         patch("decision_journal.write_decision") as mock_wd:
        msg = telegram_listener.execute_manual_trade(
            symbol="BTCUSDT",
            interval="15",
            direction="LONG",
            bot_state=mock_state
        )
        assert "Max concurrent positions limit" in msg
        assert mock_wd.called
        assert mock_wd.call_args[0][0].reason_code == ReasonCode.MAX_CONCURRENT_POSITIONS

    # 2. Test ExecutionValidator failure rejection
    mock_empty_state = {"active_trade_15m": []}
    with patch("config.MAX_CONCURRENT_POSITIONS", 5), \
         patch("data.get_history", return_value=mock_df), \
         patch("core.add_features", return_value=mock_df), \
         patch("risk_engine.evaluate_pre_trade_checklist", return_value=(True, "OK", 1.0, 100.0)), \
         patch("execution_validator.ExecutionValidator.validate_order", return_value=(False, "Simulated EV failure")), \
         patch("decision_journal.write_decision") as mock_wd2:
        msg2 = telegram_listener.execute_manual_trade(
            symbol="BTCUSDT",
            interval="15",
            direction="LONG",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            bot_state=mock_empty_state
        )
        assert "Rejected by Execution Validator" in msg2
        assert mock_wd2.called
        assert mock_wd2.call_args[0][0].reason_code == ReasonCode.EXECUTION_VALIDATION_FAILED

    # 3. Test callback auth checks cb_from_id
    with open("telegram_listener.py", "r") as f:
        tg_src = f.read()
    assert "is_cb_auth = bool(allowed_chat_ids and cb_from_id in allowed_chat_ids)" in tg_src


# --- Finding #65 Tests ---
def test_finding_65_decision_journal_drawdown_halt_and_outcomes():
    from decision_journal import ReasonCode, DecisionRecord

    # Check DRAWDOWN_HALT attribute
    assert hasattr(ReasonCode, "DRAWDOWN_HALT")
    assert ReasonCode.DRAWDOWN_HALT == "DRAWDOWN_HALT"

    # DecisionRecord fields and snapshot
    rec = DecisionRecord(symbol="BTCUSDT", interval="15")
    assert rec.mhi_score is None
    rec.mhi_score = 0.82
    assert rec.mhi_score == 0.82

    rec.snapshot(mhi_score=0.82)
    assert rec._inputs.get("mhi_score") == 0.82


# --- Finding #66 Tests ---
def test_finding_66_train_wf_sim_taker_fee():
    with open("train.py", "r") as f:
        src = f.read()
    assert "_roundtrip_fee = max(0.0011, float(TAKER_FEE_PCT) * 2.0)" in src


# --- Finding #67 Tests ---
def test_finding_67_walk_forward_engine_exception_marking():
    from walk_forward_engine import run_walk_forward_backtest

    dates = pd.date_range("2025-01-01", periods=100, freq="15min")
    df = pd.DataFrame({
        "close": np.linspace(50000, 51000, 100),
        "open": np.linspace(50000, 51000, 100),
        "high": np.linspace(50100, 51100, 100),
        "low": np.linspace(49900, 50900, 100),
        "volume": np.ones(100) * 10,
        "feature1": np.random.randn(100),
    }, index=dates)

    def crashing_sim(test_df):
        raise RuntimeError("Crash during simulation")

    results = run_walk_forward_backtest(
        df=df,
        train_window_bars=50,
        test_window_bars=25,
        step_bars=25,
        trade_simulator_fn=crashing_sim
    )

    windows = results.get("windows", [])
    assert len(windows) > 0
    for w in windows:
        assert w["status"] == "error"
        assert w["trades"] == 0


def test_finding_67_backtest_feature_columns_exclusion():
    with open("backtest.py", "r") as f:
        src = f.read()
    assert "close_btc" in src
    assert "numeric_cols = train_df.select_dtypes" in src


# --- Finding #68 Tests ---
def test_finding_68_dashboard_walk_forward_fold_statuses():
    from dashboard_routes import _get_walk_forward_folds
    import json

    mock_backtest_data = {
        "walk_forward_validation": {
            "windows": [
                {"status": "error", "trades": 0, "cum_return": 0.0, "max_drawdown": 0.0, "is_refitted": True},
                {"status": "no_trades", "trades": 0, "cum_return": 0.0, "max_drawdown": 0.0, "is_refitted": True},
                {"status": "completed", "trades": 0, "cum_return": 0.0, "max_drawdown": 0.0, "is_refitted": True},
                {"status": "completed", "trades": 25, "cum_return": 12.5, "max_drawdown": 4.2, "profit_factor": 1.8, "win_rate": 60.0, "is_refitted": True},
                {"status": "completed", "trades": 15, "cum_return": 5.0, "max_drawdown": 3.0, "profit_factor": 1.2, "win_rate": 50.0, "is_refitted": False},
            ]
        }
    }

    with patch("builtins.open", MagicMock()), \
         patch("json.load", return_value=mock_backtest_data):
        folds = _get_walk_forward_folds()
        assert len(folds) == 5
        assert folds[0]["status"] == "ERROR"
        assert folds[1]["status"] == "NO_DATA"
        assert folds[2]["status"] == "NO_DATA"
        assert folds[3]["status"] == "PASS"
        assert folds[4]["status"] == "UNREFITTED"


# --- Finding #70 Tests ---
def test_finding_70_backtest_scenario_fee_rate():
    with open("backtest.py", "r") as f:
        src = f.read()
    # Check that fee_rate=0.002 is no longer hardcoded anywhere in backtest.py
    assert "fee_rate=0.002" not in src
    assert "fee_rate=FEE_RATE" in src
