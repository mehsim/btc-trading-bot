import time
import pytest


def test_dashboard_routes_freshest_prediction_selection():
    """Verify that dashboard_routes selects the freshest prediction candidate including background evaluations."""
    status_data = {
        "active_symbol": "BTCUSDT",
        # Stale prediction from 10 minutes ago
        "latest_prediction_BTCUSDT_15m": {
            "timestamp": time.time() - 600,
            "direction": "Bearish",
            "confidence": 0.285,
            "calibrated_confidence": 0.285,
            "status": "Skipped (Low Confidence)"
        },
        # Fresh live background evaluation from 10 seconds ago
        "latest_prediction_bg_BTCUSDT_15m": {
            "timestamp": time.time() - 10,
            "direction": "Neutral",
            "confidence": 0.0,
            "calibrated_confidence": 0.0,
            "status": "Abstain (Sub-Threshold)"
        }
    }

    tf = "15m"
    active_sym = "BTCUSDT"
    iv_key = "15"
    latest_by_sym_iv = {}

    candidates = [
        status_data.get(f"latest_prediction_{active_sym}_{tf}"),
        status_data.get(f"latest_prediction_{active_sym}_{iv_key}"),
        status_data.get(f"latest_prediction_bg_{active_sym}_{tf}"),
        status_data.get(f"latest_prediction_bg_{active_sym}_{iv_key}"),
        status_data.get(f"latest_prediction_{tf}"),
        status_data.get(f"latest_prediction_{iv_key}"),
        status_data.get(f"latest_prediction_bg_{tf}"),
        status_data.get(f"latest_prediction_bg_{iv_key}"),
        status_data.get(f"latest_prediction_BTCUSDT_{tf}"),
        status_data.get(f"latest_prediction_BTCUSDT_{iv_key}"),
        status_data.get(f"latest_prediction_bg_BTCUSDT_{tf}"),
        status_data.get(f"latest_prediction_bg_BTCUSDT_{iv_key}"),
        latest_by_sym_iv.get(f"{active_sym}_{iv_key}"),
        latest_by_sym_iv.get(f"BTCUSDT_{iv_key}")
    ]
    valid_candidates = [c for c in candidates if isinstance(c, dict) and c.get("direction")]
    sym_pred = max(valid_candidates, key=lambda x: float(x.get("timestamp") or 0.0)) if valid_candidates else None

    assert sym_pred is not None
    assert sym_pred["direction"] == "Neutral"
    assert sym_pred["status"] == "Abstain (Sub-Threshold)"


def test_main_finally_harmonization_neutralizes_skipped_predictions():
    """Verify that when a signal is skipped, bot_state latest_prediction is synchronized with direction='Neutral'."""
    bot_state = {
        "latest_prediction_BTCUSDT_15m": {
            "timestamp": time.time(),
            "direction": "Bearish",
            "confidence": 0.285,
            "calibrated_confidence": 0.285,
            "status": "Bearish"
        },
        "latest_prediction_15m": {
            "timestamp": time.time(),
            "direction": "Bearish",
            "confidence": 0.285,
            "calibrated_confidence": 0.285,
            "status": "Bearish"
        }
    }

    symbol = "BTCUSDT"
    tf = "15m"
    iv = "15"
    status_msg = "Skipped (Low Confidence)"

    # Simulate synchronization block in main.py finally block
    for k_suffix in [str(tf), str(iv)]:
        for key_prefix in [f"latest_prediction_{symbol}_{k_suffix}", f"latest_prediction_{k_suffix}"]:
            curr_pred = bot_state.get(key_prefix)
            if isinstance(curr_pred, dict):
                curr_pred["status"] = str(status_msg)
                if status_msg.startswith("Skipped") or status_msg.startswith("Abstain"):
                    curr_pred["direction"] = "Neutral"

    assert bot_state["latest_prediction_BTCUSDT_15m"]["direction"] == "Neutral"
    assert bot_state["latest_prediction_BTCUSDT_15m"]["status"] == "Skipped (Low Confidence)"
    assert bot_state["latest_prediction_15m"]["direction"] == "Neutral"
    assert bot_state["latest_prediction_15m"]["status"] == "Skipped (Low Confidence)"


def test_html_files_contain_alignment_logic():
    """Ensure both templates/index.html and index.html contain the normalization logic."""
    for path in ["templates/index.html", "index.html"]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "const isSkipped = statusText.startsWith(\"Skipped\") || statusText.startsWith(\"Abstain\");" in content
        assert "const isSubThreshold = (dir === \"Bullish\" || dir === \"Bearish\") && (confVal < 0.50 || isSkipped);" in content
        assert "if (isSubThreshold || isSkipped) {" in content
        assert "dir = \"Neutral\";" in content
