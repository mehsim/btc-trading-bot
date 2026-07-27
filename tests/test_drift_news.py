import pytest
import time
from drift_detector import CUSUMDriftDetector
from news_monitor import EconomicNewsMonitor

def test_cusum_drift_detector():
    """Verify CUSUM drift detection triggers when loss rate spikes."""
    detector = CUSUMDriftDetector(threshold_H=3.0, allowance_K=0.10, target_error_mu=0.35)
    
    # 5 consecutive winning trades -> no drift
    for _ in range(5):
        is_drift, s_high, err_rate = detector.update(actual_outcome=1, predicted_confidence=0.70)
        assert not is_drift
        assert s_high == 0.0

    # 10 consecutive loss trades -> triggers drift detection
    drift_triggered = False
    for _ in range(10):
        is_drift, s_high, err_rate = detector.update(actual_outcome=0, predicted_confidence=0.70)
        if is_drift:
            drift_triggered = True
            break
            
    assert drift_triggered

def test_news_monitor_millisecond_timestamp():
    """Verify EconomicNewsMonitor correctly converts millisecond timestamps."""
    monitor = EconomicNewsMonitor()
    now_ms = time.time() * 1000.0  # Milliseconds
    
    monitor.set_upcoming_events([
        {"title": "FOMC Meeting", "timestamp": now_ms, "impact": 3}
    ])
    
    is_blackout, msg = monitor.get_news_blackout_status()
    assert is_blackout
    assert "FOMC" in msg
