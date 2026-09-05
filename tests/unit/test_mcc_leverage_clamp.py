import config

def _apply_clamp(symbol, lev_cap, mcc_val):
    thresh = getattr(config, "MCC_LEVERAGE_QUALIFICATION_THRESHOLD", 0.15)
    caps = getattr(config, "CONSERVATIVE_LEVERAGE_CAPS", {})
    if mcc_val is None or mcc_val < thresh:
        lev_cap = min(lev_cap, caps.get(symbol, caps.get("default", 3.0)))
    return lev_cap

def test_clamp_binds_below_threshold():
    assert _apply_clamp("SOLUSDT", 20.0, 0.09) == 3.0

def test_clamp_fails_closed_when_missing():
    assert _apply_clamp("SOLUSDT", 20.0, None) == 3.0

def test_clamp_inactive_above_threshold():
    assert _apply_clamp("SOLUSDT", 20.0, 0.20) == 20.0

def test_loader_sets_attribute(monkeypatch):
    import config
    monkeypatch.setattr(config, "is_manifest_degenerate", lambda m: (False, "OK"))
    from ensemble import load_ensemble_classifier
    m = load_ensemble_classifier("ensemble_trending_trend_60")
    assert hasattr(m, "manifest_mcc") and m.manifest_mcc is not None
