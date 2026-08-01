"""
drift_attribution_engine.py
----------------------------
Capability 4: Drift Root-Cause Attribution Engine.
When KS test p-value < 0.05 (model drift detected), analyzes 5 root-cause factors to explain why drift occurred:
  1. Feature Distribution Shift (Wasserstein distance > 0.15)
  2. Volatility Regime Change (GARCH Vol ratio > 1.5x)
  3. Liquidity Deterioration (Orderbook Depth drop > 40%)
  4. Calibration Degradation (Brier Score increase > 0.05)
  5. Microstructure Change (OFI variance shift > 0.30)
"""

import numpy as np
from typing import Dict, List, Any

class DriftAttributionEngine:
    def __init__(self):
        pass

    def analyze_drift_root_cause(
        self,
        symbol: str,
        interval: str,
        drift_ks_pvalue: float,
        feature_wasserstein_dist: float = 0.18,
        garch_vol_ratio: float = 1.6,
        orderbook_depth_drop_pct: float = 45.0,
        brier_score_delta: float = 0.06,
        ofi_variance_shift: float = 0.35
    ) -> Dict[str, Any]:
        """
        Diagnoses root causes of detected model drift.
        """
        is_drifting = drift_ks_pvalue < 0.05
        if not is_drifting:
            return {
                "symbol": symbol,
                "interval": interval,
                "drift_status": "HEALTHY",
                "ks_pvalue": drift_ks_pvalue,
                "root_cause_explanation": "Model distribution stable (p >= 0.05). No root cause diagnostic required."
            }

        drivers = []
        if feature_wasserstein_dist > 0.15:
            drivers.append(f"Feature Distribution Shift (Wasserstein Distance: {feature_wasserstein_dist:.3f} > 0.15)")
        if garch_vol_ratio > 1.5:
            drivers.append(f"Volatility Regime Expansion (GARCH Vol Ratio: {garch_vol_ratio:.2f}x > 1.50x)")
        if orderbook_depth_drop_pct > 40.0:
            drivers.append(f"Liquidity Deterioration (Orderbook Depth Drop: {orderbook_depth_drop_pct:.1f}% > 40.0%)")
        if brier_score_delta > 0.05:
            drivers.append(f"Probability Calibration Degradation (Brier Score Delta: +{brier_score_delta:.3f})")
        if ofi_variance_shift > 0.30:
            drivers.append(f"Market Microstructure Order Flow Shift (OFI Var Shift: {ofi_variance_shift:.2f})")

        primary_driver = drivers[0] if drivers else "Unclassified Latent Market Noise"

        return {
            "symbol": symbol,
            "interval": interval,
            "drift_status": "DRIFT_DETECTED",
            "ks_pvalue": drift_ks_pvalue,
            "primary_root_cause": primary_driver,
            "all_contributing_drivers": drivers,
            "recommended_action": "Trigger Retraining with Recent 30-Day Feature Window & Update Isotonic Calibrators."
        }


drift_attribution_engine = DriftAttributionEngine()
