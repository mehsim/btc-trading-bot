"""
Unit tests for Independent Performance Audit & Reproducibility Verifier.
"""

import pytest
from reproducibility_verifier import reproducibility_verifier

def test_reproducibility_verifier():
    audit_res = reproducibility_verifier.compute_verified_metrics()
    assert audit_res["reproducibility_status"] == "INDEPENDENTLY_VERIFIED"
    assert "verified_metrics" in audit_res
    
    metrics = audit_res["verified_metrics"]
    assert metrics["sharpe_ratio"] > 1.5
    assert metrics["sortino_ratio"] > 2.0
    assert metrics["win_rate_pct"] > 50.0
    assert len(audit_res["cryptographic_certificate_hash"]) == 64
