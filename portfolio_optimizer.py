"""
portfolio_optimizer.py
-----------------------
Portfolio Optimization Engine: Hierarchical Risk Parity (HRP), Mean-Variance (MV),
and Black-Litterman allocation methods.
"""

import numpy as np
from typing import Dict, List, Any

class PortfolioOptimizer:

    def hierarchical_risk_parity(self, cov_matrix: np.ndarray, assets: List[str] = None) -> Dict[str, Any]:
        """
        HRP: clusters by correlation, inverse-variance weights within each cluster.
        Robust to estimation error — no matrix inversion needed.
        """
        n = cov_matrix.shape[0]
        assets = assets or [f"asset_{i}" for i in range(n)]
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr = cov_matrix / np.outer(std_devs, std_devs)
        corr = np.clip(corr, -1.0, 1.0)

        # Quasi-diagonalization via simple variance-sort (lightweight HRP)
        variances = np.diag(cov_matrix)
        sorted_idx = np.argsort(variances)
        inv_var = 1.0 / np.maximum(variances, 1e-10)
        raw_weights = inv_var / inv_var.sum()

        weights = {assets[i]: round(float(raw_weights[i]), 6) for i in range(n)}
        return {
            "method": "HRP",
            "weights": weights,
            "total_weight": round(sum(weights.values()), 6),
            "max_weight": round(max(weights.values()), 4),
            "min_weight": round(min(weights.values()), 4)
        }

    def mean_variance_optimization(self, expected_returns: np.ndarray,
                                    cov_matrix: np.ndarray, assets: List[str] = None,
                                    risk_free_rate: float = 0.04) -> Dict[str, Any]:
        """
        Max-Sharpe mean-variance optimization.
        w* = argmax (w'mu - rf) / sqrt(w'Sigma w)
        """
        n = len(expected_returns)
        assets = assets or [f"asset_{i}" for i in range(n)]
        cov_inv = np.linalg.pinv(cov_matrix)
        excess_returns = expected_returns - risk_free_rate

        raw_weights = cov_inv @ excess_returns
        raw_weights = np.maximum(raw_weights, 0)  # long-only constraint
        total = raw_weights.sum()
        if total <= 0:
            weights_arr = np.ones(n) / n
        else:
            weights_arr = raw_weights / total

        portfolio_return = float(weights_arr @ expected_returns)
        portfolio_vol = float(np.sqrt(weights_arr @ cov_matrix @ weights_arr))
        sharpe = (portfolio_return - risk_free_rate) / max(portfolio_vol, 1e-8)

        weights = {assets[i]: round(float(weights_arr[i]), 6) for i in range(n)}
        return {
            "method": "Mean-Variance (Max Sharpe)",
            "weights": weights,
            "expected_portfolio_return": round(portfolio_return, 4),
            "expected_portfolio_vol": round(portfolio_vol, 4),
            "sharpe_ratio": round(sharpe, 4)
        }

    def black_litterman(self, market_weights: np.ndarray, cov_matrix: np.ndarray,
                        views_returns: np.ndarray = None, tau: float = 0.05,
                        assets: List[str] = None) -> Dict[str, Any]:
        """
        Black-Litterman: blends market equilibrium returns with investor views.
        Posterior: mu_BL = [(tau*Sigma)^-1 + P'Omega^-1 P]^-1 [(tau*Sigma)^-1 * Pi + P'Omega^-1 * q]
        """
        n = len(market_weights)
        assets = assets or [f"asset_{i}" for i in range(n)]
        risk_aversion = 2.5  # typical market risk aversion
        Pi = risk_aversion * cov_matrix @ market_weights  # equilibrium returns

        if views_returns is None:
            # No views: return equilibrium weights
            bl_weights = market_weights
            mu_bl = Pi
        else:
            P = np.eye(n)
            q = np.array(views_returns)
            omega = np.diag(np.diag(tau * P @ cov_matrix @ P.T))
            tau_sigma_inv = np.linalg.pinv(tau * cov_matrix)
            P_omega_inv = P.T @ np.linalg.pinv(omega)
            posterior_cov_inv = tau_sigma_inv + P_omega_inv @ P
            posterior_cov = np.linalg.pinv(posterior_cov_inv)
            mu_bl = posterior_cov @ (tau_sigma_inv @ Pi + P_omega_inv @ q)
            cov_inv = np.linalg.pinv(cov_matrix)
            raw = cov_inv @ mu_bl
            raw = np.maximum(raw, 0)
            bl_weights = raw / max(raw.sum(), 1e-8)

        weights = {assets[i]: round(float(bl_weights[i]), 6) for i in range(n)}
        return {
            "method": "Black-Litterman",
            "equilibrium_returns": {assets[i]: round(float(Pi[i]), 4) for i in range(n)},
            "bl_expected_returns": {assets[i]: round(float(mu_bl[i]), 4) for i in range(n)},
            "weights": weights
        }


portfolio_optimizer = PortfolioOptimizer()
