import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class PortfolioRiskEngine:
    def __init__(self, var_confidence: float = 0.99, max_var_pct: float = 0.05):
        self.var_confidence = var_confidence  # 99% 1-day VaR
        self.z_score = 2.33  # Z-score for 99% normal distribution
        self.max_var_pct = max_var_pct  # Max 5% equity VaR limit

    def calculate_parametric_var(self, open_positions: List[Dict], returns_df: pd.DataFrame, total_equity: float) -> Tuple[float, float, bool]:
        """
        Computes 99% 1-day Parametric Value at Risk (VaR):
        VaR_99 = 2.33 * portfolio_std_dev * portfolio_value
        Returns: (var_dollars, var_pct_equity, is_within_limit)
        """
        if not open_positions or returns_df is None or returns_df.empty or total_equity <= 0:
            return 0.0, 0.0, True

        symbol_sizes = {}
        for p in open_positions:
            sym = p.get("symbol")
            if sym and sym in returns_df.columns:
                symbol_sizes[sym] = symbol_sizes.get(sym, 0.0) + float(p.get("position_size_usd", 0.0))

        symbols = list(symbol_sizes.keys())
        if not symbols:
            return 0.0, 0.0, True

        weights = np.array([symbol_sizes[s] for s in symbols])
        portfolio_val = float(np.sum(weights))
        if portfolio_val <= 0:
            return 0.0, 0.0, True

        weight_vector = weights / max(1e-9, portfolio_val)
        sub_returns = returns_df[symbols].dropna()
        if sub_returns.empty or len(sub_returns) < 10:
            return 0.0, 0.0, True

        try:
            from sklearn.covariance import LedoitWolf
            cov_matrix = LedoitWolf().fit(sub_returns.values).covariance_
        except Exception:
            cov_matrix = sub_returns.cov().values

        port_variance = float(np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector)))
        port_std_dev = float(np.sqrt(max(1e-8, port_variance)))


        var_dollars = float(self.z_score * port_std_dev * portfolio_val)
        var_pct_equity = float(var_dollars / max(1e-9, total_equity))
        is_within_limit = var_pct_equity <= self.max_var_pct

        return var_dollars, var_pct_equity, is_within_limit

    def calculate_mcr(self, candidate_symbol: str, candidate_size_usd: float, open_positions: List[Dict], returns_df: pd.DataFrame) -> Tuple[float, bool]:
        """
        Computes Marginal Contribution to Risk (MCR):
        MCR = Variance(Portfolio + Candidate) - Variance(Portfolio)
        Returns: (mcr_value, is_mcr_approved)
        """
        if returns_df is None or returns_df.empty:
            return 0.0, True

        # Construct full position set including the candidate trade
        all_candidate_positions = []
        cand_found = False
        for p in (open_positions or []):
            if p.get("symbol") == candidate_symbol:
                all_candidate_positions.append({
                    "symbol": candidate_symbol,
                    "position_size_usd": p.get("position_size_usd", 0.0) + candidate_size_usd
                })
                cand_found = True
            else:
                all_candidate_positions.append(p)

        if not cand_found:
            all_candidate_positions.append({
                "symbol": candidate_symbol,
                "position_size_usd": candidate_size_usd
            })

        cand_sym_set = {p.get("symbol") for p in all_candidate_positions}
        symbols = [col for col in returns_df.columns if col in cand_sym_set]
        if candidate_symbol not in symbols:
            return 0.0, True


        sub_returns = returns_df[symbols].dropna()
        if len(sub_returns) < 10:
            return 0.0, True

        cov_matrix = sub_returns.cov()

        # Variance without candidate
        weights_curr = np.array([next((p.get("position_size_usd", 0.0) for p in open_positions if p.get("symbol") == s), 0.0) for s in symbols])
        tot_curr = float(np.sum(weights_curr))
        var_curr = float(np.dot(weights_curr.T, np.dot(cov_matrix.values, weights_curr))) if tot_curr > 0 else 0.0

        # Variance with candidate
        weights_new = np.array([next((p.get("position_size_usd", 0.0) for p in all_candidate_positions if p.get("symbol") == s), 0.0) for s in symbols])
        var_new = float(np.dot(weights_new.T, np.dot(cov_matrix.values, weights_new)))

        mcr = var_new - var_curr
        # Reject if addition increases total variance by > 2.0% (always approve initial position when tot_curr <= 0)
        is_approved = True if tot_curr <= 0 else (mcr <= (0.02 * max(1e-5, var_new)))

        return mcr, is_approved

    def calculate_pca_factor_loadings(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Runs Principal Component Analysis (PCA) on asset returns to determine market factor loading.
        """
        if returns_df is None or len(returns_df.columns) < 3 or len(returns_df) < 30:
            return {"pc1_explained_variance": 0.50}

        try:
            cov = returns_df.cov()
            eigenvalues, _ = np.linalg.eigh(cov.values)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)[::-1]
            total_var = float(np.sum(eigenvalues))
            pc1_share = float(eigenvalues[0] / max(1e-9, total_var)) if total_var > 0 else 0.50
            return {"pc1_explained_variance": pc1_share}
        except Exception:
            return {"pc1_explained_variance": 0.50}


portfolio_risk_engine = PortfolioRiskEngine()
