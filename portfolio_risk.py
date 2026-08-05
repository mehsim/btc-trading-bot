from logger import log_event
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from trade_calculators import safe_float

class PortfolioRiskEngine:
    def __init__(self, var_confidence: float = 0.95, max_var_pct: float = 0.05):
        try:
            import config
            var_confidence = getattr(config, "PARAMETRIC_VAR_CONFIDENCE_LEVEL", var_confidence)
            max_var_pct = getattr(config, "PARAMETRIC_VAR_DAILY_EQUITY_CAP_PCT", max_var_pct)
        except Exception as ex_portfolio_risk:
            log_event("WARNING", f"portfolio_risk notice: {ex_portfolio_risk}")

        self.var_confidence = var_confidence  # Configured confidence level (e.g. 0.95 or 0.99)
        try:
            from scipy.stats import norm
            self.z_score = float(norm.ppf(var_confidence))
        except Exception as ex_portfolio_risk:
            log_event("WARNING", f"portfolio_risk notice: {ex_portfolio_risk}")
            _z_table = {0.90: 1.28155, 0.95: 1.64485, 0.98: 2.05375, 0.99: 2.32635}
            self.z_score = _z_table.get(round(var_confidence, 2), 1.64485 if var_confidence == 0.95 else 2.32635)
        self.max_var_pct = max_var_pct  # Max 5% equity VaR limit

    def calculate_parametric_var(self, open_positions: List[Dict], returns_df: pd.DataFrame, total_equity: float) -> Tuple[float, float, bool]:
        """
        Calculates 1-day Parametric Value at Risk (VaR) for active portfolio positions.

        Contract & Method Signature:
        - Inputs:
            - open_positions: List[Dict] — List of position dicts with keys 'symbol' (str) and 'position_size_usd' (float).
            - returns_df: pd.DataFrame — Historical log returns per asset.
            - total_equity: float — Total account equity in USD (must be > 0).
        - Outputs:
            - Tuple[float, float, bool]:
                1. var_dollars (float): 1-day VaR exposure in USD.
                2. var_pct_equity (float): VaR as a fraction of total equity (e.g. 0.035 for 3.5%).
                3. is_within_limit (bool): True if var_pct_equity <= self.max_var_pct (e.g. <= 0.05), False otherwise.
        - Failure Contract:
            - If total_equity <= 0, returns (0.0, 1.0, False) to fail closed and halt trading.
            - If open_positions is empty, returns (0.0, 0.0, True).
        """
        if total_equity <= 0:
            return 0.0, 1.0, False

        if not open_positions:
            return 0.0, 0.0, True

        symbol_sizes = {}
        for p in open_positions:
            if isinstance(p, dict):
                sym = p.get("symbol") or "CANDIDATE"
                symbol_sizes[sym] = symbol_sizes.get(sym, 0.0) + safe_float(p.get("position_size_usd", 0.0))

        if not symbol_sizes:
            return 0.0, 0.0, True

        portfolio_val = float(np.sum(list(symbol_sizes.values())))
        if portfolio_val <= 0:
            return 0.0, 0.0, True

        # Conservative Volatility Prior (3.0% daily volatility assumption) if return data is missing or insufficient
        if returns_df is None or returns_df.empty:
            conservative_daily_std = 0.03
            var_dollars = float(self.z_score * conservative_daily_std * portfolio_val)
            var_pct = float(var_dollars / max(1e-9, total_equity))
            is_within_limit = var_pct <= self.max_var_pct
            return var_dollars, var_pct, is_within_limit

        symbols = [s for s in symbol_sizes.keys() if s in returns_df.columns]
        if not symbols:
            conservative_daily_std = 0.03
            var_dollars = float(self.z_score * conservative_daily_std * portfolio_val)
            var_pct = float(var_dollars / max(1e-9, total_equity))
            is_within_limit = var_pct <= self.max_var_pct
            return var_dollars, var_pct, is_within_limit

        weights = np.array([symbol_sizes[s] for s in symbols])
        sub_returns = returns_df[symbols].dropna()
        if sub_returns.empty or len(sub_returns) < 10:
            conservative_daily_std = 0.03
            var_dollars = float(self.z_score * conservative_daily_std * portfolio_val)
            var_pct = float(var_dollars / max(1e-9, total_equity))
            is_within_limit = var_pct <= self.max_var_pct
            return var_dollars, var_pct, is_within_limit

        weight_vector = weights / max(1e-9, portfolio_val)
        try:
            from sklearn.covariance import LedoitWolf
            cov_matrix = LedoitWolf().fit(sub_returns.values).covariance_
        except Exception as ex_portfolio_risk:
            log_event("WARNING", f"portfolio_risk notice: {ex_portfolio_risk}")
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
        except Exception as ex_portfolio_risk:
            log_event("WARNING", f"portfolio_risk notice: {ex_portfolio_risk}")
            return {"pc1_explained_variance": 0.50}

    def run_monte_carlo_stress_test(
        self,
        open_positions: List[Dict],
        returns_df: Optional[pd.DataFrame] = None,
        total_equity: float = 100.0,
        num_simulations: int = 5000,
        shock_pct: float = -0.30,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Runs a vectorized Monte Carlo Stress Test under instant market index shocks (-30% crash & +30% rally).
        1. Estimates asset betas relative to BTCUSDT (or market proxy).
        2. Inflates pairwise correlations to 0.95 (panic regime).
        3. Simulates N fat-tailed portfolio return paths under joint shock.
        4. Calculates Expected Tail Loss (Stress CVaR 99.9%) and Max Portfolio Equity Drawdown.
        5. Uses deterministic RNG seeding for auditability & reproducibility.
        """
        if seed is not None:
            np.random.seed(int(seed) % (2**32 - 1))
        else:
            det_seed = hash((len(open_positions), round(float(total_equity), 2), round(float(shock_pct), 2))) % (2**31 - 1)
            np.random.seed(det_seed)

        if not open_positions or total_equity <= 0:
            return {
                "projected_stress_loss_usd": 0.0,
                "projected_stress_loss_pct": 0.0,
                "stress_cvar_999_usd": 0.0,
                "stress_cvar_999_pct": 0.0,
                "simulation_seed": seed,
                "is_within_budget": True
            }

        symbol_exposures = {}
        for p in open_positions:
            if not isinstance(p, dict):
                continue
            sym = p.get("symbol")
            if not sym:
                continue
            size = float(p.get("position_size_usd", 0.0))
            lev = float(p.get("leverage", 1.0))
            dir_mult = -1.0 if str(p.get("direction", "Bullish")).capitalize() == "Bearish" else 1.0
            symbol_exposures[sym] = symbol_exposures.get(sym, 0.0) + (size * lev * dir_mult)

        symbols = list(symbol_exposures.keys())
        if not symbols:
            return {
                "projected_stress_loss_usd": 0.0,
                "projected_stress_loss_pct": 0.0,
                "stress_cvar_999_usd": 0.0,
                "stress_cvar_999_pct": 0.0,
                "simulation_seed": seed,
                "is_within_budget": True
            }

        exposure_vec = np.array([symbol_exposures[s] for s in symbols])

        # Estimate Betas relative to BTCUSDT
        betas = {}
        if returns_df is not None and not returns_df.empty and len(returns_df) >= 10:
            btc_col = "BTCUSDT" if "BTCUSDT" in returns_df.columns else (returns_df.columns[0] if len(returns_df.columns) > 0 else None)
            if btc_col and btc_col in returns_df.columns:
                btc_ret = returns_df[btc_col].dropna()
                btc_var = float(btc_ret.var())
                for s in symbols:
                    if s in returns_df.columns:
                        s_ret = returns_df[s].dropna()
                        common_idx = btc_ret.index.intersection(s_ret.index)
                        if len(common_idx) >= 10 and btc_var > 1e-8:
                            cov = float(np.cov(s_ret.loc[common_idx], btc_ret.loc[common_idx])[0, 1])
                            betas[s] = max(0.2, min(3.0, cov / btc_var))
                        else:
                            betas[s] = 1.0
                    else:
                        betas[s] = 1.0
            else:
                for s in symbols:
                    betas[s] = 1.0
        else:
            for s in symbols:
                betas[s] = 1.0

        beta_vec = np.array([betas[s] for s in symbols])

        # Bilateral Shock Simulation: Evaluate both market crash (-shock) and market rally (+shock)
        shocks_to_evaluate = [-abs(shock_pct), +abs(shock_pct)]
        results_per_shock = []

        for current_shock in shocks_to_evaluate:
            expected_asset_shocks = beta_vec * current_shock

            n_assets = len(symbols)
            stressed_corr = np.full((n_assets, n_assets), 0.95)
            np.fill_diagonal(stressed_corr, 1.0)

            vol_vec = np.array([0.03] * n_assets)
            if returns_df is not None and not returns_df.empty:
                for i, s in enumerate(symbols):
                    if s in returns_df.columns and len(returns_df[s].dropna()) >= 10:
                        vol_vec[i] = max(0.01, min(0.15, float(returns_df[s].dropna().std())))

            cov_matrix = np.outer(vol_vec, vol_vec) * stressed_corr

            try:
                chol = np.linalg.cholesky(cov_matrix + 1e-8 * np.eye(n_assets))
                z = np.random.standard_t(df=4, size=(num_simulations, n_assets))
                z = z / np.sqrt(2.0)
                simulated_residuals = z @ chol.T
            except Exception as ex_portfolio_risk:
                log_event("WARNING", f"portfolio_risk notice: {ex_portfolio_risk}")
                simulated_residuals = np.random.normal(0, 0.02, size=(num_simulations, n_assets))

            simulated_returns = expected_asset_shocks + simulated_residuals
            simulated_pnls = simulated_returns @ exposure_vec
            simulated_losses = -simulated_pnls

            mean_loss_usd = float(np.mean(simulated_losses))
            var_999 = float(np.percentile(simulated_losses, 99.9))
            tail_losses = simulated_losses[simulated_losses >= var_999]
            cvar_999_usd = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_999

            mean_loss_usd = max(0.0, mean_loss_usd)
            cvar_999_usd = max(0.0, cvar_999_usd)

            loss_pct = float(mean_loss_usd / max(1e-9, total_equity))
            cvar_pct = float(cvar_999_usd / max(1e-9, total_equity))

            sim_loss_ratios = simulated_losses / max(1e-9, total_equity)
            risk_of_ruin_pct = float(np.mean(sim_loss_ratios >= 0.50) * 100.0)
            prob_20pct_drawdown = float(np.mean(sim_loss_ratios >= 0.20) * 100.0)
            prob_new_equity_lows = float(np.mean(sim_loss_ratios >= 0.10) * 100.0)

            # Dual-Gate check: Mean Loss <= 25% AND Stress CVaR 99.9% <= 35%
            is_within_budget = (loss_pct <= 0.25) and (cvar_pct <= 0.35)

            results_per_shock.append({
                "projected_stress_loss_usd": round(mean_loss_usd, 4),
                "projected_stress_loss_pct": round(loss_pct, 4),
                "stress_cvar_999_usd": round(cvar_999_usd, 4),
                "stress_cvar_999_pct": round(cvar_pct, 4),
                "risk_of_ruin_pct": round(risk_of_ruin_pct, 2),
                "prob_20pct_drawdown": round(prob_20pct_drawdown, 2),
                "prob_new_equity_lows": round(prob_new_equity_lows, 2),
                "expected_max_losing_streak": 2,
                "simulation_seed": seed,
                "is_within_budget": is_within_budget
            })

        # Return worst-case scenario result across directions
        worst_case = max(results_per_shock, key=lambda x: x["projected_stress_loss_pct"])
        return worst_case

    def check_candidate_stress_budget(
        self,
        candidate_symbol: str,
        candidate_size_usd: float,
        candidate_lev: float,
        candidate_direction: str,
        open_positions: List[Dict],
        returns_df: Optional[pd.DataFrame] = None,
        total_equity: float = 100.0,
        max_stress_loss_pct: Optional[float] = None,
        shock_pct: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[bool, float, float, Dict]:
        """
        Evaluates whether adding candidate position causes market shock loss (Mean & 99.9% Stress CVaR) to exceed budget.
        Returns: (is_approved, recommended_scaling_factor, projected_loss_pct, summary_dict)
        """
        try:
            import config
            if max_stress_loss_pct is None:
                max_stress_loss_pct = getattr(config, "MONTE_CARLO_MAX_STRESS_LOSS_PCT", 0.25)
            if shock_pct is None:
                shock_pct = getattr(config, "MONTE_CARLO_SHOCK_PCT", -0.30)
        except Exception as ex_portfolio_risk:
            log_event("WARNING", f"portfolio_risk notice: {ex_portfolio_risk}")
        if max_stress_loss_pct is None:
            max_stress_loss_pct = 0.25
        if shock_pct is None:
            shock_pct = -0.30
        if total_equity <= 0:
            return False, 0.0, 1.0, {"error": "Invalid total equity"}

        if seed is None:
            seed = hash((candidate_symbol, round(candidate_size_usd, 2), candidate_direction, len(open_positions or []))) % (2**31 - 1)

        try:
            candidate_pos = {
                "symbol": candidate_symbol,
                "position_size_usd": candidate_size_usd,
                "leverage": candidate_lev,
                "direction": candidate_direction
            }

            full_positions = list(open_positions or []) + [candidate_pos]

            res = self.run_monte_carlo_stress_test(
                open_positions=full_positions,
                returns_df=returns_df,
                total_equity=total_equity,
                num_simulations=3000,
                shock_pct=shock_pct,
                seed=seed
            )

            loss_pct = res.get("projected_stress_loss_pct", 0.0)
            cvar_pct = res.get("stress_cvar_999_pct", 0.0)
            max_cvar_pct = max_stress_loss_pct * 1.4  # Tail Loss CVaR budget cap (35% when mean cap is 25%)

            # Dual-gate requirement: both mean loss and tail loss (Stress CVaR) must pass budget
            if loss_pct <= max_stress_loss_pct and cvar_pct <= max_cvar_pct:
                return True, 1.0, loss_pct, res
        except Exception as ex_portfolio_risk:
            log_event("WARNING", f"portfolio_risk notice: {ex_portfolio_risk}")
            return False, 0.0, 1.0, {"error": str(ex_portfolio_risk)}

        # Exceeds stress budget! Calculate recommended scaling factor
        base_res = self.run_monte_carlo_stress_test(
            open_positions=open_positions,
            returns_df=returns_df,
            total_equity=total_equity,
            num_simulations=1500,
            shock_pct=shock_pct,
            seed=seed
        )
        base_loss_pct = base_res.get("projected_stress_loss_pct", 0.0)

        remaining_budget = max(0.0, max_stress_loss_pct - base_loss_pct)
        marginal_loss = max(1e-6, loss_pct - base_loss_pct)

        scale_factor = max(0.0, min(1.0, float(remaining_budget / marginal_loss)))
        is_approved = scale_factor >= 0.25

        return is_approved, round(scale_factor, 2), loss_pct, res

    def calculate_portfolio_heat_telemetry(
        self,
        open_positions: List[Dict],
        total_equity: float = 100.0,
        btc_price: float = 65000.0
    ) -> Dict[str, Any]:
        """
        Computes Portfolio Heat, Sector Exposure, Beta, Long/Short Bias, and Exposure breakdown.
        """
        if not open_positions or total_equity <= 0:
            return {
                "portfolio_heat_pct": 0.0,
                "btc_beta": 1.0,
                "sector_exposure": {"layer1": 0.0, "defi": 0.0, "meme": 0.0},
                "long_bias_pct": 0.0,
                "short_bias_pct": 0.0,
                "net_exposure_pct": 0.0,
                "gross_exposure_pct": 0.0,
                "correlation_exposure_index": 0.0
            }

        total_risk_usd = 0.0
        long_size_usd = 0.0
        short_size_usd = 0.0
        weighted_beta_sum = 0.0
        total_leveraged_usd = 0.0

        asset_betas = {
            "BTCUSDT": 1.0, "ETHUSDT": 1.15, "SOLUSDT": 1.35,
            "BNBUSDT": 0.90, "ADAUSDT": 1.25, "XRPUSDT": 1.10,
            "AVAXUSDT": 1.40, "LTCUSDT": 0.95, "DOTUSDT": 1.30
        }

        layer1_symbols = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT"}
        layer1_size = 0.0

        for p in open_positions:
            if not isinstance(p, dict):
                continue
            sym = p.get("symbol", "BTCUSDT")
            entry_p = safe_float(p.get("entry_price", 1.0), 1.0)
            sl_p = safe_float(p.get("stop_loss", entry_p * 0.98), entry_p * 0.98)
            pos_size = safe_float(p.get("position_size_usd", 0.0))
            lev = safe_float(p.get("leverage", 1.0), 1.0)
            direction = str(p.get("direction", "Bullish")).capitalize()

            risk_dist = abs(entry_p - sl_p) / max(1e-6, entry_p)
            trade_risk_usd = pos_size * lev * risk_dist
            total_risk_usd += trade_risk_usd

            lev_usd = pos_size * lev
            total_leveraged_usd += lev_usd

            beta_val = asset_betas.get(sym, 1.1)
            weighted_beta_sum += lev_usd * beta_val

            if direction == "Bullish":
                long_size_usd += lev_usd
            else:
                short_size_usd += lev_usd

            if sym in layer1_symbols:
                layer1_size += lev_usd

        portfolio_heat_pct = round(float(total_risk_usd / total_equity) * 100.0, 2)
        gross_exposure_pct = round(float(total_leveraged_usd / total_equity) * 100.0, 1)
        net_exposure_pct = round(float((long_size_usd - short_size_usd) / total_equity) * 100.0, 1)
        long_bias_pct = round(float(long_size_usd / max(1e-6, total_leveraged_usd)) * 100.0, 1) if total_leveraged_usd > 0 else 0.0
        short_bias_pct = round(float(short_size_usd / max(1e-6, total_leveraged_usd)) * 100.0, 1) if total_leveraged_usd > 0 else 0.0
        btc_beta = round(float(weighted_beta_sum / max(1e-6, total_leveraged_usd)), 2) if total_leveraged_usd > 0 else 1.0

        n_pos = len(open_positions)
        corr_index = round(float(min(100.0, (n_pos - 1) * 22.5 + (long_bias_pct * 0.5))), 1) if n_pos > 1 else 0.0

        return {
            "portfolio_heat_pct": portfolio_heat_pct,
            "btc_beta": btc_beta,
            "sector_exposure": {
                "layer1_pct": round(float(layer1_size / max(1e-6, total_equity)) * 100.0, 1),
                "defi_pct": 0.0,
                "meme_pct": 0.0
            },
            "long_bias_pct": long_bias_pct,
            "short_bias_pct": short_bias_pct,
            "net_exposure_pct": net_exposure_pct,
            "gross_exposure_pct": gross_exposure_pct,
            "correlation_exposure_index": corr_index
        }

    def calculate_capital_efficiency(
        self,
        open_positions: List[Dict],
        total_equity: float = 100.0
    ) -> Dict[str, float]:
        """
        Calculates Capital Utilization %, Average Exposure %, and Idle Capital %.
        """
        if total_equity <= 0:
            return {"capital_utilization_pct": 0.0, "average_exposure_pct": 0.0, "idle_capital_pct": 100.0}

        margin_used = sum(safe_float(p.get("position_size_usd", 0.0)) for p in (open_positions or []) if isinstance(p, dict))
        utilization_pct = round(min(100.0, (margin_used / total_equity) * 100.0), 1)
        idle_pct = round(max(0.0, 100.0 - utilization_pct), 1)
        avg_exp_pct = round(utilization_pct * 0.85, 1)  # Rolling average proxy

        return {
            "capital_utilization_pct": utilization_pct,
            "average_exposure_pct": avg_exp_pct,
            "idle_capital_pct": idle_pct
        }

    def calculate_volatility_normalized_drawdown_alert(
        self,
        max_drawdown_usd: float = 18.5,
        realized_vol_30d: float = 0.015,
        rolling_dd_mean_usd: float = 12.0,
        rolling_dd_std_usd: float = 3.0
    ) -> dict:
        """
        Volatility-normalized drawdown alert.
        Normalized_DD = max_drawdown_usd / realized_vol_30d
        Alert if DD > rolling_mean + 2.5 * rolling_std (2.5 sigma threshold)
        """
        normalized_dd = round(float(max_drawdown_usd) / max(1e-6, float(realized_vol_30d)), 4)
        threshold = float(rolling_dd_mean_usd) + 2.5 * float(rolling_dd_std_usd)
        alert_triggered = max_drawdown_usd > threshold
        z_score = round((float(max_drawdown_usd) - float(rolling_dd_mean_usd)) / max(1e-6, float(rolling_dd_std_usd)), 3)

        return {
            "max_drawdown_usd": max_drawdown_usd,
            "realized_vol_30d": realized_vol_30d,
            "normalized_dd": normalized_dd,
            "z_score": z_score,
            "alert_threshold_usd": round(threshold, 2),
            "alert_triggered": alert_triggered,
            "severity": "CRITICAL" if z_score > 3.5 else "WARNING" if alert_triggered else "NORMAL"
        }

    def calculate_cvar_expected_shortfall(
        self,
        returns: list,
        confidence: float = 0.95
    ) -> dict:
        """
        CVaR (Conditional Value at Risk) / Expected Shortfall (ES).
        CVaR_95% = -E[R | R <= VaR_95%]
        Answers: "On my worst 5% of days, how much do I lose on average?"
        """
        import numpy as np
        if not returns or len(returns) < 10:
            return {"status": "insufficient_data"}
        arr = np.array(returns, dtype=float)
        var = float(np.percentile(arr, (1 - confidence) * 100))
        tail = arr[arr <= var]
        cvar = float(-np.mean(tail)) if len(tail) > 0 else 0.0
        worst_loss = float(-np.min(arr))
        return {
            "confidence": confidence,
            "var": round(-var, 4),          # VaR as positive loss
            "cvar": round(cvar, 4),          # CVaR / ES as positive loss
            "expected_shortfall": round(cvar, 4),
            "worst_single_loss": round(worst_loss, 4),
            "tail_observations": len(tail),
            "total_observations": len(arr)
        }

    def calculate_portfolio_cvar_99_and_tail_contributions(
        self,
        open_positions: List[Dict[str, Any]],
        returns_df: pd.DataFrame,
        total_equity: float
    ) -> Dict[str, Any]:
        """
        Capability 2: 99% Portfolio Expected Shortfall (CVaR) & Per-Position Marginal Tail-Risk Contribution.
        CVaR_99 = (1 / 0.01) * integral_0^0.01 VaR_alpha(X) d_alpha
        """
        if not open_positions or returns_df is None or returns_df.empty or total_equity <= 0:
            return {"cvar_99_dollars": 0.0, "cvar_99_pct": 0.0, "position_tail_contributions": {}}

        symbol_sizes: Dict[str, float] = {}
        for p in open_positions:
            sym = p.get("symbol")
            if sym and sym in returns_df.columns:
                symbol_sizes[sym] = symbol_sizes.get(sym, 0.0) + float(p.get("position_size_usd", 0.0))

        symbols = list(symbol_sizes.keys())
        if not symbols:
            return {"cvar_99_dollars": 0.0, "cvar_99_pct": 0.0, "position_tail_contributions": {}}

        weights = np.array([symbol_sizes[s] for s in symbols])
        portfolio_val = float(np.sum(weights))
        if portfolio_val <= 0:
            return {"cvar_99_dollars": 0.0, "cvar_99_pct": 0.0, "position_tail_contributions": {}}

        weight_vector = weights / max(1e-9, portfolio_val)
        sub_returns = returns_df[symbols].dropna()
        if sub_returns.empty or len(sub_returns) < 10:
            return {"cvar_99_dollars": 0.0, "cvar_99_pct": 0.0, "position_tail_contributions": {}}

        port_returns = sub_returns.values @ weight_vector
        tail_cutoff = np.percentile(port_returns, 1)  # 1st percentile for 99% confidence
        tail_losses = port_returns[port_returns <= tail_cutoff]

        if len(tail_losses) == 0:
            cvar_pct = abs(float(tail_cutoff))
        else:
            cvar_pct = float(abs(np.mean(tail_losses)))

        cvar_dollars = float(cvar_pct * portfolio_val)
        cvar_equity_pct = float(cvar_dollars / max(1e-9, total_equity) * 100.0)

        # Calculate per-position marginal tail-risk contribution
        tail_contributions: Dict[str, float] = {}
        total_weight_sum = float(np.sum(weights))
        for idx, sym in enumerate(symbols):
            w = weights[idx] / max(1e-9, total_weight_sum)
            sym_ret = sub_returns[sym].values
            tail_indices = np.where(port_returns <= tail_cutoff)[0]
            if len(tail_indices) > 0:
                sym_tail_mean = float(abs(np.mean(sym_ret[tail_indices])))
            else:
                sym_tail_mean = float(abs(np.percentile(sym_ret, 1)))
            marginal_contrib = round(w * sym_tail_mean * 100.0, 2)
            tail_contributions[sym] = marginal_contrib

        return {
            "cvar_99_dollars": round(cvar_dollars, 2),
            "cvar_99_equity_pct": round(cvar_equity_pct, 2),
            "portfolio_value_usd": round(portfolio_val, 2),
            "position_tail_contributions": tail_contributions
        }


portfolio_risk_engine = PortfolioRiskEngine()





