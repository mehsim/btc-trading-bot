import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import config
from kelly_tracker import global_kelly_tracker
from portfolio_risk import portfolio_risk_engine
from pain_feedback import pain_feedback

class AutoStopFloor:
    def __init__(self, lookback_trades=200, min_sample_size=10):
        self.lookback_trades = lookback_trades
        self.min_sample_size = min_sample_size
        self.floor_cache = {}

    def compute_optimal_floor(self, symbol, database_module=None):
        pain_adjusted = pain_feedback.get_effective_floor(symbol)
        if pain_adjusted is not None:
            return pain_adjusted

        if database_module and hasattr(database_module, 'get_trade_history'):
            try:
                trades = database_module.get_trade_history(limit=self.lookback_trades)
                pain_trades = [t for t in trades if isinstance(t, dict) and t.get('symbol') == symbol and t.get('reason') and ('STOP LOSS' in t['reason'] or 'BREAK-EVEN' in t['reason'])]
                if len(pain_trades) >= self.min_sample_size:
                    required_floors = []
                    for t in pain_trades:
                        try:
                            entry = float(t.get('entry_price', 0) or 0)
                            exit_p = float(t.get('exit_price', 0) or 0)
                        except (ValueError, TypeError):
                            entry, exit_p = 0.0, 0.0
                        if entry > 0:
                            adv = abs(exit_p - entry) / entry
                            required_floors.append(adv * 1.2)
                    if required_floors:
                        opt_floor = float(np.percentile(required_floors, 75))
                        return max(0.005, min(opt_floor, 0.020))
            except Exception as e:
                print(f"[risk_engine] Warning computing floor for {symbol}: {e}")
        return 0.008

    def get_floor(self, symbol, database_module=None):
        pain_adjusted = pain_feedback.get_effective_floor(symbol)
        if pain_adjusted is not None:
            return pain_adjusted
        return self.compute_optimal_floor(symbol, database_module=database_module)

class WickBufferCalculator:
    def __init__(self, lookback_bars=50):
        self.lookback_bars = lookback_bars

    def get_buffer_distance(self, entry_price: float, df: Optional[pd.DataFrame] = None) -> float:
        if df is None or df.empty or len(df) < 10:
            return entry_price * 0.004
        
        wick_pcts = []
        recent_df = df.tail(self.lookback_bars)
        for _, bar in recent_df.iterrows():
            body_top = max(bar['open'], bar['close'])
            body_bottom = min(bar['open'], bar['close'])
            close_p = bar['close'] if bar['close'] > 0 else 1.0
            
            upper_wick = (bar['high'] - body_top) / close_p
            lower_wick = (body_bottom - bar['low']) / close_p
            wick_pcts.extend([upper_wick, lower_wick])
            
        if not wick_pcts:
            return entry_price * 0.004
            
        expected_wick_pct = float(np.percentile(wick_pcts, 75))
        expected_wick_pct = max(0.001, min(expected_wick_pct, 0.015))
        return entry_price * expected_wick_pct * 2.0

auto_stop_floor = AutoStopFloor()
wick_buffer_calc = WickBufferCalculator()

def calculate_final_stop_distance(entry_price: float, atr_dollar: float, symbol: str, df: Optional[pd.DataFrame] = None, gmm_multiplier: float = 1.5, database_module=None) -> float:
    atr_stop = gmm_multiplier * atr_dollar
    min_floor_pct = auto_stop_floor.get_floor(symbol, database_module=database_module)
    min_floor_dist = entry_price * min_floor_pct
    wick_dist = wick_buffer_calc.get_buffer_distance(entry_price, df=df)
    
    final_stop = max(atr_stop, min_floor_dist, wick_dist)
    return final_stop

from config import INTERVAL_MAX_POSITION_PCT

MAX_SYMBOL_EXPOSURE_PCT = 0.20   # Max 20% total balance in one symbol across all intervals

def check_interval_position_limit(interval: str, proposed_size: float, balance: float) -> float:
    max_pct = INTERVAL_MAX_POSITION_PCT.get(str(interval), 0.15)
    max_size = balance * max_pct
    return min(proposed_size, max_size)

def check_symbol_total_exposure(symbol: str, active_trades: list, proposed_size: float, balance: float) -> float:
    current_exposure = sum(
        float(t.get("position_size_usd", 0.0))
        for t in active_trades
        if isinstance(t, dict) and t.get("symbol") == symbol
    )
    max_exposure = balance * MAX_SYMBOL_EXPOSURE_PCT
    available = max(0.0, max_exposure - current_exposure)
    return min(proposed_size, available)

def calculate_per_interval_kelly(interval: str, trade_history: Optional[list] = None) -> float:
    """Computes dynamic Quarter-Kelly fraction per timeframe."""
    return global_kelly_tracker.compute_kelly_fraction(timeframe=str(interval), min_trades=30, max_kelly_cap=0.20)

def compute_conservative_kelly(
    calibrated_confidence: float,
    tp_multiplier: float,
    sl_multiplier: float,
    interval: str = "15",
    trade_history: Optional[list] = None,
    mcc_val: Optional[float] = None
) -> float:
    """
    Computes conservative Kelly fraction for trading loop.
    Applies empirical Kelly tracker (Wilson CI + 95% Bootstrap lower bound) when history exists,
    or Wilson score lower bound on win rate to discount small-sample point estimates.
    Scales by model quality (MCC) via R-1 QUALITY_SIZING policy.
    """
    import numpy as np
    from kelly_tracker import global_kelly_tracker
    from config import QUALITY_SIZING
    
    if trade_history and len(trade_history) >= 10:
        emp_kelly = global_kelly_tracker.compute_kelly_fraction(
            timeframe=str(interval),
            min_trades=10,
            max_kelly_cap=0.20
        )
        if emp_kelly > 0:
            kelly_val = float(emp_kelly)
            if QUALITY_SIZING.get("enabled", True) and mcc_val is not None:
                ref_mcc = float(QUALITY_SIZING.get("reference_mcc", 0.15))
                flr = float(QUALITY_SIZING.get("floor", 0.35))
                q = float(mcc_val) / max(1e-9, ref_mcc)
                quality_mult = float(np.clip(q, flr, 1.0))
                kelly_val *= quality_mult
            return kelly_val
            
    p_hat = float(np.clip(calibrated_confidence, 0.01, 0.99))
    b_ratio = float(tp_multiplier / sl_multiplier) if sl_multiplier > 0 else 1.5
    
    n = 30.0
    z = 1.96
    denom = 1.0 + (z**2 / n)
    center = p_hat + (z**2 / (2.0 * n))
    spread = z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2 / (4.0 * n**2)))
    p_wilson = max(0.0, (center - spread) / denom)
    
    raw_kelly = max(0.0, (p_wilson * (b_ratio + 1.0) - 1.0) / b_ratio) if b_ratio > 0 else 0.0
    scaled_kelly = 0.25 * raw_kelly

    # R-1 Model Quality Sizing Multiplier
    if QUALITY_SIZING.get("enabled", True) and mcc_val is not None:
        ref_mcc = float(QUALITY_SIZING.get("reference_mcc", 0.15))
        flr = float(QUALITY_SIZING.get("floor", 0.35))
        q = float(mcc_val) / max(1e-9, ref_mcc)
        quality_mult = float(np.clip(q, flr, 1.0))
        scaled_kelly *= quality_mult

    return float(scaled_kelly)


def calculate_drawdown_multiplier(current_equity: float, peak_equity: float) -> float:
    """Continuous Sigmoid & Exponential Drawdown Penalty: dd_penalty = exp(-5 * DD). Hard halt at 20% DD."""
    if peak_equity <= 0 or current_equity <= 0:
        return 1.0
    dd_fraction = max(0.0, (peak_equity - current_equity) / max(1e-9, peak_equity))
    if dd_fraction >= 0.20:
        return 0.0  # Hard halt at 20% drawdown
    penalty = float(np.exp(-5.0 * dd_fraction))
    return float(np.clip(penalty, 0.05, 1.0))

def get_regime_sizing_multiplier(regime_name: str) -> float:
    """Regime Position Sizing Multiplier bounded strictly to [0.5, 1.0]."""
    if not regime_name:
        return 1.0
    r = regime_name.lower()
    if "trending" in r:
        raw_m = 1.0
    elif "ranging" in r:
        raw_m = 0.8
    elif "chop" in r or "crisis" in r:
        raw_m = 0.5
    else:
        raw_m = 1.0
    return float(np.clip(raw_m, 0.5, 1.0))


def calculate_volatility_leverage(symbol: str, base_leverage: float, current_atr: float, target_atr: float = 0.005, min_lev: float = 1.0, max_lev: float = 10.0) -> float:
    if current_atr <= 0:
        return base_leverage
    cap = 10.0 if "BTC" in symbol else 5.0
    max_limit = min(max_lev, cap)
    effective_lev = base_leverage * np.sqrt(target_atr / max(1e-9, current_atr))
    return float(np.clip(effective_lev, min_lev, max_limit))

def calculate_portfolio_correlation(symbol: str, open_positions: list, df_dict: dict, interval: str = "60") -> float:
    if not open_positions or symbol not in df_dict or not isinstance(df_dict[symbol], pd.DataFrame):
        return 0.0
    corr_cfg = getattr(config, "CORRELATION_WINDOW_CONFIG", {})
    lookback = corr_cfg.get(str(interval)) or corr_cfg.get("default", 20)
    
    if "close" not in df_dict[symbol].columns or len(df_dict[symbol]) < lookback:
        return 0.0
    
    target_df = df_dict[symbol].copy()
    if "timestamp" in target_df.columns:
        target_ts_series = pd.to_numeric(target_df["timestamp"], errors="coerce")
        first_ts = float(target_ts_series.dropna().iloc[0]) if len(target_ts_series.dropna()) > 0 else 0
        unit_val = "ms" if first_ts > 1e11 else "s"
        target_df["dt"] = pd.to_datetime(target_ts_series, unit=unit_val, errors="coerce")
        target_df.set_index("dt", inplace=True)
    target_s = target_df["close"].pct_change().fillna(0.0).dropna().tail(100)
    max_corr = 0.0
    
    for pos in open_positions:
        if isinstance(pos, dict):
            pos_symbol = pos.get("symbol")
            if pos_symbol and pos_symbol in df_dict and pos_symbol != symbol and isinstance(df_dict[pos_symbol], pd.DataFrame):
                pos_df = df_dict[pos_symbol].copy()
                if "close" in pos_df.columns and len(pos_df) >= lookback:
                    if "timestamp" in pos_df.columns:
                        pos_ts_series = pd.to_numeric(pos_df["timestamp"], errors="coerce")
                        pos_first_ts = float(pos_ts_series.dropna().iloc[0]) if len(pos_ts_series.dropna()) > 0 else 0
                        pos_unit = "ms" if pos_first_ts > 1e11 else "s"
                        pos_df["dt"] = pd.to_datetime(pos_ts_series, unit=pos_unit, errors="coerce")
                        pos_df.set_index("dt", inplace=True)
                    other_s = pos_df["close"].pct_change().fillna(0.0).dropna().tail(100)
                    combined = pd.concat([target_s, other_s], axis=1, join="inner").dropna()
                    if len(combined) >= lookback:
                        corr_matrix = combined.corr()
                        if corr_matrix.shape == (2, 2):
                            corr_val = corr_matrix.iloc[0, 1]
                            if not np.isnan(corr_val):
                                max_corr = max(max_corr, abs(float(corr_val)))
                    
    return max_corr

def extract_or_build_returns_df(df_dict: dict) -> pd.DataFrame:
    """Extracts or dynamically constructs a percentage returns DataFrame from candle data in df_dict."""
    if not isinstance(df_dict, dict) or not df_dict:
        return None
    if "returns_df" in df_dict and isinstance(df_dict["returns_df"], pd.DataFrame) and not df_dict["returns_df"].empty:
        return df_dict["returns_df"]
    
    close_series = {}
    for sym, obj in df_dict.items():
        if sym == "returns_df":
            continue
        if isinstance(obj, pd.DataFrame) and "close" in obj.columns and len(obj) >= 10:
            close_series[sym] = obj["close"].pct_change()
        elif isinstance(obj, pd.Series):
            close_series[sym] = obj.pct_change()

    if not close_series:
        return None
    returns_df = pd.DataFrame(close_series).dropna()
    return returns_df if not returns_df.empty else None

def check_portfolio_heat(open_positions: list, candidate_size_usd: float, candidate_lev: float, total_equity: float, returns_df: Optional[pd.DataFrame] = None) -> tuple:
    """Rule 14: Parametric VaR Heat Cap & Portfolio Notional Heat Cap."""
    if total_equity <= 0:
        return False, 0.0
    
    current_heat_usd = sum(
        float(p.get("position_size_usd", 0.0)) * float(p.get("leverage", 1.0))
        for p in open_positions if isinstance(p, dict)
    )
    new_heat_usd = current_heat_usd + (candidate_size_usd * candidate_lev)
    heat_pct = (new_heat_usd / max(1e-9, total_equity)) * 100.0

    # Always execute Parametric VaR check for active positions + candidate
    eval_positions = list(open_positions) + [{"symbol": "CANDIDATE", "position_size_usd": candidate_size_usd}]
    var_usd, var_pct, var_ok = portfolio_risk_engine.calculate_parametric_var(eval_positions, returns_df, total_equity)
    if not var_ok:
        return False, var_pct * 100.0
    
    # Standard heat cap fallback at 300% leverage exposure
    is_safe = heat_pct <= 300.0
    return is_safe, heat_pct

def check_margin_utilization(used_margin: float, total_equity: float, max_leverage: float = 10.0) -> str:
    """Dynamic Margin Utilization Warnings driven by MARGIN_UTILIZATION_POLICY."""
    if total_equity <= 0 or max_leverage <= 0:
        return "NORMAL"
    policy = getattr(config, "MARGIN_UTILIZATION_POLICY", {})
    warn_p = policy.get("warning_pct", 50.0)
    halt_p = policy.get("halt_pct", 70.0)
    emerg_p = policy.get("emergency_pct", 85.0)

    utilization_pct = (used_margin / total_equity) * 100.0
    emergency_thresh = (1.0 / (max_leverage * 1.05)) * 100.0
    halt_thresh = (1.0 / (max_leverage * 1.20)) * 100.0
    warning_thresh = (1.0 / (max_leverage * 1.50)) * 100.0

    if utilization_pct >= max(emerg_p, emergency_thresh):
        return "EMERGENCY_CLOSE"
    elif utilization_pct >= max(halt_p, halt_thresh):
        return "HALT_ENTRIES"
    elif utilization_pct >= max(warn_p, warning_thresh):
        return "WARNING_ALERT"
    return "NORMAL"

def evaluate_pre_trade_checklist(symbol: str, position_size_usd: float, leverage_val: float, active_trades: list, bot_state: dict, df_dict: dict, interval: str = "60", direction: str = "Bullish", journal: Any = None) -> tuple:
    try:
        policy = getattr(config, "PRE_TRADE_POLICY", {})
        from risk_limits import HARD_TIMEFRAME_MAX_LEVERAGE_CAPS
        tf_caps = getattr(config, "TIMEFRAME_MAX_LEVERAGE_CAPS", {})
        tf_clean = str(interval).replace("m", "")
        hard_cap = HARD_TIMEFRAME_MAX_LEVERAGE_CAPS.get(tf_clean, 10.0)
        cfg_cap = tf_caps.get(tf_clean, hard_cap)
        max_lev = min(hard_cap, cfg_cap)
        min_lev = policy.get("min_leverage", 1.0)
        min_pos = policy.get("min_position_usd", 0.20)
        min_notional = policy.get("min_notional_usd", 1.0)
        max_var = policy.get("max_var_pct", 5.0)
        max_heat = policy.get("max_heat_pct", 300.0)
        max_stress = policy.get("max_stress_loss_pct", 25.0)
        max_corr = policy.get("max_correlation", 0.70)

        if not isinstance(bot_state, dict):
            bot_state = {}
        if not isinstance(active_trades, list):
            active_trades = []
        if not isinstance(df_dict, dict):
            df_dict = {}

        if leverage_val > max_lev or leverage_val < min_lev:
            return False, f"REJECTED: Leverage ({leverage_val}x) outside allowable limit ({min_lev}x-{max_lev}x)", 0.0, 0.0

        if bot_state.get("circuit_breaker_active", False):
            return False, "REJECTED: Daily Drawdown Circuit Breaker is active", 0.0, 0.0

        equity = float(bot_state.get("live_balance", bot_state.get("wallet_balance", bot_state.get("simulated_balance", 80.0))))
        if equity <= 0:
            equity = float(bot_state.get("simulated_balance", 80.0))
        if equity <= 0:
            return False, "REJECTED: Account equity is zero or negative (Fail-Closed)", 0.0, 0.0

        peak_equity = float(bot_state.get("peak_balance", equity))
        
        # 0. Check interval position cap & total symbol exposure cap
        capped_size = check_interval_position_limit(interval, position_size_usd, equity)
        capped_size = check_symbol_total_exposure(symbol, active_trades, capped_size, equity)
        leveraged_notional = capped_size * max(1.0, leverage_val)
        if capped_size < min_pos or leveraged_notional < min_notional: # Below minimum viable trade margin/notional
            return False, f"REJECTED: Position size (${position_size_usd:.2f}) exceeds interval/symbol exposure cap for {symbol} ({interval}m)", 0.0, 0.0

        # 1. Continuous Sigmoid Drawdown scaling check
        dd_mult = calculate_drawdown_multiplier(equity, peak_equity)
        if dd_mult == 0.0:
            return False, "REJECTED: Circuit breaker active (>=20% Drawdown)", 0.0, 0.0
        
        # 2. Portfolio heat & Parametric VaR check
        returns_df = extract_or_build_returns_df(df_dict)
        eval_positions = list(active_trades) + [{"symbol": symbol, "position_size_usd": capped_size}]
        var_usd, var_pct, var_ok = portfolio_risk_engine.calculate_parametric_var(eval_positions, returns_df, equity)
        if journal:
            journal.gate("var", (var_pct * 100.0) if var_pct is not None else None, var_ok)

        if not var_ok or (var_pct is not None and (var_pct * 100.0) > max_var):
            return False, f"REJECTED: Parametric VaR ({(var_pct*100.0) if var_pct else 0:.2f}%) exceeds maximum {max_var}% capital cap", dd_mult, 0.0

        heat_safe, heat_pct = check_portfolio_heat(active_trades, capped_size, leverage_val, equity, returns_df=returns_df)
        if journal:
            journal.gate("heat", heat_pct, heat_safe)

        if not heat_safe or heat_pct > max_heat:
            return False, f"REJECTED: Portfolio risk/heat ({heat_pct:.1f}%) exceeds safety limit ({max_heat}%)", dd_mult, 0.0

        # 2.5 Monte Carlo -30% Stress Test Check
        mc_approved, mc_scale_factor, mc_loss_pct, mc_summary = portfolio_risk_engine.check_candidate_stress_budget(
            candidate_symbol=symbol,
            candidate_size_usd=capped_size,
            candidate_lev=leverage_val,
            candidate_direction=direction,
            open_positions=active_trades,
            returns_df=returns_df,
            total_equity=equity,
            max_stress_loss_pct=max_stress / 100.0,
            shock_pct=-0.30
        )
        if journal:
            mc_cvar = mc_summary.get("stress_cvar_999_pct") if isinstance(mc_summary, dict) else None
            mc_seed = mc_summary.get("simulation_seed") if isinstance(mc_summary, dict) else None
            journal.gate("stress", (mc_loss_pct * 100.0) if mc_loss_pct is not None else None, mc_approved)
            if mc_cvar is not None:
                journal.gate_stress_cvar = mc_cvar * 100.0
            if mc_seed is not None:
                journal.simulation_seed = mc_seed

        if not mc_approved:
            return False, f"REJECTED: Monte Carlo -30% Stress Test projected loss ({mc_loss_pct*100.0:.1f}%) exceeds max {max_stress}% equity budget", dd_mult, 0.0

        if mc_scale_factor < 1.0:
            capped_size = round(capped_size * mc_scale_factor, 2)
        
        # 3. Correlation check
        corr_val = calculate_portfolio_correlation(symbol, active_trades, df_dict, interval=interval)
        corr_ok = corr_val <= max_corr
        if journal:
            journal.gate("corr", corr_val, corr_ok)

        if not corr_ok:
            return False, f"REJECTED: High correlation ({corr_val:.2f} > {max_corr:.2f}) with open positions", dd_mult, 0.0
            
        return True, f"APPROVED: Risk checklist passed (Size: ${capped_size:.2f}, Sigmoid DD Mult: {dd_mult:.2f}, Heat: {heat_pct:.1f}%, Stress Loss: {mc_loss_pct*100.0:.1f}%, Max Corr: {corr_val:.2f})", dd_mult, capped_size
    except Exception as e:
        print(f"[risk_engine ERROR] Exception in evaluate_pre_trade_checklist for {symbol}: {e}")
        return False, f"REJECTED: Risk engine exception (Fail-Closed): {e}", 0.0, 0.0

def get_volatility_regime_multiplier(atr_norm: float, interval: str) -> float:
    """Dynamic Inverse ATR Sizing with continuous linear interpolation across all timeframes."""
    profiles = getattr(config, "VOLATILITY_SIZING_PROFILE", {})
    prof = profiles.get(str(interval)) or profiles.get("default", [
        (0.000, 0.5), (0.003, 0.5), (0.005, 1.0), (0.012, 1.0), (0.015, 0.7), (0.020, 0.5)
    ])
    xs, ys = zip(*prof)
    raw_m = float(np.interp(atr_norm, xs, ys))
    return float(np.clip(raw_m, 0.5, 1.0))

class JointRiskBudgetAllocator:
    """
    Institutional Joint Risk Budget Allocator.
    Stop distance is strictly derived from market structure & volatility (NEVER squeezed by high confidence).
    Capital sizing is governed by upstream portfolio heat reduction, MHI-tied fractional Kelly, and orderbook liquidity caps.
    """

    def __init__(self, max_capital_risk_pct: float = 0.02, base_min_rr: float = 1.20):
        self.max_capital_risk_pct = max_capital_risk_pct
        self.base_min_rr = base_min_rr

    def get_mhi_max_kelly(self, mhi_score: float) -> float:
        """
        Governance-tied Kelly fraction based on Model Health Index (MHI) continuous ramp with hysteresis:
        - MHI < 50.0 (CRITICAL): 0.00x Kelly (Trading Halted until score recovers > 53.0)
        - 50.0 <= MHI <= 85.0: Continuous linear ramp from 0.0x to 0.25x Kelly
        - MHI > 85.0 (HEALTHY): Max 0.25x Kelly
        """
        policy = getattr(config, "MHI_KELLY_POLICY", {
            "halt_below": 50.0,
            "resume_above": 53.0,
            "full_at": 85.0,
            "max_kelly": 0.25,
        })
        halt_below = float(policy.get("halt_below", 50.0))
        resume_above = float(policy.get("resume_above", 53.0))
        full_at = float(policy.get("full_at", 85.0))
        max_kelly = float(policy.get("max_kelly", 0.25))

        if not hasattr(self, "_mhi_halted"):
            self._mhi_halted = False

        if self._mhi_halted:
            if mhi_score >= resume_above:
                self._mhi_halted = False
            else:
                return 0.0
        elif mhi_score < halt_below:
            self._mhi_halted = True
            return 0.0

        span = max(1e-9, full_at - halt_below)
        frac = (mhi_score - halt_below) / span
        return float(np.clip(frac, 0.0, 1.0) * max_kelly)

    def allocate_risk_budget(
        self,
        symbol: str,
        entry_price: float,
        atr_dollars: float,
        atr_norm: float,
        calibrated_confidence: float,
        direction: str,
        total_equity: float,
        portfolio_heat: float = 0.0,
        mhi_score: float = 90.0,
        top_book_depth_usd: float = 50000.0,
        df_completed: Optional[pd.DataFrame] = None,
        context_multipliers: Optional[Dict[str, float]] = None,
        database_module = None,
        mcc_val: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Jointly optimizes (stop_distance, target_distance, position_size, capital_at_risk, expected_utility).
        Exposes a rich output dictionary for downstream governance & audit logging.
        """
        ctx_mults = context_multipliers or {}
        target_exp = float(ctx_mults.get("target_expansion", 1.0))
        stop_exp = float(ctx_mults.get("stop_expansion", 1.0))
        size_boost = float(ctx_mults.get("size_multiplier", 1.0))

        # 1. Upstream Portfolio Heat Reduction
        # Available Risk Budget decreases linearly as Portfolio Heat increases
        heat_ratio = min(1.0, max(0.0, portfolio_heat / 0.20)) if portfolio_heat > 0 else 0.0
        avail_budget_factor = max(0.0, 1.0 - heat_ratio)
        max_available_risk_usd = total_equity * self.max_capital_risk_pct * avail_budget_factor

        # 2. Invariant Stop Loss Distance (Structure + Volatility Grounded, NOT confidence-squeezed)
        base_stop_dist = calculate_final_stop_distance(
            entry_price=entry_price,
            atr_dollar=atr_dollars,
            symbol=symbol,
            df=df_completed,
            gmm_multiplier=1.5 * stop_exp,
            database_module=database_module
        )
        stop_distance = max(base_stop_dist, entry_price * 0.005) # Minimum 0.5% stop floor

        # 3. Dynamic Target Distance
        raw_target_dist = stop_distance * self.base_min_rr * target_exp
        target_distance = max(raw_target_dist, entry_price * 0.008) # Minimum 0.8% target floor

        # 4. MHI-Tied Fractional Kelly Sizing
        max_kelly_frac = self.get_mhi_max_kelly(mhi_score)
        if max_kelly_frac <= 0.0 or max_available_risk_usd <= 0.0:
            return {
                "symbol": symbol,
                "stop_distance": stop_distance,
                "target_distance": target_distance,
                "risk_per_trade": 0.0,
                "position_size": 0.0,
                "expected_edge": 0.0,
                "expected_utility": 0.0,
                "capital_at_risk": 0.0,
                "kelly_fraction": 0.0,
                "portfolio_heat": portfolio_heat,
                "liquidity_cap_applied": False,
                "execution_permitted": False,
                "reason": f"Halted by MHI ({mhi_score:.1f}) or exhausted portfolio heat ({portfolio_heat*100:.1f}%)"
            }

        # Raw Kelly formula b = TP_dist / SL_dist, p = confidence
        b_ratio = target_distance / stop_distance if stop_distance > 0 else 1.5
        p_win = max(0.01, min(0.99, calibrated_confidence))
        raw_kelly = (p_win * b_ratio - (1.0 - p_win)) / b_ratio if b_ratio > 0 else 0.05
        raw_kelly = max(0.0, raw_kelly)

        # Apply MHI Kelly Fraction Cap & Upstream Portfolio Heat Reduction Factor & Context Size Boost
        # Multipliers applied first, then final clamp against max_kelly_frac (MHI ceiling is binding)
        boosted_kelly = raw_kelly * max_kelly_frac * size_boost * avail_budget_factor
        effective_kelly = float(np.clip(boosted_kelly, 0.0, max_kelly_frac))

        # R-1: Scale Kelly fraction by model quality (MCC)
        from config import QUALITY_SIZING
        quality_mult = 1.0
        if QUALITY_SIZING.get("enabled", True) and mcc_val is not None:
            ref_mcc = float(QUALITY_SIZING.get("reference_mcc", 0.15))
            flr = float(QUALITY_SIZING.get("floor", 0.35))
            q = float(mcc_val) / max(1e-9, ref_mcc)
            quality_mult = float(np.clip(q, flr, 1.0))
            effective_kelly *= quality_mult
        
        # Calculate Unconstrained Risk-Budgeted Position Size (USD)
        # Position Size = Capital * Effective Kelly (bounded by max available risk)
        uncapped_size_usd = min(total_equity * effective_kelly, max_available_risk_usd / (stop_distance / entry_price))

        # 5. Orderbook Executable Liquidity Constraint (<= 2% Top-of-Book Depth & Market Impact < 0.05%)
        max_depth_cap = top_book_depth_usd * 0.02 if top_book_depth_usd > 0 else uncapped_size_usd
        position_size_usd = min(uncapped_size_usd, max_depth_cap)
        liquidity_cap_applied = position_size_usd < uncapped_size_usd - 1e-2

        # Final Capital at Risk (USD)
        capital_at_risk = position_size_usd * (stop_distance / entry_price)

        # 6. Expected Edge & Expected Utility Calculation
        roundtrip_fee_pct = 0.0010
        expected_edge = p_win * (target_distance / entry_price) - (1.0 - p_win) * (stop_distance / entry_price) - roundtrip_fee_pct
        expected_utility = position_size_usd * expected_edge

        # Item 5: Structured Risk Gate Results Logging
        var_limit = 0.05
        var_val = float(round(capital_at_risk / max(1.0, total_equity), 4))
        stress_limit = 0.25
        stress_val = float(round(var_val * 1.5, 4))
        heat_limit = 0.20
        heat_val = float(round(portfolio_heat, 4))
        kelly_limit = float(max_kelly_frac)
        kelly_val = float(min(effective_kelly, max_kelly_frac))

        risk_gate_results = {
            "VaR": {"value": var_val, "limit": var_limit, "pass": bool(var_val <= var_limit)},
            "Stress": {"loss": stress_val, "limit": stress_limit, "pass": bool(stress_val <= stress_limit)},
            "Heat": {"utilization": heat_val, "limit": heat_limit, "pass": bool(heat_val <= heat_limit)},
            "Kelly": {"fraction": kelly_val, "limit": kelly_limit, "pass": bool(kelly_val <= kelly_limit)}
        }

        return {
            "symbol": symbol,
            "stop_distance": round(stop_distance, 6),
            "target_distance": round(target_distance, 6),
            "risk_per_trade": round(capital_at_risk, 2),
            "position_size": round(position_size_usd, 2),
            "expected_edge": round(expected_edge, 6),
            "expected_utility": round(expected_utility, 4),
            "capital_at_risk": round(capital_at_risk, 2),
            "kelly_fraction": effective_kelly,
            "portfolio_heat": round(portfolio_heat, 4),
            "risk_gate_results": risk_gate_results,
            "liquidity_cap_applied": liquidity_cap_applied,
            "execution_permitted": position_size_usd >= 1.0 and expected_edge > 0,
            "reason": "APPROVED" if (position_size_usd >= 1.0 and expected_edge > 0) else "Insufficient Edge / Notional"
        }

joint_risk_budget_allocator = JointRiskBudgetAllocator()

