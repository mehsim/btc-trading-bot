import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import config
from risk_limits import HARD_MAX_DRAWDOWN_HALT_PCT, HARD_MAX_RISK_PER_TRADE_PCT
from kelly_tracker import global_kelly_tracker
from portfolio_risk import portfolio_risk_engine
from pain_feedback import pain_feedback
from database import log_event

class AutoStopFloor:
    def __init__(self, lookback_trades=200, min_sample_size=10):
        self.lookback_trades = lookback_trades
        self.min_sample_size = min_sample_size
        self.floor_cache = {}

    def compute_optimal_floor(self, symbol, database_module=None, interval=None):
        pain_adjusted = pain_feedback.get_effective_floor(symbol, interval=interval)
        if pain_adjusted is not None:
            return pain_adjusted

        if database_module and hasattr(database_module, 'get_trade_history'):
            try:
                trades = database_module.get_trade_history(limit=self.lookback_trades)
                pain_trades = [t for t in trades if isinstance(t, dict) and t.get('symbol') == symbol and t.get('reason') and ('STOP LOSS' in t['reason'] or 'BREAK-EVEN' in t['reason'])]
                if interval:
                    iv_trades = [t for t in pain_trades if str(t.get('interval')) == str(interval)]
                    if len(iv_trades) >= self.min_sample_size:
                        pain_trades = iv_trades
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
                        cfg_flr = float(config.MIN_SL_PCT_CONFIG.get(str(interval), config.MIN_SL_PCT_CONFIG.get("default", 0.008)))
                        return max(cfg_flr, min(opt_floor, 0.020))
            except Exception as e:
                print(f"[risk_engine] Warning computing floor for {symbol}: {e}")
        return float(config.MIN_SL_PCT_CONFIG.get(str(interval), config.MIN_SL_PCT_CONFIG.get("default", 0.008)))

    def get_floor(self, symbol, database_module=None, interval=None):
        pain_adjusted = pain_feedback.get_effective_floor(symbol, interval=interval)
        if pain_adjusted is not None:
            return pain_adjusted
        return self.compute_optimal_floor(symbol, database_module=database_module, interval=interval)

class WickBufferCalculator:
    def __init__(self, lookback_bars=50):
        self.lookback_bars = lookback_bars

    def get_buffer_distance(self, entry_price: float, df: Optional[pd.DataFrame] = None) -> float:
        if df is None or df.empty or len(df) < 10:
            return entry_price * 0.004
            
        wick_pcts = []
        for idx in range(max(0, len(df) - self.lookback_bars), len(df)):
            row = df.iloc[idx]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            body_size = abs(c - o)
            total_range = h - l
            if total_range > 0:
                wick_size = total_range - body_size
                wick_pcts.append(wick_size / entry_price)
                
        if not wick_pcts:
            return entry_price * 0.004
            
        expected_wick_pct = float(np.percentile(wick_pcts, 75))
        expected_wick_pct = max(0.001, min(expected_wick_pct, 0.015))
        return entry_price * expected_wick_pct * 2.0

auto_stop_floor = AutoStopFloor()
wick_buffer_calc = WickBufferCalculator()

def calculate_final_stop_distance(entry_price: float, atr_dollar: float, symbol: str, df: Optional[pd.DataFrame] = None, gmm_multiplier: float = 1.5, database_module=None, interval: Optional[str] = None) -> float:
    atr_stop = gmm_multiplier * atr_dollar
    min_floor_pct = auto_stop_floor.get_floor(symbol, database_module=database_module, interval=interval)
    cfg_floor_pct = float(config.MIN_SL_PCT_CONFIG.get(str(interval), config.MIN_SL_PCT_CONFIG.get("default", 0.008))) if interval else 0.005
    min_floor_pct = max(min_floor_pct, cfg_floor_pct)
    min_floor_dist = entry_price * min_floor_pct
    wick_dist = wick_buffer_calc.get_buffer_distance(entry_price, df=df)
    
    final_stop = max(atr_stop, min_floor_dist, wick_dist)
    return final_stop

from config import INTERVAL_MAX_POSITION_PCT, MAX_SYMBOL_EXPOSURE_PCT

def check_interval_position_limit(interval: str, proposed_size: float, balance: float, max_lev: float = 1.0) -> float:
    max_pct = INTERVAL_MAX_POSITION_PCT.get(str(interval), 0.15)
    max_size = balance * max_pct * max_lev
    return min(proposed_size, max_size)

def check_symbol_total_exposure(symbol: str, active_trades: list, proposed_size: float, balance: float, max_lev: float = 1.0) -> float:
    current_exposure = sum(
        float(t.get("position_size_usd", 0.0)) * float(t.get("leverage", 1.0))
        for t in (active_trades or [])
        if isinstance(t, dict) and t.get("symbol") == symbol
    )
    max_exposure = balance * MAX_SYMBOL_EXPOSURE_PCT * max_lev
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
    mcc_val: Optional[float] = None,
    haircut: Optional[float] = None,
    atr_norm: Optional[float] = None
) -> float:
    """
    Computes conservative Kelly fraction for trading loop.
    Applies empirical Kelly tracker (Wilson CI + 95% Bootstrap lower bound) when history exists,
    or Wilson score lower bound on win rate to discount small-sample point estimates.
    Scales by model quality (MCC) via R-1 QUALITY_SIZING policy.
    """
    import numpy as np
    from kelly_tracker import global_kelly_tracker
    from config import QUALITY_SIZING, REALIZED_RR_HAIRCUT
    haircut = haircut if haircut is not None else getattr(config, "REALIZED_RR_HAIRCUT", 0.28)
    
    # Compute effective geometry & payoff ratio in consistent units (Finding #49 / #38)
    eff_tp = float(tp_multiplier) * haircut
    eff_sl = max(1e-6, float(sl_multiplier))
    roundtrip_cost = 0.0010
    atr_norm_val = atr_norm if atr_norm is not None else getattr(config, "TARGET_VOLATILITY_ATR", 0.005)
    cost_in_atr = roundtrip_cost / max(1e-4, float(atr_norm_val))
    b_ratio = max(0.01, (eff_tp - cost_in_atr) / eff_sl)
    geom_p_star = 1.0 / (b_ratio + 1.0)

    realized_wr = None
    p_hat = float(np.clip(calibrated_confidence, 0.01, 0.99))
    if trade_history and len(trade_history) >= 10:
        win_count = sum(1 for t in trade_history if float(t.get("pnl_usd", 0.0) or 0.0) > 0 or float(t.get("return_pct", 0.0) or 0.0) > 0 or t.get("success"))
        realized_wr = float(win_count) / float(len(trade_history))
        p_hat = min(p_hat, realized_wr)
        # Finding #38: Geometry gate - if expected win rate cannot clear order geometry break-even, fail closed
        if min(float(calibrated_confidence), realized_wr) <= geom_p_star:
            return 0.0

    # Trade geometry Quarter-Kelly (Finding #52)
    if p_hat <= geom_p_star:
        return 0.0
    q_kelly_geom = max(0.0, 0.25 * (p_hat * (b_ratio + 1.0) - 1.0) / b_ratio)
    if q_kelly_geom <= 0.0:
        return 0.0

    if trade_history and len(trade_history) >= 10:
        emp_kelly = global_kelly_tracker.compute_kelly_fraction(
            timeframe=str(interval),
            min_trades=10,
            max_kelly_cap=0.20,
            insufficient_as_none=True
        )
        if emp_kelly is not None:
            if emp_kelly <= 0.0:
                # Finding #163 & #71: Measured negative or zero empirical edge -> Fail-closed! Abstain (0.0)
                # instead of falling through to confidence-based prior.
                return 0.0
            # Finding #52: Empirical Kelly cannot exceed trade geometry Quarter-Kelly
            kelly_val = min(float(emp_kelly), q_kelly_geom)
            if QUALITY_SIZING.get("enabled", True) and mcc_val is not None:
                ref_mcc = float(QUALITY_SIZING.get("reference_mcc", 0.15))
                flr = float(QUALITY_SIZING.get("floor", 0.35))
                q = float(mcc_val) / max(1e-9, ref_mcc)
                quality_mult = float(np.clip(q, flr, 1.0))
                kelly_val *= quality_mult
            return kelly_val

    raw_kelly = max(0.0, (p_hat * (b_ratio + 1.0) - 1.0) / b_ratio)
    scaled_kelly = 0.25 * raw_kelly

    # R-1 Model Quality Sizing Multiplier
    if QUALITY_SIZING.get("enabled", True) and mcc_val is not None:
        ref_mcc = float(QUALITY_SIZING.get("reference_mcc", 0.15))
        flr = float(QUALITY_SIZING.get("floor", 0.35))
        q = float(mcc_val) / max(1e-9, ref_mcc)
        quality_mult = float(np.clip(q, flr, 1.0))
        scaled_kelly *= quality_mult

    return float(scaled_kelly)


calculate_conservative_kelly = compute_conservative_kelly


def calculate_drawdown_multiplier(current_equity: float, peak_equity: float) -> float:
    """Continuous Sigmoid & Exponential Drawdown Penalty: dd_penalty = exp(-5 * DD). Hard halt at HARD_MAX_DRAWDOWN_HALT_PCT DD."""
    if peak_equity <= 0 or current_equity <= 0:
        return 1.0
    dd_fraction = max(0.0, (peak_equity - current_equity) / max(1e-9, peak_equity))
    if dd_fraction >= HARD_MAX_DRAWDOWN_HALT_PCT:
        return 0.0  # Hard halt at maximum drawdown limit
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


def get_timeframe_sizing_multiplier(interval: str) -> float:
    """
    Timeframe Capital Allocation Multiplier.
    Weights higher-timeframe trends (4h/2h/1h) with higher capital priority
    while scaling down lower-timeframe micro-signals (15m/30m) to minimize fee friction.
    """
    iv_str = str(interval).replace("m", "").replace("h", "")
    tf_weights = {
        "240": 1.25, # 4h
        "120": 1.10, # 2h
        "60": 1.00,  # 1h
        "30": 0.75,  # 30m
        "15": 0.60,  # 15m
    }
    return float(tf_weights.get(iv_str, 1.0))


def get_timeframe_stop_multiplier(interval: str) -> float:
    """
    Timeframe-Adaptive ATR Stop Distance Multiplier.
    Provides wider structural cushions for high-timeframe swing trades (4H/2H)
    to absorb intraday flash wicks without premature stopouts.
    """
    iv_str = str(interval).replace("m", "").replace("h", "")
    tf_stop_mults = {
        "240": 1.35, # 4h: 1.35x ATR Stop Cushion
        "120": 1.15, # 2h: 1.15x ATR Stop Cushion
        "60": 1.00,  # 1h: 1.00x ATR
        "30": 0.80,  # 30m: 0.80x ATR
        "15": 0.80,  # 15m: 0.80x ATR
        "5": 0.70,   # 5m: 0.70x ATR
    }
    return float(tf_stop_mults.get(iv_str, 1.0))


calculate_timeframe_stop_multiplier = get_timeframe_stop_multiplier


def calibrate_dynamic_sl_multiplier(
    interval: str,
    realized_volatility: Optional[float] = None,
    recent_slippage: Optional[float] = None,
    base_multiplier: Optional[float] = None
) -> float:
    """
    Finding #157: Dynamic Stop Calibration.
    Calculates calibrated ATR stop multiplier value locally without mutating global
    config.DYNAMIC_SL_MULTIPLIER directly (preventing concurrent request race conditions).
    """
    if base_multiplier is None:
        base_multiplier = get_timeframe_stop_multiplier(interval)

    # Use global configured multiplier as scaling factor without mutating it
    cfg_base = float(getattr(config, "DYNAMIC_SL_MULTIPLIER", 1.0))
    multiplier = float(base_multiplier) * cfg_base

    if realized_volatility is not None and realized_volatility > 0:
        vol_scalar = np.clip(realized_volatility / 0.02, 0.8, 1.5)
        multiplier *= float(vol_scalar)

    if recent_slippage is not None and recent_slippage > 0:
        slip_buffer = 1.0 + min(0.3, recent_slippage * 50.0)
        multiplier *= float(slip_buffer)

    return float(np.clip(multiplier, 0.5, 2.5))


def recalculate_dynamic_sl_multiplier(
    interval: str,
    realized_volatility: Optional[float] = None,
    recent_slippage: Optional[float] = None,
    base_multiplier: Optional[float] = None
) -> float:
    """Alias for calibrate_dynamic_sl_multiplier returning calibrated value without global config mutation."""
    return calibrate_dynamic_sl_multiplier(
        interval=interval,
        realized_volatility=realized_volatility,
        recent_slippage=recent_slippage,
        base_multiplier=base_multiplier
    )


def calculate_volatility_leverage(symbol: str, base_leverage: float, current_atr: float, target_atr: float = 0.005, min_lev: float = 1.0, max_lev: float = 10.0) -> float:
    if current_atr <= 0 or not np.isfinite(current_atr):
        return float(base_leverage)
    if target_atr <= 0 or not np.isfinite(target_atr):
        return float(base_leverage)
    if not np.isfinite(base_leverage) or base_leverage <= 0:
        return float(min_lev)
    cap = 10.0 if "BTC" in symbol else 5.0
    max_limit = min(max_lev, cap)
    effective_lev = base_leverage * np.sqrt(target_atr / max(1e-9, current_atr))
    if not np.isfinite(effective_lev):
        return float(base_leverage)
    return float(np.clip(effective_lev, min_lev, max_limit))

def _get_returns_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or "close" not in df.columns or len(df) < 10:
        return None
    close_series = pd.to_numeric(df["close"], errors="coerce")
    s = close_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "timestamp" in df.columns:
        s.index = pd.to_numeric(df["timestamp"], errors="coerce")
    return s.tail(100)

def calculate_portfolio_correlation(symbol: str, open_positions: list, df_dict: dict, interval: str = "60", candidate_direction: str = "Bullish", direction: Optional[str] = None) -> float:
    if direction is not None:
        candidate_direction = direction
    """
    Calculates maximum portfolio correlation with active positions (Finding #87).
    - If candle data for candidate or open positions is missing, returns conservative fallback prior (0.80).
    - Evaluates signed correlation relative to trade direction:
      Opposite direction with positive correlation acts as a hedge (negative effective correlation) and is credited.
    """
    if not open_positions:
        return 0.0
    if not df_dict or symbol not in df_dict or not isinstance(df_dict[symbol], pd.DataFrame):
        log_event("WARNING", f"[Portfolio Correlation Guard] Candidate {symbol} missing candle data in df_dict — applying conservative fallback prior (0.80).")
        return 0.80
    corr_cfg = getattr(config, "CORRELATION_WINDOW_CONFIG", {})
    lookback = corr_cfg.get(str(interval), corr_cfg.get("default", 20))
    
    target_s = _get_returns_series(df_dict[symbol])
    if target_s is None or len(target_s) < lookback:
        log_event("WARNING", f"[Portfolio Correlation Guard] Candidate {symbol} return series too short ({len(target_s) if target_s is not None else 0} < {lookback}) — applying conservative fallback prior (0.80).")
        return 0.80
    
    cand_dir_str = str(candidate_direction).title()
    cand_is_long = cand_dir_str in ["Bullish", "Long", "Buy"]
    max_corr = None
    has_valid_comparison = False

    for pos in open_positions:
        if isinstance(pos, dict):
            pos_symbol = pos.get("symbol")
            if not pos_symbol or pos_symbol == symbol:
                continue
            pos_dir_str = str(pos.get("direction", "Bullish")).title()
            pos_is_long = pos_dir_str in ["Bullish", "Long", "Buy"]
            # Multiplier: +1 if same direction (correlated = risk concentration), -1 if opposite direction (correlated = hedge)
            net_mult = 1.0 if (cand_is_long == pos_is_long) else -1.0

            if pos_symbol not in df_dict or not isinstance(df_dict[pos_symbol], pd.DataFrame):
                log_event("WARNING", f"[Portfolio Correlation Guard] Position {pos_symbol} missing from df_dict — assuming conservative correlation (0.80).")
                effective_corr = 0.80 if net_mult > 0 else 0.0  # Finding #58: Missing data must never credit hedge
                max_corr = effective_corr if max_corr is None else max(max_corr, effective_corr)
                has_valid_comparison = True
                continue

            other_s = _get_returns_series(df_dict[pos_symbol])
            if other_s is None or len(other_s) < lookback:
                log_event("WARNING", f"[Portfolio Correlation Guard] Position {pos_symbol} history too short — assuming conservative correlation (0.80).")
                effective_corr = 0.80 if net_mult > 0 else 0.0  # Finding #58: Short history must never credit hedge
                max_corr = effective_corr if max_corr is None else max(max_corr, effective_corr)
                has_valid_comparison = True
                continue

            combined = pd.concat([target_s, other_s], axis=1, join="inner").dropna()
            if len(combined) >= lookback:
                # Check for zero volatility (zero standard deviation)
                std_0 = float(combined.iloc[:, 0].std())
                std_1 = float(combined.iloc[:, 1].std())
                if std_0 <= 1e-8 or std_1 <= 1e-8 or not np.isfinite(std_0) or not np.isfinite(std_1):
                    # Zero volatility: correlation is zero (orthogonal flat movement)
                    corr_val = 0.0
                else:
                    corr_matrix = combined.corr()
                    corr_val = corr_matrix.iloc[0, 1] if corr_matrix.shape == (2, 2) else 0.0
                if not np.isnan(corr_val) and np.isfinite(corr_val):
                    effective_corr = float(corr_val) * net_mult
                    max_corr = effective_corr if max_corr is None else max(max_corr, effective_corr)
                    has_valid_comparison = True
            else:
                effective_corr = 0.80 if net_mult > 0 else 0.0  # Finding #58: Missing overlap must never credit hedge
                max_corr = effective_corr if max_corr is None else max(max_corr, effective_corr)
                has_valid_comparison = True
                    
    # If open positions exist but none could be checked (e.g. all missing), return conservative prior
    if not has_valid_comparison and len([p for p in open_positions if isinstance(p, dict) and p.get("symbol") != symbol]) > 0:
        return 0.80

    return max_corr if max_corr is not None else 0.0

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
            s = obj["close"].pct_change()
            if "timestamp" in obj.columns:
                s.index = obj["timestamp"]
            close_series[sym] = s
        elif isinstance(obj, pd.Series):
            close_series[sym] = obj.pct_change()

    if not close_series:
        return None
    returns_df = pd.DataFrame(close_series).dropna()
    if len(returns_df) < 10:
        return None
    return returns_df if not returns_df.empty else None

def check_portfolio_heat(open_positions: list, candidate_size_usd: float, candidate_lev: float, total_equity: float, returns_df: Optional[pd.DataFrame] = None, interval_minutes: int = 1440) -> tuple:
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
    eval_positions = list(open_positions) + [{"symbol": "CANDIDATE", "position_size_usd": candidate_size_usd, "leverage": candidate_lev}]
    var_usd, var_pct, var_ok = portfolio_risk_engine.calculate_parametric_var(eval_positions, returns_df, total_equity, interval_minutes=interval_minutes)
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


def check_wallet_margin_utilization(candidate_margin: float, margin_info: Any) -> Tuple[bool, str]:
    """
    Finding #153: Safely evaluate candidate margin against wallet margin info.
    Handles non-dict, None, or malformed inputs without raising exceptions.
    """
    if not isinstance(margin_info, dict):
        return False, "REJECTED: Malformed or missing wallet margin info"

    try:
        total_equity = float(margin_info.get("total_equity", margin_info.get("equity", 0.0)))
        used_margin = float(margin_info.get("used_margin", margin_info.get("total_margin_used", 0.0)))
        if total_equity <= 0.0:
            return False, "REJECTED: Total equity non-positive"
        new_used = used_margin + float(candidate_margin)
        utilization_pct = (new_used / total_equity) * 100.0
        max_allowed = float(getattr(config, "MAX_WALLET_MARGIN_UTILIZATION_PCT", 85.0))
        if utilization_pct > max_allowed:
            return False, f"REJECTED: Wallet margin utilization {utilization_pct:.1f}% exceeds max {max_allowed:.1f}%"
        return True, "APPROVED"
    except (ValueError, TypeError) as e:
        return False, f"REJECTED: Margin calculation error: {e}"


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

        from collections.abc import Mapping
        if not isinstance(bot_state, (dict, Mapping)) and not hasattr(bot_state, "get"):
            bot_state = {}
        if not isinstance(active_trades, list):
            active_trades = []
        if not isinstance(df_dict, (dict, Mapping)) and not hasattr(df_dict, "get"):
            df_dict = {}

        if leverage_val > max_lev or leverage_val < min_lev:
            return False, f"REJECTED: Leverage ({leverage_val}x) outside allowable limit ({min_lev}x-{max_lev}x)", 0.0, 0.0

        if bot_state.get("circuit_breaker_active", False):
            return False, "REJECTED: Daily Drawdown Circuit Breaker is active", 0.0, 0.0

        # Resolve equity from live state keys (fail-closed if missing, zero or negative)
        live_bal = bot_state.get("live_balance")
        wallet_bal = bot_state.get("wallet_balance")
        sim_bal = bot_state.get("simulated_balance")
        gen_bal = bot_state.get("balance")

        equity = None
        for b_cand in [live_bal, wallet_bal, sim_bal, gen_bal]:
            if b_cand is not None:
                try:
                    b_val = float(b_cand)
                    if b_val > 0:
                        equity = b_val
                        break
                except (ValueError, TypeError):
                    continue

        if equity is None or equity <= 0:
            return False, "REJECTED: Account equity is missing, zero or negative (Fail-Closed)", 0.0, 0.0

        peak_equity = None
        raw_peak = bot_state.get("peak_balance")
        if raw_peak is not None:
            try:
                p_val = float(raw_peak)
                if p_val > 0:
                    peak_equity = max(equity, p_val)
            except (ValueError, TypeError):
                pass

        if peak_equity is None:
            peak_equity = equity
        
        # 0. Check interval position cap & total symbol exposure cap on true economic notional
        raw_notional = position_size_usd * max(1.0, leverage_val)
        capped_notional = check_interval_position_limit(interval, raw_notional, equity, max_lev=max(1.0, leverage_val))
        capped_notional = check_symbol_total_exposure(symbol, active_trades, capped_notional, equity, max_lev=max(1.0, leverage_val))
        capped_size = capped_notional / max(1.0, leverage_val)
        leveraged_notional = capped_notional
        if capped_size < min_pos or leveraged_notional < min_notional: # Below minimum viable trade margin/notional
            return False, f"REJECTED: Position size (${position_size_usd:.2f}) exceeds interval/symbol exposure cap for {symbol} ({interval}m)", 0.0, 0.0

        # 1. Continuous Sigmoid Drawdown scaling check
        dd_mult = calculate_drawdown_multiplier(equity, peak_equity)
        if dd_mult == 0.0:
            return False, "REJECTED: Circuit breaker active (>=20% Drawdown)", 0.0, 0.0
        
        # 2. Portfolio heat & Parametric VaR check (scaled to 1-day horizon, Finding #85)
        try:
            int_str = str(interval).lower().replace("m", "").replace("h", "")
            interval_mins = int(float(int_str) * (60 if "h" in str(interval).lower() else 1))
        except Exception:
            interval_mins = 60

        returns_df = extract_or_build_returns_df(df_dict)
        eval_positions = list(active_trades) + [{"symbol": symbol, "position_size_usd": capped_size, "leverage": leverage_val}]
        var_usd, var_pct, var_ok = portfolio_risk_engine.calculate_parametric_var(eval_positions, returns_df, equity, interval_minutes=interval_mins)
        if journal:
            journal.gate("var", (var_pct * 100.0) if var_pct is not None else None, var_ok)

        if not var_ok or (var_pct is not None and (var_pct * 100.0) > max_var):
            return False, f"REJECTED: Parametric VaR ({(var_pct*100.0) if var_pct else 0:.2f}%) exceeds maximum {max_var}% capital cap", dd_mult, 0.0

        heat_safe, heat_pct = check_portfolio_heat(active_trades, capped_size, leverage_val, equity, returns_df=returns_df, interval_minutes=interval_mins)
        if journal:
            journal.gate("heat", heat_pct, heat_safe)

        if not heat_safe or heat_pct > max_heat:
            return False, f"REJECTED: Portfolio risk/heat ({heat_pct:.1f}%) exceeds safety limit ({max_heat}%)", dd_mult, 0.0

        # 2.3 Net Directional Beta / Portfolio Delta Capping (Finding #86)
        max_dir_cap = getattr(config, "MAX_DIRECTIONAL_RATIO", 1.25)
        dir_ok, dir_ratio, dir_reason = portfolio_risk_engine.check_directional_budget(
            proposed_direction=direction,
            proposed_size_usd=capped_size,
            open_positions=active_trades,
            total_equity=equity,
            max_directional_ratio=max_dir_cap,
            proposed_leverage=leverage_val
        )
        if not dir_ok:
            return False, dir_reason, dd_mult, 0.0

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
        
        # 3. Correlation check (Direction-aware signed correlation, Finding #87)
        corr_val = calculate_portfolio_correlation(symbol, active_trades, df_dict, interval=interval, candidate_direction=direction)
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
        mcc_val: Optional[float] = None,
        stop_distance: Optional[float] = None,
        target_distance: Optional[float] = None,
        interval: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Jointly optimizes (stop_distance, target_distance, position_size, capital_at_risk, expected_utility).
        Exposes a rich output dictionary for downstream governance & audit logging.
        """
        ctx_mults = context_multipliers or {}
        target_exp = float(ctx_mults.get("target_expansion", 1.0))
        stop_exp = float(ctx_mults.get("stop_expansion", 1.0))
        size_boost = float(ctx_mults.get("size_multiplier", 1.0))

        # 1. Upstream Portfolio Heat Reduction with High-Conviction Ladder Expansion
        # Base heat ceiling is 0.30 (30%); expands dynamically to 0.40 (40%) when calibrated confidence >= 0.55
        heat_ceiling = 0.40 if calibrated_confidence >= 0.55 else 0.30
        heat_ratio = min(1.0, max(0.0, portfolio_heat / heat_ceiling)) if portfolio_heat > 0 else 0.0
        avail_budget_factor = max(0.0, 1.0 - heat_ratio)
        max_available_risk_usd = total_equity * self.max_capital_risk_pct * avail_budget_factor

        # 2. Invariant Stop Loss Distance (Structure + Volatility Grounded, NOT confidence-squeezed)
        if stop_distance is not None and stop_distance > 0:
            stop_dist = max(float(stop_distance), entry_price * 0.002)
        else:
            base_stop_dist = calculate_final_stop_distance(
                entry_price=entry_price,
                atr_dollar=atr_dollars,
                symbol=symbol,
                df=df_completed,
                gmm_multiplier=1.5 * stop_exp,
                database_module=database_module,
                interval=str(interval or "60")
            )
            stop_dist = max(base_stop_dist, entry_price * 0.005) # Minimum 0.5% stop floor
        stop_distance = stop_dist

        # 3. Dynamic Target Distance
        if target_distance is not None and target_distance > 0:
            target_dist = max(float(target_distance), entry_price * 0.003)
        else:
            raw_target_dist = stop_distance * self.base_min_rr * target_exp
            target_dist = max(raw_target_dist, entry_price * 0.008) # Minimum 0.8% target floor
        target_distance = target_dist

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

        # Raw Kelly formula b = (TP_dist * haircut) / SL_dist, p = confidence (Finding #99, #71, #27, #29)
        haircut = getattr(config, "REALIZED_RR_HAIRCUT", 0.28)
        eff_target_dist = target_distance * haircut if target_distance > 0 else 0.0
        b_ratio = eff_target_dist / stop_distance if stop_distance > 0 else 1.5
        p_win = max(0.01, min(0.99, calibrated_confidence))

        # Finding #71: Align win rate with realized empirical outcomes
        realized_wr = None
        if df_completed is not None and len(df_completed) >= 10:
            if "pnl_usd" in df_completed.columns:
                realized_wr = float((df_completed["pnl_usd"] > 0).mean())
            elif "success" in df_completed.columns:
                realized_wr = float((df_completed["success"] > 0).mean())
                
        if realized_wr is not None:
            p_win = min(p_win, realized_wr)

        p_star = 1.0 / (b_ratio + 1.0) if b_ratio > 0 else 0.5
        if p_win <= p_star:
            raw_kelly = 0.0
        else:
            raw_kelly = (p_win * b_ratio - (1.0 - p_win)) / b_ratio if b_ratio > 0 else 0.0
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
        # Position Size = Capital * Effective Kelly / stop_pct (bounded by max available risk / stop_pct) (Finding #95)
        stop_pct = max(1e-6, stop_distance / entry_price)
        uncapped_size_usd = min((total_equity * effective_kelly) / stop_pct, max_available_risk_usd / stop_pct)

        # 5. Orderbook Executable Liquidity Constraint (<= 2% Top-of-Book Depth & Market Impact < 0.05%)
        max_depth_cap = top_book_depth_usd * 0.02 if top_book_depth_usd > 0 else uncapped_size_usd
        position_size_usd = min(uncapped_size_usd, max_depth_cap)
        liquidity_cap_applied = position_size_usd < uncapped_size_usd - 1e-2

        # Final Capital at Risk (USD) bounded by HARD_MAX_RISK_PER_TRADE_PCT
        max_hard_risk_usd = total_equity * HARD_MAX_RISK_PER_TRADE_PCT
        stop_pct = max(1e-6, stop_distance / entry_price)
        capital_at_risk = position_size_usd * stop_pct
        if capital_at_risk > max_hard_risk_usd:
            position_size_usd = max_hard_risk_usd / stop_pct
            capital_at_risk = position_size_usd * stop_pct

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
            "position_size_usd": round(position_size_usd, 2),
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


def calculate_atr_risk_parity_size(
    symbol: str,
    price: float,
    atr_dollars: float,
    sl_multiplier: float = 1.0,
    target_risk_usd: float = 10.0,
    max_position_size_usd: float = 500.0,
    leverage: float = 3.0
) -> Dict[str, Any]:
    """
    Calculates volatility-equalized position size (ATR Risk Parity).
    Ensures that a 1.0x ATR stop loss results in the exact same dollar loss
    regardless of whether trading low-volatility BTC or high-beta altcoins (SOL/AVAX/DOGE).
    """
    price = max(1e-6, float(price))
    atr_dollars = max(price * 0.005, float(atr_dollars))
    sl_dist = max(1e-6, float(sl_multiplier) * atr_dollars)
    stop_pct = sl_dist / price
    
    # Position Size (USD) such that: Position Size * Stop % = Target Risk USD
    ideal_size_usd = target_risk_usd / max(1e-4, stop_pct)
    capped_size_usd = min(ideal_size_usd, max_position_size_usd)
    effective_dollar_risk = capped_size_usd * stop_pct
    
    return {
        "symbol": symbol,
        "price": price,
        "atr_dollars": atr_dollars,
        "stop_distance_pct": round(stop_pct * 100.0, 3),
        "ideal_size_usd": round(ideal_size_usd, 2),
        "position_size_usd": round(capped_size_usd, 2),
        "dollar_risk_at_stop": round(effective_dollar_risk, 2),
        "leverage": leverage
    }


def calculate_anti_martingale_risk_multiplier(
    current_equity: float,
    peak_equity: float,
    recent_trades: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Computes dynamic Anti-Martingale equity curve risk scaling:
    - Compounding Mode (Hot Streak / High-Watermark): Multiplier = 1.25x to 1.50x
    - Normal Mode: Multiplier = 1.0x
    - Drawdown Defense Mode: Multiplier = 0.50x to 0.75x
    """
    current_equity = max(1.0, float(current_equity))
    if isinstance(peak_equity, dict):
        peak_val = float(peak_equity.get("peak_balance", peak_equity.get("peak_wallet_balance", current_equity)) or current_equity)
    else:
        peak_val = float(peak_equity or current_equity)
    peak_equity = max(current_equity, peak_val)
    
    drawdown_pct = (peak_equity - current_equity) / peak_equity
    
    recent_wins = 0
    total_recent = 0
    if recent_trades:
        for t in recent_trades[-5:]:
            if isinstance(t, dict):
                total_recent += 1
                succ = str(t.get("success", "")).lower()
                pnl = float(t.get("pnl_usd", 0.0) or 0.0)
                if succ in ["true", "1", "yes"] or pnl > 0.0:
                    recent_wins += 1
                    
    win_rate_recent = (recent_wins / total_recent) if total_recent >= 3 else 0.50
    
    if drawdown_pct >= 0.05:
        multiplier = 0.50
        regime = "SEVERE_DRAWDOWN_DEFENSE"
    elif drawdown_pct >= 0.025:
        multiplier = 0.75
        regime = "MODERATE_DRAWDOWN_DEFENSE"
    elif drawdown_pct < 0.015 and win_rate_recent >= 0.60:
        multiplier = 1.25 if win_rate_recent < 0.80 else 1.50
        regime = "HOT_STREAK_COMPOUNDING"
    else:
        multiplier = 1.00
        regime = "STANDARD_RISK"
        
    return {
        "multiplier": multiplier,
        "regime": regime,
        "drawdown_pct": round(drawdown_pct * 100.0, 2),
        "recent_win_rate_pct": round(win_rate_recent * 100.0, 1)
    }



