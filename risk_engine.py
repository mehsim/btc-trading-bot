import numpy as np
import pandas as pd
from kelly_tracker import global_kelly_tracker
from portfolio_risk import portfolio_risk_engine

INTERVAL_MAX_POSITION_PCT = {
    "5": 0.05,
    "15": 0.08,   # 8% max for 15m
    "30": 0.10,   # 10% max for 30m
    "60": 0.20,   # 20% max for 60m (optimized Half-Kelly)
    "120": 0.20   # 20% max for 120m (optimized Half-Kelly)
}

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

def calculate_per_interval_kelly(interval: str, trade_history: list = None) -> float:
    """Computes dynamic Quarter-Kelly fraction per timeframe."""
    return global_kelly_tracker.compute_kelly_fraction(timeframe=str(interval), min_trades=10, max_kelly_cap=0.20)

def calculate_drawdown_multiplier(current_equity: float, peak_equity: float) -> float:
    """Rule 15: Continuous Sigmoid Drawdown Curve with 20% Hard Halt."""
    if peak_equity <= 0 or current_equity <= 0:
        return 1.0
    drawdown_pct = (peak_equity - current_equity) / peak_equity * 100.0
    if drawdown_pct >= 20.0:
        return 0.0  # Circuit breaker hard halt
    # Sigmoid decay centered at 12.5% drawdown
    sigmoid_mult = 1.0 / (1.0 + np.exp(10.0 * (drawdown_pct - 12.5) / 20.0))
    return float(np.clip(sigmoid_mult, 0.05, 1.0))

def calculate_volatility_leverage(symbol: str, base_leverage: float, current_atr: float, target_atr: float = 0.005, min_lev: float = 1.0, max_lev: float = 10.0) -> float:
    if current_atr <= 0:
        return base_leverage
    cap = 10.0 if "BTC" in symbol else 5.0
    max_limit = min(max_lev, cap)
    effective_lev = base_leverage * np.sqrt(target_atr / current_atr)
    return float(np.clip(effective_lev, min_lev, max_limit))

def calculate_portfolio_correlation(symbol: str, open_positions: list, df_dict: dict) -> float:
    if not open_positions or symbol not in df_dict or not isinstance(df_dict[symbol], pd.DataFrame):
        return 0.0
    if "close" not in df_dict[symbol].columns or len(df_dict[symbol]) < 20:
        return 0.0
    
    target_df = df_dict[symbol].copy()
    if "timestamp" in target_df.columns:
        first_ts = target_df["timestamp"].iloc[0]
        unit_str = "ms" if first_ts > 1e11 else "s"
        target_df["dt"] = pd.to_datetime(target_df["timestamp"], unit=unit_str, errors="coerce")
        target_s = target_df.set_index("dt")["close"].pct_change().dropna().iloc[-100:]
    else:
        target_s = target_df["close"].pct_change().dropna().iloc[-100:]
    max_corr = 0.0
    
    for pos in open_positions:
        if isinstance(pos, dict):
            pos_symbol = pos.get("symbol")
            if pos_symbol and pos_symbol in df_dict and pos_symbol != symbol and isinstance(df_dict[pos_symbol], pd.DataFrame):
                pos_df = df_dict[pos_symbol].copy()
                if "close" in pos_df.columns and len(pos_df) >= 20:
                    if "timestamp" in pos_df.columns:
                        first_ts_other = pos_df["timestamp"].iloc[0]
                        unit_other = "ms" if first_ts_other > 1e11 else "s"
                        pos_df["dt"] = pd.to_datetime(pos_df["timestamp"], unit=unit_other, errors="coerce")
                        other_s = pos_df.set_index("dt")["close"].pct_change().dropna().iloc[-100:]
                    else:
                        other_s = pos_df["close"].pct_change().dropna().iloc[-100:]
                    combined = pd.concat([target_s, other_s], axis=1).dropna()
                    if len(combined) >= 20:
                        corr_matrix = combined.corr()
                        if corr_matrix.shape == (2, 2):
                            corr_val = corr_matrix.iloc[0, 1]
                            if not np.isnan(corr_val):
                                max_corr = max(max_corr, abs(float(corr_val)))
                    
    return max_corr

def check_portfolio_heat(open_positions: list, candidate_size_usd: float, candidate_lev: float, total_equity: float, returns_df: pd.DataFrame = None) -> tuple:
    """Rule 14: 99% 1-day Parametric VaR Heat Cap (Max 5% equity VaR)."""
    if total_equity <= 0:
        return False, 0.0
    
    current_heat_usd = sum(
        float(p.get("position_size_usd", 0.0)) * float(p.get("leverage", 1.0))
        for p in open_positions if isinstance(p, dict)
    )
    new_heat_usd = current_heat_usd + (candidate_size_usd * candidate_lev)
    heat_pct = (new_heat_usd / total_equity) * 100.0

    # Parametric VaR calculation if returns history is available
    if returns_df is not None and not returns_df.empty:
        var_usd, var_pct, var_ok = portfolio_risk_engine.calculate_parametric_var(open_positions, returns_df, total_equity)
        if not var_ok:
            return False, var_pct * 100.0
    
    # Standard heat cap fallback at 300% leverage exposure
    is_safe = heat_pct <= 300.0
    return is_safe, heat_pct

def check_margin_utilization(used_margin: float, total_equity: float, max_leverage: float = 10.0) -> str:
    """Rule 17: Dynamic Margin Utilization Warnings scaled by Leverage Tier."""
    if total_equity <= 0 or max_leverage <= 0:
        return "NORMAL"
    utilization_pct = (used_margin / total_equity) * 100.0
    emergency_thresh = (1.0 / (max_leverage * 1.05)) * 100.0
    halt_thresh = (1.0 / (max_leverage * 1.20)) * 100.0
    warning_thresh = (1.0 / (max_leverage * 1.50)) * 100.0

    if utilization_pct >= max(85.0, emergency_thresh):
        return "EMERGENCY_CLOSE"
    elif utilization_pct >= max(70.0, halt_thresh):
        return "HALT_ENTRIES"
    elif utilization_pct >= max(50.0, warning_thresh):
        return "WARNING_ALERT"
    return "NORMAL"

def evaluate_pre_trade_checklist(symbol: str, position_size_usd: float, leverage_val: float, active_trades: list, bot_state: dict, df_dict: dict, interval: str = "60") -> tuple:
    equity = float(bot_state.get("live_balance", bot_state.get("wallet_balance", bot_state.get("simulated_balance", 80.0))))
    if equity <= 0:
        equity = float(bot_state.get("simulated_balance", 80.0))
    peak_equity = float(bot_state.get("peak_balance", equity))
    
    # 0. Check interval position cap & total symbol exposure cap
    capped_size = check_interval_position_limit(interval, position_size_usd, equity)
    capped_size = check_symbol_total_exposure(symbol, active_trades, capped_size, equity)
    if capped_size < 1.0: # Below minimum trade size ($1)
        return False, f"REJECTED: Position size (${position_size_usd:.2f}) exceeds interval/symbol exposure cap for {symbol} ({interval}m)", 0.0, 0.0
        
    # 1. Continuous Sigmoid Drawdown scaling check
    dd_mult = calculate_drawdown_multiplier(equity, peak_equity)
    if dd_mult == 0.0:
        return False, "REJECTED: Circuit breaker active (>=20% Drawdown)", 0.0, 0.0
    
    # 2. Portfolio heat & Parametric VaR check
    returns_df = df_dict.get("returns_df") if isinstance(df_dict, dict) else None
    heat_safe, heat_pct = check_portfolio_heat(active_trades, capped_size, leverage_val, equity, returns_df=returns_df)
    if not heat_safe:
        return False, f"REJECTED: Portfolio risk/heat ({heat_pct:.1f}%) exceeds safety limit", dd_mult, 0.0
    
    # 3. Correlation check
    corr_val = calculate_portfolio_correlation(symbol, active_trades, df_dict)
    if corr_val > 0.7:
        return False, f"REJECTED: High correlation ({corr_val:.2f} > 0.70) with open positions", dd_mult, 0.0
        
    return True, f"APPROVED: Risk checklist passed (Size: ${capped_size:.2f}, Sigmoid DD Mult: {dd_mult:.2f}, Heat: {heat_pct:.1f}%, Max Corr: {corr_val:.2f})", dd_mult, capped_size

def get_volatility_regime_multiplier(atr_norm: float, interval: str) -> float:
    """Rule 16: Dynamic Inverse ATR Percentile Sizing."""
    if str(interval) in ["15", "30"]:
        if atr_norm > 0.02:           # Extreme volatility (>2% per candle)
            return 0.5                  # Cut size in half for safety
        elif atr_norm > 0.015:        # High volatility
            return 0.7
        elif 0.005 <= atr_norm <= 0.012: # Sweet spot for 15M/30M trend efficiency
            return 1.2                  # Boost size by +20%
        elif atr_norm < 0.003:        # Flat chop / dead market
            return 0.3                  # Heavy reduction
    return 1.0
