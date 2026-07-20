import numpy as np
import pandas as pd
import time
import json

def calculate_drawdown_multiplier(current_equity: float, peak_equity: float) -> float:
    if peak_equity <= 0:
        return 1.0
    drawdown_pct = (peak_equity - current_equity) / peak_equity * 100.0
    if drawdown_pct >= 20.0:
        return 0.0  # Circuit breaker halt
    elif drawdown_pct >= 15.0:
        return 0.25
    elif drawdown_pct >= 10.0:
        return 0.50
    elif drawdown_pct >= 5.0:
        return 0.75
    else:
        return 1.0

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
    
    target_returns = df_dict[symbol]["close"].pct_change().dropna().iloc[-100:]
    max_corr = 0.0
    
    for pos in open_positions:
        if isinstance(pos, dict):
            pos_symbol = pos.get("symbol")
            if pos_symbol and pos_symbol in df_dict and pos_symbol != symbol and isinstance(df_dict[pos_symbol], pd.DataFrame):
                if "close" in df_dict[pos_symbol].columns:
                    other_returns = df_dict[pos_symbol]["close"].pct_change().dropna().iloc[-100:]
                    combined = pd.concat([target_returns, other_returns], axis=1).dropna()
                    if len(combined) >= 20:
                        corr_matrix = combined.corr()
                        if corr_matrix.shape == (2, 2):
                            corr_val = corr_matrix.iloc[0, 1]
                            if not np.isnan(corr_val):
                                max_corr = max(max_corr, abs(float(corr_val)))
                    
    return max_corr

def check_portfolio_heat(open_positions: list, candidate_size_usd: float, candidate_lev: float, total_equity: float) -> tuple:
    if total_equity <= 0:
        return False, 0.0
    
    current_heat_usd = sum(
        float(p.get("position_size_usd", 0.0)) * float(p.get("leverage", 1.0))
        for p in open_positions if isinstance(p, dict)
    )
    new_heat_usd = current_heat_usd + (candidate_size_usd * candidate_lev)
    heat_pct = (new_heat_usd / total_equity) * 100.0
    
    # Cap total portfolio heat at 300% (3.0x equity exposure)
    is_safe = heat_pct <= 300.0
    return is_safe, heat_pct

def check_margin_utilization(used_margin: float, total_equity: float) -> str:
    if total_equity <= 0:
        return "NORMAL"
    utilization_pct = (used_margin / total_equity) * 100.0
    if utilization_pct >= 85.0:
        return "EMERGENCY_CLOSE"
    elif utilization_pct >= 70.0:
        return "HALT_ENTRIES"
    elif utilization_pct >= 50.0:
        return "WARNING_ALERT"
    return "NORMAL"

def evaluate_pre_trade_checklist(symbol: str, position_size_usd: float, leverage_val: float, active_trades: list, bot_state: dict, df_dict: dict) -> tuple:
    equity = float(bot_state.get("simulated_balance", 80.0))
    peak_equity = float(bot_state.get("peak_balance", equity))
    
    # 1. Drawdown scaling check
    dd_mult = calculate_drawdown_multiplier(equity, peak_equity)
    if dd_mult == 0.0:
        return False, "REJECTED: Circuit breaker active (>=20% Drawdown)", 0.0
    
    # 2. Portfolio heat check
    heat_safe, heat_pct = check_portfolio_heat(active_trades, position_size_usd, leverage_val, equity)
    if not heat_safe:
        return False, f"REJECTED: Portfolio heat ({heat_pct:.1f}%) exceeds 300% cap", dd_mult
    
    # 3. Correlation check
    corr_val = calculate_portfolio_correlation(symbol, active_trades, df_dict)
    if corr_val > 0.7:
        return False, f"REJECTED: High correlation ({corr_val:.2f} > 0.70) with open positions", dd_mult
        
    return True, f"APPROVED: Risk checklist passed (Drawdown Mult: {dd_mult:.2f}, Heat: {heat_pct:.1f}%, Max Corr: {corr_val:.2f})", dd_mult
