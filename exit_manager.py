"""
exit_manager.py
----------------
Modular Active Position Exit & Trailing Stop Engine.
Extracted from main.py to decouple position lifecycle management, break-even updates,
trailing stop logic, scale-outs, and exit executions.
"""

import time
from logger import log_event
from trade_calculators import calculate_break_even_stop
from gmm_trail import gmm_trailing_engine
from decay_calibrator import decay_calibrator


def compute_trailing_multiplier(active_trade, tf, current_adx):
    """Computes ADX & time-decayed trailing stop multiplier."""
    trailing_multiplier = gmm_trailing_engine.calculate_gmm_trailing_multiplier(current_adx)
    entry_time_ms = active_trade.get("entry_time")
    if entry_time_ms:
        trade_age_hours = max(0.0, (time.time() - (entry_time_ms / 1000.0)) / 3600.0)
        start_decay_h, decay_rate_unit = decay_calibrator.get_decay_start_and_rate(tf)
        if trade_age_hours > start_decay_h:
            decay_rate = min(0.30, decay_rate_unit * ((trade_age_hours - start_decay_h) / 2.0))
            trailing_multiplier = trailing_multiplier * (1.0 - decay_rate)
    return trailing_multiplier


def evaluate_trailing_and_break_even(
    active_symbol,
    iv,
    tf,
    direction,
    entry_price,
    current_price,
    highest_price,
    lowest_price,
    stop_loss,
    break_even_triggered,
    atr_dollars,
    position_size_usd,
    active_trade,
    required_be_dist,
    trailing_multiplier,
    update_sl_fn,
    trade_mode="live"
):
    """Evaluates trailing stop tightening and break-even triggers for an active open trade."""
    active_trades_updated = False
    trade_leverage = float(active_trade.get("leverage", 1.0))

    if direction == "Bullish":
        if current_price > highest_price:
            highest_price = current_price
            active_trade["highest_price"] = highest_price

        # Trailing stop only activates once trade has moved at least 0.8x ATR into profit
        min_trail_profit_hurdle = entry_price + 0.8 * atr_dollars
        if highest_price >= min_trail_profit_hurdle:
            atr_sl = highest_price - trailing_multiplier * atr_dollars
            # Ensure trailing stop never trails tighter than 0.4x ATR below current price
            max_tight_sl = current_price - 0.4 * atr_dollars
            potential_sl = min(atr_sl, max_tight_sl)

            if break_even_triggered:
                be_floor = calculate_break_even_stop("Bullish", entry_price, current_price, atr_dollars)
                potential_sl = max(potential_sl, be_floor)

            if potential_sl > stop_loss:
                if trade_mode != "simulation":
                    success = update_sl_fn(active_symbol, potential_sl, active_trade)
                    if success:
                        stop_loss = potential_sl
                        active_trade["stop_loss"] = stop_loss
                        active_trades_updated = True
                        gross_r = (stop_loss - entry_price) / max(1e-4, atr_dollars)
                        net_pnl_est = (stop_loss - entry_price) / max(1e-4, entry_price) * position_size_usd * trade_leverage
                        print(f"[{active_symbol} {iv}m Trailing Engine] Mode: TRAILING | Direction: Bullish | Entry: {entry_price:.4f} | New SL: {stop_loss:.4f} | Gross Locked R: {gross_r:+.2f}R | Net Expected PnL: ${net_pnl_est:+.2f}")
                else:
                    stop_loss = potential_sl
                    active_trade["stop_loss"] = stop_loss
                    active_trades_updated = True

        if not break_even_triggered and current_price >= entry_price + required_be_dist:
            target_sl = calculate_break_even_stop(direction, entry_price, current_price, atr_dollars)
            if trade_mode != "simulation":
                success = update_sl_fn(active_symbol, target_sl, active_trade)
                if success:
                    break_even_triggered = True
                    active_trade["break_even_triggered"] = True
                    stop_loss = target_sl
                    active_trade["stop_loss"] = stop_loss
                    active_trades_updated = True
            else:
                break_even_triggered = True
                active_trade["break_even_triggered"] = True
                stop_loss = target_sl
                active_trade["stop_loss"] = stop_loss
                active_trades_updated = True

    else:  # Bearish (Short)
        if current_price < lowest_price:
            lowest_price = current_price
            active_trade["lowest_price"] = lowest_price

        # Trailing stop only activates once trade has moved at least 0.8x ATR into profit
        min_trail_profit_hurdle = entry_price - 0.8 * atr_dollars
        if lowest_price <= min_trail_profit_hurdle:
            atr_sl = lowest_price + trailing_multiplier * atr_dollars
            # Ensure trailing stop never trails tighter than 0.4x ATR above current price
            max_tight_sl = current_price + 0.4 * atr_dollars
            potential_sl = max(atr_sl, max_tight_sl)

            if break_even_triggered:
                be_floor = calculate_break_even_stop("Bearish", entry_price, current_price, atr_dollars)
                potential_sl = min(potential_sl, be_floor)

            if potential_sl < stop_loss:
                if trade_mode != "simulation":
                    success = update_sl_fn(active_symbol, potential_sl, active_trade)
                    if success:
                        stop_loss = potential_sl
                        active_trade["stop_loss"] = stop_loss
                        active_trades_updated = True
                        gross_r = (entry_price - stop_loss) / max(1e-4, atr_dollars)
                        net_pnl_est = (entry_price - stop_loss) / max(1e-4, entry_price) * position_size_usd * trade_leverage
                        print(f"[{active_symbol} {iv}m Trailing Engine] Mode: TRAILING | Direction: Bearish | Entry: {entry_price:.4f} | New SL: {stop_loss:.4f} | Gross Locked R: {gross_r:+.2f}R | Net Expected PnL: ${net_pnl_est:+.2f}")
                else:
                    stop_loss = potential_sl
                    active_trade["stop_loss"] = stop_loss
                    active_trades_updated = True

        if not break_even_triggered and current_price <= entry_price - required_be_dist:
            target_sl = calculate_break_even_stop(direction, entry_price, current_price, atr_dollars)
            if trade_mode != "simulation":
                success = update_sl_fn(active_symbol, target_sl, active_trade)
                if success:
                    break_even_triggered = True
                    active_trade["break_even_triggered"] = True
                    stop_loss = target_sl
                    active_trade["stop_loss"] = stop_loss
                    active_trades_updated = True
            else:
                break_even_triggered = True
                active_trade["break_even_triggered"] = True
                stop_loss = target_sl
                active_trade["stop_loss"] = stop_loss
                active_trades_updated = True

    return {
        "highest_price": highest_price,
        "lowest_price": lowest_price,
        "stop_loss": stop_loss,
        "break_even_triggered": break_even_triggered,
        "active_trades_updated": active_trades_updated
    }
