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


def compute_dynamic_trail_params(iv, tf, entry_price, atr_dollars, current_adx, regime="Trending"):
    """
    Computes dynamic, adaptive profit hurdle and minimum trailing buffer based on:
    - Timeframe scale (scalp 15m/30m vs swing 1h/4h)
    - Market regime & ADX trend strength
    - Volatility (ATR dollars)
    """
    # 1. Base Timeframe Hurdle Multipliers
    tf_hurdle_map = {
        "15": 0.60,
        "30": 0.75,
        "60": 0.90,
        "120": 1.05,
        "240": 1.20,
        "360": 1.35
    }
    base_hurdle_mult = tf_hurdle_map.get(str(iv), 0.85)

    # 2. ADX Trend Strength & Regime Adaptation
    # Strong trends (ADX >= 30) give more breathing room (+20%) to ride multi-candle momentum.
    # Ranging/Chop (ADX <= 20) triggers trailing sooner (-15%) to bank quick mean-reversion profits.
    if current_adx >= 30.0:
        regime_factor = 1.20
    elif current_adx <= 20.0 or "Ranging" in regime:
        regime_factor = 0.85
    else:
        regime_factor = 1.0

    dynamic_hurdle_mult = base_hurdle_mult * regime_factor
    profit_hurdle_dist = dynamic_hurdle_mult * atr_dollars

    # 3. Dynamic Minimum Protective Trail Buffer
    # Ensures the trailing stop never suffocates within market noise or bid-ask spread
    # In choppy/low ADX markets, wider safety margin (0.45x ATR); in strong trends, 0.35x ATR
    adx_clamp = max(10.0, min(50.0, float(current_adx)))
    trail_buffer_mult = 0.35 + 0.15 * ((50.0 - adx_clamp) / 40.0)
    min_trail_buffer = max(0.0015 * entry_price, trail_buffer_mult * atr_dollars)

    return profit_hurdle_dist, min_trail_buffer


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
    """Evaluates fully dynamic trailing stop tightening and break-even triggers for an active open trade."""
    active_trades_updated = False
    trade_leverage = float(active_trade.get("leverage", 1.0))
    current_adx = float(active_trade.get("adx", 22.0))
    regime = str(active_trade.get("entry_regime", "Trending"))

    # Compute fully adaptive hurdle and safety buffer
    profit_hurdle_dist, min_trail_buffer = compute_dynamic_trail_params(
        iv=iv,
        tf=tf,
        entry_price=entry_price,
        atr_dollars=atr_dollars,
        current_adx=current_adx,
        regime=regime
    )

    if direction == "Bullish":
        if current_price > highest_price:
            highest_price = current_price
            active_trade["highest_price"] = highest_price

        # Trailing stop only activates once trade has cleared the dynamic profit hurdle
        min_trail_profit_hurdle = entry_price + profit_hurdle_dist
        if highest_price >= min_trail_profit_hurdle:
            atr_sl = highest_price - trailing_multiplier * atr_dollars
            # Dynamic safety clamp: trailing stop never suffocates within min_trail_buffer of current price
            max_tight_sl = current_price - min_trail_buffer
            potential_sl = min(atr_sl, max_tight_sl)

            if break_even_triggered:
                be_floor = calculate_break_even_stop("Bullish", entry_price, current_price, atr_dollars, interval=iv)
                potential_sl = max(potential_sl, be_floor)
            potential_sl = min(potential_sl, current_price * 0.9995)

            if potential_sl > stop_loss:
                if trade_mode != "simulation":
                    success = update_sl_fn(active_symbol, potential_sl, active_trade)
                    if success:
                        stop_loss = potential_sl
                        active_trade["stop_loss"] = stop_loss
                        active_trades_updated = True
                        gross_r = (stop_loss - entry_price) / max(1e-4, atr_dollars)
                        net_pnl_est = (stop_loss - entry_price) / max(1e-4, entry_price) * position_size_usd * trade_leverage
                        print(f"[{active_symbol} {iv}m Dynamic Trailing] Direction: Bullish | Entry: {entry_price:.4f} | New SL: {stop_loss:.4f} | Locked: {gross_r:+.2f}R | Est PnL: ${net_pnl_est:+.2f}")
                else:
                    stop_loss = potential_sl
                    active_trade["stop_loss"] = stop_loss
                    active_trades_updated = True

        if not break_even_triggered and current_price >= entry_price + required_be_dist:
            target_sl = calculate_break_even_stop(direction, entry_price, current_price, atr_dollars, interval=iv)
            if trade_mode != "simulation":
                success = update_sl_fn(active_symbol, target_sl, active_trade)
                if success:
                    break_even_triggered = True
                    active_trade["break_even_triggered"] = True
                    stop_loss = target_sl
                    active_trade["stop_loss"] = stop_loss
                    active_trades_updated = True
                    print(f"[{active_symbol} {iv}m Dynamic Break-Even] Moved SL to cost-aware break-even: {target_sl:.4f}")
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

        # Trailing stop only activates once trade has cleared the dynamic profit hurdle
        min_trail_profit_hurdle = entry_price - profit_hurdle_dist
        if lowest_price <= min_trail_profit_hurdle:
            atr_sl = lowest_price + trailing_multiplier * atr_dollars
            # Dynamic safety clamp: trailing stop never suffocates within min_trail_buffer of current price
            max_tight_sl = current_price + min_trail_buffer
            potential_sl = max(atr_sl, max_tight_sl)

            if break_even_triggered:
                be_floor = calculate_break_even_stop("Bearish", entry_price, current_price, atr_dollars, interval=iv)
                potential_sl = min(potential_sl, be_floor)
            potential_sl = max(potential_sl, current_price * 1.0005)

            if potential_sl < stop_loss:
                if trade_mode != "simulation":
                    success = update_sl_fn(active_symbol, potential_sl, active_trade)
                    if success:
                        stop_loss = potential_sl
                        active_trade["stop_loss"] = stop_loss
                        active_trades_updated = True
                        gross_r = (entry_price - stop_loss) / max(1e-4, atr_dollars)
                        net_pnl_est = (entry_price - stop_loss) / max(1e-4, entry_price) * position_size_usd * trade_leverage
                        print(f"[{active_symbol} {iv}m Dynamic Trailing] Direction: Bearish | Entry: {entry_price:.4f} | New SL: {stop_loss:.4f} | Locked: {gross_r:+.2f}R | Est PnL: ${net_pnl_est:+.2f}")
                else:
                    stop_loss = potential_sl
                    active_trade["stop_loss"] = stop_loss
                    active_trades_updated = True

        if not break_even_triggered and current_price <= entry_price - required_be_dist:
            target_sl = calculate_break_even_stop(direction, entry_price, current_price, atr_dollars, interval=iv)
            if trade_mode != "simulation":
                success = update_sl_fn(active_symbol, target_sl, active_trade)
                if success:
                    break_even_triggered = True
                    active_trade["break_even_triggered"] = True
                    stop_loss = target_sl
                    active_trade["stop_loss"] = stop_loss
                    active_trades_updated = True
                    print(f"[{active_symbol} {iv}m Dynamic Break-Even] Moved SL to cost-aware break-even: {target_sl:.4f}")
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
