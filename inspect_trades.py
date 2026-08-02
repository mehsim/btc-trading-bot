import json
import sqlite3
import datetime

data = json.load(open('dashboard_history.json'))
trades = data.get('trade_history', [])[-10:]

print('=== LAST 10 CLOSED TRADES IN DASHBOARD HISTORY ===\n')
for i, t in enumerate(trades, 1):
    symbol = t.get('symbol')
    interval = t.get('interval')
    direction = t.get('direction')
    entry = float(t.get('entry_price') or 0.0)
    exit_p = float(t.get('exit_price') or 0.0)
    tp = float(t.get('take_profit') or 0.0)
    sl = float(t.get('stop_loss') or 0.0)
    atr = float(t.get('atr_dollars') or 0.0)
    pnl = float(t.get('pnl_usd') or 0.0)
    reason = t.get('reason')
    exit_ts = float(t.get('exit_time') or 0.0)
    exit_str = datetime.datetime.fromtimestamp(exit_ts, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if exit_ts else 'N/A'
    
    if atr > 0 and entry > 0:
        tp_dist_r = abs(tp - entry) / atr if tp > 0 else 0.0
        sl_dist_r = abs(entry - sl) / atr if sl > 0 else 0.0
    else:
        tp_dist_r = 0.0
        sl_dist_r = 0.0

    print(f"[{i}] {symbol} ({interval}m) - {direction}")
    print(f"    Exit Time  : {exit_str}")
    print(f"    Entry Price: {entry:.4f}")
    print(f"    Exit Price : {exit_p:.4f}")
    print(f"    Take Profit: {tp:.4f} (Dist: {tp_dist_r:.2f}x ATR)")
    print(f"    Stop Loss  : {sl:.4f} (Dist: {sl_dist_r:.2f}x ATR)")
    print(f"    ATR ($)    : {atr:.4f}")
    print(f"    PnL ($)    : ${pnl:.4f}")
    print(f"    Reason     : {reason}")
    print('-'*60)
