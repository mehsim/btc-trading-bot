import os
import sys
import json
import sqlite3
import datetime
import argparse

# Configuration paths
WORKSPACE = "/home/ubuntu/btc-trading-bot" if os.path.exists("/home/ubuntu/btc-trading-bot") else os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(WORKSPACE, "trading_bot.db")
KLINE_DB_FILE = os.path.join(WORKSPACE, "kline_cache.db")

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate skipped trades for a specific day.")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD format (UTC). Defaults to today.")
    return parser.parse_args()

def calculate_atr_at_time(kline_conn, symbol, interval, entry_time_ms):
    cur = kline_conn.cursor()
    cur.execute("""
        SELECT open, high, low, close 
        FROM kline_data 
        WHERE symbol = ? AND interval = ? AND timestamp <= ? 
        ORDER BY timestamp DESC LIMIT 50
    """, (symbol, interval, entry_time_ms))
    kline_rows = cur.fetchall()
    if len(kline_rows) < 15:
        return 0.0
    kline_rows = kline_rows[::-1]
    
    trs = []
    for i in range(len(kline_rows)):
        high = kline_rows[i][1]
        low = kline_rows[i][2]
        if i == 0:
            tr = high - low
        else:
            prev_close = kline_rows[i-1][3]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        
    atr = trs[0]
    for val in trs[1:]:
        atr = (atr * 13 + val) / 14
    return atr

def main():
    args = parse_args()
    
    # Get date window
    if args.date:
        target_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = datetime.datetime.now(datetime.UTC).date()
        
    start_dt = datetime.datetime.combine(target_date, datetime.time.min, tzinfo=datetime.UTC)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max, tzinfo=datetime.UTC)
    
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()
    
    print(f"=== Skipped Trades Simulation for {target_date} (UTC) ===")
    print(f"Time Window: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not os.path.exists(DB_FILE) or not os.path.exists(KLINE_DB_FILE):
        print(f"Error: Database files not found in {WORKSPACE}")
        sys.exit(1)
        
    conn = sqlite3.connect(DB_FILE)
    kline_conn = sqlite3.connect(KLINE_DB_FILE)
    
    c = conn.cursor()
    c.execute('SELECT raw_data FROM predictions WHERE timestamp >= ? AND timestamp <= ?', (start_ts, end_ts))
    rows = c.fetchall()
    
    skipped_trades = []
    for r in rows:
        try:
            d = json.loads(r[0])
            status = d.get("status", "")
            if "skip" in status.lower():
                skipped_trades.append(d)
        except Exception:
            pass
            
    print(f"Total Skipped Predictions Found: {len(skipped_trades)}")
    if not skipped_trades:
        print("No skipped trades to simulate.")
        return
        
    results = []
    total_pnl = 0.0
    
    for t in skipped_trades:
        symbol = t['symbol']
        interval = t['interval']
        direction = t['direction']
        entry_price = t['ref_price']
        ts_raw = t['timestamp']
        entry_time_ms = int(ts_raw) if ts_raw > 1e11 else int(ts_raw * 1000)

        
        atr = calculate_atr_at_time(kline_conn, symbol, interval, entry_time_ms)
        if atr == 0.0:
            atr = entry_price * 0.01
            
        calibrated_confidence = t.get('calibrated_confidence', t.get('confidence', 0.5))
        dynamic_threshold = t.get('dynamic_threshold', 0.58)
        sl_multiplier = 1.5
        if calibrated_confidence > dynamic_threshold and dynamic_threshold < 1.0:
            confidence_ratio = (calibrated_confidence - dynamic_threshold) / (1.0 - dynamic_threshold)
            sl_multiplier_adjusted = sl_multiplier * (1.0 - 0.3 * confidence_ratio)
        else:
            sl_multiplier_adjusted = sl_multiplier
            
        tp_multiplier = 1.25
        
        if direction == 'Bullish':
            sl = entry_price - sl_multiplier_adjusted * atr
            tp = entry_price + tp_multiplier * atr
            scale_out = entry_price + 1.0 * atr
        else:
            sl = entry_price + sl_multiplier_adjusted * atr
            tp = entry_price - tp_multiplier * atr
            scale_out = entry_price - 1.0 * atr
            
        cur = kline_conn.cursor()
        cur.execute('''
            SELECT timestamp, open, high, low, close 
            FROM kline_data 
            WHERE symbol = ? AND interval = ? AND timestamp > ? 
            ORDER BY timestamp ASC LIMIT 24
        ''', (symbol, interval, entry_time_ms))
        future_candles = cur.fetchall()
        
        outcome = 'Expired'
        pnl = 0.0
        half_closed = False
        exit_price = None
        
        for candle in future_candles:
            c_time, c_open, c_high, c_low, c_close = candle
            
            if not half_closed:
                if direction == 'Bullish' and c_high >= scale_out:
                    half_closed = True
                    pnl += 0.5 * (scale_out - entry_price) / entry_price
                    sl = entry_price
                elif direction == 'Bearish' and c_low <= scale_out:
                    half_closed = True
                    pnl += 0.5 * (entry_price - scale_out) / entry_price
                    sl = entry_price
                    
            if direction == 'Bullish':
                if c_low <= sl:
                    outcome = 'Stop Loss'
                    exit_price = sl
                    if half_closed:
                        pass
                    else:
                        pnl = - (entry_price - sl) / entry_price
                    break
                elif c_high >= tp:
                    outcome = 'Take Profit'
                    exit_price = tp
                    if half_closed:
                        pnl += 0.5 * (tp - entry_price) / entry_price
                    else:
                        pnl = (tp - entry_price) / entry_price
                    break
            else:
                if c_high >= sl:
                    outcome = 'Stop Loss'
                    exit_price = sl
                    if half_closed:
                        pass
                    else:
                        pnl = - (sl - entry_price) / entry_price
                    break
                elif c_low <= tp:
                    outcome = 'Take Profit'
                    exit_price = tp
                    if half_closed:
                        pnl += 0.5 * (entry_price - tp) / entry_price
                    else:
                        pnl = (entry_price - tp) / entry_price
                    break
        else:
            if len(future_candles) > 0:
                final_close = future_candles[-1][4]
                exit_price = final_close
                multiplier = 0.5 if half_closed else 1.0
                if direction == 'Bullish':
                    pnl += multiplier * (final_close - entry_price) / entry_price
                else:
                    pnl += multiplier * (entry_price - final_close) / entry_price
                outcome = 'Expired'
            else:
                outcome = 'No Data'
                pnl = 0.0
                
        pnl_pct = pnl * 100
        total_pnl += pnl_pct
        
        dt_str = datetime.datetime.fromtimestamp(entry_time_sec, datetime.UTC).strftime('%H:%M')
        results.append({
            'time': dt_str,
            'symbol': symbol,
            'interval': interval,
            'direction': direction,
            'entry': entry_price,
            'exit': exit_price,
            'status': status.replace('Skipped ', ''),
            'outcome': outcome,
            'pnl_pct': pnl_pct
        })
        
    # Group results by outcome
    outcomes_count = {}
    for r in results:
        outcomes_count[r['outcome']] = outcomes_count.get(r['outcome'], 0) + 1
        
    print("| Time (UTC) | Symbol | TF | Direction | Entry | Exit | Reason | Outcome | PnL % |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    # Show first 15 rows to keep summary readable, summarize remainder
    for r in results[:15]:
        print(f"| {r['time']} | {r['symbol']} | {r['interval']}m | {r['direction']} | {r['entry']:.4f} | {r['exit']:.4f} | {r['status']} | {r['outcome']} | {r['pnl_pct']:+.4f}% |")
    
    if len(results) > 15:
        print(f"| ... | ... | ... | ... | ... | ... | ... | ... | ... |")
        print(f"| *Note* | *and {len(results)-15} more trades* | | | | | | | |")
        
    print(f"\nTotal Simulated PnL: {total_pnl:+.4f}%")
    print("Outcomes Breakdown:")
    for k, v in outcomes_count.items():
        print(f"  - {k}: {v} trades")

if __name__ == "__main__":
    main()
