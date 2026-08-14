import sqlite3, database
from datetime import datetime, timezone

conn = sqlite3.connect('trading_bot.db')
cur = conn.cursor()

print('=== NON-NEUTRAL DECISIONS LAST 1 HOUR ===')
cur.execute("SELECT ts, symbol, interval, direction FROM decision_journal WHERE ts > strftime('%s','now','-1 hour') AND direction != 'Neutral' ORDER BY ts DESC LIMIT 20")
rows = cur.fetchall()
if rows:
    for r in rows:
        dt = datetime.fromtimestamp(r[0], timezone.utc).strftime('%H:%M UTC')
        print(f'• {dt} | {r[1]} | {r[2]}m | {r[3]}')
else:
    preds = database.get_prediction_history(limit=200)
    non_neutral = [p for p in (preds or []) if p.get('direction') not in ['Neutral', 'Hold']]
    for p in sorted(non_neutral, key=lambda x: x.get('candle_timestamp', 0), reverse=True)[:10]:
        ts = p.get('candle_timestamp', 0)
        dt = datetime.fromtimestamp(ts/1000, timezone.utc).strftime('%H:%M UTC') if ts > 1e11 else str(ts)
        sym = p.get('symbol'); iv = p.get('interval'); d = p.get('direction'); st = p.get('status')
        print(f'• {dt} | {sym} | {iv}m | {d} | {st}')

print()
print('=== ACTIVE TRADES ===')
cur.execute('SELECT symbol, direction, entry_time FROM active_trades')
trades = cur.fetchall()
print(trades if trades else 'No active trades')
conn.close()
