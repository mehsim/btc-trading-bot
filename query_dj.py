import sqlite3
from datetime import datetime, timezone

conn = sqlite3.connect('/home/ubuntu/btc-trading-bot/trading_bot.db')
cur = conn.cursor()

# Show last 2 hours of 15m decisions (no minute filter, all records)
cur.execute("""
SELECT
    datetime(CAST(ts AS INTEGER),'unixepoch') as dt,
    symbol,
    interval,
    direction,
    ROUND(calibrated_conf, 3) as conf,
    outcome,
    reject_reason
FROM decision_journal
WHERE interval='15'
  AND ts > (strftime('%s','now') - 7200)
ORDER BY ts DESC
""")
rows = cur.fetchall()

print(f"{'timestamp (UTC)':<22} {'symbol':<10} {'iv':<4} {'direction':<10} {'conf':>6} {'outcome':<10} {'reason'}")
print('-' * 90)
if rows:
    for r in rows:
        print(f"{str(r[0]):<22} {str(r[1]):<10} {str(r[2]):<4} {str(r[3]):<10} {str(r[4]):>6} {str(r[5]):<10} {str(r[6])}")
else:
    print("(no 15m rows in the last 2 hours)")

# Also show last 5 rows regardless of time
print("\n--- Last 5 rows in decision_journal (any interval) ---")
cur.execute("""
SELECT
    datetime(CAST(ts AS INTEGER),'unixepoch') as dt,
    symbol, interval, direction,
    ROUND(calibrated_conf, 3) as conf,
    outcome, reject_reason
FROM decision_journal
ORDER BY ts DESC LIMIT 5
""")
for r in cur.fetchall():
    print(r)

# Show the ts range
cur.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM decision_journal WHERE interval='15'")
mn, mx, cnt = cur.fetchone()
print(f"\n15m rows: {cnt}")
print(f"  earliest ts: {mn} => {datetime.fromtimestamp(mn, tz=timezone.utc)}")
print(f"  latest ts:   {mx} => {datetime.fromtimestamp(mx, tz=timezone.utc)}")
print(f"  server now:  {datetime.now(tz=timezone.utc)}")

conn.close()
