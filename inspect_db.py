import sqlite3

conn = sqlite3.connect('/home/ubuntu/btc-trading-bot/trading_bot.db')
cur = conn.cursor()

# List all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

# Check decision_journal
if 'decision_journal' in tables:
    cur.execute("PRAGMA table_info(decision_journal)")
    cols = cur.fetchall()
    print("Columns:", [c[1] for c in cols])
    cur.execute("SELECT COUNT(*) FROM decision_journal")
    print("Total rows:", cur.fetchone()[0])
    cur.execute("SELECT * FROM decision_journal ORDER BY rowid DESC LIMIT 5")
    rows = cur.fetchall()
    print("Last 5 rows:")
    for r in rows:
        print(r)
else:
    print("decision_journal table NOT FOUND")
    for t in tables:
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        cnt = cur.fetchone()[0]
        print(f"  {t}: {cnt} rows")

conn.close()
