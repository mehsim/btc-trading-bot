# BTC Trading Bot - Mitigation Implementation Plan

This document outlines the detailed code modifications required to implement the audit recommendations: **Thread Safety**, **SQLite Migration**, and **Volatility-Based Order Routing**.

---

## 1. Thread Lock Implementation (`main.py`)

### A. Define the Global State Lock
Near the top of `main.py` (e.g., around line 150), define a re-entrant lock to serialize state modifications:
```python
import threading

# Re-entrant lock for thread-safe access to bot_state and file IO
bot_state_lock = threading.RLock()
```

### B. Wrap State Updates & File Writes
Wrap all state changes and logging writes inside `with bot_state_lock:` context managers:

#### 1. In `save_history()`:
```python
def save_history():
    with bot_state_lock:
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(bot_state, f, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save history: {e}")
```

#### 2. In trade execution & updates (e.g. where positions are modified/appended):
```python
with bot_state_lock:
    bot_state[active_trade_key].append(new_trade)
    bot_state["simulated_balance"] = new_balance
    save_history()
```

---

## 2. SQLite Migration Plan (`data.py` & `main.py`)

### A. Create Database Schema
Create a new file `db.py` (or embed inside `data.py`) to manage connection setup:
```python
import sqlite3

def init_db(db_path="bot_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Trades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS completed_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            symbol TEXT,
            interval TEXT,
            direction TEXT,
            qty REAL,
            entry_price REAL,
            exit_price REAL,
            pnl_usd REAL,
            outcome TEXT,
            exit_reason TEXT
        )
    """)
    
    # 2. Market cache table (OHLCV + OI + Funding)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_cache (
            timestamp INTEGER PRIMARY KEY,
            symbol TEXT,
            interval TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            open_interest REAL,
            funding_rate REAL,
            fear_greed REAL
        )
    """)
    
    conn.commit()
    conn.close()
```

### B. Refactor Caching to SQLite
Replace file reads/writes in `data.py` with SQL execution:
```python
def save_candles_to_db(df, symbol, interval, db_path="bot_data.db"):
    conn = sqlite3.connect(db_path)
    df.to_sql("market_cache", conn, if_exists="append", index=False, chunksize=1000)
    conn.close()
```

---

## 3. Volatility-Based Order Routing

### A. Define Volatility Safeguard Constant
In `main.py` configurations, define the maximum allowable normal volatility threshold for market orders:
```python
# Maximum normal volatility (ATR / Price) to allow market entry fallback (1.5%)
MAX_VOLATILITY_THRESHOLD = 0.015
```

### B. Inject Safeguard Check Before Market Fallback
Modify the order placement block in `main.py` (around line 4820):
```python
# Fallback to Market order if all limit order chases failed
if not bybit_success:
    atr_norm = float(latest_candle.get("ATR_norm", 0.0))
    if atr_norm >= MAX_VOLATILITY_THRESHOLD:
        print(f"[{symbol} {iv}m API BLOCK] Limit chases failed. Market fallback blocked due to extreme volatility (ATR_norm: {atr_norm:.4f} >= {MAX_VOLATILITY_THRESHOLD:.4f}).")
        status_msg = "Skipped (Volatility Block)"
    else:
        print(f"[{symbol} {iv}m API] All Limit Maker chases failed. Falling back to Market order...")
        order_res = place_bybit_order(symbol, side, qty_str)
        # ... proceed with market execution logic ...
```
