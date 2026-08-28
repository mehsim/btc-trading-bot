#!/usr/bin/env python3
"""
Simulate the latest 20 skipped low confidence trades for 15M and 60M models.
Pulls from both predictions table and decision_journal to ensure exact 20 trade evaluation.
"""
import sqlite3
import json
import time
import pandas as pd
import numpy as np
from bybit_client import bybit_get_request
from config import TIMEFRAME_CONFIG

def get_candidates(db_path, interval, n=20):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    unique_candidates = []
    seen = set()
    current_time_ms = int(time.time() * 1000)
    lookahead_ms = int(interval) * 60 * 1000 * 6  # at least 6 bars completed
    
    # 1. First check predictions table
    cur.execute("""
        SELECT raw_data FROM predictions 
        WHERE interval = ? 
        ORDER BY timestamp DESC
    """, (str(interval),))
    
    for r in cur.fetchall():
        try:
            d = json.loads(r[0])
            st = str(d.get("status", ""))
            dir_val = str(d.get("direction", ""))
            if "Low Confidence" in st and dir_val in ["Bullish", "Bearish"]:
                sym = d.get("symbol")
                c_ts = d.get("candle_timestamp") or d.get("timestamp") or 0
                c_ts_ms = int(c_ts) if c_ts > 1e11 else int(c_ts * 1000)
                
                if current_time_ms - c_ts_ms < lookahead_ms:
                    continue
                    
                key = (sym, c_ts_ms)
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append({
                        "candle_ts_ms": c_ts_ms,
                        "symbol": sym,
                        "interval": str(interval),
                        "direction": dir_val,
                        "calibrated_conf": float(d.get("calibrated_confidence") or d.get("confidence") or 0.40),
                        "raw_conf": float(d.get("raw_confidence") or 0.50),
                        "regime": "Ranging"
                    })
                if len(unique_candidates) >= n:
                    break
        except Exception:
            pass
            
    # 2. If needed, supplement from decision_journal
    if len(unique_candidates) < n:
        cur.execute("""
            SELECT candle_timestamp, ts, symbol, interval, direction, calibrated_conf, raw_confidence, regime
            FROM decision_journal
            WHERE interval = ? AND (reject_reason LIKE '%Low Confidence%' OR reject_reason LIKE '%confidence%') AND direction IN ('Bullish', 'Bearish')
            ORDER BY ts DESC
        """, (str(interval),))
        for r in cur.fetchall():
            sym = r[2]
            c_ts = r[0] or int(r[1] * 1000)
            c_ts_ms = int(c_ts) if c_ts > 1e11 else int(c_ts * 1000)
            
            if current_time_ms - c_ts_ms < lookahead_ms:
                continue
                
            key = (sym, c_ts_ms)
            if key not in seen:
                seen.add(key)
                unique_candidates.append({
                    "candle_ts_ms": c_ts_ms,
                    "symbol": sym,
                    "interval": str(interval),
                    "direction": r[4],
                    "calibrated_conf": float(r[5] or 0.40),
                    "raw_conf": float(r[6] or 0.50),
                    "regime": r[7] or "Ranging"
                })
            if len(unique_candidates) >= n:
                break
                
    con.close()
    return unique_candidates[:n]

def fetch_subsequent_candles(symbol, interval, start_ts_ms, limit=20):
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": str(interval),
        "start": int(start_ts_ms),
        "limit": limit
    }
    res = bybit_get_request("/v5/market/kline", params)
    if not res or res.get("retCode") != 0:
        return None
    raw_list = res.get("result", {}).get("list", [])
    if not raw_list:
        return None
    candles = []
    for item in reversed(raw_list):
        candles.append({
            "timestamp_ms": int(item[0]),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5])
        })
    df = pd.DataFrame(candles)
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - df["prev_close"]).abs(), (df["low"] - df["prev_close"]).abs()))
    df["atr"] = df["tr"].rolling(14).mean().fillna(df["close"] * 0.008)
    return df

def simulate_batch(candidates, interval):
    if not candidates:
        return []
        
    cfg = TIMEFRAME_CONFIG.get(str(interval), TIMEFRAME_CONFIG.get("15"))
    results = []
    
    for c in candidates:
        sym = c["symbol"]
        c_ts = c["candle_ts_ms"]
        direction = c["direction"]
        regime = c["regime"]
        
        df = fetch_subsequent_candles(sym, interval, c_ts, limit=25)
        if df is None or len(df) < 2:
            continue
            
        entry_row = df.iloc[0]
        entry_price = float(entry_row["close"])
        atr = float(entry_row["atr"])
        if np.isnan(atr) or atr <= 0:
            atr = entry_price * 0.008
            
        is_trending = "Trending" in regime
        tp_mult = float(cfg.get("tp_mult_trending" if is_trending else "tp_mult_ranging", 1.5))
        sl_mult = float(cfg.get("sl_mult", 1.0))
        lookahead = min(int(cfg.get("lookahead", 12)), len(df) - 1)
        
        if direction == "Bullish":
            tp_price = entry_price + (atr * tp_mult)
            sl_price = entry_price - (atr * sl_mult)
        else:
            tp_price = entry_price - (atr * tp_mult)
            sl_price = entry_price + (atr * sl_mult)
            
        subsequent = df.iloc[1 : lookahead + 1]
        if len(subsequent) == 0:
            continue
            
        exit_reason = "TIMEOUT"
        exit_price = float(subsequent.iloc[-1]["close"])
        
        for _, bar in subsequent.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            
            if direction == "Bullish":
                if high >= tp_price:
                    exit_reason = "TP HIT"
                    exit_price = tp_price
                    break
                elif low <= sl_price:
                    exit_reason = "SL HIT"
                    exit_price = sl_price
                    break
            else: # Bearish
                if low <= tp_price:
                    exit_reason = "TP HIT"
                    exit_price = tp_price
                    break
                elif high >= sl_price:
                    exit_reason = "SL HIT"
                    exit_price = sl_price
                    break
                    
        if direction == "Bullish":
            raw_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            raw_pct = (entry_price - exit_price) / entry_price * 100.0
            
        net_pct = raw_pct - 0.15  # 0.12% fees + 0.03% slippage
        is_win = net_pct > 0
        
        time_str = time.strftime("%Y-%m-%d %H:%M", time.gmtime(c_ts / 1000))
        
        results.append({
            "symbol": sym,
            "candle_time": time_str,
            "direction": direction,
            "calibrated_conf": c["calibrated_conf"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "exit_reason": exit_reason,
            "raw_pct": raw_pct,
            "net_pct": net_pct,
            "is_win": is_win
        })
        time.sleep(0.04) # API rate limit respect
        
    return results

if __name__ == "__main__":
    db_file = "trading_bot.db"
    
    cands_15 = get_candidates(db_file, 15, 20)
    res_15 = simulate_batch(cands_15, 15)
    
    cands_60 = get_candidates(db_file, 60, 20)
    res_60 = simulate_batch(cands_60, 60)
    
    print("\n" + "="*88)
    print(f"📊 15-MINUTE MODEL: SIMULATION OF LATEST {len(res_15)} SKIPPED (LOW CONFIDENCE) TRADES")
    print("="*88)
    if res_15:
        wins_15 = sum(1 for r in res_15 if r["is_win"])
        tot_pnl_15 = sum(r["net_pct"] for r in res_15)
        wr_15 = (wins_15 / len(res_15)) * 100.0
        print(f"Win Rate: {wins_15}/{len(res_15)} ({wr_15:.1f}%) | Cumulative Net PnL: {tot_pnl_15:+.2f}%\n")
        print(f"{'#':<3} {'Time (UTC)':<17} {'Symbol':<9} {'Dir':<8} {'CalConf':<8} {'Entry':<10} {'Exit':<10} {'Reason':<8} {'Net PnL':<9} {'Result'}")
        print("-" * 92)
        for i, r in enumerate(res_15, 1):
            res_str = "✅ WIN" if r["is_win"] else "❌ LOSS"
            print(f"{i:<3} {r['candle_time']:<17} {r['symbol']:<9} {r['direction']:<8} {r['calibrated_conf']*100:.1f}%   {r['entry_price']:<10.4f} {r['exit_price']:<10.4f} {r['exit_reason']:<8} {r['net_pct']:+6.2f}%   {res_str}")

    print("\n" + "="*88)
    print(f"📊 60-MINUTE MODEL: SIMULATION OF LATEST {len(res_60)} SKIPPED (LOW CONFIDENCE) TRADES")
    print("="*88)
    if res_60:
        wins_60 = sum(1 for r in res_60 if r["is_win"])
        tot_pnl_60 = sum(r["net_pct"] for r in res_60)
        wr_60 = (wins_60 / len(res_60)) * 100.0
        print(f"Win Rate: {wins_60}/{len(res_60)} ({wr_60:.1f}%) | Cumulative Net PnL: {tot_pnl_60:+.2f}%\n")
        print(f"{'#':<3} {'Time (UTC)':<17} {'Symbol':<9} {'Dir':<8} {'CalConf':<8} {'Entry':<10} {'Exit':<10} {'Reason':<8} {'Net PnL':<9} {'Result'}")
        print("-" * 92)
        for i, r in enumerate(res_60, 1):
            res_str = "✅ WIN" if r["is_win"] else "❌ LOSS"
            print(f"{i:<3} {r['candle_time']:<17} {r['symbol']:<9} {r['direction']:<8} {r['calibrated_conf']*100:.1f}%   {r['entry_price']:<10.4f} {r['exit_price']:<10.4f} {r['exit_reason']:<8} {r['net_pct']:+6.2f}%   {res_str}")
