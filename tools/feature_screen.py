import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data import get_history, merge_derivatives_sentiment_features
from core import add_features

parser = argparse.ArgumentParser(description="Pooled Feature Screener")
parser.add_argument("--interval", type=str, default="60")
parser.add_argument("--pages", type=int, default=40)
parser.add_argument("--lookahead", type=int, default=36)
args = parser.parse_args()

IV = str(args.interval)
PAGES = int(args.pages)
HORIZON = int(args.lookahead)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
           "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

print(f"Fetching and engineering features for {len(SYMBOLS)} symbols (Interval={IV}m, Pages={PAGES}, Horizon={HORIZON} bars)...")
frames = []
for sym in SYMBOLS:
    try:
        d = get_history(symbol=sym, interval=IV, limit=1000, pages=PAGES)
        if d is None or len(d) < 500:
            print(f"  skip {sym}")
            continue
        d["close_btc"] = d["close"]
        d = merge_derivatives_sentiment_features(d, symbol=sym, interval=IV)
        d = add_features(d)
        d["fwd"] = (d["close"].shift(-HORIZON) / d["close"] - 1.0) * 100.0
        d = d.dropna(subset=["fwd"]).iloc[::HORIZON].reset_index(drop=True)  # non-overlapping, PER SYMBOL
        d["symbol"] = sym
        frames.append(d)
        print(f"  {sym}: {len(d)} independent obs")
    except Exception as e:
        print(f"  error loading {sym}: {e}")

df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
print(f"\npooled: {len(df)} independent observations\n")
split = int(len(df) * 0.8)

# Candidate features
candidate_features = [
    c for c in df.columns 
    if c not in ["open_time", "timestamp", "open", "high", "low", "close", "volume", "close_btc", "fwd", "target", "target_price", "symbol"] 
    and pd.api.types.is_numeric_dtype(df[c])
]

def screen(frame, col):
    f = frame[[col, "fwd", "symbol"]].dropna()
    if len(f) < 200 or f[col].nunique() < 20:
        return None
    r = f.groupby("symbol")[col].rank(pct=True)  # rank WITHIN symbol
    try:
        dec = pd.qcut(r, 10, labels=False, duplicates="drop")
    except Exception:
        return None
    g = f.groupby(dec)["fwd"].agg(["mean", "count"])
    if len(g) < 5:
        return None
    spread = g["mean"].iloc[-1] - g["mean"].iloc[0]
    se = f["fwd"].std() * np.sqrt(1.0 / g["count"].iloc[-1] + 1.0 / g["count"].iloc[0])
    mono = np.corrcoef(np.arange(len(g)), g["mean"].values)[0, 1]
    return spread, mono, se

rows = []
for feat in candidate_features:
    a = screen(df.iloc[:split], feat)
    b = screen(df.iloc[split:], feat)
    if a is None or b is None:
        continue
    te_t = b[0] / b[2] if b[2] > 0 else 0.0
    rows.append((feat, a[0], a[1], b[0], b[1], b[2], te_t))

out = pd.DataFrame(rows, columns=["feature", "tr_spread%", "tr_mono", "te_spread%", "te_mono", "te_SE%", "te_t"])
out["abs_te_t"] = out["te_t"].abs()
out = out.sort_values(by="abs_te_t", ascending=False).drop(columns=["abs_te_t"])

print("=" * 90)
print(f"POOLED MULTI-ASSET FEATURE SCREENING REPORT (Interval={IV}m, Horizon={HORIZON}b, {len(SYMBOLS)} symbols, {len(df)} independent obs)")
print("=" * 90)
print(out.head(35).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# Identify robust candidates
winners = out[
    (np.sign(out["tr_spread%"]) == np.sign(out["te_spread%"])) &
    (out["te_t"].abs() >= 2.0) &
    (out["tr_mono"].abs() >= 0.6) &
    (out["te_mono"].abs() >= 0.6)
]

print("\n" + "=" * 90)
print("QUALIFIED ALPHA CANDIDATES (Same sign both splits, |te_t| >= 2.0, |mono| >= 0.6 in both):")
print("=" * 90)
if len(winners) > 0:
    print(winners.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
else:
    print("None qualified (|te_t| < 2.0 across all tested features - consistent with random noise).")
print("=" * 90 + "\n")
