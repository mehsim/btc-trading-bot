import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from logger import log_event

parser = argparse.ArgumentParser(description="Pooled Feature Screener with Cross-Sectional Residuals")
parser.add_argument("--interval", type=str, default="60")
parser.add_argument("--pages", type=int, default=40)
parser.add_argument("--lookahead", type=int, default=36)
parser.add_argument("--xs", dest="xs", action="store_true", default=True, help="Use cross-sectional basket-demeaned forward returns (default: True)")
parser.add_argument("--no-xs", dest="xs", action="store_false", help="Use raw absolute forward returns")
parser.add_argument("--start-ts", type=str, default=None, help="Filter pooled dataset to timestamp >= start_ts (ISO date or epoch ms)")
parser.add_argument("--mode", type=str, default="all", choices=["all", "price_only", "derivatives_only"],
                    help="Screening mode: 'all', 'price_only' (excludes OI/funding), or 'derivatives_only' (auto-windowed)")
args = parser.parse_args()

IV = str(args.interval)
PAGES = int(args.pages)
HORIZON = int(args.lookahead)
USE_XS = bool(args.xs)
MODE = str(args.mode)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
           "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

# Coverage boundaries for derivatives
OI_COVERAGE_START_MS = 1748304000000       # ~2025-05-27 UTC (451 days)
FUNDING_COVERAGE_START_MS = 1778544000000  # ~2026-05-12 UTC (101 days)

def parse_start_ts(val):
    if not val:
        return None
    try:
        if str(val).isdigit():
            ts = int(val)
            return ts if ts > 1e11 else ts * 1000
        dt = pd.to_datetime(val, utc=True)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None

START_TS_MS = parse_start_ts(args.start_ts)
if MODE == "derivatives_only" and START_TS_MS is None:
    START_TS_MS = FUNDING_COVERAGE_START_MS
    print(f"[Derivatives Mode] Auto-restricting to verified funding coverage window (>= 2026-05-12)...")

print(f"Fetching and engineering features for {len(SYMBOLS)} symbols (Interval={IV}m, Pages={PAGES}, Horizon={HORIZON}b, XS={USE_XS}, Mode={MODE})...")
frames = []
for sym in SYMBOLS:
    try:
        d = get_history(symbol=sym, interval=IV, limit=1000, pages=PAGES)
        if d is None or len(d) < 500:
            print(f"  skip {sym}")
            continue
        d["close_btc"] = d["close"]
        d = merge_derivatives_sentiment_features(d, symbol=sym, interval=IV)
        d = add_features(d, symbol=sym, interval=IV)
        d["fwd"] = (d["close"].shift(-HORIZON) / d["close"] - 1.0) * 100.0
        # Retain contiguous series for time-alignment (do not decimate per-symbol)
        d = d.dropna(subset=["fwd"]).reset_index(drop=True)
        d["symbol"] = sym
        frames.append(d)
        print(f"  {sym}: {len(d)} raw forward observations")
    except Exception as e:
        print(f"  error loading {sym}: {e}")

if not frames:
    print("Error: No data frames successfully loaded.")
    sys.exit(1)

df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

# 1. Compute cross-sectional basket-demeaned return
if USE_XS:
    df["fwd_xs"] = df["fwd"] - df.groupby("timestamp")["fwd"].transform("mean")
    target_col = "fwd_xs"
else:
    target_col = "fwd"

# 2. Apply start-ts filter if specified
if START_TS_MS is not None:
    initial_len = len(df)
    df = df[df["timestamp"] >= START_TS_MS].reset_index(drop=True)
    print(f"Filtered timestamp >= {START_TS_MS}: {initial_len} -> {len(df)} observations")

# 3. Decimate on shared timestamp grid for non-overlapping observations
shared_stamps = sorted(df["timestamp"].unique())
keep_stamps = set(shared_stamps[::HORIZON])
df = df[df["timestamp"].isin(keep_stamps)].reset_index(drop=True)

print(f"\nPooled dataset: {len(df)} non-overlapping independent observations across {len(df['symbol'].unique())} symbols (Target={target_col})\n")
split = int(len(df) * 0.8)

# Candidate features
DERIVATIVE_PREFIXES = ["funding", "oi", "open_interest", "cvd", "ofi", "liq", "orderbook", "ob_"]
EXCLUDE_COLS = [
    "open_time", "timestamp", "open", "high", "low", "close", "volume",
    "close_btc", "fwd", "fwd_xs", "target", "target_price", "target_trend",
    "target_direction", "future_ret", "symbol"
]

all_candidate_features = [
    c for c in df.columns 
    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
]

if MODE == "price_only":
    candidate_features = [
        c for c in all_candidate_features 
        if not any(dp in c.lower() for dp in DERIVATIVE_PREFIXES)
    ]
elif MODE == "derivatives_only":
    candidate_features = [
        c for c in all_candidate_features 
        if any(dp in c.lower() for dp in DERIVATIVE_PREFIXES)
    ]
else:
    candidate_features = all_candidate_features

def screen(frame, col, target_name):
    f = frame[[col, target_name, "symbol"]].dropna()
    if len(f) < 200:
        return None, None, None, None, f"INSUFFICIENT_OBS ({len(f)}<200)"
    
    n_unique = f[col].nunique()
    if n_unique < 20:
        return None, None, None, None, f"LOW_CARDINALITY (n_unique={n_unique}<20)"
    
    # Tie-mass guard: if most common value represents >20% of data, reject degenerate blob
    tie_frac = float(f[col].value_counts(normalize=True).iloc[0])
    if tie_frac > 0.20:
        return None, None, None, None, f"EXCLUDED-DEGENERATE (tie_mass {tie_frac*100:.1f}% > 20%)"

    r = f.groupby("symbol")[col].rank(pct=True)  # within-symbol percentile rank
    try:
        dec = pd.qcut(r, 10, labels=False, duplicates="drop")
    except Exception as ex_qcut:
        return None, None, None, None, f"QCUT_ERROR ({ex_qcut})"
        
    g = f.groupby(dec)[target_name].agg(["mean", "count"])
    n_deciles = len(g)
    if n_deciles < 5:
        return None, None, None, n_deciles, f"QCUT_COLLAPSE (n_deciles={n_deciles}<5)"
        
    spread = g["mean"].iloc[-1] - g["mean"].iloc[0]
    se = f[target_name].std() * np.sqrt(1.0 / g["count"].iloc[-1] + 1.0 / g["count"].iloc[0])
    mono = float(np.corrcoef(np.arange(len(g)), g["mean"].values)[0, 1])
    return spread, mono, se, n_deciles, None

rows = []
exclusions = []

for feat in candidate_features:
    a_spread, a_mono, a_se, a_dec, a_err = screen(df.iloc[:split], feat, target_col)
    b_spread, b_mono, b_se, b_dec, b_err = screen(df.iloc[split:], feat, target_col)
    
    if a_err is not None or b_err is not None:
        err_msg = a_err if a_err is not None else b_err
        split_name = "Train" if a_err is not None else "Test"
        n_unq = df[feat].nunique()
        exclusions.append({
            "feature": feat,
            "split": split_name,
            "reason": err_msg,
            "n_unique": n_unq
        })
        continue

    te_t = b_spread / b_se if b_se > 0 else 0.0
    rows.append((feat, a_spread, a_mono, a_dec, b_spread, b_mono, b_se, b_dec, te_t))

# Print Exclusions Report
if exclusions:
    print("=" * 90)
    print(f"FEATURE EXCLUSIONS REPORT ({len(exclusions)} features excluded from evaluation)")
    print("=" * 90)
    ex_df = pd.DataFrame(exclusions)
    print(ex_df.to_string(index=False))
    print("=" * 90 + "\n")

out = pd.DataFrame(rows, columns=["feature", "tr_spread%", "tr_mono", "tr_dec", "te_spread%", "te_mono", "te_SE%", "te_dec", "te_t"])
if not out.empty:
    out["abs_te_t"] = out["te_t"].abs()
    out = out.sort_values(by="abs_te_t", ascending=False).drop(columns=["abs_te_t"])

    print("=" * 90)
    print(f"POOLED MULTI-ASSET FEATURE SCREENING REPORT (Interval={IV}m, Horizon={HORIZON}b, Target={target_col}, Mode={MODE})")
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
else:
    print("No features passed the validity screen.")
