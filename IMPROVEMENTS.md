# BTC Trading Bot - Improvements & Refactor Log (June 29, 2026)

This document summarizes the architecture upgrades, feature updates, layout optimizations, and performance improvements implemented today.

---

## 1. Timeframe Restructuring & Interval Refactor
- **Active Intervals Configured**: The bot now operates exclusively on **1h (`60`)**, **2h (`120`)**, **4h (`240`)**, and **6h (`360`)** intervals.
- **Deprecations**: Removed all references, parameters, and configurations for the old `5m` and `15m` timeframes.
- **Lookback History Extension**:
  - Increased lookback dataset length to retrieve a minimum of **20 pages** (~20,000 candles).
  - Translates to a lookback size of **833 days** for 1H, **1,666 days** for 2H, **3,333 days** for 4H, and **5,000 days** for 6H (substantially exceeding the 100-day minimum target).
- **Ensemble Model Training**:
  - Ran dynamic Optuna hyperparameter optimization and cross-validation for all intervals.
  - Successfully built, calibrated, and saved new ensemble models (`CatBoost`, `LightGBM`, `XGBoost`, and meta-regime classifiers) for both trending and ranging regimes.

---

## 2. Completed Trades History & Pop-up Details Modal
- **Interactive Popup Modal**:
  - Added a responsive CSS overlay popup modal (`#trade-modal`) in `index.html`.
  - Clicking on any row in the **Completed Trades History** table immediately displays execution metrics: entry/exit prices, net returns, realized PnL, outcome, execution time, and exit reason.
- **Investment Metrics Tracking**:
  - Added columns displaying **Invested (USD)** (simulated Kelly-sized position capital) and **PnL (USD)** (realized returns in dollar amount after round-trip trading fees).
- **Timeframe Filtering**:
  - Completed trades are filtered on the frontend and backend to strictly show active intervals (1h, 2h, 4h, and 6h).

---

## 3. Web Dashboard Layout Enhancements
- **Horizontal Strategy Grid**:
  - Upgraded the strategy cards at the top of the dashboard to a **5-column grid layout** (Spot Price + 4 Strategy columns) to show all 4 active intervals horizontally without wrapping.
- **Spacing Optimization**:
  - Moved the **Completed Trades History** card outside of the 2-column `.main-panels-grid` container. It now spans the full width of the dashboard, giving ample horizontal space for the new metrics columns.
- **Win Rate Bugfix**:
  - Re-mapped the win rate subcards from the old `5`/`15` slots to the new active slots (`tf-winrate-60`, `tf-winrate-120`, `tf-winrate-240`, `tf-winrate-360`).

---

## 4. Manual Position Closure (Force Close Button)
- **Backend manual exit endpoint**:
  - Implemented `/api/close_trade` in `main.py` which force-exits any active position at the current live price, computes real-time PnL, logs `Manual Exit (Force Closed)` to the stdout console logs, and writes details to history.
- **Close Button Locations**:
  - Added a red **Manual Close Position** button inside the active position cards at the top of the dashboard.
  - Added a red **Force Close** button next to the `TRADED` badge inside the **ML Predictions Tracker** table row for any pending trade.

---

## 5. Deployment & State Persistence
- **Hugging Face Live Sync & Dataset Backup**:
  - Implemented automatic sync logic inside `load_history()` in `main.py`. On local startup, it queries the live Hugging Face Space API (`/api/status`) to pull predictions, trades, and balances.
  - Implemented a programmatic cloud backup/restore to a private Hugging Face Dataset (`{space_id}-history`) utilizing the Space's `HF_TOKEN` secret to prevent data loss across container rebuilds/restarts.
- **Git Tracking**:
  - Re-tracked `dashboard_history.json` in Git with starting history so the bot does not start completely blank in environments without secrets configured.

---

## 6. Advanced Trading Logic & Prediction Accuracy Upgrades (Evening Session)
- **Pakistan Standard Time (PKT) Logs**:
  - Forced PKT (UTC+5) across all print wrapper console logs, websocket fallback messages, candle close detections, pre-trade report logs, and the daily drawdown/profit resets.
- **Dynamic Sizing & Safety Leverage (1x to 125x)**:
  - Enabled dynamic leverage scaled linearly from `1.0x` (at 50% ML confidence) to `125.0x` (at 100% ML confidence).
  - Implemented a safety liquidation guard that automatically caps leverage such that the distance to the Stop Loss (`0.75 * ATR`) never exceeds `90.0%` of the maintenance margin.
  - Integrated leverage values into active position cards, completed trades table rows, and the details popup modal.
- **Daily Profit Goal ($1000)**:
  - Implemented a daily profit goal circuit breaker. If the day's realized PnL reaches **$1,000**, the bot pauses trading for the remainder of the day (evaluated on PKT day boundary).
- **Refined Regime Switching & Ensemble Weights**:
  - Replaced simple average voting with weighted averaging based on ADX regimes:
    - **Trending Regime (ADX >= 20)**: Weight is shifted towards **CatBoost** (XGB: 30%, LGB: 20%, Cat: 50%).
    - **Ranging Regime (ADX < 20)**: Weight is shifted towards **LightGBM** (XGB: 30%, LGB: 50%, Cat: 20%).
- **Open Interest (OI) Delta Confirmation**:
  - Added a new pre-trade confluence check (**Check 15**) that monitors the percentage change of Open Interest. If OI drops by more than `2.0%` over the last candle, the check fails (preventing entries during trend exhaustion).
- **Dynamic ATR Sizing (Take Profit Multipliers)**:
  - Implemented volatility-adaptive TP multipliers. Low volatility scales the multiplier up to `3.0x` to capture breakout extensions; high volatility scales it down to `0.9x - 1.5x` to lock in profits early.

---

## 7. API Reliability, Proxy, & Trading Integrity Upgrades (July 2026)
- **Hugging Face Proxy Optimization**:
  - Implemented a **3x automatic retry loop with exponential backoff** in REST API helpers (`bybit_get_request`, `bybit_post_request`) to absorb transient proxy connection drops and latency spikes.
  - Isolated proxy traffic to Bybit trading endpoints only, routing 90% of requests (model history candle downloads, sentiment feeds) directly for low latency and zero proxy bandwidth exhaustion.
  - Automatically bypasses Hugging Face internal proxies if no explicit `BYBIT_PROXY` secret is defined.
- **Partial Fill Sizing on Entry**:
  - Upgraded order execution to query the actual average fill price and executed quantity (`cumExecQty`) from order details.
  - Dynamically resizes scale-out limit orders and metadata metrics based on the actual filled amount instead of target size.
- **Exchange Side Mismatch Guard**:
  - Added a self-healing mismatch guard during Bybit position sync. If the bot's local direction (e.g. Bearish) mismatches the actual exchange side (e.g. Buy/Long due to partial fills or manual trading), it immediately force-closes the position and discards the trade to prevent unhedged SL/TP targets.
- **De-duplication Logic**:
  - Added timeframe de-duplication inside position syncs. The bot will automatically discard duplicate active trade references for the same symbol to avoid corrupted, doubled Assumed PnL calculations on the dashboard.
- **Testnet Price & TP/SL Syncing**:
  - Corrected testnet pricing anomalies by extracting contract `lastPrice` instead of `indexPrice`.
  - Added real-time synchronization of active position TP/SL parameters directly from the exchange.
