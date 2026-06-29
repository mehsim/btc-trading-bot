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
- **Hugging Face Live Sync**:
  - Implemented automatic sync logic inside `load_history()` in `main.py`. On local startup, it queries the live Hugging Face Space API (`/api/status`) to pull predictions, trades, and balances to synchronize local states.
- **Environment Safety**:
  - The sync engine skips self-requests on the Hugging Face container itself (via the `SPACE_ID` check) to prevent loops and timeouts.
- **Persistent Storage**:
  - Configured `HISTORY_FILE` to automatically use the `/data/dashboard_history.json` persistent mount if present, ensuring data is not lost during Space rebuilds/restarts.
- **Git Ignore Fixes**:
  - Removed `dashboard_history.json` from git tracking and ignored it to prevent local empty history files from overwriting live history files during new commits/pushes.
