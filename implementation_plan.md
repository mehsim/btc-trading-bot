# Implementation Plan: Phase 1 + Phase 2 + Phase 3 — Comprehensive Resiliency, Risk & Optimization

This plan details the technical steps to resolve systemic vulnerabilities of the trading bot by decoupling high-impact processes, securing state, protecting execution from market volatility, implementing advanced mathematical/ML updates (CVaR risk sizing, stress-test covariance, adaptive meta-learning, and regime-specific registries), and introducing key performance optimizations (rate-limit backoff and GPU acceleration).

## User Review Required

> [!IMPORTANT]
> **Taker Execution Fee Impact**: Falling back to Taker IOC (Immediate-or-Cancel) orders during high volatility spikes will incur Bybit's taker fee (0.055%) instead of the maker fee (0.02% or rebate). This is an insurance premium paid to prevent getting filled on toxic flow.
> 
> **Separate Worker Architecture**: We propose running the retraining task on a decoupled background systemd service (`retrain-worker`) using the same VPS but with strict CPU affinity (`taskset`) and memory limits (`cgroups`) if we stay on a single host.

---

## Proposed Changes

### 1. Decouple Analytics & Execution

#### [NEW] [retrain_worker.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/retrain_worker.py)
* Extract the training orchestrator code from `main.py` into a separate, independent script.
* Configure a systemd worker service (`trading-bot-retrainer.service`) that executes `retrain_worker.py` sequentially once a week.
* Set CPU limits using systemd configuration (`CPUQuota=50%` and `MemoryLimit=800M`) to guarantee retraining never starves the main trading bot.

#### [MODIFY] [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py)
* Remove the in-process `run_daily_journal_scheduler` and retraining threads to keep the runtime lightweight.
* Maintain only the hot-reload file listener to load weights when updated by the worker.
* **Simple Benefit**: Keeps the trading bot fast and prevents missed trades while the models retrain. One crash on retraining won't take down the trading loop.

---

### 2. State & Database Durability

#### [MODIFY] [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py)
* Implement automated daily database snapshots:
  * At UTC midnight, create a compressed backup of `trading_bot.db` and `trade_journal.csv`.
  * Upload the backup to an external, secure cloud bucket (e.g. AWS S3) using the `boto3` library.
* Add database integrity verification checks (`PRAGMA integrity_check;`) on startup.
* **Simple Benefit**: Protects your trade history and metrics so you can fully restore them if the server crashes.

---

### 3. Execution Protection: Volatility Taker IOC Fallback

#### [MODIFY] [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py)
* Add a **Volatility Monitor** inside the order placement manager:
  * Calculate the 1-minute Average True Range (ATR) normalized by price.
  * Define a volatility threshold (e.g., standard deviation of 1m ATR > 3.0).
* **Execution Logic**:
  * If volatility is **Normal**: Place Maker-Limit orders and use limit chasing (default behavior).
  * If volatility is **Extreme**: Place a Taker IOC order immediately to fill instantly or cancel, bypassing the limit-chasing queue.
* **Simple Benefit**: Prevents buying a "falling knife" during flash crashes by executing entries/exits instantly instead of waiting in line while price collapses.

---

### 4. Persistent OI & Funding Rate Cache *(Phase 2 Addition)*

#### [MODIFY] [data.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/data.py)
* Create two persistent CSV cache files inside `kline_cache/`:
  * `kline_cache/oi_{symbol}_{interval}.csv` — Open Interest history
  * `kline_cache/funding_{symbol}.csv` — Funding Rate history
* On startup: load the cached CSV, then fetch only the **delta** (new candles since last cached timestamp) from Bybit — not the full history.
* **Simple Benefit**: Bot startup drops from **3-5 minutes → under 10 seconds** on a warm restart, reducing down-time.

---

### 5. `/api/health` System Health Endpoint *(Phase 2 Addition)*

#### [MODIFY] [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py)
* Add a new Flask route `GET /api/health` that returns a JSON snapshot of uptime, API status, and WebSocket status.
* **Simple Benefit**: Lets you quickly check if the bot is running normally without logging into the server, and allows integration with alert monitors (e.g. UptimeRobot) to notify your phone if the bot dies.

---

### 6. Portfolio Risk Sizing: CVaR & Expected Shortfall (ES) *(Phase 2 Addition)*

#### [MODIFY] [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py)
* Replace standard Quarter-Kelly scaling with a **Conditional Value-at-Risk (CVaR) / Expected Shortfall (ES)** risk limit.
* Calculate historical lookback returns (e.g., last 30 days) of tradeable assets.
* Implement a parametric CVaR sizing rule:
  $$\text{Max Risk Size} = \frac{\text{Daily Loss Budget (e.g. 5% of balance)}}{\text{CVaR}_{\alpha}(R)}$$
  where $\alpha = 0.95$ and $R$ represents historical asset returns.
* **Simple Benefit**: Prevents the bot from over-sizing trades (over-leveraging) when markets are highly unpredictable, protecting account capital from heavy drawdowns.

---

### 7. Dynamic Stress-Test Covariance Matrix *(Phase 2 Addition)*

#### [MODIFY] [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py)
* Update `calculate_covariance_multiplier` to support stress scenarios.
* Monitor standard deviation of 1-hour ATR. If volatility exceeds a threshold (e.g. z-score > 2.0), trigger **Stress-Test Mode**.
* Under Stress-Test Mode, override the static `CORRELATION_MAP` and set all cross-asset correlations to a default of **`0.95`** (approaching perfect correlation, simulating a market panic where all coins move together).
* **Simple Benefit**: Automatically downscales trade sizes if multiple correlated assets (like BTC and ETH) are moving in the same direction, avoiding a massive joint liquidation during market crashes.

---

### 8. Adaptive Meta-Learner Weight Decay *(Phase 3 Addition)*

#### [MODIFY] [ensemble.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/ensemble.py)
* Update the fit logic of `EnsembleClassifier` and `EnsembleRegressor` to accept time-weighted sample distributions.
* Define an exponential decay factor $\lambda$ based on age in days:
  $$w_i = e^{-\lambda \cdot t_i}$$
  where $t_i$ is the age of the candle/validation-sample relative to the most recent record.
* Train the Stacking Meta-Learner using these weights.
* **Simple Benefit**: Prioritizes model performance from the last 14 days so the bot adapts quickly to new market trends rather than relying on stale historical success.

---

### 9. Regime-Specific Feature Registry *(Phase 3 Addition)*

#### [MODIFY] [train.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/train.py) & [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py)
* Modify the feature selection process to split features by regime.
* Save two independent JSON feature registries:
  * `selected_features_{interval}_trending.json`
  * `selected_features_{interval}_ranging.json`
* Prevent adversarial validation from dropping key trend features during consolidation phases.
* **Simple Benefit**: Stops the bot from losing its "memory" of trend indicators when switching between range-bound and trending markets, improving prediction accuracy.

---

### 10. Adaptive API Rate Limiter & Dynamic Sleep *(Optimization)*

#### [MODIFY] [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py)
* Update `bybit_get_request` and `bybit_post_request` response handlers to read Bybit API headers:
  * `X-Bapi-Limit-Status` (remaining calls allowed in current window)
  * `X-Bapi-Limit` (total limit for endpoint category)
* If `X-Bapi-Limit-Status` falls below **20%** of `X-Bapi-Limit`, introduce an adaptive delay before returning control.
* **Simple Benefit**: Guarantees crucial commands (like setting Stop Loss or exiting positions) never fail due to hitting Bybit's API rate limits.

---

### 11. GPU Retraining Optimization *(Optimization)*

#### [MODIFY] [train.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/train.py)
* Ensure XGBoost, LightGBM, and CatBoost models explicitly configure and utilize GPU hardware limits on the retraining worker.
* Modify model parameter maps to use `device='cuda'`, `tree_method='hist'` (XGBoost) / `device='gpu'` (LightGBM) / `task_type='GPU'` (CatBoost) when GPU resources are detected.
* **Simple Benefit**: Weekly model retraining takes minutes instead of hours, freeing up CPU resources for the live trading loop.

---

## Verification Plan

### Automated Verification
* **IOC Volatility Test**: Mock Bybit orderbook volatility in a local test script to confirm that the order placement manager correctly routes limit vs. taker IOC executions based on standard deviation limits.
* **Worker Isolation Test**: Run `retrain_worker.py` under CPU constraint and monitor the main trading bot's thread cycle times to verify zero lag in WebSocket events.
* **Cache Hit Test**: Delete the OI/Funding CSVs, restart the bot, confirm fresh fetch. Then restart again and confirm startup reads from disk only (no API calls for historical blocks).
* **Health Endpoint Test**: `curl http://localhost:<PORT>/api/health` and verify JSON response includes uptime, API status, and model weight age.
* **CVaR Sizing Test**: Verify that simulating a high-volatility tail return downscales the position size below the standard Kelly allocation.
* **Covariance Stress Test**: Mock extreme volatility conditions and verify that the covariance multiplier decreases automatically as correlations are stressed to `0.95`.
* **Regime Feature Slicing Test**: Verify that switching regimes correctly dynamically changes the columns/shape of `X_live` passed to the models.
* **Rate Limiter Test**: Mock low rate limit headers (`X-Bapi-Limit-Status: 5` out of `100`) and verify that requests auto-introduce dynamic sleeping.
* **GPU Retraining Test**: Verify that running the training script correctly checks for and registers GPU parameters if hardware is present.

### Manual Verification
* Trigger a manual model weight reload to verify that hot-reloading performs correctly when triggered by a separate process.
* Confirm that daily database snapshots are successfully uploaded to external storage.
* Visually inspect `/api/health` in a browser to confirm all fields are populated correctly.
