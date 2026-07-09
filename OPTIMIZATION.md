# BTC Trading Bot - Performance & Architecture Optimization Guide

This document outlines structural, mathematical, and algorithmic optimizations for the **BTC Trading Bot**. Each section provides **concrete implementation details** for the current codebase and highlights the **pros (benefits)** of applying these upgrades.

---

## 1. Data Pipeline & Feature Engineering Optimization

### A. Persistent Derivatives & Sentiment Caching
*   **Implementation Details**:
    *   Currently, `get_history` caches market klines to CSV. However, derivatives features (Open Interest, Funding Rate) and sentiment metrics (Fear & Greed) are queried live online via `get_bybit_oi_history`, `get_bybit_funding_history`, and `get_fear_and_greed_history`.
    *   **Action**: Modify `data.py` to create separate persistent CSV caches for OI, funding rates, and Fear & Greed indices inside `kline_cache/`. During startup, load these cached DataFrames and merge them locally, fetching only data from the last cached timestamp to the current time.
*   **Pros**:
    *   **Speed**: Startup time drops from several minutes to under 5 seconds since large historical blocks are read from disk.
    *   **Resiliency**: Prevents API limit exhaustion or temporary timeouts from external sources (e.g. Fear & Greed API) during model re-training or backtests.

### B. Vectorized Technical Analysis
*   **Implementation Details**:
    *   Ensure all indicators in `data.py` (e.g. EMA crossover, Bollinger Bands, ATR, ADX, Kalman Filters) are computed using highly optimized vectorized libraries (`numpy` and `pandas`) rather than python loops.
    *   Replace sequential iteration or `.apply()` calls with NumPy operations where possible. For instance, for rolling Kalman calculations, compile the loop using `Numba` (`@jit(nopython=True)`) to run at near-C speeds.
*   **Pros**:
    *   **Computation Efficiency**: Speeds up training data preprocessing by 10x-50x when analyzing 40,000+ historical candles.
    *   **Lower Memory Footprint**: Reduces overhead, making the bot lightweight enough to run easily on minimal cloud hosts or Hugging Face basic containers.

### C. Rate Limit Management & Dynamic Sleep
*   **Implementation Details**:
    *   Introduce an adaptive API rate-limiter in REST helpers (`bybit_public_get` / `bybit_post_request`). Read Bybit's response headers (`X-Bapi-Limit`, `X-Bapi-Limit-Status`, `X-Bapi-Limit-Reset-Timestamp`) to dynamically adjust sleep times or postpone non-urgent data requests (e.g. daily news sentiment pulls) if rate limits are nearing exhaustion.
*   **Pros**:
    *   **Reliability**: Guarantees that critical execution tasks (e.g. setting Stop Loss or force-closing a position) will never fail due to `HTTP 429 Too Many Requests`.

---

## 2. Machine Learning Model & Hyperparameter Tuning

### A. Extended Hyperparameter Search & Regularization
*   **Implementation Details**:
    *   In `train.py`, increase the Optuna search trials (e.g. from `n_trials=15` to `n_trials=100+`) to find optimal configurations.
    *   Expand search spaces to include tree regularization terms:
        *   **XGBoost**: `reg_alpha` (L1), `reg_lambda` (L2), `min_child_weight`, and `gamma` (minimum loss reduction).
        *   **LightGBM**: `reg_alpha`, `reg_lambda`, `min_child_samples`, and `colsample_bytree`.
        *   **CatBoost**: `l2_leaf_reg`, `random_strength`, and `bagging_temperature`.
*   **Pros**:
    *   **Generalization**: Regularization reduces model variance, avoiding over-fitting on noise or micro-trends.
    *   **Robustness**: Finds more stable param combinations that hold up in varying live market conditions.

### B. Purged & Embargoed Time-Series Cross-Validation
*   **Implementation Details**:
    *   Because our models predict over a future window (lookahead of 10 to 16 candles), labels in consecutive steps are highly correlated and overlap in time.
    *   **Action**: Integrate `PurgedEmbargoTimeSeriesSplit` from `ensemble.py` into the training loop in `train.py`. Set `purged_gap` equal to the model's lookahead window (e.g. 16 hours for 6h timeframe) and `embargo_gap` to a similar span to ensure training and validation sets have zero overlapping dates or residual influence.
*   **Pros**:
    *   **Leakage Prevention**: Eliminates time-series information leakage, giving a realistic evaluation of backtest performance.
    *   **Accurate Calibration**: Model metrics (accuracy, confidence) align closely with live performance.

### C. Automated Feature Selection & Dimension Reduction
*   **Implementation Details**:
    *   Our features array lists 70+ technical, derivative, and lag elements. Having too many features leads to the "curse of dimensionality".
    *   **Action**: Implement Recursive Feature Elimination with Cross-Validation (`RFECV`) or SHAP (SHapley Additive exPlanations) value thresholding. Keep only features that rank in the top 20-30 for prediction power. Save the selected feature list inside the interval config (e.g., `selected_features_60.json`).
*   **Pros**:
    *   **Leaner Models**: Decreases training and evaluation times.
    *   **Less Noise**: Removing redundant features reduces model noise and prevents overfitting to historical anomalies.

### D. GPU Acceleration
*   **Implementation Details**:
    *   Add hardware configuration parameters in model definitions:
        *   **XGBoost**: `tree_method='hist'`, `device='cuda'` (if GPU is available).
        *   **CatBoost**: `task_type='GPU'`.
        *   **LightGBM**: `device='gpu'`.
*   **Pros**:
    *   **Fast Iteration**: Drastically decreases model training time (from hours to minutes), enabling rapid backtesting of new strategies.

---

## 3. Execution Latency & Order Routing

### A. Transition to Private Websocket Streams
*   **Implementation Details**:
    *   Currently, the bot fetches execution updates and active positions via polling REST endpoints.
    *   **Action**: Establish a secure connection to Bybit's **Private V5 WebSocket API** (endpoint: `wss://stream.bybit.com/v5/private` or `wss://stream-testnet.bybit.com/v5/private`). Subscribe to channels:
        *   `position`: For real-time updates on size, entry price, and leverage.
        *   `execution`: For immediate fills.
        *   `order`: For tracking open order status changes.
*   **Pros**:
    *   **Ultra-Low Latency**: Receives fill confirmation in <10ms compared to 500ms - 2000ms polling latency.
    *   **Eliminates API Limits**: Minimizes REST API calls, avoiding rate limits.

### B. Co-Location
*   **Implementation Details**:
    *   Deploy the bot on virtual private servers (VPS) located close to Bybit's primary server centers. Bybit servers are located in **AWS Tokyo (ap-northeast-1)** or **Singapore (ap-southeast-1)**.
*   **Pros**:
    *   **Lower Ping**: Reduces network ping from ~150ms (from US/European VPS) to <2ms.
    *   **Slippage Prevention**: Fast order execution minimizes slippage on stop-loss triggers or breakout entries.

### C. Active Limit Orders & Queue Positioning
*   **Implementation Details**:
    *   Replace market execution orders with passive limit orders placed at the top of the order book (using real-time bid/ask values from the WebSocket orderbook stream).
    *   If the limit order is not filled within a short timeout (e.g. 15-30 seconds), modify or cancel the order.
*   **Pros**:
    *   **Fee Savings**: Maker fees on Bybit are significantly lower than taker fees (e.g. Maker 0.02% vs Taker 0.055%). Over thousands of trades, this saving directly boosts profitability.

---

## 4. Risk Management & Position Sizing

### A. Fractional Kelly Criterion
*   **Implementation Details**:
    *   Instead of static size allocation, calculate optimal size dynamically:
        $$f^* = \frac{p \cdot b - (1 - p)}{b}$$
        Where $p$ is the model's calibrated probability (win rate), and $b$ is the reward-to-risk ratio (Take Profit distance / Stop Loss distance).
    *   Apply a safety factor (e.g. half-Kelly or quarter-Kelly, $0.25 \cdot f^*$) to protect capital.
*   **Pros**:
    *   **Mathematical Growth**: Mathematically maximizes the growth rate of capital over time.
    *   **Risk Mitigation**: Automatically reduces sizes when model confidence ($p$) is low or win rates drop.

### B. Daily Drawdown Circuit Breakers
*   **Implementation Details**:
    *   Maintain a tracking variable for daily equity.
    *   If the realized or paper losses for the current PKT day exceed a fixed threshold (e.g., 3.0% of total equity), cancel all pending orders, close all positions, and pause trading execution until the next day boundary reset.
*   **Pros**:
    *   **Tail Risk Protection**: Protects the account from black-swan events, API bugs, or sudden regime shifts that occur faster than indicators adapt.

### C. Volatility-Adaptive Risk Adjustments
*   **Implementation Details**:
    *   Adjust stop-loss distance dynamically:
        $$\text{SL Distance} = \text{ATR} \times \text{Multiplier} \times (1 + \text{Vol\_Z\_Score})$$
        Where `Vol_Z_Score` is the z-score of volatility relative to its 30-day average.
*   **Pros**:
    *   **Noise Filter**: Widens stops during periods of high noise (preventing premature exits) and tightens them during low-volatility conditions to protect capital.

---

## 5. System Reliability, Logging, & Backup

### A. Non-Blocking Uploader Threads
*   **Implementation Details**:
    *   Currently, cloud backup of dashboard metrics and trade history is triggered inside `load_history()` or active sync loops.
    *   **Action**: Move Hugging Face syncs and data exports to a background daemon thread or an asynchronous task queue. Never let the backup process block the main market analysis and execution loop.
*   **Pros**:
    *   **No Execution Lag**: Ensures that database writes or cloud backups never cause latency spikes during crucial execution moments.

### B. Healthchecks & Telegram Alerts
*   **Implementation Details**:
    *   Integrate a Telegram/Slack webhook inside the logging handler. Write a health-check endpoint `/api/health` that returns simple uptime statistics, API connection status, and model weight validity.
    *   Push notifications for:
        *   Trade entries, target prices, and leverage values.
        *   Stop-loss or take-profit hits.
        *   System exceptions or mismatches between local state and Bybit.
*   **Pros**:
    *   **Instant Visibility**: Enables remote monitoring and quick manual intervention if the server crashes or runs into execution failures.
