# System Fixes & Security Hardening Log

This document tracks technical fixes, security hardening updates, and structural enhancements applied to the BTC Trading Bot repository.

---

## Log of Applied Fixes

### 1. Dashboard Authentication & API Security Hardening (F-13)
* **Date**: August 4, 2026
* **Affected Files**:
  * [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py#L2276)
  * [dashboard_routes.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/dashboard_routes.py#L120)
  * [.env.example](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/.env.example#L5)
  * [tests/test_dashboard_auth.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/tests/test_dashboard_auth.py)

#### Problem Description
* **Fail-Open Security Flaw**: When `DASHBOARD_API_KEY` was empty or unconfigured, the authentication decorator bypassed authentication (`if expected_key:` evaluated to `False`), returning HTTP 200 for administrative action endpoints.
* **Credential Exposure via Query Params**: `require_api_key` permitted credentials in URL query strings (`?api_key=...`), exposing keys in access logs, browser history, and referer headers.
* **Hardcoded Default Credential**: `require_admin_key` in `dashboard_routes.py` defaulted to `"btc_bot_admin_secure_key_2026"`, a hardcoded credential committed in `.env.example`.

#### Solution & Code Changes
1. **Unconditional 401 Enforcement (Fail-Closed)**:
   Removed `if expected_key:` guards. Unset or empty `DASHBOARD_API_KEY` now unconditionally returns **401 Unauthorized** for all administrative action endpoints.
2. **Header-Only Authorization**:
   Removed `request.args.get("api_key")`. API keys must be provided via the `X-API-KEY` HTTP header and are evaluated in constant-time via `hmac.compare_digest`.
3. **Removed Hardcoded Defaults**:
   Removed fallback `"btc_bot_admin_secure_key_2026"` from code and updated `.env.example` to standard `your_dashboard_api_key` placeholder.
4. **Decorator Consolidation**:
   Unified `require_admin_key = require_api_key` and decorated all state-changing endpoints (`/killswitch`, `/api/terminate`, `/api/retrain`, `/api/close_trade`, `/api/partial_exit_trade`, `/api/close_all_trades`, `/api/toggle_bot`, `/api/reset_circuit_breaker`, `/api/clear_history`, `/api/test_email`).
5. **Verification**:
   Expanded unit test suite in [tests/test_dashboard_auth.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/tests/test_dashboard_auth.py) (7/7 tests passed). Live deployment verified on AWS Singapore instance `47.129.153.199` (`curl -i -X POST /killswitch` returns `HTTP 401 UNAUTHORIZED`).

---

### 2. Active Trades Partial Exit ("½ Exit") Feature
* **Date**: August 4, 2026
* **Affected Files**:
  * [templates/index.html](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/templates/index.html#L2285)
  * [main.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/main.py#L3746)

#### Feature Overview
* Added a **"½ Exit"** button next to each open position in the Active Trades table.
* Allows extracting **50% of the invested margin + 50% of current accrued profit** while maintaining the remaining 50% position open with updated initial margin tracking.
* On live/testnet mode, executes a 50% `reduce_only` market order on Bybit; on simulation mode, updates simulated account balance.
* Displays a `"50% Done"` badge to prevent double partial exits on the same trade.

---

### 3. Feature Matrix Alignment & Retraining Enhancements
* **Date**: August 4, 2026
* **Affected Files**:
  * [ensemble.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/ensemble.py#L125)
  * [signal_evaluator.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/signal_evaluator.py#L88)
  * [train.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/train.py#L1507)

#### Fix Details
1. **Positional Model Feature Slicing & Zero-Padding**:
   Handled LightGBM/XGBoost models with positional `Column_N` feature names. Extra features are safely truncated, and missing features on stale models are zero-padded to match expected input shapes.
2. **Feature List Fallback**:
   Added a fallback in [signal_evaluator.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/signal_evaluator.py#L88) when `selected_features` JSON has fewer columns than model expected features.
3. **Multi-Timeframe Pipeline Inclusion**:
   Added `240` (4H timeframe) to `intervals_to_train` in [train.py](file:///Users/mehsimkhurshid/Downloads/btc-trading-bot/train.py#L1507) to ensure 4H models are retrained alongside 15m, 30m, 60m, and 120m.
4. **Retraining Performance Optimization**:
   Optimized `train.py` hyperparameters for lower RAM/CPU consumption (`PAGES=5`, Optuna `n_trials=3`, `n_estimators=80`, low-priority `nice -n 19` execution).

---

## Next / Pending Fixes Log

| Issue ID | Category | Description | Status | Target File(s) |
| :--- | :--- | :--- | :--- | :--- |
| **NEXT-01** | Security / Audit | Audit remaining API endpoints for strict CORS and payload schema validation | ⏳ Pending | `dashboard_routes.py`, `main.py` |
| **NEXT-02** | MLOps | Automatic retrain trigger upon persistent feature shape drift | ⏳ Pending | `mlops_engine.py`, `retrain_worker.py` |
| **NEXT-03** | Risk Engine | Dynamic slippage estimator based on orderbook depth L2 telemetry | ⏳ Pending | `risk_engine.py`, `bybit_client.py` |

---
