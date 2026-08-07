# BTC Algorithmic Trading Bot — System Architecture & Operations Runbook

## 1. System Overview & Architecture Design

The BTC Algorithmic Trading Bot is an institutional-grade, multi-timeframe quantitative trading engine designed for Binance / Bybit USDT perpetual contracts. It combines machine learning ensemble classifiers, regression price targets, regime classification, dynamic risk controls, and automated governance contracts.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MARKET DATA PIPELINE                              │
│             Binance / Bybit REST & Public WebSockets                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-TIMEFRAME FEATURE PIPELINE                      │
│   (15m, 30m, 60m, 120m, 240m, 360m — Derivs, OB L2, Microstructure, TA)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 REGIME CLASSIFIER & REGIME HYSTERESIS                       │
│    ADX Hysteresis (REGIME_ADX_ENTER_BY_INTERVAL) → Trending vs Ranging      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             C-1 MODEL GOVERNANCE & PREDICTIVE FLOOR GATES                   │
│         Verify Manifest, Contract Hash, & MCC Floor (>= 0.05)             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SIGNAL EVALUATOR & CONFLUENCE ENGINE                    │
│    Ensemble Probability → Calibrated Confidence → Macro Confluence Check   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RISK ENGINE & POSITION SIZING PIPELINE                   │
│   Conservative Kelly → R-1 Quality Sizing (MCC / 0.15) → MHI Ramp → CVaR  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BYBIT EXECUTION & REST/WS                          │
│     Limit / Taker Orders, SL/TP Multipliers, Liquidation Target Alignment    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Governance & Predictive Floors (C-1 Contract)

To prevent unverified or sub-statistical models from taking trades, the bot enforces hard fail-closed governance invariants:

1. **Predictive Floor (`min_mcc: 0.05`)**: Any model with a cross-validated Matthew's Correlation Coefficient (MCC) below 0.05 is classified as statistical noise and causes an automatic fail-closed **ABSTAIN** response.
2. **Balanced Accuracy Floor (`min_balanced_accuracy: 0.36`)**: 3-class classification baseline must beat random chance (0.333).
3. **MCC Regression Tolerance (`mcc_regression_tolerance: 0.010`)**: Prevents challenger regression while absorbing Optuna cross-validation variance.
4. **Model Contract Hash & Barrier Config Check**: Load-time verification ensures live serving config matches the exact triple-barrier parameters used during Optuna model training (`tp_mult_trending`, `tp_mult_ranging`, `sl_mult`, `lookahead`, `regime_adx_enter`).

---

## 3. Multi-Timeframe Regime Classification

The bot separates market conditions into **Trending** and **Ranging** regimes using ADX Hysteresis per timeframe (`config.py`):

```python
REGIME_ADX_ENTER_BY_INTERVAL = {
    "15": 32.0,   # Preserves 15m ranging model routing (MCC 0.1542, +270% return)
    "30": 32.0,
    "60": 22.0,   # Matches 60m training labeller threshold (routes 60m ADX 22.4 to MCC 0.0646)
    "120": 28.0,
    "240": 28.0,  # 240m ADX 20.7 stays Ranging & abstains (MCC 0.0408 < 0.05 floor)
    "360": 28.0,
}
```

- **Hysteresis Logic**: Switching from Ranging to Trending requires ADX $\ge$ `REGIME_ADX_ENTER_BY_INTERVAL[tf]`. Switching back requires ADX $\le$ `REGIME_ADX_EXIT_BY_INTERVAL[tf]`. This prevents regime flapping on choppy candle boundaries.

### 3.1 Directional-Mass Thresholds (15m & 30m Signal Evaluator)

To suppress high-frequency noise and false breakouts, fast timeframes (15m and 30m) enforce a **0.52 Directional-Mass Threshold**:
- **Bullish Signal**: Requires `prob_bullish >= 0.52` (signals in `[0.500, 0.519]` remain `Neutral`).
- **Bearish Signal**: Requires `prob_bearish >= 0.52` (signals in `[0.500, 0.519]` remain `Neutral`).
- **Operational Note**: Operators auditing signal logs must note that 15m/30m probabilities between 50.0% and 51.9% are intentionally filtered to `Neutral` by design to prevent chop overtrading.

---

## 4. Risk Management & Position Sizing Pipeline

Position sizing is governed by a multi-layered quantitative risk framework:

```python
# Step 1: Conservative Kelly Sizing
scaled_kelly = risk_engine.compute_conservative_kelly(...)

# Step 2: R-1 Model Quality Scaling
quality_mult = np.clip(mcc_val / 0.15, 0.35, 1.0)
kelly_fraction *= quality_mult

# Step 3: Model Health Index (MHI) Scaling
mhi_scale = np.clip((mhi_score - 50.0) / 40.0, 0.0, 1.0)
position_size_usd *= mhi_scale

# Step 4: CVaR Tail Loss Constraints
position_size_usd = min(position_size_usd, max_cvar_allowed_size)
```

- **R-1 Quality Sizing**: Capital allocation scales dynamically by measured predictive content. Models at the fleet benchmark (15m Ranging @ MCC `0.1542`) receive 1.00x sizing, while lower quality models (60m Trending @ MCC `0.0646`) scale down smoothly to ~0.43x.
- **R-3 Kill Criteria**: Evaluated at $\ge 250$ trades. Triggers automatic model retirement if drawdown $> 25\%$ or Sharpe $< 0.50$.
- **R-5 Retraining Noise Protection**: Prevents noisy automated retraining unless MHI drops below 70.0.

---

## 5. Operations & Troubleshooting Runbook

### Primary AWS Deployment Details
- **Instance IP**: `47.129.153.199` (AWS Singapore)
- **SSH Key**: `/Users/mehsimkhurshid/Downloads/singapore-key.pem`
- **Systemd Service**: `trading-bot`

### Daily Commands Quick-Reference

1. **Check Live Service Status**:
   ```bash
   ssh -i /Users/mehsimkhurshid/Downloads/singapore-key.pem ubuntu@47.129.153.199 "sudo systemctl status trading-bot"
   ```

2. **Tail Live Logs**:
   ```bash
   ssh -i /Users/mehsimkhurshid/Downloads/singapore-key.pem ubuntu@47.129.153.199 "sudo journalctl -u trading-bot -f"
   ```

3. **Query Recent Trades PnL**:
   ```bash
   ssh -i /Users/mehsimkhurshid/Downloads/singapore-key.pem ubuntu@47.129.153.199 "cd ~/btc-trading-bot && python3 -c \"
   import sqlite3
   conn = sqlite3.connect('trading_bot.db')
   cur = conn.cursor()
   cur.execute('SELECT interval, COUNT(*), ROUND(AVG(change_pct),3), ROUND(SUM(change_pct),2) FROM completed_trades GROUP BY 1 ORDER BY 4 DESC;')
   for r in cur.fetchall(): print(r)
   conn.close()
   \""
   ```

4. **Deploy Latest Code & Restart Service**:
   ```bash
   ssh -i /Users/mehsimkhurshid/Downloads/singapore-key.pem ubuntu@47.129.153.199 "cd ~/btc-trading-bot && git fetch origin && git reset --hard origin/main && sudo systemctl restart trading-bot"
   ```

5. **Run Governance & Kill Criteria Audit**:
   ```bash
   python3 tools/evaluate_kill_criteria.py
   python3 tools/measure_calibration.py
   ```
