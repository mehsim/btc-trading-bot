# Comprehensive Codebase Audit: All Hardcoded Logic & Automation Roadmap

This audit documents **all 28 hardcoded constants, static thresholds, fixed multipliers, and manual rules** discovered across every module in the trading bot codebase, along with data-driven and machine-learning automation strategies for each.

---

## 1. `main.py` — Orchestrator & Execution Engine

| # | Hardcoded Logic / Threshold | File Line(s) | Current Static Value | Automated Data-Driven Alternative |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Asset Class Tiers** | L7162–L7168 | BTC/ETH (0.5%), Mid-Caps (0.8%), Alts (1.2%) | **Dynamic Volatility Clustering**: Cluster coins using 30-day Parkinson volatility & order book liquidity depth. |
| **2** | **Time-Decayed Trail Threshold** | L7167 | 4.0 hours start age, 0.05 decay factor | **Adaptive Half-Life Decay**: Base decay on median trade duration from historical trades per timeframe. |
| **3** | **Daily Circuit Breaker** | L7738–L7742 | Fixed `7.0%` daily drawdown halt limit | **GARCH Volatility Regime Scaling**: Adjust daily circuit breaker dynamically based on 30-day GARCH market volatility. |
| **4** | **Daily Profit Goal** | L7745–L7748 | Fixed `$1000.0` daily profit goal | **Rolling Equity Target**: Scale goal dynamically to `5.0%` of current live equity. |
| **5** | **News Blackout Window** | L7760 | Fixed `15 minutes` (`900s`) pre/post news | **Impact-Weighted News Window**: Scale blackout from 5 mins (minor CPI) to 45 mins (FOMC decisions). |
| **6** | **ADX Trend Multipliers** | L7160–L7165 | ADX $\ge$ 25 (1.5x), ADX < 18 (0.9x) | **Continuous GMM Multipliers**: Interpolate trailing distance smoothly using continuous GMM regime probabilities. |
| **7** | **Stagnation Age & Dev** | L7429–L7435 | `0.6x` lookahead duration & `0.5x ATR` dev | **Volume Decay Stagnation**: Exit when 4-hour volume falls below 15th percentile of 30-day volume. |
| **8** | **Scale-Out Target** | L7387 | 50% scale-out at `1.0x ATR` profit | **Order Book Pool Target**: Scale out dynamically at nearest order book depth / liquidation pool cluster. |
| **9** | **Fee Buffer** | L7231 | Fixed `0.0005` (0.05%) fee buffer | **Bybit REST Fee API**: Fetch live VIP tier fee schedule (Maker vs Taker) dynamically via API. |
| **10**| **TP Progress Lock** | L7256 | 50% profit lock at `40%` TP progress | **ATR Profit Step-Lock**: Dynamically lock profit using trailing Fibonacci retracement levels (38.2%, 50%, 61.8%). |

---

## 2. `risk_engine.py` — Risk & Capital Sizing

| # | Hardcoded Logic / Threshold | File Line(s) | Current Static Value | Automated Data-Driven Alternative |
| :--- | :--- | :--- | :--- | :--- |
| **11**| **Interval Position Caps** | L6–L12 | 5m (5%), 15m (8%), 30m (10%), 60m/120m (20%) | **Dynamic Half-Kelly Sizing**: Compute Kelly Fraction continuously based on rolling 100-trade performance. |
| **12**| **Symbol Exposure Cap** | L14, L27 | Fixed `20.0%` max equity in one coin | **Marginal Contribution to Risk (MCR)**: Cap position size using full portfolio covariance matrix. |
| **13**| **Correlation Cutoff** | L162 | Fixed `0.70` max correlation limit | **PCA Factor Exposure**: Restrict entries if portfolio principal component exposure to BTC/ETH exceeds 80%. |
| **14**| **Portfolio Heat Cap** | L123 | Fixed `300.0%` (3.0x equity exposure) | **Parametric VaR / Expected Shortfall**: Limit portfolio heat so 99% 1-day Parametric VaR never exceeds 5% of equity. |
| **15**| **Drawdown Scaling Tiers**| L59–L68 | 5% (-25%), 10% (-50%), 15% (-75%), 20% (Halt)| **Continuous Sigmoid Curve**: Smooth decay function $f(dd) = \frac{1}{1 + e^{k(dd - dd_{mid})}}$. |
| **16**| **Volatility Regime Scaler**| L170–L178 | >2% (0.5x), >1.5% (0.7x), 0.5-1.2% (1.2x), <0.3% (0.3x) | **Adaptive Volatility Density**: Scale position size continuously by inverse normalized ATR percentile. |
| **17**| **Margin Utilization Warning**| L130–L136 | 85% Emergency, 70% Halt, 50% Warning | **Dynamic Maintenance Margin Buffer**: Scale warning levels based on current leverage tier. |

---

## 3. `feature_pipeline.py` & `features.py` — Feature Engineering

| # | Hardcoded Logic / Threshold | File Line(s) | Current Static Value | Automated Data-Driven Alternative |
| :--- | :--- | :--- | :--- | :--- |
| **18**| **Outlier Threshold** | L4–L12 | Fixed $z = 4.0$, rolling window `30` | **Adaptive Median Absolute Deviation (MAD)**: Dynamic outlier bounds based on rolling IQR. |
| **19**| **Imputation FFill Limit** | L22 | Fixed `ffill(limit=5)` | **Spline / Kalman Filter Imputation**: Use Kalman filtering for continuous indicator state estimation. |
| **20**| **Sentiment Decay Factor** | L34 | Fixed `0.95` decay multiplier per period | **Half-Life Sentiment Decay**: Estimate decay constant $\lambda$ from news sentiment price response half-life. |
| **21**| **Multicollinearity Cutoff**| L41 | Fixed correlation cutoff `0.85` | **Variance Inflation Factor (VIF)**: Automate feature selection by dropping features with VIF $> 10$. |
| **22**| **Indicator Windows** | Various | EMA (9, 21), RSI (14), ADX (14), Bollinger (20, 2.0) | **Fast Fourier Transform (FFT)**: Adapt indicator periods dynamically to match dominant market cycle length. |

---

## 4. `mlops_engine.py` & `meta_model_engine.py` — Telemetry & Meta-Models

| # | Hardcoded Logic / Threshold | File Line(s) | Current Static Value | Automated Data-Driven Alternative |
| :--- | :--- | :--- | :--- | :--- |
| **23**| **Bayesian Cold-Start Priors**| L247–L262 | CI threshold 75, Win rate prior 0.65/0.68 | **Empirical Bayes Estimation**: Estimate priors automatically from out-of-fold cross-validation results. |
| **24**| **Model Drift Alerts** | L304–L306 | Accuracy `< 45%`, High-Conf Win Rate `< 55%` | **CUSUM / Page-Hinkley Drift Test**: Statistical process control test for real-time drift detection. |
| **25**| **Bayesian Odds Update** | L5–L11 | Prior prob `0.50`, Likelihood ratio `1.25` | **Logistic Calibration Odds**: Estimate likelihood ratios from historical out-of-fold error distributions. |
| **26**| **Meta-Gatekeeper Filters**| L19–L21 | Primary conf `< 0.55`, Win rate `< 40.0%` | **Supervised Meta-Classifier**: Train a secondary XGBoost model to predict signal profitability directly. |

---

## 5. `retrain_pipeline.py` & `order_state_machine.py` — Pipeline & Execution

| # | Hardcoded Logic / Threshold | File Line(s) | Current Static Value | Automated Data-Driven Alternative |
| :--- | :--- | :--- | :--- | :--- |
| **27**| **Retrain Schedule** | L13–L18 | 15m (3 days), 30m (5 days), 60m/120m (7 days) | **Concept Drift Trigger**: Retrain on-demand only when PSI $> 0.25$ or CUSUM drift test fires. |
| **28**| **Idempotency TTL & Backoff**| L18, L49 | TTL `60s`, Base delay `0.5s`, Max delay `30s` | **Dynamic Network Latency Backoff**: Adapt backoff delay based on rolling API ping / HTTP response latency. |

---

## Strategic Automation Roadmap

1. **Phase 1: Dynamic Half-Kelly & Sigmoid Drawdown Curve** (`risk_engine.py`)
2. **Phase 2: Dynamic Volatility Clustering & Order Book TP** (`main.py`)
3. **Phase 3: Dominant Cycle FFT Indicators & VIF Filtering** (`features.py` & `feature_pipeline.py`)
