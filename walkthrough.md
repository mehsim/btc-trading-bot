# Walkthrough: AWS Singapore Migration

This walkthrough documents the technical steps, verification procedures, and latency improvements of migrating the Bybit trading bot from AWS Tokyo (`ap-northeast-1`) to AWS Singapore (`ap-southeast-1`).

---

## 1. Summary of Actions Completed

### 🇸🇬 Singapore Server Provisioning
* **Instance**: Launched a clean `t3.micro` instance in AWS Singapore (`ap-southeast-1`) running Ubuntu 24.04.
* **Security Group**: Configured to open TCP port 22 (SSH) and port 5000 (Flask dashboard if needed).
* **Package Setup**: Installed system-level dependencies including `python3-pip`, `python3-venv`, `redis-server`, `git`, and `curl`. Enabled Redis server to run co-located on port 6379.

### 🧹 Disk Space and Virtualenv Setup
* **Disk Optimization**: Temporarily removed the swap file to increase disk space, using a custom `TMPDIR` variable to complete the python `pip` package installation without hitting `tmpfs` RAM disk storage limits.
* **Venv Setup**: Configured a virtual environment `.venv` and successfully installed all model training and execution packages.

### 📦 Database & State Migration
* **State Recovery**: Stopped the active bot daemon in Tokyo, created a backup archive of `trading_bot.db`, `kline_cache.db`, and `dashboard_history.json`, and transferred it to Singapore.
* **Trained Weights Migration**: Transferred all trained weights, classifiers (`ensemble_*.json`), stacking parameters (`meta_*.json`), and calibrators (`calibrator_*.json`) from the Tokyo instance to the Singapore instance.

### ⚙️ Daemon Management
* **Systemd Service**: Configured the `/etc/systemd/system/trading-bot.service` daemon and verified it runs automatically.

---

## 2. Performance Assessment

We performed round-trip latency checks to Bybit's API server time endpoint (`/v5/market/time`) to verify the speed improvements:

| Location | Request Type | Average RTT Latency | Latency Reduction |
| :--- | :--- | :--- | :--- |
| **AWS Tokyo (Before)** | Cold/Warm HTTPS | 113 ms | Baseline |
| **AWS Singapore (After)** | Cold HTTPS (No connection reuse) | 27 ms | **76% reduction** |
| **AWS Singapore (After)** | **Warm HTTPS (Persistent Session)** | **5 ms** | **95.6% reduction (22x faster)** |

> [!IMPORTANT]
> Running the bot in AWS Singapore colocates the trading bot in the exact same AWS region as Bybit's matching engines, lowering warm HTTPS latency to **`5 ms`** and ensuring ultra-fast order executions.
