# 🚀 BTC Trading Bot — Complete Deployment & Setup Guide

This guide provides step-by-step instructions for installing, configuring, training, and running the **BTC Trading Bot** from scratch after cloning or pulling the repository.

---

## 📋 Prerequisites & System Requirements

* **Operating System**: Linux (Ubuntu 20.04/22.04 recommended) or macOS
* **Python**: Python 3.10 or higher
* **Redis**: Local or remote Redis server for state management
* **Dependencies**: `git`, `python3-venv`, `build-essential`, `curl`

---

## 📥 Step 1: Clone Repository & Virtual Environment

```bash
# 1. Clone the repository
git clone https://github.com/mehsim/btc-trading-bot.git
cd btc-trading-bot

# 2. Create Python virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate
```

---

## 📦 Step 2: Install Required Dependencies

```bash
# Upgrade pip and install all required python packages
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: Ensure Redis server is installed and active on your system:
> ```bash
> sudo apt update && sudo apt install -y redis-server
> sudo systemctl enable --now redis
> ```

---

## 🔑 Step 3: Configure Environment Variables

Create a `.env` file in the root project directory:

```bash
cat << 'EOF' > .env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID_HERE

# Bybit API Credentials (Live Trading & Wallet Equity Sync)
BYBIT_API_KEY=YOUR_BYBIT_API_KEY_HERE
BYBIT_API_SECRET=YOUR_BYBIT_API_SECRET_HERE

# Bot Mode: 'simulation' (Paper Trading) or 'live' (Real Trading)
TRADE_MODE=simulation

# News & Sentiment API Key (Optional)
FINNHUB_TOKEN=free
EOF
```

---

## 🗄️ Step 4: Initialize Database & Verify Schema

Run the database initialization script to create SQLite tables and historical order flow schemas:

```bash
.venv/bin/python3 -c "import data; data.init_db(); print('✅ Database initialized successfully!')"
```

---

## 🧠 Step 5: Model Training (If Model Weights are Missing)

To train initial machine learning ensemble models (XGBoost + LightGBM + CatBoost) across active timeframes:

```bash
# Retrain all timeframes (15m, 30m, 60m, 120m)
.venv/bin/python3 retrain_worker.py
```

*Or train a single timeframe individually:*
```bash
.venv/bin/python3 train.py --interval 60 --pages 20
```

---

## ▶️ Step 6: Start the Trading Bot

### Option A: Run Directly in Terminal
```bash
.venv/bin/python3 main.py
```

### Option B: Run as a Background Systemd Service (Recommended for Production)

1. Create a systemd service file:
```bash
sudo nano /etc/systemd/system/trading-bot.service
```

2. Paste the following configuration:
```ini
[Unit]
Description=BTC Trading Bot Service
After=network.target redis.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/btc-trading-bot
EnvironmentFile=/home/ubuntu/btc-trading-bot/.env
ExecStart=/home/ubuntu/btc-trading-bot/.venv/bin/python3 main.py
ExecStartPost=/bin/sh -c '. /home/ubuntu/btc-trading-bot/.env && curl -s -X POST https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage -d chat_id=$TELEGRAM_CHAT_ID -d text="✅ BTC Bot started successfully."'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service
sudo systemctl start trading-bot.service
```

---

## 📲 Useful Telegram Bot Commands

Once running, send these commands to your bot on Telegram:
* `/summary` — View 24h performance, win rate, and health report
* `/active` — View all currently open active trades
* `/balance` — Check live wallet equity and available margin
* `/profit` — View historical P&L stats
* `/stop_all` — Emergency stop bot and close active trades
* `/start_bot` — Resume trading engine
