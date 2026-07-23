#!/bin/bash
# BTC Bot Weekly Health Check Script

echo "=============================================="
echo "===      BTC Bot Weekly Health Check       ==="
echo "=============================================="
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# 1. Service status
SERVICE_STATUS=$(systemctl is-active trading-bot.service)
if [ "$SERVICE_STATUS" = "active" ]; then
    echo "✅ Service Status: ACTIVE"
else
    echo "⚠️ SERVICE DOWN: Status is $SERVICE_STATUS"
fi

# 2. Disk space
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "⚠️ Disk Space High: ${DISK_USAGE}% used"
else
    echo "✅ Disk Space: ${DISK_USAGE}% used (Normal)"
fi

# 3. Model files check (15M & 30M)
MODEL_COUNT_15=$(ls /home/ubuntu/btc-trading-bot/ensemble_*_15_*.json 2>/dev/null | wc -l)
MODEL_COUNT_30=$(ls /home/ubuntu/btc-trading-bot/ensemble_*_30_*.json 2>/dev/null | wc -l)
if [ "$MODEL_COUNT_15" -ge 12 ] && [ "$MODEL_COUNT_30" -ge 12 ]; then
    echo "✅ 15M & 30M Models: COMPLETE (15M: $MODEL_COUNT_15, 30M: $MODEL_COUNT_30 files present)"
else
    echo "⚠️ Model Files Check: 15M ($MODEL_COUNT_15/12), 30M ($MODEL_COUNT_30/12)"
fi

# 4. Recent error log check (last 7 days)
ERROR_COUNT=$(journalctl -u trading-bot --since "7 days ago" | grep -c "ERROR" || true)
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "⚠️ Recent Errors (7 days): $ERROR_COUNT errors detected"
else
    echo "✅ Recent Errors (7 days): $ERROR_COUNT errors (Acceptable)"
fi

# 5. Service Start Timestamp
UPTIME_TS=$(systemctl show trading-bot.service --property=ActiveEnterTimestamp | cut -d= -f2)
echo "ℹ️  Service Running Since: $UPTIME_TS"

echo "=============================================="
echo "===             Check Complete             ==="
echo "=============================================="
