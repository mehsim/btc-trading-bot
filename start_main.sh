#!/bin/bash
cd ~/btc-trading-bot
pkill -9 -f 'python.*main.py' || true
sleep 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export LOKY_MAX_CPU_COUNT=1
export JOBLIB_MULTIPROCESSING=0
export PYTHONUNBUFFERED=1

nohup .venv/bin/python -u main.py > ~/main.log 2>&1 &
echo "Started Bot PID: $!"
