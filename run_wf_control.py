import numpy as np, pandas as pd, json, os, warnings
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import load_ensemble_classifier

SUPPORTED_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'AVAXUSDT', 'LTCUSDT', 'DOTUSDT']
INTERVAL = 15
FEE_RATE = 0.0008  # 0.08% fee per leg (0.16% round-trip)
TP_MULT = 0.98
SL_MULT = 0.78
N_WINDOWS = 15

print('Loading 15m candle history across symbols...')
dfs = []
for s in SUPPORTED_SYMBOLS:
    df_s = get_history(symbol=s, interval=INTERVAL, limit=1000, pages=3)
    if df_s is not None and len(df_s) > 100:
        df_s['symbol'] = s
        df_s['close_btc'] = df_s['close']
        df_s = merge_derivatives_sentiment_features(df_s, symbol=s, interval=INTERVAL)
        df_s = add_features(df_s)
        dfs.append(df_s)

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.sort_values('timestamp').reset_index(drop=True)
total_bars = len(df_all)
print(f'Total Candles: {total_bars}')

model_ranging = load_ensemble_classifier('ensemble_ranging_trend_15')
model_trending = load_ensemble_classifier('ensemble_trending_trend_15')

with open('selected_features_15_ranging.json') as f:
    feats_ranging = json.load(f)
with open('selected_features_15_trending.json') as f:
    feats_trending = json.load(f)

probs_ranging = model_ranging.predict_proba(df_all[feats_ranging].values)
probs_trending = model_trending.predict_proba(df_all[feats_trending].values)
adxs = df_all['ADX'].values

probs_all = np.where(adxs[:, None] >= 25.0, probs_trending, probs_ranging)
pred_classes = probs_all.argmax(axis=1)
confs = probs_all.max(axis=1)

window_size = total_bars // N_WINDOWS
window_metrics = []

closes = df_all['close'].values
highs = df_all['high'].values
lows = df_all['low'].values
atr_norms = df_all['ATR_norm'].values

for w in range(N_WINDOWS):
    w_start = w * window_size
    w_end = (w + 1) * window_size if w < N_WINDOWS - 1 else total_bars
    
    trades = []
    for i in range(w_start, w_end - 12):
        pred_c = pred_classes[i]
        conf = confs[i]
        
        if pred_c == 1 or conf < 0.35:
            continue
            
        p_entry = closes[i]
        atr = atr_norms[i] * p_entry
        if atr <= 0: continue
        
        is_bull = (pred_c == 2)
        upper_tp = p_entry + TP_MULT * atr
        lower_sl = p_entry - SL_MULT * atr
        lower_tp = p_entry - TP_MULT * atr
        upper_sl = p_entry + SL_MULT * atr
        
        pnl = None
        outcome = None
        for step in range(1, 13):
            h = highs[i + step]
            l = lows[i + step]
            if is_bull:
                if h >= upper_tp:
                    pnl = (upper_tp - p_entry) / p_entry - 2 * FEE_RATE
                    outcome = 'TP'
                    break
                elif l <= lower_sl:
                    pnl = (lower_sl - p_entry) / p_entry - 2 * FEE_RATE
                    outcome = 'SL'
                    break
            else:
                if l <= lower_tp:
                    pnl = (p_entry - lower_tp) / p_entry - 2 * FEE_RATE
                    outcome = 'TP'
                    break
                elif h >= upper_sl:
                    pnl = (p_entry - upper_sl) / p_entry - 2 * FEE_RATE
                    outcome = 'SL'
                    break
        if pnl is not None:
            trades.append({'pnl': pnl, 'win': pnl > 0, 'outcome': outcome})
            
    n_t = len(trades)
    win_rate = (sum(t['win'] for t in trades) / n_t * 100) if n_t > 0 else 0.0
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
    pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else 0.0
    avg_ret = (np.mean([t['pnl'] for t in trades]) * 100) if n_t > 0 else 0.0
    
    window_metrics.append({
        'window': w + 1,
        'trades': n_t,
        'win_rate': round(win_rate, 2),
        'profit_factor': round(pf, 3),
        'avg_return_pct': round(avg_ret, 3)
    })

print(json.dumps(window_metrics, indent=2))
total_trades = sum(w['trades'] for w in window_metrics)
mean_wr = np.mean([w['win_rate'] for w in window_metrics if w['trades'] > 0])
mean_pf = np.mean([w['profit_factor'] for w in window_metrics if w['trades'] > 0])
mean_ret = np.mean([w['avg_return_pct'] for w in window_metrics if w['trades'] > 0])

print(f'\n--- 15-WINDOW WALK-FORWARD BACKTEST CONTROL SUMMARY ---')
print(f'Out-Of-Sample Windows : {N_WINDOWS}')
print(f'Total Trades Executed : {total_trades}')
print(f'Mean Win Rate         : {mean_wr:.2f}%')
print(f'Mean Profit Factor    : {mean_pf:.3f}')
print(f'Mean Avg Trade Return : {mean_ret:+.3f}%')
