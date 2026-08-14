import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from tools.run_wf_60m_gated_pass import run_simulation as run_sim_std
from tools.run_wf_60m_gated_shift_pass import run_simulation as run_sim_shift

print("=== CALCULATING WINDOW ROBUSTNESS METRICS ===")

# 1. Standard Dataset (1,000 candles)
df_std, _, _, _ = run_sim_std(is_flipped_mode=False)

# Sort windows by trade count to find the 2 largest windows
df_std_sorted = df_std.sort_values("taken", ascending=False)
largest_2_std = df_std_sorted.head(2)["window"].tolist()

# A. PF excluding the two largest windows
df_std_ex_largest = df_std[~df_std["window"].isin(largest_2_std)]
net_pnl_ex_largest_std = df_std_ex_largest["net_pnl"].sum()

# Aggregate trade PnLs for windows excluding largest 2
pnls_ex_largest_std = []
for idx, row in df_std_ex_largest.iterrows():
    if row["taken"] > 0:
        pnls_ex_largest_std.append(row["net_pnl"])

gross_win_ex_std = sum(p for p in pnls_ex_largest_std if p > 0)
gross_loss_ex_std = abs(sum(p for p in pnls_ex_largest_std if p < 0))
pf_ex_largest_std = (gross_win_ex_std / gross_loss_ex_std) if gross_loss_ex_std > 0 else (999.0 if gross_win_ex_std > 0 else 0.0)

# B. PF across only windows with <20 trades
df_std_lt20 = df_std[df_std["taken"] < 20]
pnls_lt20_std = [row["net_pnl"] for _, row in df_std_lt20.iterrows() if row["taken"] > 0]
gross_win_lt20_std = sum(p for p in pnls_lt20_std if p > 0)
gross_loss_lt20_std = abs(sum(p for p in pnls_lt20_std if p < 0))
pf_lt20_std = (gross_win_lt20_std / gross_loss_lt20_std) if gross_loss_lt20_std > 0 else (999.0 if gross_win_lt20_std > 0 else 0.0)

print("\n--- STANDARD GATED DATASET (1,000 CANDLES / 15 WINDOWS) ---")
print(f"Two Largest Windows by Trade Count : Windows {largest_2_std} (Taken: {[int(c) for c in df_std_sorted.head(2)['taken'].tolist()]})")
print(f"• PF excluding 2 largest windows   : {pf_ex_largest_std:.3f} (Net PnL Sum: {net_pnl_ex_largest_std:+.4f})")
print(f"• PF across windows with <20 trades : {pf_lt20_std:.3f} (Active Windows: {len(df_std_lt20[df_std_lt20['taken'] > 0])})")


# 2. Shifted Dataset (1,500 candles)
df_shift, _, _, _ = run_sim_shift(is_flipped_mode=False)

df_shift_sorted = df_shift.sort_values("taken", ascending=False)
largest_2_shift = df_shift_sorted.head(2)["window"].tolist()

df_shift_ex_largest = df_shift[~df_shift["window"].isin(largest_2_shift)]
net_pnl_ex_largest_shift = df_shift_ex_largest["net_pnl"].sum()

pnls_ex_largest_shift = [row["net_pnl"] for _, row in df_shift_ex_largest.iterrows() if row["taken"] > 0]
gross_win_ex_shift = sum(p for p in pnls_ex_largest_shift if p > 0)
gross_loss_ex_shift = abs(sum(p for p in pnls_ex_largest_shift if p < 0))
pf_ex_largest_shift = (gross_win_ex_shift / gross_loss_ex_shift) if gross_loss_ex_shift > 0 else (999.0 if gross_win_ex_shift > 0 else 0.0)

df_shift_lt20 = df_shift[df_shift["taken"] < 20]
pnls_lt20_shift = [row["net_pnl"] for _, row in df_shift_lt20.iterrows() if row["taken"] > 0]
gross_win_lt20_shift = sum(p for p in pnls_lt20_shift if p > 0)
gross_loss_lt20_shift = abs(sum(p for p in pnls_lt20_shift if p < 0))
pf_lt20_shift = (gross_win_lt20_shift / gross_loss_lt20_shift) if gross_loss_lt20_shift > 0 else (999.0 if gross_win_lt20_shift > 0 else 0.0)

print("\n--- OUT-OF-SAMPLE SHIFTED DATASET (1,500 CANDLES / 15 WINDOWS) ---")
print(f"Two Largest Windows by Trade Count : Windows {largest_2_shift} (Taken: {[int(c) for c in df_shift_sorted.head(2)['taken'].tolist()]})")
print(f"• PF excluding 2 largest windows   : {pf_ex_largest_shift:.3f} (Net PnL Sum: {net_pnl_ex_largest_shift:+.4f})")
print(f"• PF across windows with <20 trades : {pf_lt20_shift:.3f} (Active Windows: {len(df_shift_lt20[df_shift_lt20['taken'] > 0])})")
