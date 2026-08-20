import numpy as np
import pandas as pd
from typing import Tuple, Dict

class FastFourierCycleDetector:
    def __init__(self, sample_window: int = 256):
        self.sample_window = sample_window

    def find_dominant_cycle_period(self, close_series: pd.Series) -> int:
        """
        Rule 22: Runs Fast Fourier Transform (FFT) on closing prices (256-bar window)
        to find dominant market frequency and convert to cycle period (clamped 5 to 50 bars).
        """
        if close_series is None or len(close_series) < 64:
            return 14  # Standard default period

        sub_series = close_series.dropna().tail(self.sample_window).values
        if len(sub_series) < 64:
            return 14

        # Detrend price series using linear regression to isolate cyclical oscillations
        x = np.arange(len(sub_series))
        detrended = sub_series - np.polyval(np.polyfit(x, sub_series, 1), x)

        # Compute FFT magnitude spectrum
        fft_values = np.abs(np.fft.rfft(detrended))
        frequencies = np.fft.rfftfreq(len(sub_series))

        # Exclude DC component (freq = 0) and ultra-low frequencies (< 2 cycles per window)
        fft_values[:2] = 0.0

        if np.max(fft_values) == 0:
            return 14

        dominant_idx = int(np.argmax(fft_values))
        dominant_freq = float(frequencies[dominant_idx])

        if dominant_freq <= 0:
            return 14

        raw_period = int(round(1.0 / dominant_freq))
        # Clamp dynamic window between 5 bars and 50 bars
        return int(np.clip(raw_period, 5, 50))

    def get_dynamic_indicator_windows(self, close_series: pd.Series) -> Tuple[int, int, int]:
        """
        Returns dynamic indicator periods (fast_ema_period, rsi_period, bollinger_window)
        """
        dom_period = self.find_dominant_cycle_period(close_series)
        fast_ema = max(5, int(round(dom_period * 0.6)))
        rsi_period = dom_period
        bollinger_window = max(10, int(round(dom_period * 1.4)))

        return fast_ema, rsi_period, bollinger_window

cycle_detector = FastFourierCycleDetector()

from production_regime_engine import production_regime_engine

def detect_market_regime_with_hysteresis(df: pd.DataFrame, symbol: str = "DEFAULT", interval: str = "60") -> str:
    """
    Evaluates market regime using ProductionRegimeEngine with ADX hysteresis.
    - Transition to TRENDING when ADX > 26.0 and volatility_ratio > 1.2
    - Transition back to RANGING when ADX < 22.0 or volatility_ratio < 0.8
    """
    if df is None or len(df) < 14 or "ADX" not in df.columns or "ATR_norm" not in df.columns:
        return "RANGING"

    adx_val = float(df["ADX"].iloc[-1])
    atr_norm = float(df["ATR_norm"].iloc[-1])
    atr_hist_median = float(df["ATR_norm"].iloc[:-1].median()) if len(df) > 1 else float(atr_norm)
    vol_ratio = atr_norm / max(1e-9, atr_hist_median)

    if "choppiness_index" in df.columns:
        chop_val = float(df["choppiness_index"].iloc[-1])
    elif "CHOP" in df.columns:
        chop_val = float(df["CHOP"].iloc[-1])
    else:
        try:
            from trade_calculators import choppiness_index
            chop_series = choppiness_index(df, window=14)
            chop_val = float(chop_series.iloc[-1]) if hasattr(chop_series, "iloc") else 50.0
        except Exception:
            chop_val = 50.0
    bb_width = float(df["BB_width_norm"].iloc[-1] * 100.0) if "BB_width_norm" in df.columns else (
        float((df["BB_high"].iloc[-1] - df["BB_low"].iloc[-1]) / max(1e-9, df["close"].iloc[-1]) * 100.0) if ("BB_high" in df.columns and "BB_low" in df.columns and "close" in df.columns) else 20.0
    )
    vol_20d = float(df["volume"].iloc[-1] / max(1e-9, df["volume"].tail(20).mean())) if ("volume" in df.columns and len(df) >= 20) else 1.0

    return production_regime_engine.update_regime(
        symbol=symbol,
        interval=interval,
        adx_value=adx_val,
        volatility_ratio=vol_ratio,
        choppiness=chop_val,
        bb_width_pct=bb_width,
        volume_ratio_20d=vol_20d
    )
