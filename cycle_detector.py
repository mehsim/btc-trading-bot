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
