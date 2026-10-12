"""
Baseline Forecasting Models for Benchmark Comparison
Includes Repeat-Period (Daily/Weekly Lag) and Rolling Mean Baselines.
"""

import numpy as np
import pandas as pd


class RepeatPeriodBaseline:
    """
    Repeat-Period baseline forecaster.
    Repeats observations from exactly one period prior (e.g. 24 hours or 7 days prior).
    """
    def __init__(self, period_steps: int = 96):
        """
        period_steps: Number of timesteps in a full cycle.
        For 15-min intervals:
        - 24 hours = 96 steps.
        - 7 days = 672 steps.
        """
        self.period_steps = period_steps

    def predict(self, history: np.ndarray, horizon: int = 96) -> np.ndarray:
        """
        Generates predictions for `horizon` steps based on historical period repeats.
        """
        if len(history) < self.period_steps:
            raise ValueError(f"History length ({len(history)}) must be >= period_steps ({self.period_steps})")

        # Extract the last `period_steps` readings
        cycle = history[-self.period_steps:]
        
        # Tile/repeat to match required horizon
        repeats_needed = int(np.ceil(horizon / self.period_steps))
        tiled = np.tile(cycle, repeats_needed)
        return tiled[:horizon]


class RollingMeanBaseline:
    """
    Moving Average baseline forecaster.
    """
    def __init__(self, window: int = 48):
        self.window = window

    def predict(self, history: np.ndarray, horizon: int = 96) -> np.ndarray:
        mean_val = np.mean(history[-self.window:])
        return np.full(shape=(horizon,), fill_value=mean_val)

