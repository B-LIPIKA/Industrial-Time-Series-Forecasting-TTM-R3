"""
Industrial Inflow Time-Series Synthetic Generator & Chronological Splitter
Generates realistic 15-minute industrial sensor inflow telemetry with diurnal cycles,
weekly operating shifts, process noise, and stochastic inflow surge spikes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def generate_industrial_inflow_dataset(
    start_date: str = "2025-01-01",
    days: int = 180,
    sampling_interval_minutes: int = 15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a multivariate industrial inflow sensor dataset.
    
    Parameters:
    - start_date: ISO start date string.
    - days: Total duration in days.
    - sampling_interval_minutes: Interval between consecutive sensor readings (default 15m).
    - seed: Random seed for reproducibility.
    
    Returns:
    - pd.DataFrame containing datetime index, inflow_rate (m3/h), pressure_bar,
      turbidity_ntu, is_spike_event flag.
    """
    np.random.seed(seed)
    timestamps = pd.date_range(
        start=start_date,
        periods=days * 24 * (60 // sampling_interval_minutes),
        freq=f"{sampling_interval_minutes}min"
    )
    n = len(timestamps)

    # 1. Base Diurnal Seasonality (Factory Operating Shifts)
    hour = timestamps.hour + timestamps.minute / 60.0
    daily_pattern = 40.0 + 25.0 * np.sin(2 * np.pi * (hour - 6) / 24) + 10.0 * np.sin(4 * np.pi * hour / 24)

    # 2. Weekly Seasonality (Workweek vs Weekend duty cycles)
    day_of_week = timestamps.dayofweek
    weekday_factor = np.where(day_of_week < 5, 1.15, 0.65)

    base_inflow = daily_pattern * weekday_factor

    # 3. Macro Industrial Trend (Gradual seasonal shifts)
    trend = np.linspace(0, 15, n)

    # 4. Realistic Sensor Measurement Noise
    noise = np.random.normal(0, 2.5, n)

    inflow_rate = np.maximum(5.0, base_inflow + trend + noise)

    # 5. Injection of Extreme Inflow Spikes (Sudden process surge / extreme runoff)
    # Approx 1.5% of timestamps experience surges
    spike_mask = np.random.random(n) < 0.015
    spike_amplitudes = np.random.uniform(40.0, 95.0, n)
    inflow_rate = np.where(spike_mask, inflow_rate + spike_amplitudes, inflow_rate)

    # Correlated Secondary Telemetry (Pressure & Turbidity)
    pressure_bar = 3.5 + 0.03 * inflow_rate + np.random.normal(0, 0.2, n)
    turbidity_ntu = 12.0 + 0.15 * inflow_rate + np.random.normal(0, 1.5, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "inflow_rate": np.round(inflow_rate, 2),
        "pressure_bar": np.round(pressure_bar, 2),
        "turbidity_ntu": np.round(turbidity_ntu, 2),
        "is_spike": spike_mask.astype(int)
    })
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Dict[str, pd.DataFrame]:
    """
    Performs strictly chronological train/validation/test split to prevent temporal data leakage.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": df.iloc[:train_end].copy(),
        "val": df.iloc[train_end:val_end].copy(),
        "test": df.iloc[val_end:].copy()
    }
    return splits


if __name__ == "__main__":
    data = generate_industrial_inflow_dataset(days=90)
    splits = chronological_split(data)
    print(f"Generated {len(data)} sensor observations.")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"Total spike events: {data['is_spike'].sum()}")
