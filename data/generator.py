"""
Industrial Thermal & Temperature Time-Series Synthetic Generator & Chronological Splitter
Generates realistic 10-minute industrial thermal sensor telemetry with heating/cooling cycles,
weekly operating shifts, process thermal inertia, and stochastic thermal runaway spikes.
"""

import numpy as np
import pandas as pd
from typing import Dict


def generate_industrial_thermal_dataset(
    start_date: str = "2025-01-01",
    days: int = 180,
    sampling_interval_minutes: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a multivariate industrial thermal dynamics sensor dataset.
    
    Parameters:
    - start_date: ISO start date string.
    - days: Total duration in days.
    - sampling_interval_minutes: Interval between consecutive sensor readings (default 10m).
    - seed: Random seed for reproducibility.
    
    Returns:
    - pd.DataFrame containing datetime index, core_temperature_c, heater_power_kw,
      cooling_flow_lpm, is_thermal_spike flag.
    """
    np.random.seed(seed)
    timestamps = pd.date_range(
        start=start_date,
        periods=days * 24 * (60 // sampling_interval_minutes),
        freq=f"{sampling_interval_minutes}min"
    )
    n = len(timestamps)

    # 1. Base Diurnal Thermal Curve (Plant Heating & Batch Cycles)
    hour = timestamps.hour + timestamps.minute / 60.0
    daily_pattern = 65.0 + 12.0 * np.sin(2 * np.pi * (hour - 6) / 24) + 4.0 * np.sin(4 * np.pi * hour / 24)

    # 2. Operating Duty Factor (Weekday production vs weekend idle cooling)
    day_of_week = timestamps.dayofweek
    duty_factor = np.where(day_of_week < 5, 1.10, 0.75)

    base_temp = daily_pattern * duty_factor

    # 3. Macro Seasonal Ambient Drift
    ambient_drift = np.linspace(0, 8.0, n)

    # 4. Realistic Thermal Measurement Noise (Thermocouple variance)
    sensor_noise = np.random.normal(0, 1.2, n)

    temp_series = np.maximum(20.0, base_temp + ambient_drift + sensor_noise)

    # 5. Injection of Stochastic Thermal Runaway / Exothermic Surge Spikes
    # Approx 1.5% of timestamps experience thermal surge events
    spike_mask = np.random.random(n) < 0.015
    spike_amplitudes = np.random.uniform(25.0, 55.0, n)
    core_temperature_c = np.where(spike_mask, temp_series + spike_amplitudes, temp_series)

    # 6. Correlated Secondary Telemetry (Heater Power kW & Coolant Flow L/min)
    heater_power_kw = np.maximum(0.0, 20.0 + 0.25 * core_temperature_c + np.random.normal(0, 1.5, n))
    cooling_flow_lpm = np.maximum(5.0, 35.0 + 0.18 * core_temperature_c + np.random.normal(0, 2.0, n))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "core_temperature_c": np.round(core_temperature_c, 2),
        "heater_power_kw": np.round(heater_power_kw, 2),
        "cooling_flow_lpm": np.round(cooling_flow_lpm, 2),
        "is_thermal_spike": spike_mask.astype(int)
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
    data = generate_industrial_thermal_dataset(days=90)
    splits = chronological_split(data)
    print(f"Generated {len(data)} thermal sensor observations.")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"Total thermal spike events: {data['is_thermal_spike'].sum()}")
