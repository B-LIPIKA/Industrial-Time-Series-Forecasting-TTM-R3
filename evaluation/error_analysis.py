"""
Inflow Error Analysis & Regime Segregation
Analyzes model performance across Normal Operating Regimes vs Extreme Inflow Spikes,
and evaluates performance degradation across extending forecast horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from evaluation.metrics import calculate_metrics


def perform_regime_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spike_flags: np.ndarray
) -> Dict[str, Any]:
    """
    Segregates errors between normal operating conditions and extreme inflow spikes.
    
    Parameters:
    - y_true: Ground truth values.
    - y_pred: Predicted values.
    - spike_flags: Binary array indicating presence of surge spike event (1 = spike, 0 = normal).
    
    Returns:
    - Structured dictionary with metrics for Normal, Spike, and Combined regimes.
    """
    normal_mask = (spike_flags == 0)
    spike_mask = (spike_flags == 1)

    overall_metrics = calculate_metrics(y_true, y_pred)
    
    normal_metrics = calculate_metrics(y_true[normal_mask], y_pred[normal_mask]) if np.any(normal_mask) else {}
    spike_metrics = calculate_metrics(y_true[spike_mask], y_pred[spike_mask]) if np.any(spike_mask) else {}

    # Calculate degradation percentage under extreme surges
    degradation_pct = None
    if "MAE" in normal_metrics and "MAE" in spike_metrics and normal_metrics["MAE"] > 0:
        degradation_pct = round(((spike_metrics["MAE"] - normal_metrics["MAE"]) / normal_metrics["MAE"]) * 100.0, 1)

    return {
        "overall": overall_metrics,
        "normal_regime": {
            **normal_metrics,
            "sample_count": int(np.sum(normal_mask))
        },
        "spike_regime": {
            **spike_metrics,
            "sample_count": int(np.sum(spike_mask)),
            "mae_degradation_pct": degradation_pct
        },
        "insights": (
            f"Under normal operating conditions, TTM-R3 achieves MAE of {normal_metrics.get('MAE', 0)} m3/h. "
            f"During non-linear industrial surge spikes, error increases by {degradation_pct}%, "
            f"highlighting the need for hybrid threshold-triggered safety buffers in downstream pump controllers."
        )
    }


def horizon_breakdown_analysis(
    y_true_matrix: np.ndarray,
    y_pred_matrix: np.ndarray
) -> pd.DataFrame:
    """
    Evaluates MAE and RMSE at key forecasting horizons:
    - 1 Hour (4 steps)
    - 4 Hours (16 steps)
    - 12 Hours (48 steps)
    - 24 Hours (96 steps)
    """
    horizons = {
        "1_Hour (4 steps)": 4,
        "4_Hours (16 steps)": 16,
        "12_Hours (48 steps)": 48,
        "24_Hours (96 steps)": min(96, y_true_matrix.shape[1])
    }

    records = []
    for label, step in horizons.items():
        sub_true = y_true_matrix[:, :step].flatten()
        sub_pred = y_pred_matrix[:, :step].flatten()
        m = calculate_metrics(sub_true, sub_pred)
        records.append({
            "Horizon": label,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "sMAPE_%": m["sMAPE_percent"]
        })

    return pd.DataFrame(records)
