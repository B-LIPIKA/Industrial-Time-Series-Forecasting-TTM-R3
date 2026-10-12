"""
Evaluation Metrics for Industrial Time-Series Forecasting
Implements MAE, RMSE, MAPE, and sMAPE.
"""

import numpy as np
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard industrial time-series accuracy metrics.
    
    Parameters:
    - y_true: Ground truth observations array.
    - y_pred: Model predictions array.
    
    Returns:
    - Dict with MAE, RMSE, MAPE (%), and sMAPE (%).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # 1. MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # 2. RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # 3. MAPE (with epsilon guard against div-by-zero)
    epsilon = 1e-5
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100.0)

    # 4. sMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(np.mean(np.abs(y_pred - y_true) / np.maximum(denominator, epsilon)) * 100.0)

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE_percent": round(mape, 2),
        "sMAPE_percent": round(smape, 2)
    }

