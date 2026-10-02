"""
Unit Tests for Industrial Time-Series Forecasting Pipeline
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from data.generator import generate_industrial_inflow_dataset, chronological_split
from models.baselines import RepeatPeriodBaseline, RollingMeanBaseline
from models.ttm_model import GraniteTTMR3Forecaster, IndustrialTTMFineTuningPipeline
from evaluation.metrics import calculate_metrics
from evaluation.error_analysis import perform_regime_error_analysis
from app.main import app


client = TestClient(app)


def test_dataset_generation():
    df = generate_industrial_inflow_dataset(days=10, seed=123)
    assert len(df) == 10 * 24 * 4  # 960 rows for 15-min intervals
    assert "inflow_rate" in df.columns
    assert "is_spike" in df.columns

    splits = chronological_split(df, train_ratio=0.7, val_ratio=0.15)
    assert len(splits["train"]) == int(len(df) * 0.7)
    assert len(splits["test"]) > 0


def test_baseline_forecasting():
    history = np.sin(np.linspace(0, 10, 192))
    baseline = RepeatPeriodBaseline(period_steps=96)
    preds = baseline.predict(history, horizon=96)
    assert len(preds) == 96
    # Check that lag matches exactly
    np.testing.assert_allclose(preds, history[-96:])


def test_metrics_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 39.0])
    m = calculate_metrics(y_true, y_pred)
    assert m["MAE"] == 2.0
    assert m["RMSE"] == 2.0
    assert m["MAPE_percent"] > 0
    assert m["sMAPE_percent"] > 0


def test_regime_error_analysis():
    y_true = np.array([10.0, 20.0, 100.0])
    y_pred = np.array([10.0, 20.0, 70.0])
    spikes = np.array([0, 0, 1])
    report = perform_regime_error_analysis(y_true, y_pred, spikes)
    assert report["normal_regime"]["MAE"] == 0.0
    assert report["spike_regime"]["MAE"] == 30.0
    assert "insights" in report


def test_ttm_model_inference():
    context_len = 192
    horizon_len = 96
    pipeline = IndustrialTTMFineTuningPipeline(context_len=context_len, horizon_len=horizon_len)
    history = np.random.normal(50.0, 5.0, context_len)
    preds = pipeline.forecast(history)
    assert len(preds) == horizon_len


def test_fastapi_endpoints():
    # 1. Health
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. Ingest Telemetry
    res_ingest = client.post("/api/v1/telemetry/ingest", json={
        "inflow_rate": 52.3,
        "pressure_bar": 3.9,
        "turbidity_ntu": 15.1
    })
    assert res_ingest.status_code == 201

    # 3. Generate Forecast
    res_pred = client.post("/api/v1/forecast/predict", json={"horizon_steps": 48})
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert len(data["predictions"]) == 48
    assert data["forecast_horizon_hours"] == 12.0

    # 4. Benchmark Metrics
    res_bench = client.get("/api/v1/metrics/benchmark")
    assert res_bench.status_code == 200
    assert "IBM_Granite_TTM_R3" in res_bench.json()["models"]
