import pytest
import pandas as pd
from src.pipeline import load_and_preprocess_dataset, calculate_feature_metrics, scale_features

def test_load_and_preprocess_dataset():
    X, y = load_and_preprocess_dataset()
    assert not X.empty
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0

def test_calculate_feature_metrics():
    X, _ = load_and_preprocess_dataset()
    metrics = calculate_feature_metrics(X)
    assert isinstance(metrics, pd.DataFrame)
    assert "mean" in metrics.columns
    assert "std" in metrics.columns

def test_scale_features():
    X, _ = load_and_preprocess_dataset()
    scaled = scale_features(X)
    assert isinstance(scaled, pd.DataFrame)
    assert scaled.shape == X.shape
    assert list(scaled.columns) == list(X.columns)
    assert scaled.index.equals(X.index)
    assert abs(scaled.mean().mean()) < 1e-10
    assert abs(scaled.std(ddof=0).mean() - 1.0) < 1e-10 