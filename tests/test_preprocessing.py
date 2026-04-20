"""
Tests for preprocessing pipeline (Phase 2 stub)
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import get_preprocessing_pipeline


def test_preprocessing_pipeline_smoke():
    """Smoke test: pipeline builds without errors."""
    categorical_cols = ["gender", "Contract", "PaymentMethod"]
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    preprocessor = get_preprocessing_pipeline(categorical_cols, numerical_cols)
    assert preprocessor is not None


def test_preprocessing_pipeline_transforms():
    """Check pipeline transforms sample data correctly."""
    categorical_cols = ["Contract"]
    numerical_cols = ["tenure", "MonthlyCharges"]

    df = pd.DataFrame({
        "Contract": ["Month-to-month", "Two year", "Month-to-month"],
        "tenure": [1, 24, 6],
        "MonthlyCharges": [29.85, 56.95, 53.85],
    })

    preprocessor = get_preprocessing_pipeline(categorical_cols, numerical_cols)
    result = preprocessor.fit_transform(df)
    assert result.shape[0] == 3  # 3 rows preserved
