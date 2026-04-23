"""
Shared fixtures for all tests.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing."""
    return {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 12.0,
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 1,
        "OnlineSecurity": 0,
        "OnlineBackup": 1,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 1,
        "StreamingMovies": 1,
        "Contract": 0,
        "PaperlessBilling": 1,
        "PaymentMethod": 2,
        "MonthlyCharges": 55.5,
        "TotalCharges": 660.0
    }


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return pd.DataFrame({
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 0, 1, 0],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [12, 24, 6, 36],
        "PhoneService": ["Yes", "Yes", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "Contract": ["Month-to-month", "Two year", "Month-to-month", "One year"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
        "TotalCharges": [358.2, 1366.8, 323.1, 1522.8],
        "Churn": ["No", "No", "Yes", "No"]
    })


@pytest.fixture
def sample_predictions():
    """Sample prediction results for testing."""
    return pd.DataFrame({
        "customer_id": ["1", "2", "3", "4"],
        "churn_prediction": [0, 1, 0, 0],
        "churn_probability": [0.25, 0.85, 0.35, 0.15]
    })
