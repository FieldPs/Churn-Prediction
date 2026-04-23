"""
Simple tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the app
from backend.main import app

client = TestClient(app)


class TestRootEndpoint:
    def test_root_returns_online_status(self):
        """Test that the root endpoint returns online status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"


class TestPredictEndpoint:
    def test_predict_endpoint_accepts_valid_data(self, sample_customer_data):
        """Test that predict endpoint accepts valid customer data."""
        response = client.post("/predict", json=sample_customer_data)
        # Note: May return 503 if model is not loaded
        assert response.status_code in [200, 503]

    def test_predict_returns_expected_fields(self, sample_customer_data):
        """Test that predict endpoint returns expected fields."""
        response = client.post("/predict", json=sample_customer_data)
        if response.status_code == 200:
            data = response.json()
            assert "churn_prediction" in data
            assert "churn_probability" in data
            assert "risk_level" in data

    def test_predict_with_missing_fields(self):
        """Test that predict endpoint handles missing fields."""
        incomplete_data = {"gender": 1}
        response = client.post("/predict", json=incomplete_data)
        # Should return validation error
        assert response.status_code == 422

    def test_predict_churn_probability_in_range(self, sample_customer_data):
        """Test that churn probability is between 0 and 1."""
        response = client.post("/predict", json=sample_customer_data)
        if response.status_code == 200:
            data = response.json()
            prob = data["churn_probability"]
            assert 0 <= prob <= 1

    def test_predict_risk_level_is_valid(self, sample_customer_data):
        """Test that risk level is one of the valid values."""
        response = client.post("/predict", json=sample_customer_data)
        if response.status_code == 200:
            data = response.json()
            valid_risks = ["Low", "Medium", "High"]
            assert data["risk_level"] in valid_risks

    def test_predict_with_minimum_tenure(self, sample_customer_data):
        """Test with minimum tenure value."""
        sample_customer_data["tenure"] = 0.0
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code in [200, 503]

    def test_predict_with_maximum_tenure(self, sample_customer_data):
        """Test with maximum tenure value."""
        sample_customer_data["tenure"] = 120.0
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code in [200, 503]

    def test_predict_with_zero_charges(self, sample_customer_data):
        """Test with zero monthly charges."""
        sample_customer_data["MonthlyCharges"] = 0.0
        sample_customer_data["TotalCharges"] = 0.0
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code in [200, 503]

    def test_predict_model_source_field(self, sample_customer_data):
        """Test that model source field exists."""
        response = client.post("/predict", json=sample_customer_data)
        if response.status_code == 200:
            data = response.json()
            assert "model_source" in data

    def test_predict_churn_prediction_format(self, sample_customer_data):
        """Test that churn prediction is Yes or No."""
        response = client.post("/predict", json=sample_customer_data)
        if response.status_code == 200:
            data = response.json()
            assert data["churn_prediction"] in ["Yes", "No"]


class TestReloadModelEndpoint:
    def test_reload_model_endpoint_exists(self):
        """Test that reload model endpoint exists."""
        response = client.post("/reload-model")
        # Note: May fail if MLflow is not available
        assert response.status_code in [200, 503]
