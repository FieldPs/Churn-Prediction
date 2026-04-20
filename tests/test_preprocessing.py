"""
Tests for the ML pipeline scripts.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_ingestion import clean_data
from preprocessing import label_encode_binary_cols, build_preprocessor, LABEL_ENCODE_COLS, OHE_COLS, NUMERICAL_COLS


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Minimal raw DataFrame that mimics the Telco Churn CSV structure."""
    return pd.DataFrame({
        "customerID":    ["1", "2", "3", "4"],
        "gender":        ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 0, 1, 0],
        "Partner":       ["Yes", "No", "Yes", "No"],
        "Dependents":    ["No", "No", "Yes", "No"],
        "tenure":        [0, 12, 24, 6],      # row 0 should be dropped (tenure==0)
        "PhoneService":  ["Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "No", "Yes"],
        "OnlineBackup":   ["Yes", "No", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes"],
        "TechSupport":    ["No", "No", "Yes", "No"],
        "StreamingTV":    ["No", "Yes", "No", "No"],
        "StreamingMovies": ["No", "Yes", "No", "No"],
        "Contract":       ["Month-to-month", "Two year", "Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod":  ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
        "TotalCharges":   ["0", "683.4", "1290.4", "253.8"],  # whitespace/string intentional
        "Churn":          ["No", "No", "Yes", "No"],
    })


# ─── data_ingestion tests ──────────────────────────────────────────────────────

class TestCleanData:
    def test_drops_customer_id(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert "customerID" not in cleaned.columns

    def test_drops_zero_tenure_rows(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert 0 not in cleaned["tenure"].values

    def test_total_charges_is_numeric(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])

    def test_no_nulls_after_cleaning(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert cleaned.isnull().sum().sum() == 0

    def test_senior_citizen_mapped_to_str(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert cleaned["SeniorCitizen"].dtype == object
        assert set(cleaned["SeniorCitizen"].unique()).issubset({"Yes", "No"})


# ─── preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessor:
    def test_builds_without_error(self):
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_label_encode_binary_cols(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        encoded = label_encode_binary_cols(cleaned)
        # All label-encoded cols should now be numeric
        for col in LABEL_ENCODE_COLS:
            if col in encoded.columns:
                assert pd.api.types.is_numeric_dtype(encoded[col]), f"{col} not numeric after encoding"

    def test_preprocessor_output_shape(self, sample_raw_df):
        """Preprocessor output should have more columns than input (due to OHE)."""
        cleaned = clean_data(sample_raw_df)
        encoded = label_encode_binary_cols(cleaned)
        X = encoded.drop(columns=["Churn"])
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(X)
        assert result.shape[0] == len(X)
        assert result.shape[1] > len(X.columns)   # OHE expands columns
