"""
Simple tests for dataset validation.
"""

import pytest
import pandas as pd
import numpy as np


class TestDatasetStructure:
    def test_dataset_has_required_columns(self, sample_dataset):
        """Test that dataset has required columns."""
        required_cols = ["gender", "tenure", "MonthlyCharges", "Churn"]
        for col in required_cols:
            assert col in sample_dataset.columns

    def test_dataset_has_rows(self, sample_dataset):
        """Test that dataset is not empty."""
        assert len(sample_dataset) > 0

    def test_dataset_no_null_values(self, sample_dataset):
        """Test that dataset has no null values."""
        assert sample_dataset.isnull().sum().sum() == 0

    def test_dataset_has_partner_column(self, sample_dataset):
        """Test that Partner column exists."""
        assert "Partner" in sample_dataset.columns

    def test_dataset_has_contract_column(self, sample_dataset):
        """Test that Contract column exists."""
        assert "Contract" in sample_dataset.columns

    def test_dataset_has_internet_service_column(self, sample_dataset):
        """Test that InternetService column exists."""
        assert "InternetService" in sample_dataset.columns


class TestDataTypes:
    def test_tenure_is_numeric(self, sample_dataset):
        """Test that tenure column is numeric."""
        assert pd.api.types.is_numeric_dtype(sample_dataset["tenure"])

    def test_monthly_charges_is_numeric(self, sample_dataset):
        """Test that MonthlyCharges column is numeric."""
        assert pd.api.types.is_numeric_dtype(sample_dataset["MonthlyCharges"])

    def test_churn_is_binary(self, sample_dataset):
        """Test that Churn column contains only Yes/No values."""
        valid_values = ["Yes", "No", 0, 1]
        assert all(val in valid_values for val in sample_dataset["Churn"].unique())

    def test_total_charges_is_numeric(self, sample_dataset):
        """Test that TotalCharges column is numeric."""
        if "TotalCharges" in sample_dataset.columns:
            assert pd.api.types.is_numeric_dtype(sample_dataset["TotalCharges"])

    def test_gender_is_string_or_categorical(self, sample_dataset):
        """Test that gender column is string or categorical."""
        assert sample_dataset["gender"].dtype == object or pd.api.types.is_categorical_dtype(sample_dataset["gender"])

    def test_contract_is_string_or_categorical(self, sample_dataset):
        """Test that Contract column is string or categorical."""
        assert sample_dataset["Contract"].dtype == object or pd.api.types.is_categorical_dtype(sample_dataset["Contract"])


class TestDataRanges:
    def test_tenure_is_non_negative(self, sample_dataset):
        """Test that tenure values are non-negative."""
        assert all(sample_dataset["tenure"] >= 0)

    def test_monthly_charges_is_positive(self, sample_dataset):
        """Test that MonthlyCharges values are positive."""
        assert all(sample_dataset["MonthlyCharges"] > 0)

    def test_total_charges_matches_monthly(self, sample_dataset):
        """Test that TotalCharges is roughly consistent with MonthlyCharges * tenure."""
        if "TotalCharges" in sample_dataset.columns:
            for _, row in sample_dataset.iterrows():
                expected = row["MonthlyCharges"] * row["tenure"]
                actual = row.get("TotalCharges", expected)
                # Allow 10% tolerance
                assert abs(expected - actual) <= expected * 0.1

    def test_tenure_max_value(self, sample_dataset):
        """Test that tenure does not exceed reasonable maximum."""
        assert all(sample_dataset["tenure"] <= 120)  # Max 120 months (10 years)

    def test_total_charges_is_non_negative(self, sample_dataset):
        """Test that TotalCharges values are non-negative."""
        if "TotalCharges" in sample_dataset.columns:
            assert all(sample_dataset["TotalCharges"] >= 0)
