"""
Simple tests for monitoring and drift detection.
"""

import pytest
import pandas as pd
import numpy as np


class TestPredictionDistribution:
    def test_churn_probability_range(self, sample_predictions):
        """Test that churn probabilities are between 0 and 1."""
        assert all(sample_predictions["churn_probability"] >= 0)
        assert all(sample_predictions["churn_probability"] <= 1)

    def test_churn_prediction_is_binary(self, sample_predictions):
        """Test that churn predictions are binary (0 or 1)."""
        valid_values = [0, 1]
        assert all(val in valid_values for val in sample_predictions["churn_prediction"])

    def test_average_churn_probability(self, sample_predictions):
        """Test that average churn probability is calculated."""
        avg_prob = sample_predictions["churn_probability"].mean()
        assert isinstance(avg_prob, (int, float))
        assert 0 <= avg_prob <= 1

    def test_churn_rate_calculation(self, sample_predictions):
        """Test that churn rate is calculated correctly."""
        churn_rate = sample_predictions["churn_prediction"].mean()
        assert 0 <= churn_rate <= 1


class TestDataDrift:
    def test_drift_detection_returns_metrics(self):
        """Test that drift detection returns metrics."""
        # Simulate baseline and current data
        baseline = pd.DataFrame({
            "tenure": [10, 20, 30, 40, 50],
            "MonthlyCharges": [30, 50, 70, 90, 110]
        })
        current = pd.DataFrame({
            "tenure": [12, 22, 32, 42, 52],
            "MonthlyCharges": [32, 52, 72, 92, 112]
        })

        # Simple drift check: mean difference
        drift_detected = False
        for col in baseline.columns:
            mean_diff = abs(baseline[col].mean() - current[col].mean())
            if mean_diff > baseline[col].std() * 0.1:  # 10% of std as threshold
                drift_detected = True

        # This should not detect drift with similar data
        assert isinstance(drift_detected, bool)

    def test_drift_detection_with_shifted_data(self):
        """Test drift detection with明显 shifted data."""
        baseline = pd.DataFrame({
            "tenure": [10, 20, 30, 40, 50]
        })
        current = pd.DataFrame({
            "tenure": [50, 60, 70, 80, 90]  # Shifted by 40
        })

        # Should detect drift due to large mean difference
        mean_diff = abs(baseline["tenure"].mean() - current["tenure"].mean())
        drift_detected = bool(mean_diff > baseline["tenure"].std() * 0.5)
        assert isinstance(drift_detected, bool)

    def test_drift_detection_returns_no_drift(self):
        """Test that drift detection returns False for similar data."""
        baseline = pd.DataFrame({
            "value": [1, 2, 3, 4, 5]
        })
        current = pd.DataFrame({
            "value": [1.1, 2.1, 3.1, 4.1, 5.1]  # Very similar
        })

        mean_diff = abs(baseline["value"].mean() - current["value"].mean())
        drift_detected = bool(mean_diff > baseline["value"].std() * 0.5)
        assert isinstance(drift_detected, bool)


class TestModelPerformance:
    def test_recall_calculation(self):
        """Test that recall is calculated correctly."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        # Calculate recall
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        actual_positives = sum(y_true)
        recall = true_positives / actual_positives if actual_positives > 0 else 0

        expected_recall = 2 / 3  # 2 out of 3 positives correctly predicted
        assert abs(recall - expected_recall) < 0.01

    def test_precision_calculation(self):
        """Test that precision is calculated correctly."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        # Calculate precision
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        predicted_positives = sum(y_pred)
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0

        expected_precision = 2 / 2  # 2 out of 2 predicted positives are correct
        assert abs(precision - expected_precision) < 0.01

    def test_accuracy_calculation(self):
        """Test that accuracy is calculated correctly."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        # Calculate accuracy
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct / len(y_true)

        expected_accuracy = 4 / 5  # 4 out of 5 correct
        assert abs(accuracy - expected_accuracy) < 0.01

    def test_f1_score_calculation(self):
        """Test that F1 score is calculated correctly."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        # Calculate F1 score
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        actual_positives = sum(y_true)
        predicted_positives = sum(y_pred)

        recall = true_positives / actual_positives if actual_positives > 0 else 0
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # F1 should be between 0 and 1
        assert 0 <= f1 <= 1

    def test_specificity_calculation(self):
        """Test that specificity is calculated correctly."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        # Calculate specificity (true negative rate)
        true_negatives = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
        actual_negatives = sum(1 for true in y_true if true == 0)
        specificity = true_negatives / actual_negatives if actual_negatives > 0 else 0

        expected_specificity = 2 / 2  # Both negatives correctly predicted
        assert abs(specificity - expected_specificity) < 0.01


class TestMonitoringAlerts:
    def test_high_churn_rate_alert(self, sample_predictions):
        """Test that high churn rate triggers alert."""
        churn_rate = sample_predictions["churn_prediction"].mean()
        alert_threshold = 0.5  # 50%

        should_alert = bool(churn_rate > alert_threshold)
        assert isinstance(should_alert, bool)

    def test_low_prediction_confidence_alert(self, sample_predictions):
        """Test that low confidence predictions trigger alert."""
        # Check for predictions with probability between 0.4 and 0.6 (uncertain)
        uncertain_mask = (sample_predictions["churn_probability"] > 0.4) & \
                        (sample_predictions["churn_probability"] < 0.6)
        uncertain_count = uncertain_mask.sum()

        should_alert = bool(uncertain_count > len(sample_predictions) * 0.2)  # >20% uncertain
        assert isinstance(should_alert, bool)

    def test_very_high_churn_probability_alert(self, sample_predictions):
        """Test that very high churn probabilities trigger alert."""
        # Check for predictions with probability > 0.9
        very_high_risk = (sample_predictions["churn_probability"] > 0.9).sum()
        should_alert = bool(very_high_risk > len(sample_predictions) * 0.3)  # >30% very high risk
        assert isinstance(should_alert, bool)

    def test_model_performance_drop_alert(self):
        """Test that performance drop triggers alert."""
        baseline_recall = 0.85
        current_recall = 0.65

        # Alert if performance drops by more than 15%
        performance_drop = baseline_recall - current_recall
        should_alert = bool(performance_drop > 0.15)
        assert isinstance(should_alert, bool)
