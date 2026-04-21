import sys
import os
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from batch_predict import batch_predict, REQUIRED_COLUMNS


PROJECT_DIR = os.getenv("PROJECT_DIR", ".")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_DIR, "data"))
OUTPUT_PATH = os.getenv("PREDICTION_OUTPUT_PATH", os.path.join(DATA_DIR, "predictions.csv"))

default_input = os.path.join(DATA_DIR, "current_customers.csv")
if not os.path.exists(default_input):
    default_input = os.path.join(DATA_DIR, "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

INPUT_PATH = os.getenv("CURRENT_INPUT_PATH", default_input)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(PROJECT_DIR, "models", "voting_classifier.pkl"))
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", os.path.join(PROJECT_DIR, "models", "preprocessor.pkl"))

class DummyPreprocessor:
    def transform(self, X):
        return X

class DummyModel:
    def predict_proba(self, X):
        import numpy as np
        # Return random probabilities for 2 classes, ensuring column 1 (churn) exists
        probs = np.random.rand(len(X), 2)
        # Normalize so they sum to 1
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

original_exists = os.path.exists

def custom_exists(path):
    # Pretend the model and preprocessor files always exist
    if path in [MODEL_PATH, PREPROCESSOR_PATH]:
        return True
    return original_exists(path)

def custom_joblib_load(path):
    if path == MODEL_PATH:
        return DummyModel()
    if path == PREPROCESSOR_PATH:
        return DummyPreprocessor()
    raise FileNotFoundError(f"Mock didn't expect to load {path}")

@patch("os.path.exists", side_effect=custom_exists)
@patch("joblib.load", side_effect=custom_joblib_load)
def test_batch_predict_smoke(mock_load, mock_exists):
    import tempfile
    
    # 1. Create a dummy dataframe to bypass DVC missing data in CI
    dummy_data = {
        "customerID": ["1234-ABCDE", "5678-FGHIJ"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [12, 1],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "No"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "Month-to-month"],
        "PaperlessBilling": ["Yes", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": [600.0, 80.0]
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        dummy_df.to_csv(tmp.name, index=False)
        dummy_input_path = tmp.name

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    try:
        # 2. Run prediction using the dummy input
        result = batch_predict(
            input_path=dummy_input_path,
            output_path=OUTPUT_PATH,
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
        )

        assert os.path.exists(OUTPUT_PATH), f"Output file was not created: {OUTPUT_PATH}"
        assert len(result) > 0, "Returned prediction result is empty"

        pred_df = pd.read_csv(OUTPUT_PATH)
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in pred_df.columns]
        assert not missing_cols, f"Missing required columns: {missing_cols}"
        assert not pred_df.empty, "predictions.csv is empty"
        assert pred_df["churn_probability"].between(0, 1).all(), "Invalid probability values detected"
    
    finally:
        # Clean up temporary input file
        if os.path.exists(dummy_input_path):
            os.remove(dummy_input_path)