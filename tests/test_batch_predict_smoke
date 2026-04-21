import os
import pandas as pd

from batch_predict import batch_predict, REQUIRED_COLUMNS


PROJECT_DIR = os.getenv("PROJECT_DIR", ".")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_DIR, "data"))
OUTPUT_PATH = os.getenv("PREDICTION_OUTPUT_PATH", os.path.join(DATA_DIR, "predictions.csv"))
INPUT_PATH = os.getenv("CURRENT_INPUT_PATH", os.path.join(DATA_DIR, "current_customers.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(PROJECT_DIR, "models", "voting_classifier.pkl"))
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", os.path.join(PROJECT_DIR, "models", "preprocessor.pkl"))


def test_batch_predict_smoke():
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    result = batch_predict(
        input_path=INPUT_PATH,
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