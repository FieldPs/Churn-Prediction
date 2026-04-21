import os
import pandas as pd
import joblib


DEFAULT_PROJECT_DIR = os.getenv("PROJECT_DIR", ".")
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", os.path.join(DEFAULT_PROJECT_DIR, "data"))
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(DEFAULT_PROJECT_DIR, "models"))

CURRENT_INPUT_PATH = os.getenv(
    "CURRENT_INPUT_PATH",
    os.path.join(DEFAULT_DATA_DIR, "current_customers.csv"),
)
PREDICTION_OUTPUT_PATH = os.getenv(
    "PREDICTION_OUTPUT_PATH",
    os.path.join(DEFAULT_DATA_DIR, "predictions.csv"),
)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(DEFAULT_MODEL_DIR, "voting_classifier.pkl"),
)
PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH",
    os.path.join(DEFAULT_MODEL_DIR, "preprocessor.pkl"),
)


REQUIRED_COLUMNS = [
    "churn_probability",
    "churn_prediction",
    "risk_level",
]


def assign_risk(prob):
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def batch_predict(
    input_path=CURRENT_INPUT_PATH,
    output_path=PREDICTION_OUTPUT_PATH,
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    df = pd.read_csv(input_path)
    X = df.drop(columns=["customerID", "Churn"], errors="ignore")

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["TotalCharges"] = X["TotalCharges"].fillna(X["TotalCharges"].median())

    X_processed = preprocessor.transform(X)
    probs = model.predict_proba(X_processed)[:, 1]
    preds = (probs >= 0.50).astype(int)

    result = df.copy()
    result["churn_probability"] = probs
    result["churn_prediction"] = preds
    result["risk_level"] = result["churn_probability"].apply(assign_risk)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    result_df = batch_predict()
    print(f"Prediction completed. Output saved to: {PREDICTION_OUTPUT_PATH}")
    print(result_df.head())