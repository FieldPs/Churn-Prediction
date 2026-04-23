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

DEMOGRAPHIC_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents"]

LABEL_ENCODE_MAPS = {
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

OHE_REVERSE_MAPS = {
    "InternetService": {0: "DSL", 1: "Fiber optic", 2: "No"},
    "Contract": {0: "Month-to-month", 1: "One year", 2: "Two year"},
    "PaymentMethod": {
        0: "Bank transfer (automatic)",
        1: "Credit card (automatic)",
        2: "Electronic check",
        3: "Mailed check",
    },
}


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


def _map_label_column(series, mapping, col_name):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    mapped = series.astype(str).str.strip().map(mapping)
    if mapped.isna().any():
        unknown = sorted(series[mapped.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unsupported values in column '{col_name}': {unknown[:5]}")
    return mapped


def _normalize_ohe_column(series, reverse_map, col_name):
    if pd.api.types.is_numeric_dtype(series):
        mapped = pd.to_numeric(series, errors="coerce").map(reverse_map)
        if mapped.isna().any():
            bad = sorted(series[mapped.isna()].dropna().astype(int).unique().tolist())
            raise ValueError(f"Unsupported encoded values in column '{col_name}': {bad[:5]}")
        return mapped

    numeric_try = pd.to_numeric(series, errors="coerce")
    if numeric_try.notna().all():
        mapped = numeric_try.astype(int).map(reverse_map)
        if mapped.isna().any():
            bad = sorted(numeric_try[mapped.isna()].dropna().astype(int).unique().tolist())
            raise ValueError(f"Unsupported encoded values in column '{col_name}': {bad[:5]}")
        return mapped

    return series.astype(str)


def _prepare_features(input_df, preprocessor):
    X = input_df.copy()
    X = X.drop(columns=["customerID", "Churn"], errors="ignore")

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        median_val = X["TotalCharges"].median()
        X["TotalCharges"] = X["TotalCharges"].fillna(0.0 if pd.isna(median_val) else median_val)

    for col, mapping in LABEL_ENCODE_MAPS.items():
        if col in X.columns:
            X[col] = _map_label_column(X[col], mapping, col)

    for col, reverse_map in OHE_REVERSE_MAPS.items():
        if col in X.columns:
            X[col] = _normalize_ohe_column(X[col], reverse_map, col)

    X = X.drop(columns=DEMOGRAPHIC_COLS, errors="ignore")

    if hasattr(preprocessor, "feature_names_in_"):
        expected_cols = list(preprocessor.feature_names_in_)
        missing_cols = [col for col in expected_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        X = X[expected_cols]

    if X.isna().any().any():
        missing_info = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN values found after feature preparation in columns: {missing_info}")

    return X


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
    X = _prepare_features(df, preprocessor)
    X_processed = preprocessor.transform(X)
    probs = model.predict_proba(X_processed)[:, 1]
    preds = (probs >= 0.50).astype(int)

    result = df.copy()
    result["churn_probability"] = probs
    result["churn_prediction"] = ["Yes" if p == 1 else "No" for p in preds]
    result["risk_level"] = result["churn_probability"].apply(assign_risk)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    result_df = batch_predict()
    print(f"Prediction completed. Output saved to: {PREDICTION_OUTPUT_PATH}")
    print(result_df.head())