"""
evaluate.py
-----------
Phase 2 — Post-training evaluation.

Performs:
  1. Cross-group Evaluation: Recall/F1 broken down by key customer segments
     (Contract type, SeniorCitizen, tenure group)
  2. Churn Risk Score distribution: histogram of predicted probabilities
  3. Confusion matrix log (text-based for CI/batch runs)

NOTE: Bias check — Gender is deliberately NOT used as a grouping feature
to maintain fairness (Phase 6 requirement).
"""

import logging
import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "voting_classifier.pkl")
RECALL_THRESHOLD = 0.80


# ─── Group Evaluation ─────────────────────────────────────────────────────────

def evaluate_by_group(
    model,
    X_test_raw: pd.DataFrame,
    X_test_transformed: np.ndarray,
    y_test: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute Recall, Precision, F1 per subgroup value of group_col.

    Args:
        X_test_raw: Original (un-preprocessed) test DataFrame for grouping.
        X_test_transformed: Preprocessed numpy array to run predictions on.
        y_test: Ground truth labels.
        group_col: Column name to group by (e.g., "Contract").

    Returns:
        DataFrame with per-group metrics.
    """
    if group_col not in X_test_raw.columns:
        logger.warning(f"Column '{group_col}' not found in test data — skipping group eval.")
        return pd.DataFrame()

    predictions = model.predict(X_test_transformed)
    results = []

    for group_val in X_test_raw[group_col].unique():
        mask = (X_test_raw[group_col] == group_val).values
        if mask.sum() == 0:
            continue
        y_grp = y_test[mask]
        p_grp = predictions[mask]
        results.append({
            "group":     group_col,
            "value":     group_val,
            "n_samples": int(mask.sum()),
            "recall":    round(recall_score(y_grp, p_grp, zero_division=0), 4),
            "precision": round(precision_score(y_grp, p_grp, zero_division=0), 4),
            "f1":        round(f1_score(y_grp, p_grp, zero_division=0), 4),
        })

    df_results = pd.DataFrame(results)
    logger.info(f"\nGroup evaluation — {group_col}:\n{df_results.to_string(index=False)}")
    return df_results


# ─── Churn Risk Score Distribution ───────────────────────────────────────────

def churn_risk_distribution(model, X_test_transformed: np.ndarray) -> np.ndarray:
    """
    Return predicted churn probabilities (class=1) for all test samples.
    Useful for business risk-tiering (High/Medium/Low churn risk).
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_transformed)[:, 1]
    else:
        logger.warning("Model does not support predict_proba — using hard predictions.")
        probs = model.predict(X_test_transformed).astype(float)

    high_risk   = (probs >= 0.70).sum()
    medium_risk = ((probs >= 0.40) & (probs < 0.70)).sum()
    low_risk    = (probs < 0.40).sum()

    logger.info(f"Churn Risk Distribution:")
    logger.info(f"  High   (≥70%): {high_risk} customers ({high_risk / len(probs):.1%})")
    logger.info(f"  Medium (40-70%): {medium_risk} customers ({medium_risk / len(probs):.1%})")
    logger.info(f"  Low    (<40%): {low_risk} customers ({low_risk / len(probs):.1%})")
    return probs


# ─── Main Evaluate Function ───────────────────────────────────────────────────

def evaluate(X_test_transformed: np.ndarray, y_test: np.ndarray,
             X_test_raw: pd.DataFrame = None, model=None) -> dict:
    """
    Run full evaluation pipeline. Loads model from disk if not provided.

    Returns:
        dict of overall metrics + per-group DataFrames
    """
    if model is None:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test_transformed)
    probs = churn_risk_distribution(model, X_test_transformed)

    # ── Overall metrics ────────────────────────────────────────────────────
    metrics = {
        "recall":    recall_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "f1":        f1_score(y_test, predictions),
    }
    logger.info(f"\n{classification_report(y_test, predictions, target_names=['No Churn', 'Churn'])}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")

    passed = metrics["recall"] >= RECALL_THRESHOLD
    logger.info(f"Recall threshold ({RECALL_THRESHOLD}) passed: {passed}")

    # ── Group evaluation (skip if raw df not provided) ─────────────────────
    group_results = {}
    if X_test_raw is not None:
        # NOTE: Gender intentionally excluded for fairness (Phase 6)
        for group_col in ["Contract", "SeniorCitizen"]:
            group_results[group_col] = evaluate_by_group(
                model, X_test_raw, X_test_transformed, y_test, group_col
            )

    return {
        **metrics,
        "threshold_passed": passed,
        "churn_probabilities": probs,
        "group_results": group_results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import ingest
    from preprocessing import preprocess

    df_raw = ingest()
    X_train, X_test, y_train, y_test, _ = preprocess(df_raw, save_preprocessor=False)

    # Keep a raw copy for group evaluation (before preprocessing)
    from sklearn.model_selection import train_test_split
    from preprocessing import label_encode_binary_cols, TARGET_COL
    df_encoded = label_encode_binary_cols(df_raw)
    X_raw = df_encoded.drop(columns=[TARGET_COL])
    _, X_test_raw, _, _ = train_test_split(X_raw, df_encoded[TARGET_COL], test_size=0.30, random_state=40, stratify=df_encoded[TARGET_COL])

    results = evaluate(X_test, y_test, X_test_raw=X_test_raw)
    print(f"\nRecall threshold passed: {results['threshold_passed']}")
