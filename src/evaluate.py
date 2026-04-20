"""
evaluate.py
-----------
Phase 2 - Modularize ML Pipeline

Performs:
  - Cross-group Evaluation (by demographic / contract type)
  - Churn Risk Score distribution analysis

TODO: Full implementation — to be refactored from notebook.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_by_group(model, X_test: pd.DataFrame, y_test: pd.Series, group_col: str):
    """Compute Recall and F1 per subgroup for fairness analysis."""
    # TODO: Full implementation — refactor from notebook
    raise NotImplementedError("Full implementation in Phase 2")


def churn_risk_distribution(model, X_test: pd.DataFrame):
    """Plot and return distribution of predicted churn probabilities."""
    # TODO: Full implementation — refactor from notebook
    raise NotImplementedError("Full implementation in Phase 2")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluate script placeholder — Phase 2 implementation pending.")
