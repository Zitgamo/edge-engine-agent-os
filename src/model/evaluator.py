from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

log = logging.getLogger(__name__)

from src.model.schema import FEATURE_COLS, TARGET_COL


class ModelEvaluator:
    def evaluate(self, model: xgb.XGBClassifier, df: pd.DataFrame, target_col: str | None = None) -> dict[str, float]:
        col = target_col or TARGET_COL
        df = df.dropna(subset=FEATURE_COLS + [col]).copy()
        X = df[FEATURE_COLS]
        y = df[col]

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_prob),
        }

        log.info("Evaluation metrics: %s", metrics)
        return metrics
