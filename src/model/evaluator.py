from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "return_5d", "return_20d", "return_60d",
    "rs_5d", "rs_20d", "rs_60d",
    "atr", "atr_pct",
    "volume_surge", "volume_surge_flag",
    "vndusd", "sbv_rate", "cpi_mom",
    "pe_ratio", "pb_ratio", "roe", "rev_growth", "earn_growth",
    "profit_margin", "debt_equity", "div_yield", "log_mcap", "forward_pe",
]

TARGET_COL = "outperform_5d"


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
