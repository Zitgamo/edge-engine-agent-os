from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

log = logging.getLogger(__name__)

from src.model.schema import FEATURE_COLS, TARGET_COL


class ModelEvaluator:
    """Evaluate whichever target schema the caller explicitly supplies."""

    def evaluate(self, model: xgb.XGBClassifier, df: pd.DataFrame, target_col: str | None = None) -> dict[str, float]:
        col = target_col or TARGET_COL
        df = df.dropna(subset=FEATURE_COLS + [col]).copy()
        if df.empty:
            raise ValueError(f"No rows available to evaluate target {col}")
        X = df[FEATURE_COLS].fillna(0)
        y = df[col]

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        roc_auc = roc_auc_score(y, y_prob) if y.nunique() > 1 else 0.5
        positive_rate = float(y.mean())
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc,
            "positive_rate": positive_rate,
            "majority_baseline_accuracy": max(positive_rate, 1.0 - positive_rate),
        }

        # The production use case is cross-sectional top-k ranking, not
        # classifying every stock independently.  Measure the ranking edge
        # on the same horizon's continuous excess return whenever it is
        # available, so a low-probability but useful ranking is not rejected
        # solely because its class ROC-AUC is close to 0.5.
        return_col = col.replace("outperform_", "excess_return_")
        if "date" in df.columns and return_col in df.columns:
            ranking = df[["date", return_col]].copy()
            ranking["score"] = y_prob
            ranking = ranking.dropna(subset=[return_col, "date"])
            daily: list[dict[str, float]] = []
            for _, day in ranking.groupby("date"):
                if len(day) < 3:
                    continue
                top = day.nlargest(3, "score")
                top_return = float(top[return_col].mean())
                universe_return = float(day[return_col].mean())
                daily.append({
                    "top3_win_rate": float((top[return_col] > 0).mean()),
                    "top3_excess_return": top_return,
                    "universe_excess_return": universe_return,
                    "top3_spread": top_return - universe_return,
                })
            if daily:
                ranking_metrics = pd.DataFrame(daily)
                metrics.update({
                    "evaluation_dates": float(len(ranking_metrics)),
                    "top3_win_rate": float(ranking_metrics["top3_win_rate"].mean()),
                    "top3_excess_return": float(ranking_metrics["top3_excess_return"].mean()),
                    "universe_excess_return": float(ranking_metrics["universe_excess_return"].mean()),
                    "top3_spread": float(ranking_metrics["top3_spread"].mean()),
                })

        log.info("Evaluation metrics: %s", metrics)
        return metrics
