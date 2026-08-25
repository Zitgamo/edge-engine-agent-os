from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.model.evaluator import ModelEvaluator
from src.model.trainer import FEATURE_COLS, TARGET_COL, ModelTrainer


def test_model_path_override_is_used_for_primary_t20_model(tmp_path) -> None:
    config = Config()
    config.model_path = tmp_path / "custom" / "primary.json"

    assert config.model_path_for_horizon(20) == tmp_path / "custom" / "primary.json"
    assert config.model_path_for_horizon(5).as_posix() == "models/xgboost_model_h5.json"


@pytest.fixture
def training_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    n = len(dates)
    data: dict[str, list] = {"date": dates}
    for col in FEATURE_COLS:
        data[col] = [float(i % 100) / 100.0 for i in range(n)]
    data[TARGET_COL] = [i % 2 for i in range(n)]
    data["label_end_date_5d"] = dates.to_series().shift(-5).to_numpy()
    return pd.DataFrame(data)


class TestModelTrainer:
    def test_train_returns_model(self, training_df: pd.DataFrame) -> None:
        trainer = ModelTrainer()
        model = trainer.train(training_df)
        assert model is not None


class TestModelEvaluator:
    def test_evaluate_returns_metrics(self, training_df: pd.DataFrame) -> None:
        trainer = ModelTrainer()
        model = trainer.train(training_df)
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, training_df)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "positive_rate" in metrics
        assert "majority_baseline_accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_reports_top3_ranking_metrics(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        rows = []
        for day in dates:
            for ticker, score in [("AAA", 0.9), ("BBB", 0.6), ("CCC", 0.1)]:
                row = {col: 0.0 for col in FEATURE_COLS}
                row.update({
                    "date": day,
                    "outperform_5d": int(score > 0.5),
                    "excess_return_5d": score - 0.5,
                    "score": score,
                })
                rows.append(row)
        df = pd.DataFrame(rows)

        class StubModel:
            def predict(self, frame: pd.DataFrame) -> list[int]:
                return [1 if value > 0 else 0 for value in frame[FEATURE_COLS[0]]]

            def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
                # The evaluator only needs a deterministic ranking score.
                values = pd.Series(range(len(frame)), dtype=float) / max(len(frame), 1)
                return np.asarray([[1.0 - value, value] for value in values])

        metrics = ModelEvaluator().evaluate(StubModel(), df)

        assert metrics["evaluation_dates"] == 2
        assert "top3_excess_return" in metrics
        assert "top3_spread" in metrics
