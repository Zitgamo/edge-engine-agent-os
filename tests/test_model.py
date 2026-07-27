from __future__ import annotations

import pandas as pd
import pytest

from src.model.evaluator import ModelEvaluator
from src.model.trainer import FEATURE_COLS, TARGET_COL, ModelTrainer


@pytest.fixture
def training_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    n = len(dates)
    data: dict[str, list] = {"date": dates}
    for col in FEATURE_COLS:
        data[col] = [float(i % 100) / 100.0 for i in range(n)]
    data[TARGET_COL] = [i % 2 for i in range(n)]
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
        assert 0.0 <= metrics["accuracy"] <= 1.0
