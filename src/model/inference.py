from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.config import Config

log = logging.getLogger(__name__)

from src.model.schema import FEATURE_COLS


class ModelInference:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._model: xgb.XGBClassifier | None = None

    def load(self, path: Path | None = None) -> None:
        model_path = path or self.config.model_path
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_path))
        log.info("Model loaded from %s", model_path)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        df = df.copy()
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            log.warning("Missing features: %s. Filling with 0.", missing)
            for c in missing:
                df[c] = 0.0

        X = df[FEATURE_COLS].fillna(0)
        df["score"] = self._model.predict_proba(X)[:, 1]
        df["prediction"] = self._model.predict(X)
        return df
