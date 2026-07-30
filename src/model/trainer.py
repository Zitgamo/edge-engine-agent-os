from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.config import Config

log = logging.getLogger(__name__)

from src.model.schema import FEATURE_COLS, TARGET_COL


class ModelTrainer:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def train(self, df: pd.DataFrame) -> xgb.XGBClassifier:
        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False,
        )

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        log.info("Model trained successfully")
        return model
