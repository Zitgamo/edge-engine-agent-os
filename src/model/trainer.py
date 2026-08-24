from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb

from src.config import Config

log = logging.getLogger(__name__)

from src.model.schema import FEATURE_COLS, TARGET_COL, XGBOOST_PARAMS
from src.model.splits import purged_train_test_split, recent_date_window


class ModelTrainer:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def train(self, df: pd.DataFrame) -> xgb.XGBClassifier:
        horizon = int(TARGET_COL.rsplit("_", 1)[-1].removesuffix("d"))
        label_end_col = f"label_end_date_{horizon}d"
        required = FEATURE_COLS + [TARGET_COL, label_end_col, "date"]
        df = df.dropna(subset=required).copy()
        dates = sorted(pd.to_datetime(df["date"]).dt.normalize().unique())
        split = int(len(dates) * 0.8)
        if split <= 0 or split >= len(dates):
            raise ValueError("Not enough dated rows for purged model split")
        train, test = purged_train_test_split(
            df,
            test_start=pd.Timestamp(dates[split]),
            label_end_col=label_end_col,
        )
        train = recent_date_window(train, self.config.model_training_days)
        if train.empty or test.empty or train[TARGET_COL].nunique() < 2:
            raise ValueError("Purged model split does not contain usable train/test data")

        X_train = train[FEATURE_COLS]
        y_train = train[TARGET_COL]
        X_test = test[FEATURE_COLS]
        y_test = test[TARGET_COL]

        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        log.info("Model trained successfully")
        return model
