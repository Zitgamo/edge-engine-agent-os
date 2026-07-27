from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.config import Config

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
