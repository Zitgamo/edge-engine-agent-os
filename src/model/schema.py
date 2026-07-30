"""Single source of truth for feature columns and model constants."""

from __future__ import annotations

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

ENSEMBLE_HORIZONS = [1, 5, 10, 20]

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

N_PICKS = 3
HOLDING_PERIOD = 20
