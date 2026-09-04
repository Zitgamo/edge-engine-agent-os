"""Single source of truth for model features and constants.

The production model intentionally uses only point-in-time-safe technical
features plus the historical VND/USD series. Current Yahoo ``info`` snapshots
and unavailable macro releases are kept outside the supervised model because
they would leak future information into historical training rows. The
production T+20 target can
use the same executable next-open/T+2/SL/TP mechanics as the actuals tracker;
legacy close-to-close columns remain supported.
"""

from __future__ import annotations

FEATURE_COLS = [
    "return_5d", "return_20d", "return_60d",
    "rs_5d", "rs_20d", "rs_60d",
    "atr", "atr_pct",
    "volume_surge", "volume_surge_flag",
    "vndusd",
]

MODEL_VERSION = "xgboost_technical_v7_execution_max_sl05_tp10"

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
