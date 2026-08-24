from __future__ import annotations

import pandas as pd
import pytest

from src.model.splits import (
    purged_recent_train_window,
    purged_train_test_split,
    recent_date_window,
    require_columns,
    require_label_end_columns,
)


def test_purged_split_excludes_labels_reaching_test_boundary() -> None:
    dates = pd.bdate_range("2026-01-01", periods=10)
    df = pd.DataFrame({
        "date": dates,
        "label_end_date_5d": [*dates[2:], pd.NaT, pd.NaT],
        "value": range(10),
    })

    train, test = purged_train_test_split(
        df,
        test_start=dates[5],
        label_end_col="label_end_date_5d",
    )

    assert train["value"].tolist() == [0, 1, 2]
    assert test["value"].tolist() == [5, 6, 7, 8, 9]
    assert (train["label_end_date_5d"] < dates[5]).all()


def test_purged_split_requires_maturity_metadata() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"])})

    with pytest.raises(ValueError, match="purged-split columns"):
        purged_train_test_split(
            df,
            test_start="2026-01-02",
            label_end_col="label_end_date_5d",
        )


def test_persisted_features_fail_closed_without_label_end_columns() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"])})

    with pytest.raises(ValueError, match="label maturity metadata"):
        require_label_end_columns(df, [1, 5, 20])


def test_recent_date_window_keeps_trading_sessions_not_calendar_days() -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    df = pd.DataFrame({"date": dates, "value": range(5)})

    result = recent_date_window(df, 2)

    assert result["value"].tolist() == [3, 4]


def test_require_columns_reports_execution_schema_missing() -> None:
    with pytest.raises(ValueError, match="execution labels"):
        require_columns(
            pd.DataFrame(),
            ["execution_outperform_20d"],
            description="execution labels",
        )


def test_purged_recent_window_applies_maturity_and_recency() -> None:
    dates = pd.bdate_range("2026-01-01", periods=8)
    df = pd.DataFrame({
        "date": dates,
        "label_end_date_5d": dates + pd.Timedelta(days=1),
        "value": range(8),
    })

    result = purged_recent_train_window(
        df,
        test_start=dates[7],
        label_end_col="label_end_date_5d",
        max_dates=3,
    )

    assert result["value"].tolist() == [4, 5, 6]
