from __future__ import annotations

import pandas as pd
import pytest

from src.labels.execution import add_execution_labels
from src.model.targets import resolve_target_spec, target_spec


def _write_prices(path, closes, opens=None):
    dates = pd.bdate_range("2026-01-01", periods=len(closes))
    opens = opens or closes
    frame = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": [value * 1.01 for value in closes],
        "low": [value * 0.99 for value in closes],
        "close": closes,
    })
    frame.to_parquet(path)
    return dates


def test_execution_labels_match_next_open_backtest_and_full_horizon(tmp_path) -> None:
    dates = _write_prices(
        tmp_path / "AAA_raw.parquet",
        [100, 110, 115, 120, 125, 130],
        [100, 111, 116, 121, 126, 131],
    )
    _write_prices(
        tmp_path / "VNINDEX_raw.parquet",
        [1000, 1000, 1010, 1020, 1030, 1040],
        [1000, 2000, 2010, 2020, 2030, 2040],
    )
    panel = pd.DataFrame({
        "date": [dates[0], dates[-1]],
        "ticker": ["AAA", "AAA"],
    })

    result = add_execution_labels(
        panel,
        raw_data_dir=tmp_path,
        stop_loss=0.0,
        take_profit=0.0,
        holding_period=3,
    )

    row = result.iloc[0]
    expected_stock = (120 - 111) / 111
    expected_benchmark = (1020 - 2000) / 2000
    assert row["execution_excess_return_3d"] == pytest.approx(
        expected_stock - expected_benchmark
    )
    assert row["execution_outperform_3d"] == 1.0
    assert row["execution_label_end_date_3d"] == dates[3]
    assert pd.isna(result.iloc[1]["execution_excess_return_3d"])


def test_target_resolution_prefers_execution_t20_only_when_complete() -> None:
    execution = target_spec(20)
    assert execution.execution is True
    assert execution.target_col == "execution_outperform_20d"

    frame = pd.DataFrame(columns=[
        execution.target_col,
        execution.return_col,
        execution.label_end_col,
    ])
    assert resolve_target_spec(frame, 20).execution is True
    assert resolve_target_spec(pd.DataFrame(), 20).execution is False
