from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.bottom_to_now import (
    _find_recent_bottom,
    run_bottom_to_now_analysis,
)


def _write_prices(path, dates, close_values, *, benchmark=False) -> None:
    close = np.asarray(close_values, dtype=float)
    frame = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close + (5.0 if not benchmark else 10.0),
        "low": close - (1.0 if not benchmark else 2.0),
        "close": close,
        "volume": [100_000] * len(close),
    })
    frame.to_parquet(path, index=False)


def test_rolling_low_fallback_uses_the_lowest_value_not_the_last_value() -> None:
    prices = pd.DataFrame({
        "low": [100.0, 90.0, 95.0, 96.0, 97.0, 98.0, 99.0],
        "high": [101.0] * 7,
    })

    result = _find_recent_bottom(
        prices,
        lookback_bars=7,
        pivot_left_bars=1,
        pivot_right_bars=1,
        minimum_rebound=0.50,
    )

    assert result == {"index": 1, "method": "rolling_low_fallback"}


def test_bottom_to_now_report_flags_fixed_stop_recovery_and_keeps_pending_rows(
    tmp_path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dates = pd.bdate_range("2026-01-01", periods=80)
    close = np.linspace(120.0, 100.0, len(dates))
    close[60:66] = [100.0, 101.0, 103.0, 105.0, 107.0, 109.0]
    close[66] = 108.0
    close[67] = 110.0
    close[68] = 116.0
    close[69:] = 120.0
    _write_prices(raw_dir / "AAA_raw.parquet", dates, close)
    _write_prices(raw_dir / "VNINDEX_raw.parquet", dates, np.linspace(1000, 1100, len(dates)), benchmark=True)

    result = run_bottom_to_now_analysis(
        research_dir=tmp_path,
        universe=["AAA"],
        fixed_stop_loss=-0.005,
        baseline_atr_multiple=2.0,
        baseline_take_profit=0.10,
        lookback_bars=80,
        pivot_left_bars=2,
        pivot_right_bars=2,
        minimum_rebound=0.05,
        as_of=dates[-1] + pd.Timedelta(hours=16),
        save=False,
    )

    row = result["report"].iloc[0]
    assert row["ticker"] == "AAA"
    assert row["analysis_status"] == "analyzed"
    assert row["bottom_method"] == "confirmed_swing_low"
    assert row["bottom_date"] == dates[60].date().isoformat()
    assert bool(row["tp10_hit"]) is True
    assert bool(row["fixed_sl_then_tp10"]) is True
    assert result["summary"]["counts"]["tickers_entry_ready"] == 1


def test_bottom_to_now_summary_separates_pending_entry_from_exit_sample(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dates = pd.bdate_range("2026-01-01", periods=70)
    close = np.full(len(dates), 100.0)
    close[-6:] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    _write_prices(raw_dir / "AAA_raw.parquet", dates, close)
    _write_prices(raw_dir / "VNINDEX_raw.parquet", dates, np.linspace(1000, 1050, len(dates)), benchmark=True)

    result = run_bottom_to_now_analysis(
        research_dir=tmp_path,
        universe=["AAA"],
        lookback_bars=70,
        pivot_left_bars=2,
        pivot_right_bars=5,
        minimum_rebound=0.05,
        as_of=dates[-1] + pd.Timedelta(hours=16),
        save=False,
    )

    row = result["report"].iloc[0]
    assert row["analysis_status"] == "pending_entry"
    assert result["summary"]["counts"]["tickers_analyzed"] == 1
    assert result["summary"]["counts"]["tickers_pending_entry"] == 1
    assert result["summary"]["counts"]["tp10_hit"] == 0
