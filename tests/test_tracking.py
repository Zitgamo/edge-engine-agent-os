from __future__ import annotations

import pandas as pd

from src.tracking import realtime


def _ohlcv(ticker: str, dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [ticker] * len(dates),
        "date": dates,
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [99.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1_000_000] * len(dates),
    })


def test_refreshes_stale_parquet(monkeypatch, tmp_path) -> None:
    today = pd.Timestamp.today().normalize()
    stale_dates = [today - pd.Timedelta(days=2), today - pd.Timedelta(days=1)]
    fresh_dates = stale_dates + [today]
    _ohlcv("FRT", stale_dates).to_parquet(tmp_path / "FRT_raw.parquet")
    fresh = _ohlcv("FRT", fresh_dates)

    realtime._stock_cache.clear()
    monkeypatch.setattr(realtime, "_fetch_yfinance", lambda ticker: fresh)

    result = realtime.load_stock_data("FRT", str(tmp_path))

    assert result is not None
    assert result["date"].max() == today
    assert len(result) == 3


def test_uses_stale_parquet_when_refresh_fails(monkeypatch, tmp_path) -> None:
    today = pd.Timestamp.today().normalize()
    stale_dates = [today - pd.Timedelta(days=2), today - pd.Timedelta(days=1)]
    _ohlcv("VCB", stale_dates).to_parquet(tmp_path / "VCB_raw.parquet")

    realtime._stock_cache.clear()
    monkeypatch.setattr(realtime, "_fetch_yfinance", lambda ticker: None)

    result = realtime.load_stock_data("VCB", str(tmp_path))

    assert result is not None
    assert result["date"].max() == stale_dates[-1]


def test_summary_averages_overlapping_cohorts_instead_of_compounding() -> None:
    result = realtime.get_signal_summary([
        {"signal_date": "2026-07-01", "status": "HIT_TP", "pnl": 1.0, "weight": 1.0},
        {"signal_date": "2026-07-02", "status": "HIT_SL", "pnl": -0.5, "weight": 1.0},
    ])

    assert result["portfolio_pnl"] == 0.25
    assert result["total_pnl"] == 0.25


def test_tracker_uses_next_open_and_respects_settlement_delay() -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    df = _ohlcv("AAA", list(dates))
    df["open"] = [90.0, 100.0, 100.0, 100.0, 100.0]
    df["close"] = [90.0, 100.0, 100.0, 100.0, 100.0]
    df.loc[2, "low"] = 90.0  # T+2: must still be ignored
    df.loc[3, "low"] = 90.0  # T+3: SL is executable

    result = realtime.simulate_holding(
        df,
        signal_date="2026-01-01",
        entry_price=100.0,
        stop_loss=-0.05,
        take_profit=0.08,
        holding_period=4,
        settlement_delay=2,
    )

    assert result["status"] == "HIT_SL"
    assert result["exit_date"] == "2026-01-06"


def test_track_signal_infers_next_session_open(monkeypatch) -> None:
    dates = pd.bdate_range("2026-01-01", periods=3)
    df = _ohlcv("AAA", list(dates))
    df["open"] = [90.0, 101.0, 102.0]
    monkeypatch.setattr(realtime, "load_stock_data", lambda ticker, data_dir=None: df)

    result = realtime.track_signal(
        "AAA",
        "2026-01-01",
        stop_loss=0.0,
        take_profit=0.0,
        holding_period=1,
    )

    assert result["entry_price"] == 101.0
