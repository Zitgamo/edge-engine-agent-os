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
