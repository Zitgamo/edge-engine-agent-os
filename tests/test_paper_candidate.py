from __future__ import annotations

import pandas as pd

from src.research.paper_candidate import build_candidate_signals
from src.research.ticker_exit_optimizer import PROFILE_SCHEMA_VERSION


def _write_synthetic_history(path, ticker: str, dates: pd.DatetimeIndex, slope: float) -> None:
    close = 100.0 + slope * pd.Series(range(len(dates)), dtype="float64")
    frame = pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": close,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": 1_000_000,
    })
    frame.to_parquet(path / f"{ticker}_raw.parquet", index=False)


def test_candidate_uses_breadth_rs_and_atr_stop(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=100)
    _write_synthetic_history(tmp_path, "VNINDEX", dates, 0.5)
    for index, ticker in enumerate((
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "KDH", "MBB", "MSN", "MWG", "NVL", "PNJ", "POW", "SAB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    )):
        _write_synthetic_history(tmp_path, ticker, dates, 0.6 + index * 0.02)

    signals, report = build_candidate_signals(
        tmp_path,
        signal_date=dates[-1],
        min_breadth=0.60,
        top_n=3,
        atr_multiple=2.0,
        take_profit=0.10,
    )

    assert report["status"] == "passed"
    assert report["market_breadth_20d"] == 1.0
    assert len(signals) == 3
    assert signals["rank"].tolist() == [1, 2, 3]
    assert (signals["stop_loss"] < 0).all()
    assert (signals["take_profit"] == 0.10).all()
    assert set(signals["action"]) == {"PAPER_BUY"}


def test_candidate_applies_only_approved_ticker_exit_profile(tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=100)
    _write_synthetic_history(tmp_path, "VNINDEX", dates, 0.5)
    tickers = (
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "KDH", "MBB", "MSN", "MWG", "NVL", "PNJ", "POW", "SAB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    )
    for index, ticker in enumerate(tickers):
        _write_synthetic_history(tmp_path, ticker, dates, 0.6 + index * 0.02)

    document = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profiles": {
            ticker: {
                "approved": True,
                "atr_multiple": 2.5,
                "take_profit": 0.08,
                "confidence": 0.7,
            }
            for ticker in tickers
        },
    }
    signals, report = build_candidate_signals(
        tmp_path,
        signal_date=dates[-1],
        min_breadth=0.60,
        top_n=3,
        atr_multiple=2.0,
        take_profit=0.10,
        exit_profiles=document,
    )

    assert report["status"] == "passed"
    assert report["exit_profile_used_count"] == 3
    assert (signals["take_profit"] == 0.08).all()
    assert (signals["exit_profile_used"]).all()
    assert (signals["stop_loss"] < 0).all()
