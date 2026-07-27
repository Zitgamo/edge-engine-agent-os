from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

from src.config import Config
from src.data.universe import VN30_TICKERS

log = logging.getLogger(__name__)


class OHLCVCollector:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def fetch(self, ticker: str, days: int = 365) -> pd.DataFrame:
        source = self.config.data_source
        if source == "mock":
            return self._mock_data(ticker, days)
        if source == "yfinance":
            return self._fetch_yfinance(ticker, days)
        return self._mock_data(ticker, days)

    def _normalize_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df["date"] = df["date"].dt.normalize()
        return df

    def _fetch_yfinance(self, ticker: str, days: int = 365) -> pd.DataFrame:
        if ticker == "VNINDEX":
            return self._build_vnindex(days)
        yf_ticker = f"{ticker}.VN"
        log.info("Fetching %s from Yahoo Finance (%d days)", yf_ticker, days)
        t = yf.Ticker(yf_ticker)
        hist = t.history(period=f"{days}d")
        if hist.empty:
            log.warning("No data for %s, falling back to mock", yf_ticker)
            return self._mock_data(ticker, days)
        df = hist.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df = self._normalize_date(df)
        df["ticker"] = ticker
        df["volume"] = df["volume"].fillna(0).astype(int)
        return df[["ticker", "date", "open", "high", "low", "close", "volume"]].sort_values("date")

    def _build_vnindex(self, days: int = 365) -> pd.DataFrame:
        log.info("Building VNINDEX composite from VN30 constituents")
        prices: dict[str, pd.Series] = {}
        for t in VN30_TICKERS:
            try:
                df = self._fetch_yfinance(t, days)
                if not df.empty:
                    prices[t] = df.set_index("date")["close"]
            except (ValueError, OSError) as e:
                log.warning("Failed to fetch %s for VNINDEX: %s", t, e)
        if not prices:
            log.warning("No VN30 data for VNINDEX, using mock")
            return self._mock_data("VNINDEX", days)
        normalized = pd.DataFrame({t: s / s.iloc[0] * 1000 for t, s in prices.items()})
        composite = normalized.mean(axis=1).ffill().reset_index()
        composite.columns = ["date", "close"]
        composite = self._normalize_date(composite)
        composite["ticker"] = "VNINDEX"
        composite["open"] = composite["close"]
        composite["high"] = composite["close"]
        composite["low"] = composite["close"]
        composite["volume"] = 0
        return composite[["ticker", "date", "open", "high", "low", "close", "volume"]].sort_values("date")

    def _mock_data(self, ticker: str, days: int = 365) -> pd.DataFrame:
        log.info("Generating mock data for %s (%d days)", ticker, days)
        end = datetime.now(UTC)
        dates = pd.date_range(end=end, periods=days, freq="B")
        n = len(dates)
        rng = __import__("numpy").random.default_rng(hash(ticker) % (2**31))
        base = 20_000 + hash(ticker) % 200_000
        drift = 0.0002
        volatility = 0.015
        returns = drift + volatility * rng.normal(size=n)
        price = base * (1 + returns).cumprod()
        df = pd.DataFrame({
            "ticker": ticker,
            "date": dates,
            "open": price * (1 + 0.002 * rng.normal(size=n)),
            "high": price * (1 + 0.01 * abs(rng.normal(size=n))),
            "low": price * (1 - 0.01 * abs(rng.normal(size=n))),
            "close": price,
            "volume": (1_000_000 + (hash(ticker) % 5_000_000) + (rng.integers(0, 100, size=n) * 10_000)),
        })
        df = self._normalize_date(df)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(2)
        return df.sort_values("date")
