from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from src.config import Config
from src.data.universe import VN30_TICKERS
from src.time_utils import now_vn

log = logging.getLogger(__name__)


class OHLCVCollector:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.last_benchmark_source: str | None = None
        self.last_invalid_count: int = 0

    def fetch(self, ticker: str, days: int = 365) -> pd.DataFrame:
        self.last_invalid_count = 0
        source = str(self.config.data_source).strip().lower()
        if source == "mock":
            if ticker == "VNINDEX":
                self.last_benchmark_source = "mock"
            return self._mock_data(ticker, days)
        if source == "yfinance":
            return self._fetch_yfinance(ticker, days)
        if source in {"kbs", "kbs_public"}:
            return self._fetch_kbs(ticker, days)
        raise ValueError(
            f"Unsupported DATA_SOURCE={self.config.data_source!r}; "
            "expected 'yfinance', 'kbs', or 'mock'"
        )

    def _normalize_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df["date"] = df["date"].dt.normalize()
        return df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    @staticmethod
    def _history_with_retry(ticker: yf.Ticker, days: int, attempts: int = 3) -> pd.DataFrame:
        """Fetch history with bounded retries for transient Yahoo failures."""
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return ticker.history(period=f"{days}d", auto_adjust=False)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Yahoo Finance request failed after {attempts} attempts") from last_error

    def _fetch_yfinance(self, ticker: str, days: int = 365) -> pd.DataFrame:
        if ticker == "VNINDEX":
            return self._build_vnindex(days)
        yf_ticker = f"{ticker}.VN"
        log.info("Fetching %s from Yahoo Finance (%d days)", yf_ticker, days)
        t = yf.Ticker(yf_ticker)
        hist = self._history_with_retry(t, days)
        if hist.empty:
            log.warning("No data for %s, returning empty (delisted)", yf_ticker)
            return pd.DataFrame()
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

    def _fetch_kbs(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """Fetch daily OHLCV from KBS's public end-of-day endpoint."""
        end_date = now_vn().date()
        start_date = end_date - timedelta(days=max(1, int(days)))
        base_url = str(self.config.kbs_base_url).rstrip("/")
        ticker = str(ticker).upper().strip()
        if ticker == "VNINDEX":
            resource = f"index/{ticker}"
            self.last_benchmark_source = "kbs"
        else:
            resource = f"stocks/{ticker}"
        url = (
            f"{base_url}/{resource}/data_day"
            f"?sdate={start_date:%d-%m-%Y}&edate={end_date:%d-%m-%Y}"
        )
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.kbsec.com.vn/",
            "User-Agent": "Mozilla/5.0",
        }
        log.info("Fetching %s from KBS (%d days)", ticker, days)
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=float(self.config.kbs_timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"KBS request failed for {ticker}: {exc}") from exc

        rows = payload.get("data_day", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list) or not rows:
            log.warning("No KBS data for %s", ticker)
            return pd.DataFrame()

        frame = pd.DataFrame(rows).rename(columns={
            "t": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise RuntimeError(f"KBS response for {ticker} is missing columns: {missing}")

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = (
            frame["date"].isna()
            | frame[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
            | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | (frame["volume"] < 0)
            | (frame["high"] < frame["low"])
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
        )
        invalid_count = int(invalid.sum())
        self.last_invalid_count = invalid_count
        if invalid_count:
            log.warning("Dropping %d invalid KBS OHLCV rows for %s", invalid_count, ticker)
            frame = frame.loc[~invalid].copy()
        if frame.empty:
            return pd.DataFrame()

        frame["ticker"] = ticker
        frame = self._normalize_date(frame)
        frame["volume"] = frame["volume"].round().astype("int64")
        return frame[["ticker", "date", "open", "high", "low", "close", "volume"]]

    def _fetch_vnstock_index(self, days: int = 365) -> pd.DataFrame:
        """Fetch VNINDEX from VCI when Yahoo has no historical index series."""
        from vnstock.api.quote import Quote

        end = now_vn().date() + timedelta(days=1)
        start = now_vn().date() - timedelta(days=days)
        hist = Quote(symbol="VNINDEX", source="VCI", show_log=False).history(
            start=start.isoformat(),
            end=end.isoformat(),
        )
        if hist.empty:
            return pd.DataFrame()
        df = hist.reset_index().rename(columns={
            "time": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })
        if "date" not in df.columns:
            return pd.DataFrame()
        df = self._normalize_date(df)
        df["ticker"] = "VNINDEX"
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        return df[["ticker", "date", "open", "high", "low", "close", "volume"]]

    def _build_vnindex(self, days: int = 365) -> pd.DataFrame:
        # Try fetching the real VN-Index (^VNINDEX) from yfinance first
        try:
            log.info("Fetching real VNINDEX (^VNINDEX) from Yahoo Finance")
            t = yf.Ticker("^VNINDEX")
            hist = self._history_with_retry(t, days)
            if len(hist) >= 30:
                df = hist.reset_index().rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
                df = self._normalize_date(df)
                df["ticker"] = "VNINDEX"
                df["volume"] = df["volume"].fillna(0).astype(int)
                log.info("Real VNINDEX fetched: %d rows", len(df))
                self.last_benchmark_source = "yahoo"
                return df[["ticker", "date", "open", "high", "low", "close", "volume"]].sort_values("date")
        except Exception as e:
            log.warning("Real VNINDEX fetch failed: %s — trying VCI", e)

        try:
            log.info("Fetching VNINDEX from VCI via vnstock")
            df = self._fetch_vnstock_index(days)
            if len(df) >= 30:
                self.last_benchmark_source = "vnstock_vci"
                log.info("VCI VNINDEX fetched: %d rows", len(df))
                return df
        except Exception as e:
            log.warning("VCI VNINDEX fetch failed: %s", e)

        # Fallback: synthesize composite from VN30 (equal-weight, price-normalized)
        log.warning("Using SYNTHETIC VNINDEX composite (not the real market-cap-weighted index)")
        self.last_benchmark_source = "synthetic"
        prices: dict[str, pd.Series] = {}
        for tk in VN30_TICKERS:
            try:
                df = self._fetch_yfinance(tk, days)
                if not df.empty:
                    prices[tk] = df.set_index("date")["close"]
            except Exception as e:
                log.warning("Failed to fetch %s for VNINDEX: %s", tk, e)
        if not prices:
            log.warning("No VN30 data for VNINDEX, using mock")
            self.last_benchmark_source = "mock"
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
