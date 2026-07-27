from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

FUNDA_KEYS = [
    "trailingPE", "priceToBook", "returnOnEquity", "revenueGrowth",
    "earningsQuarterlyGrowth", "profitMargins", "debtToEquity",
    "dividendYield", "marketCap", "forwardPE",
]

FUNDA_FEATURE_COLS = [
    "pe_ratio", "pb_ratio", "roe", "rev_growth", "earn_growth",
    "profit_margin", "debt_equity", "div_yield", "log_mcap", "forward_pe",
]


class FundamentalFeatures:
    def __init__(self) -> None:
        self._cache: dict[str, dict[str, float | None]] = {}

    def fetch_info(self, ticker: str) -> dict[str, float | None]:
        if ticker in self._cache:
            return self._cache[ticker]
        try:
            time.sleep(0.1)
            yf_ticker = f"{ticker}.VN"
            info = yf.Ticker(yf_ticker).info
            result: dict[str, float | None] = {}
            for k in FUNDA_KEYS:
                v = info.get(k)
                result[k] = float(v) if v is not None and v != "" and v != "N/A" else None
            self._cache[ticker] = result
            log.debug("Fetched fundamental data for %s", ticker)
            return result
        except Exception as e:
            log.warning("Failed to fetch fundamental data for %s: %s", ticker, e)
            result = {k: None for k in FUNDA_KEYS}
            self._cache[ticker] = result
            return result

    def compute_ticker_features(self, ticker: str) -> dict[str, float]:
        info = self.fetch_info(ticker)
        features: dict[str, float] = {}

        pe = info.get("trailingPE")
        features["pe_ratio"] = pe if pe and 0 < pe < 200 else 15.0

        pb = info.get("priceToBook")
        features["pb_ratio"] = pb if pb and pb > 0 else 2.0

        roe = info.get("returnOnEquity")
        features["roe"] = roe if roe is not None else 0.12

        rg = info.get("revenueGrowth")
        features["rev_growth"] = rg if rg is not None else 0.0

        eg = info.get("earningsQuarterlyGrowth")
        features["earn_growth"] = eg if eg is not None else 0.0

        pm = info.get("profitMargins")
        features["profit_margin"] = pm if pm is not None else 0.1

        de = info.get("debtToEquity")
        features["debt_equity"] = de if de is not None and de >= 0 else 1.0

        dy = info.get("dividendYield")
        features["div_yield"] = dy if dy is not None else 0.0

        mc = info.get("marketCap")
        features["log_mcap"] = np.log(mc) if mc and mc > 0 else 25.0

        fpe = info.get("forwardPE")
        features["forward_pe"] = fpe if fpe and 0 < fpe < 200 else pe if pe and 0 < pe < 200 else 15.0

        return features


def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    ff = FundamentalFeatures()
    tickers = df["ticker"].unique()
    funda_map: dict[str, dict[str, float]] = {}
    for t in tickers:
        funda_map[t] = ff.compute_ticker_features(t)

    rows = []
    for _, row in df.iterrows():
        r = row.to_dict()
        funda = funda_map.get(row["ticker"], {})
        for col in FUNDA_FEATURE_COLS:
            r[col] = funda.get(col, 0.0)
        rows.append(r)

    result = pd.DataFrame(rows)
    for col in FUNDA_FEATURE_COLS:
        median_val = result[col].median()
        result[col] = result[col].fillna(median_val)

    log.info("Added fundamental features: %s", FUNDA_FEATURE_COLS)
    return result
