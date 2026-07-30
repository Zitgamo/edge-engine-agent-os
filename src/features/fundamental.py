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


class HistoricalFundamentalFeatures:
    """Compute historical PE/PB/ROE from yfinance quarterly financials + balance sheet."""

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}

    def _get_quarterly(self, ticker: str, attr: str) -> pd.DataFrame:
        t = yf.Ticker(f"{ticker}.VN")
        df = getattr(t, attr, None)
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    def fetch_quarterly_fundamentals(self, ticker: str) -> pd.DataFrame:
        fin = self._get_quarterly(ticker, "quarterly_financials")
        bs = self._get_quarterly(ticker, "quarterly_balance_sheet")
        if fin.empty and bs.empty:
            return pd.DataFrame()

        # Annual data as fallback for older periods
        afin = self._get_quarterly(ticker, "financials")
        abs_ = self._get_quarterly(ticker, "balance_sheet")

        rows = []
        # Collect all unique quarter dates
        all_dates = set()
        for df_ in [fin, bs, afin, abs_]:
            if not df_.empty:
                all_dates.update(df_.columns)

        # Helper to get value from a dataframe at a date
        def _val(df_, date_, idx):
            if df_.empty:
                return None
            if date_ not in df_.columns:
                return None
            if idx not in df_.index:
                return None
            v = df_.loc[idx, date_]
            return float(v) if pd.notna(v) else None

        for d in sorted(all_dates):
            ni = _val(fin, d, "Net Income From Continuing Operation Net Minority Interest") or \
                 _val(afin, d, "Net Income From Continuing Operation Net Minority Interest")
            rev = _val(fin, d, "Total Revenue") or _val(fin, d, "Operating Revenue") or \
                  _val(afin, d, "Total Revenue")
            equity = _val(bs, d, "Common Stock Equity") or _val(abs_, d, "Common Stock Equity")
            shares = _val(bs, d, "Ordinary Shares Number") or _val(abs_, d, "Ordinary Shares Number")
            debt = _val(bs, d, "Total Debt") or \
                   (_val(bs, d, "Long Term Debt") or 0) + (_val(bs, d, "Short Long Term Debt") or 0)

            if ni is not None and shares is not None and shares > 0:
                eps = ni / shares
            else:
                eps = None
            if equity is not None and shares is not None and shares > 0:
                bvps = equity / shares
            else:
                bvps = None
            roe_val = (ni / equity) if ni is not None and equity is not None and equity > 0 else None
            de_val = (debt / equity) if debt is not None and equity is not None and equity > 0 else None

            rows.append({
                "date": d,
                "ticker": ticker,
                "net_income": ni,
                "revenue": rev,
                "equity": equity,
                "shares": shares,
                "eps": eps,
                "bvps": bvps,
                "roe": roe_val,
                "debt_equity": de_val,
            })

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def add_historical_to_df(self, df: pd.DataFrame, price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add historical PE/PB/ROE to daily price dataframe."""
        tickers = df["ticker"].unique()
        funda_by_ticker: dict[str, pd.DataFrame] = {}

        for t in tickers:
            funda = self.fetch_quarterly_fundamentals(t)
            if funda.empty:
                continue
            prices = price_data.get(t)
            if prices is None or prices.empty:
                continue
            # Merge fundamentals with price data by forward-filling quarterly values
            prices_dates = prices[["date", "close"]].copy()
            prices_dates["date"] = pd.to_datetime(prices_dates["date"])
            funda["date"] = pd.to_datetime(funda["date"])

            merged = pd.merge_asof(
                prices_dates.sort_values("date"),
                funda.sort_values("date"),
                on="date", direction="backward",
            )

            # Compute PE and PB
            merged["pe_ratio"] = merged.apply(
                lambda r: r["close"] / r["eps"] if r["eps"] is not None and r["eps"] > 0 else None, axis=1
            )
            merged["pb_ratio"] = merged.apply(
                lambda r: r["close"] / r["bvps"] if r["bvps"] is not None and r["bvps"] > 0 else None, axis=1
            )
            # Percentile over available history
            for col in ["pe_ratio", "pb_ratio", "roe", "debt_equity"]:
                if col in merged.columns:
                    valid = merged[col].notna()
                    if valid.any():
                        merged[f"{col}_pct"] = None
                        merged.loc[valid, f"{col}_pct"] = merged.loc[valid, col].rank(pct=True)

            funda_by_ticker[t] = merged[["date", "ticker", "pe_ratio", "pb_ratio", "roe",
                                          "debt_equity", "pe_ratio_pct", "pb_ratio_pct",
                                          "roe_pct", "debt_equity_pct"]]

        if not funda_by_ticker:
            return df

        funda_all = pd.concat(funda_by_ticker.values(), ignore_index=True)
        funda_all["date"] = pd.to_datetime(funda_all["date"])
        df["date"] = pd.to_datetime(df["date"])

        result = df.merge(funda_all, on=["date", "ticker"], how="left")

        # Fill missing with current snapshot fallback
        ff = FundamentalFeatures()
        for t in tickers:
            mask = (result["ticker"] == t)
            current = ff.compute_ticker_features(t)
            for col in ["pe_ratio", "pb_ratio", "roe", "debt_equity"]:
                col_mask = mask & result[col].isna()
                if col_mask.any():
                    result.loc[col_mask, col] = current.get(col, 0.0)
                if result[col].isna().all():
                    result[col] = current.get(col, 0.0)

        for col in ["pe_ratio", "pb_ratio"]:
            if col in result.columns:
                pct_col = f"{col}_pct"
                if pct_col not in result.columns:
                    result[pct_col] = result.groupby("ticker")[col].rank(pct=True)

        return result


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
