from __future__ import annotations

import logging
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.universe import VN30_TICKERS
from src.data.collector import OHLCVCollector
from src.config import Config

log = logging.getLogger(__name__)

INVESTMENT_DEFAULTS = {
    "monthly_amount": 10_000_000,
    "frequency": "monthly",
    "start_date": "2020-01-01",
    "end_date": None,
}


def _fetch_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    # Try cached collector first to avoid yfinance rate limits
    try:
        collector = OHLCVCollector(Config())
        days = (pd.Timestamp.now() - pd.Timestamp(start)).days + 60
        df = collector.fetch(ticker, days=max(365, days))
        if df is not None and not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
            col_map = {"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"}
            for old_c, new_c in col_map.items():
                if old_c in df.columns and new_c not in df.columns:
                    df = df.rename(columns={old_c: new_c})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            if end:
                df = df[df["date"] <= pd.to_datetime(end)]
            df = df[df["date"] >= pd.to_datetime(start)]
            if not df.empty and "close" in df.columns:
                return df[["date", "open", "high", "low", "close", "volume"]]
    except Exception:
        pass

    yf_ticker = f"{ticker}.VN"
    end_str = end or datetime.now().strftime("%Y-%m-%d")
    time.sleep(1)  # rate limit prevention
    t = yf.Ticker(yf_ticker)
    hist = t.history(start=start, end=end_str)
    if hist.empty:
        return pd.DataFrame()
    df = hist.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def simulate_dca(
    prices: pd.DataFrame,
    monthly_amount: float = 10_000_000,
    frequency: str = "monthly",
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    prices = prices.copy().sort_values("date")
    offset_map = {"monthly": 30, "quarterly": 91, "yearly": 365}
    days_offset = offset_map.get(frequency, 30)

    first_date = prices["date"].min()
    last_date = prices["date"].max()
    invest_dates = pd.date_range(start=first_date, end=last_date, freq=f"{days_offset}D")
    invest_dates = [d.date() for d in invest_dates]

    rows = []
    total_shares = 0.0
    total_invested = 0.0
    cash_buffer = 0.0

    for i, row in prices.iterrows():
        d = row["date"]
        if isinstance(d, pd.Timestamp):
            d = d.date()
        price = row["close"]
        if pd.isna(price) or price <= 0:
            continue

        if d in invest_dates:
            invest = monthly_amount + cash_buffer
            shares_bought = invest / price
            total_shares += shares_bought
            total_invested += monthly_amount
            cash_buffer = invest - shares_bought * price

        portfolio_value = total_shares * price
        rows.append({
            "date": d,
            "price": price,
            "total_shares": total_shares,
            "total_invested": total_invested,
            "portfolio_value": portfolio_value,
            "cash_buffer": cash_buffer,
            "pnl": portfolio_value - total_invested,
            "pnl_pct": (portfolio_value - total_invested) / total_invested if total_invested > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def _load_fundamentals(ticker: str) -> pd.DataFrame:
    feat_path = Config.processed_data_dir / "features.parquet"
    if not feat_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(feat_path)
    ticker_col = "ticker"
    if ticker_col not in df.columns:
        return pd.DataFrame()
    tk = df[df[ticker_col] == ticker].copy()
    if tk.empty:
        return pd.DataFrame()
    tk = tk.sort_values("date")[["date", "pe_ratio", "pb_ratio", "roe", "profit_margin", "log_mcap"]].copy()
    for col in ["pe_ratio", "pb_ratio"]:
        if col in tk.columns:
            pct = tk[col].rank(pct=True)
            tk[f"{col}_pct"] = pct
    return tk


def _compute_sma_rsi(prices: pd.DataFrame, sma_period: int = 200, rsi_period: int = 14) -> pd.DataFrame:
    df = prices.copy().sort_values("date")
    df["sma200"] = df["close"].rolling(sma_period).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["above_sma200"] = (df["close"] > df["sma200"]).astype(float)
    return df


_METHODS = {
    "fixed": "DCA deu dan",
    "value": "DCA dinh gia (PE/PB)",
    "cycle": "DCA chu ky (SMA+RSI)",
}


def simulate_value_dca(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    monthly_amount: float = 10_000_000,
    frequency: str = "monthly",
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    prices = prices.copy().sort_values("date")
    offset_map = {"monthly": 30, "quarterly": 91, "yearly": 365}
    days_offset = offset_map.get(frequency, 30)
    first_date = prices["date"].min()
    last_date = prices["date"].max()
    invest_dates = pd.date_range(start=first_date, end=last_date, freq=f"{days_offset}D")
    invest_dates_set = {d.date() for d in invest_dates}

    fund_map = {}
    if not fundamentals.empty:
        for _, r in fundamentals.iterrows():
            d = r["date"]
            if isinstance(d, pd.Timestamp):
                d = d.date()
            fund_map[d] = r

    rows = []
    total_shares = 0.0
    total_invested = 0.0
    cash_buffer = 0.0

    for i, row in prices.iterrows():
        d = row["date"]
        if isinstance(d, pd.Timestamp):
            d = d.date()
        price = row["close"]
        if pd.isna(price) or price <= 0:
            continue

        if d in invest_dates_set:
            multiplier = 1.0
            if d in fund_map:
                f = fund_map[d]
                pe_pct = f.get("pe_ratio_pct", np.nan) if "pe_ratio_pct" in f.index or "pe_ratio_pct" in fundamentals.columns else np.nan
                pb_pct = f.get("pb_ratio_pct", np.nan) if "pb_ratio_pct" in f.index or "pb_ratio_pct" in fundamentals.columns else np.nan
                if pd.notna(pe_pct):
                    if pe_pct < 0.2:
                        multiplier = 1.5
                    elif pe_pct < 0.4:
                        multiplier = 1.25
                    elif pe_pct > 0.7:
                        multiplier = 0.5
                    elif pe_pct > 0.9:
                        multiplier = 0.0
                if pd.notna(pb_pct) and multiplier > 0:
                    if pb_pct < 0.2:
                        multiplier *= 1.3
                    elif pb_pct > 0.8:
                        multiplier *= 0.6

            invest = monthly_amount * multiplier + cash_buffer
            if invest <= 0:
                cash_buffer = -invest
                invest = 0
            if invest > 0:
                shares_bought = invest / price
                total_shares += shares_bought
                total_invested += monthly_amount * multiplier
                cash_buffer = invest - shares_bought * price

        portfolio_value = total_shares * price
        rows.append({
            "date": d, "price": price, "total_shares": total_shares,
            "total_invested": total_invested, "portfolio_value": portfolio_value,
            "cash_buffer": cash_buffer,
            "pnl": portfolio_value - total_invested,
            "pnl_pct": (portfolio_value - total_invested) / total_invested if total_invested > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def simulate_cycle_dca(
    prices: pd.DataFrame,
    monthly_amount: float = 10_000_000,
    frequency: str = "monthly",
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    df = _compute_sma_rsi(prices)
    offset_map = {"monthly": 30, "quarterly": 91, "yearly": 365}
    days_offset = offset_map.get(frequency, 30)
    first_date = df["date"].min()
    last_date = df["date"].max()
    invest_dates = pd.date_range(start=first_date, end=last_date, freq=f"{days_offset}D")
    invest_dates_set = {d.date() for d in invest_dates}

    rows = []
    total_shares = 0.0
    total_invested = 0.0
    cash_buffer = 0.0

    for i, row in df.iterrows():
        d = row["date"]
        if isinstance(d, pd.Timestamp):
            d = d.date()
        price = row["close"]
        if pd.isna(price) or price <= 0:
            continue

        if d in invest_dates_set:
            multiplier = 1.0
            rsi = row.get("rsi", np.nan)
            above_sma = row.get("above_sma200", np.nan)

            if pd.notna(rsi):
                if rsi < 25:
                    multiplier = 2.0
                elif rsi < 35:
                    multiplier = 1.5
                elif rsi > 70:
                    multiplier = 0.0
                elif rsi > 60:
                    multiplier = 0.5

            if pd.notna(above_sma) and multiplier > 0:
                if above_sma < 0.5:
                    multiplier *= 1.2

            invest = monthly_amount * multiplier + cash_buffer
            if invest <= 0:
                cash_buffer = -invest
                invest = 0
            if invest > 0:
                shares_bought = invest / price
                total_shares += shares_bought
                total_invested += monthly_amount * multiplier
                cash_buffer = invest - shares_bought * price

        portfolio_value = total_shares * price
        rows.append({
            "date": d, "price": price, "total_shares": total_shares,
            "total_invested": total_invested, "portfolio_value": portfolio_value,
            "cash_buffer": cash_buffer,
            "pnl": portfolio_value - total_invested,
            "pnl_pct": (portfolio_value - total_invested) / total_invested if total_invested > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def backtest_compare_methods(
    ticker: str,
    monthly_amount: float = INVESTMENT_DEFAULTS["monthly_amount"],
    frequency: str = INVESTMENT_DEFAULTS["frequency"],
    start_date: str = INVESTMENT_DEFAULTS["start_date"],
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    log.info("Backtesting methods for %s", ticker)
    prices = _fetch_prices(ticker, start_date, end_date)
    if prices.empty:
        return pd.DataFrame(), {}

    fundamentals = _load_fundamentals(ticker)
    methods = {
        "fixed": simulate_dca(prices, monthly_amount, frequency),
        "cycle": simulate_cycle_dca(prices, monthly_amount, frequency),
    }
    value_dca = simulate_value_dca(prices, fundamentals, monthly_amount, frequency)
    if not value_dca.empty:
        methods["value"] = value_dca

    results = []
    histories = {}
    for name, dca in methods.items():
        if dca.empty:
            continue
        m = compute_metrics(dca["portfolio_value"], dca["total_invested"].iloc[-1], dca["portfolio_value"].iloc[-1])
        histories[name] = dca
        results.append({
            "method": name,
            "label": _METHODS.get(name, name),
            "total_invested": m["total_invested"],
            "final_value": m["final_value"],
            "total_return": m["total_return"],
            "cagr": m["cagr"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_drawdown"],
        })

    return pd.DataFrame(results).sort_values("cagr", ascending=False), histories


def backtest_compare_all(
    tickers: list[str] | None = None,
    monthly_amount: float = INVESTMENT_DEFAULTS["monthly_amount"],
    start_date: str = INVESTMENT_DEFAULTS["start_date"],
    end_date: str | None = None,
) -> pd.DataFrame:
    if tickers is None:
        tickers = VN30_TICKERS
    rows = []
    for t in tickers:
        df, _ = backtest_compare_methods(t, monthly_amount, "monthly", start_date, end_date)
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "ticker": t,
                "method": r["method"],
                "label": r["label"],
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "max_dd": r["max_dd"],
                "final_value": r["final_value"],
                "total_invested": r["total_invested"],
            })

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        return all_df

    summary = all_df.pivot_table(index="ticker", columns="method", values="cagr").reset_index()
    summary.columns.name = None
    if "fixed" in summary and "cycle" in summary:
        summary["diff_cycle"] = summary["cycle"] - summary["fixed"]
    if "value" in summary and "fixed" in summary:
        summary["diff_value"] = summary["value"] - summary["fixed"]
    summary = summary.sort_values("diff_cycle", ascending=False) if "diff_cycle" in summary.columns else summary

    return summary


def compute_metrics(series: pd.Series, total_invested: float, final_value: float) -> dict:
    if len(series) < 2 or total_invested <= 0:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    total_return = (final_value - total_invested) / total_invested
    daily_returns = series.pct_change().dropna()
    years = len(series) / 252
    cagr = (final_value / total_invested) ** (1 / years) - 1 if years > 0 else 0.0

    rf_annual = 0.05
    rf_daily = rf_annual / 252
    excess = daily_returns - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    peak = series.cummax()
    dd = (series - peak) / peak
    max_dd = dd.min()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_value": final_value,
        "total_invested": total_invested,
        "years": round(years, 1),
    }


def backtest_tich_san(
    ticker: str,
    monthly_amount: float = INVESTMENT_DEFAULTS["monthly_amount"],
    frequency: str = INVESTMENT_DEFAULTS["frequency"],
    start_date: str = INVESTMENT_DEFAULTS["start_date"],
    end_date: str | None = None,
) -> dict:
    log.info("Backtesting %s: %s %.0fVND/month from %s", ticker, frequency, monthly_amount, start_date)
    prices = _fetch_prices(ticker, start_date, end_date)
    if prices.empty:
        log.warning("No price data for %s", ticker)
        return {"ticker": ticker, "error": "No data", "metrics": {}}

    try:
        collector = OHLCVCollector(Config())
        req_days = int((pd.Timestamp.now() - pd.Timestamp(start_date)).days * 1.2) + 60
        vnindex_raw = collector.fetch("VNINDEX", days=max(365, req_days))
        if not vnindex_raw.empty:
            vnindex_raw = vnindex_raw.sort_values("date").reset_index(drop=True)
            mask = vnindex_raw["date"] >= pd.to_datetime(start_date)
            if end_date:
                mask = mask & (vnindex_raw["date"] <= pd.to_datetime(end_date))
            vnindex_raw = vnindex_raw[mask].reset_index(drop=True)
            vnindex_prices = vnindex_raw[["date", "close"]].copy()
        else:
            vnindex_prices = _fetch_prices("VNINDEX", start_date, end_date)
    except Exception as e:
        log.warning("VNINDEX fetch failed: %s", e)
        vnindex_prices = _fetch_prices("VNINDEX", start_date, end_date)

    dca = simulate_dca(prices, monthly_amount, frequency)
    if dca.empty:
        return {"ticker": ticker, "error": "DCA simulation failed", "metrics": {}}

    vnindex_dca = simulate_dca(vnindex_prices, monthly_amount, frequency) if not vnindex_prices.empty else pd.DataFrame()

    metrics = compute_metrics(dca["portfolio_value"], dca["total_invested"].iloc[-1], dca["portfolio_value"].iloc[-1])

    active_return = 0.0
    if not vnindex_dca.empty:
        vn_metrics = compute_metrics(vnindex_dca["portfolio_value"], vnindex_dca["total_invested"].iloc[-1], vnindex_dca["portfolio_value"].iloc[-1])
        active_return = metrics["total_return"] - vn_metrics["total_return"]
        metrics["vnindex_return"] = vn_metrics["total_return"]
        metrics["vnindex_cagr"] = vn_metrics["cagr"]
        metrics["active_return"] = active_return

    result = {
        "ticker": ticker,
        "start_date": dca["date"].min(),
        "end_date": dca["date"].max(),
        "first_price": prices["close"].iloc[0],
        "last_price": prices["close"].iloc[-1],
        "price_change": (prices["close"].iloc[-1] - prices["close"].iloc[0]) / prices["close"].iloc[0],
        "metrics": metrics,
        "dca_history": dca,
        "prices": prices,
    }

    log.info("  Result: invested=%.0f, final=%.0f, return=%+.2f%%, CAGR=%+.2f%%, Sharpe=%.2f, DD=%+.2f%%",
             metrics["total_invested"], metrics["final_value"],
             metrics["total_return"] * 100, metrics["cagr"] * 100,
             metrics["sharpe"], metrics["max_drawdown"] * 100)
    if "active_return" in metrics:
        log.info("  Vs VNINDEX: active return=%+.2f%%", metrics["active_return"] * 100)

    return result


def backtest_multi(
    tickers: list[str] | None = None,
    monthly_amount: float = INVESTMENT_DEFAULTS["monthly_amount"],
    frequency: str = INVESTMENT_DEFAULTS["frequency"],
    start_date: str = INVESTMENT_DEFAULTS["start_date"],
    end_date: str | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    if tickers is None:
        tickers = VN30_TICKERS

    results = []
    for idx, t in enumerate(tickers):
        if idx > 0:
            time.sleep(0.5)
        res = backtest_tich_san(t, monthly_amount, frequency, start_date, end_date)
        if "error" not in res:
            m = res["metrics"]
            results.append({
                "ticker": t,
                "total_return": m.get("total_return", 0),
                "cagr": m.get("cagr", 0),
                "sharpe": m.get("sharpe", 0),
                "max_dd": m.get("max_drawdown", 0),
                "final_value": m.get("final_value", 0),
                "total_invested": m.get("total_invested", 0),
                "years": m.get("years", 0),
                "active_return": m.get("active_return", 0),
            })
        else:
            results.append({"ticker": t, "total_return": 0, "cagr": 0, "sharpe": 0, "max_dd": 0, "final_value": 0, "total_invested": 0, "years": 0, "active_return": 0})

    df = pd.DataFrame(results).sort_values("cagr", ascending=False)
    log.info("\n=== Tích sản Backtest: Top %d ===", top_n)
    log.info("%-6s %10s %8s %8s %8s %12s %8s", "Ticker", "Return", "CAGR", "Sharpe", "MaxDD", "FinalVal", "Active")
    log.info("-" * 70)
    for _, r in df.head(top_n).iterrows():
        log.info("%-6s %+9.2f%% %+7.2f%% %7.2f %7.2f%% %11.0f %+7.2f%%",
                 r["ticker"], r["total_return"] * 100, r["cagr"] * 100, r["sharpe"], r["max_dd"] * 100, r["final_value"], r["active_return"] * 100)

    return df


def backtest_compare_frequencies(
    ticker: str,
    total_per_year: float = 120_000_000,
    frequencies: list[str] | None = None,
    start_date: str = INVESTMENT_DEFAULTS["start_date"],
    end_date: str | None = None,
) -> pd.DataFrame:
    if frequencies is None:
        frequencies = ["monthly", "quarterly", "yearly"]

    freq_mult = {"monthly": 1 / 12, "quarterly": 1 / 4, "yearly": 1.0}

    results = []
    histories = {}
    prices = _fetch_prices(ticker, start_date, end_date)
    if prices.empty:
        return pd.DataFrame(), {}

    for freq in frequencies:
        per_amount = total_per_year * freq_mult.get(freq, 1 / 12)
        dca = simulate_dca(prices, per_amount, freq)
        if dca.empty:
            continue
        m = compute_metrics(dca["portfolio_value"], dca["total_invested"].iloc[-1], dca["portfolio_value"].iloc[-1])
        histories[freq] = dca
        results.append({
            "frequency": freq,
            "total_invested": m["total_invested"],
            "final_value": m["final_value"],
            "total_return": m["total_return"],
            "cagr": m["cagr"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_drawdown"],
            "price_change": (prices["close"].iloc[-1] - prices["close"].iloc[0]) / prices["close"].iloc[0],
        })

    return pd.DataFrame(results), histories


def compute_ranking_history(
    features: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compute AccumulationStrategy ranking for every historical date."""
    from src.strategies.accumulation import AccumulationStrategy
    strat = AccumulationStrategy()
    dates = sorted(features["date"].unique())
    rows = []
    for d in dates:
        day_df = features[features["date"] == d].copy()
        try:
            ranking = strat.rank(day_df)
            if ranking.empty:
                continue
            ranking = ranking.reset_index(drop=True)
            for _, r in ranking.iterrows():
                rnk = int(r["rank"])
                rows.append({
                    "date": d,
                    "ticker": r["ticker"],
                    "rank": rnk,
                    "score": r["score"],
                    "in_top_3": rnk <= 3,
                    "in_top_5": rnk <= 5,
                    "in_top_10": rnk <= 10,
                })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    return df.sort_values(["date", "rank"]).reset_index(drop=True)


def simulate_dca_portfolio(
    ranking_history: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    monthly_amount: float = 10_000_000,
    top_n: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Dynamic DCA: each month invest in top N ranked stocks, track P&L."""
    if ranking_history.empty:
        return pd.DataFrame()

    rh = ranking_history.copy()
    rh["date"] = pd.to_datetime(rh["date"])
    if start_date:
        rh = rh[rh["date"] >= pd.to_datetime(start_date)]
    if end_date:
        rh = rh[rh["date"] <= pd.to_datetime(end_date)]
    if rh.empty:
        return pd.DataFrame()

    invest_dates = sorted(rh["date"].unique())
    portfolio: dict[str, dict] = {}  # ticker -> shares, cost, invested
    rows = []

    for d in invest_dates:
        day_rank = rh[rh["date"] == d].sort_values("rank")
        top_tickers = set(day_rank.head(top_n)["ticker"].tolist())

        # Decide actions for current holdings
        for ticker in list(portfolio.keys()):
            h = portfolio[ticker]
            rank_row = day_rank[day_rank["ticker"] == ticker]
            if rank_row.empty:
                signal = "EXIT"
            else:
                r = rank_row.iloc[0]["rank"]
                if r <= top_n:
                    signal = "ACCUMULATE" if r <= 3 else "NORMAL"
                elif r <= top_n * 2:
                    signal = "WATCH"
                else:
                    signal = "EXIT"
            h["signal"] = signal

        # Invest in top N
        invest_amount = monthly_amount / top_n
        for ticker in top_tickers:
            price_data = prices.get(ticker)
            if price_data is None or price_data.empty:
                continue
            price_row = price_data[price_data["date"] == d]
            if price_row.empty:
                continue
            price = price_row.iloc[0]["close"]
            if pd.isna(price) or price <= 0:
                continue

            if ticker not in portfolio:
                portfolio[ticker] = {"shares": 0.0, "cost": 0.0, "invested": 0.0, "signal": "BUY"}

            h = portfolio[ticker]
            mult = 1.5 if h.get("signal") == "ACCUMULATE" else 1.0
            invest = invest_amount * mult
            shares_bought = invest / price
            h["shares"] += shares_bought
            h["invested"] += invest
            h["cost"] = h["invested"] / h["shares"] if h["shares"] > 0 else 0

        # Calculate daily P&L
        total_invested = 0.0
        total_value = 0.0
        for ticker, h in portfolio.items():
            price_data = prices.get(ticker)
            val = 0.0
            if price_data is not None and not price_data.empty:
                pr = price_data[price_data["date"] == d]
                if not pr.empty:
                    px = pr.iloc[0]["close"]
                    if pd.notna(px) and px > 0:
                        val = h["shares"] * px
            total_invested += h["invested"]
            total_value += val
            pnl = val - h["invested"]
            pnl_pct = pnl / h["invested"] if h["invested"] > 0 else 0.0
            rows.append({
                "date": d,
                "ticker": ticker,
                "shares": round(h["shares"], 2),
                "invested": h["invested"],
                "value": val,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "signal": h.get("signal", "HOLD"),
                "cost_basis": round(h["cost"], 0) if h["cost"] > 0 else 0,
            })

    return pd.DataFrame(rows)


def summarize_portfolio(
    portfolio_df: pd.DataFrame,
) -> dict:
    """Aggregate portfolio summary by date."""
    if portfolio_df.empty:
        return {}
    by_date = portfolio_df.groupby("date").agg(
        total_invested=("invested", "sum"),
        total_value=("value", "sum"),
        holdings=("ticker", "nunique"),
    ).reset_index()
    by_date["pnl"] = by_date["total_value"] - by_date["total_invested"]
    by_date["pnl_pct"] = by_date["pnl"] / by_date["total_invested"].replace(0, np.nan)
    last = by_date.iloc[-1]
    first = by_date.iloc[0]
    years = (by_date["date"].max() - by_date["date"].min()).days / 365.25
    total_ret = last["pnl_pct"]
    cagr = (last["total_value"] / last["total_invested"]) ** (1 / years) - 1 if years > 0 and last["total_invested"] > 0 else 0.0
    return {
        "total_invested": last["total_invested"],
        "total_value": last["total_value"],
        "pnl": last["pnl"],
        "pnl_pct": last["pnl_pct"],
        "cagr": cagr,
        "holdings": last["holdings"],
        "years": round(years, 1),
        "history": by_date,
    }


def print_portfolio(portfolio_df: pd.DataFrame) -> None:
    summary = summarize_portfolio(portfolio_df)
    if not summary:
        print("  No portfolio data")
        return

    print(f"\n{'='*60}")
    print(f"  DANH MUC TICH SAN")
    print(f"  Period: {summary['history']['date'].min().date()} -> {summary['history']['date'].max().date()} ({summary['years']} years)")
    print(f"{'='*60}")
    print(f"  Holdings:        {summary['holdings']}")
    print(f"  Total invested:  {summary['total_invested']:>12,.0f} VND")
    print(f"  Portfolio value: {summary['total_value']:>12,.0f} VND")
    print(f"  P&L:             {summary['pnl']:>+12,.0f} VND ({summary['pnl_pct']*100:+.2f}%)")
    print(f"  CAGR:            {summary['cagr']*100:+.2f}%")
    print(f"{'='*60}")

    latest = portfolio_df[portfolio_df["date"] == portfolio_df["date"].max()]
    print(f"\n  {'Ticker':<8} {'Signal':<12} {'Shares':>8} {'Cost':>10} {'Value':>10} {'P&L':>10} {'P&L%':>8}")
    print(f"  {'-'*68}")
    for _, r in latest.iterrows():
        signal = r['signal']
        sig_icon = {'ACCUMULATE': '+', 'NORMAL': '=', 'WATCH': '?', 'EXIT': '!', 'BUY': '+', 'HOLD': '-', 'SELL': 'v'}.get(signal, ' ')
        print(f"  {r['ticker']:<8} {sig_icon} {signal:<10} {r['shares']:>8.1f} {r['cost_basis']:>10,.0f} {r['value']:>10,.0f} {r['pnl']:>+10,.0f} {r['pnl_pct']*100:>+7.2f}%")


def print_report(result: dict) -> None:
    sys_stdout = __import__("sys").stdout
    if hasattr(sys_stdout, "reconfigure"):
        try:
            sys_stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if "error" in result:
        print(f"\n  {result['ticker']}: {result['error']}")
        return

    m = result["metrics"]
    print(f"\n{'='*60}")
    print(f"  TICH SAN REPORT: {result['ticker']}")
    print(f"  Period: {result['start_date']} -> {result['end_date']} ({m.get('years', 0)} years)")
    print(f"{'='*60}")
    print(f"  Monthly investment: {INVESTMENT_DEFAULTS['monthly_amount']:,.0f} VND")
    print(f"  Total invested:    {m['total_invested']:>12,.0f} VND")
    print(f"  Final portfolio:   {m['final_value']:>12,.0f} VND")
    print(f"  P&L:               {m['final_value'] - m['total_invested']:>+12,.0f} VND")
    print(f"  Total return:      {m['total_return']*100:>+11.2f}%")
    print(f"  CAGR:              {m['cagr']*100:>+11.2f}%")
    print(f"  Sharpe ratio:      {m['sharpe']:>11.2f}")
    print(f"  Max drawdown:      {m['max_drawdown']*100:>+11.2f}%")
    print(f"  Price change:      {result['price_change']*100:>+11.2f}%")
    if "vnindex_return" in m:
        print(f"  VNINDEX DCA return: {m['vnindex_return']*100:>+11.2f}%")
        print(f"  Active return:      {m['active_return']*100:>+11.2f}%")
    print(f"{'='*60}")
