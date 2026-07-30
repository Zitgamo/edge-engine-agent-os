from __future__ import annotations

import logging
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
    yf_ticker = f"{ticker}.VN"
    end_str = end or datetime.now().strftime("%Y-%m-%d")
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
    for t in tickers:
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
