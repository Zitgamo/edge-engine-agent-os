from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config
from src.data.collector import OHLCVCollector
from src.data.universe import filter_quality, get_ticker_universe
from src.data.validator import DataValidator
from src.features.returns import ReturnFeatures
from src.features.rs import RelativeStrength
from src.features.volatility import ATR
from src.features.volume import VolumeSurge
from src.labels.outperformance import OutperformanceLabel
from src.model.inference import ModelInference

log = logging.getLogger(__name__)

HORIZONS = [1, 3, 5, 10, 15, 20, 30, 40, 60]


def prepare_data() -> pd.DataFrame:
    config = Config()
    collector = OHLCVCollector(config)
    validator = DataValidator()
    bm = collector.fetch("VNINDEX", days=365)

    universe = get_ticker_universe()
    all_dfs: list[pd.DataFrame] = []
    for ticker in universe:
        df = collector.fetch(ticker, days=365)
        df = filter_quality(df, ticker)
        if df is None:
            continue
        errors = validator.validate(df)
        if errors:
            log.warning("Skipping %s: %s", ticker, errors)
            continue
        all_dfs.append(df)

    rf = ReturnFeatures()
    rs = RelativeStrength()
    atr = ATR()
    vs = VolumeSurge()

    feature_dfs = []
    for df in all_dfs:
        d = df.sort_values("date").copy()
        d = rf.compute(d)
        d = rs.compute(d, bm)
        d = atr.compute(d)
        d = vs.compute(d)
        feature_dfs.append(d)
    features = pd.concat(feature_dfs, ignore_index=True)

    label_parts = []
    for df in all_dfs:
        labelled = df.copy()
        for h in HORIZONS:
            labelled = OutperformanceLabel().compute(labelled, bm, horizon=h)
        label_parts.append(
            labelled[
                ["date", "ticker"]
                + [f"excess_return_{h}d" for h in HORIZONS]
                + [f"outperform_{h}d" for h in HORIZONS]
            ]
        )

    labels = pd.concat(label_parts, ignore_index=True)
    data = features.merge(labels, on=["date", "ticker"], how="left")
    return data


def run_simulation() -> None:
    df = pd.read_parquet("data/processed/features.parquet")
    exist_er = [c for c in df.columns if c.startswith("excess_return_")]
    if not exist_er:
        log.info("Preparing multi-horizon data...")
        df = prepare_data()
        df.to_parquet("data/processed/features.parquet")

    if "score" not in df.columns:
        log.info("Running inference...")
        inf = ModelInference(Config())
        inf.load()
        df = inf.predict(df)

    available = sorted(set(int(c.split("_")[-1].replace("d", "")) for c in df.columns if c.startswith("excess_return_")))
    all_results: list[dict] = []

    for n in (list(range(1, 11)) + [15, 20]):
        daily_rets: dict[int, list[float]] = {h: [] for h in available}

        for d in sorted(df["date"].unique()):
            day = df[df["date"] == d]
            picks = day.sort_values("score", ascending=False).head(n)
            if picks.empty:
                continue
            for h in available:
                vals = picks[f"excess_return_{h}d"].dropna()
                if len(vals) > 0:
                    daily_rets[h].append(vals.mean())

        for h in available:
            rets = pd.Series(daily_rets[h])
            if len(rets) < 10:
                continue
            win_rate = (rets > 0).mean()
            avg_ret = rets.mean()
            cum_ret = (1 + rets).prod() - 1
            sharpe = (avg_ret / rets.std() * np.sqrt(252 / h)) if rets.std() > 0 else 0.0
            peak = (1 + rets).cummax()
            dd = ((1 + rets).cumprod() - peak) / peak
            max_dd = dd.min()

            all_results.append({
                "n": n, "horizon": h, "days": len(rets),
                "win_rate": win_rate, "avg_return": avg_ret,
                "cumulative_return": cum_ret,
                "sharpe": sharpe, "max_dd": max_dd,
            })

    res = pd.DataFrame(all_results)

    print(f"{'N':>3} | {'T+n':>5} | {'Days':>5} | {'WinRate':>8} | {'AvgRet':>10} | {'CumRet':>12} | {'Sharpe':>7} | {'MaxDD':>7}")
    print("=" * 80)
    for n in sorted(res["n"].unique()):
        nrows = res[res["n"] == n].sort_values("horizon")
        for _, r in nrows.iterrows():
            print(f"{r['n']:>3} | T+{r['horizon']:<3} | {r['days']:>5} | {r['win_rate']:>7.1%} | {r['avg_return']:>+9.2%} | {r['cumulative_return']:>+11.2%} | {r['sharpe']:>+6.2f} | {r['max_dd']:>6.2%}")
        print()

    print("=" * 70)
    print("OPTIMAL HOLDING PERIOD (by Sharpe per portfolio size)")
    print("=" * 70)
    for n in sorted(res["n"].unique()):
        best = res[res["n"] == n].sort_values("sharpe", ascending=False).iloc[0]
        print(f"N={n:>2}: T+{best['horizon']:>2} | Sharpe {best['sharpe']:>+6.2f} | Win {best['win_rate']:>6.1%} | Avg {best['avg_return']:>+7.2%} | Cum {best['cumulative_return']:>+8.2%}")

    print()
    print("=" * 70)
    print("OPTIMAL HOLDING PERIOD (by AvgReturn per portfolio size)")
    print("=" * 70)
    for n in sorted(res["n"].unique()):
        best = res[res["n"] == n].sort_values("avg_return", ascending=False).iloc[0]
        print(f"N={n:>2}: T+{best['horizon']:>2} | Avg {best['avg_return']:>+7.2%} | Sharpe {best['sharpe']:>+6.2f} | Win {best['win_rate']:>6.1%} | Cum {best['cumulative_return']:>+8.2%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_simulation()
