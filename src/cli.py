from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _configure_console_encoding() -> None:
    """Keep Vietnamese CLI help usable on Windows code pages."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            continue


def _merge_history_frames(
    local: pd.DataFrame,
    cloud: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    limit: int,
) -> pd.DataFrame:
    """Merge local and cloud history without losing older cloud-only rows."""
    frames: list[pd.DataFrame] = []
    for source_priority, frame in ((1, local), (0, cloud)):
        if frame is None or frame.empty or any(
            column not in frame.columns for column in key_columns
        ):
            continue
        current = frame.copy()
        if "signal_date" in current.columns:
            current["signal_date"] = pd.to_datetime(
                current["signal_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        if "ticker" in current.columns:
            current["ticker"] = current["ticker"].astype(str)
        current["__source_priority"] = source_priority
        realized_columns = [
            column
            for column in (
                "execution_excess_return",
                "actual_excess_return_20d",
                "actual_excess_return",
                "actual_excess_return_5d",
                "actual_outperform",
            )
            if column in current.columns
        ]
        current["__completeness"] = (
            current[realized_columns].notna().sum(axis=1)
            if realized_columns
            else 0
        )
        frames.append(current)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["__completeness", "__source_priority"],
        ascending=[False, False],
        kind="stable",
    )
    combined = combined.drop_duplicates(list(key_columns), keep="first")
    sort_columns = [column for column in ("signal_date", "rank") if column in combined.columns]
    if sort_columns:
        if "rank" in combined.columns:
            combined["rank"] = pd.to_numeric(combined["rank"], errors="coerce")
        combined = combined.sort_values(
            sort_columns,
            ascending=[False, True][: len(sort_columns)],
            kind="stable",
        )
    return combined.drop(columns=["__source_priority", "__completeness"], errors="ignore").head(limit)


def _load_signal_history(limit: int) -> pd.DataFrame:
    """Load the union of local and Supabase signals for CLI history views."""
    from src.database import get_signals

    local = get_signals(limit=limit)
    cloud = pd.DataFrame()
    try:
        from src.supabase_client import get_client

        client = get_client()
        if client is not None:
            cloud = pd.DataFrame(client.get_signals(limit=limit))
    except Exception as exc:  # pragma: no cover - network/config dependent
        log.warning("Could not load signal history from Supabase: %s", exc)
    return _merge_history_frames(
        local,
        cloud,
        key_columns=("signal_date", "ticker"),
        limit=limit,
    )


def _load_actual_history(limit: int = 500) -> pd.DataFrame:
    """Load the union of local and cloud actuals for CLI summaries."""
    from src.database import get_conn, init_db

    init_db()
    conn = get_conn()
    local = pd.read_sql_query(
        """SELECT a.signal_date, a.ticker,
                  COALESCE(a.actual_excess_return_20d, a.actual_excess_return_5d)
                  AS execution_excess_return,
                  a.actual_excess_return_5d, a.actual_excess_return_20d,
                  a.actual_stock_return, a.benchmark_return,
                  a.actual_outperform,
                  s.score, s.ensemble_score, s.rank
           FROM actuals a
           LEFT JOIN signals s ON a.signal_date = s.signal_date AND a.ticker = s.ticker
           ORDER BY a.signal_date DESC
           LIMIT ?""",
        conn,
        params=(limit,),
    )
    conn.close()

    cloud = pd.DataFrame()
    try:
        from src.supabase_client import get_client

        client = get_client()
        if client is not None:
            cloud = pd.DataFrame(client.get_actuals(limit=limit))
    except Exception as exc:  # pragma: no cover - network/config dependent
        log.warning("Could not load performance history from Supabase: %s", exc)

    return _merge_history_frames(
        local,
        cloud,
        key_columns=("signal_date", "ticker"),
        limit=limit,
    )


def main() -> None:
    _configure_console_encoding()
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli <command>")
        print("Commands:")
        print("  pipeline          Run full pipeline")
        print("  pipeline-force    Rebuild/train/publish latest closed session")
        print("  backfill          Backfill actual T+20 performance")
        print("  summary           Show performance summary")
        print("  signal            Show latest signal")
        print("  backtest-score    Walk-forward: top 3 vs bottom 3")
        print("  history           Show full history + tracker")
        print("  ceiling           Ceiling/floor context analysis for tickers")
        print("  telegram-test     Test Telegram notification")
        print("  sync-cloud        Sync local SQLite data to Supabase")
        print("  accumulation      Tích sản backtest (use: accumulation HPG or accumulation all)")
        print("  strategy-attribution  Show de-duplicated realized strategy attribution")
        print("  paper-test        Show cost-aware independent basket readiness")
        return

    cmd = sys.argv[1]

    if cmd == "pipeline":
        from src.pipeline import run_pipeline
        run_pipeline()

    elif cmd == "pipeline-force":
        from src.pipeline import run_pipeline
        run_pipeline(force=True)

    elif cmd == "backfill":
        from src.database import backfill_actuals
        count = backfill_actuals(holding_period=20)
        log.info("Backfilled %d actuals", count)

    elif cmd == "strategies":
        from src.strategies import StrategyManager
        sm = StrategyManager()
        sm.backfill_strategy_actuals()
        sm.show_comparison()
        weights = sm.get_strategy_weights()
        print("\nEnsemble weights (no hard switch):")
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {name:<20} weight={w:.2f}")

    elif cmd == "strategy-attribution":
        from src.research.strategy_attribution import load_realized_strategy_attribution

        attribution = load_realized_strategy_attribution(prefer_cloud=True)
        if attribution.empty:
            print("No realized strategy trades yet.")
        else:
            print(attribution.to_string(index=False))

    elif cmd == "paper-test":
        from src.config import Config
        from src.research.paper_test import load_paper_test_readiness

        config = Config()
        report = load_paper_test_readiness(prefer_cloud=True)
        print(
            f"Paper-test cost={config.round_trip_cost:.2%}; "
            "a basket requires 3 executable picks; minimum=30, target=50 baskets"
        )
        if report.empty:
            print("No strategy signals or complete executable baskets yet.")
        else:
            print(report.to_string(index=False))

    elif cmd == "summary":
        actuals = _load_actual_history()
        if actuals.empty:
            print("No performance data yet. Run 'pipeline' first (backfill is automatic now).")
            return
        from src.actuals import add_execution_excess_column
        actuals = add_execution_excess_column(actuals)
        actuals["execution_excess_return"] = pd.to_numeric(
            actuals["execution_excess_return"], errors="coerce"
        ).astype(float)
        actuals = actuals[actuals["execution_excess_return"].notna()].copy()
        if actuals.empty:
            print("No realized performance data yet. Run 'pipeline' first.")
            return
        total = len(actuals)
        win_rate = actuals["actual_outperform"].mean()
        avg_ret = actuals["execution_excess_return"].mean()
        sharpe = (avg_ret / actuals["execution_excess_return"].std() * np.sqrt(252 / 20)) if actuals["execution_excess_return"].std() > 0 else 0.0
        best = actuals.loc[actuals["execution_excess_return"].idxmax(), "ticker"] if total > 0 else "N/A"
        worst = actuals.loc[actuals["execution_excess_return"].idxmin(), "ticker"] if total > 0 else "N/A"
        dates = actuals["signal_date"].nunique()
        print(f"\n{'='*55}")
        print(f"  PERFORMANCE SUMMARY (from {total} actuals across {dates} trading days)")
        print(f"{'='*55}")
        print(f"  Win Rate:      {win_rate:>7.1%}   ({actuals['actual_outperform'].sum():.0f}/{total} wins)")
        print(f"  Avg Excess Ret:{avg_ret:>+7.2%}   per T+20 trade")
        if "actual_stock_return" in actuals.columns:
            stock_return = pd.to_numeric(actuals["actual_stock_return"], errors="coerce").dropna()
            if not stock_return.empty:
                absolute_wr = float((stock_return > 0).mean())
                avg_stock_return = float(stock_return.mean())
                wins = stock_return[stock_return > 0]
                losses = stock_return[stock_return < 0]
                realized_rr = (
                    float(wins.mean() / abs(losses.mean()))
                    if not wins.empty and not losses.empty and losses.mean() != 0
                    else float("nan")
                )
                print(f"  Absolute ROI:  {avg_stock_return:>+7.2%}   avg net stock return")
                print(f"  Absolute WR:   {absolute_wr:>7.1%}   profitable stock trades")
                print(f"  Realized R:R:  {realized_rr:>7.2f}   avg win / avg loss")
        if "benchmark_return" in actuals.columns:
            benchmark_return = pd.to_numeric(actuals["benchmark_return"], errors="coerce").dropna()
            if not benchmark_return.empty:
                print(f"  Avg VNINDEX:   {benchmark_return.mean():>+7.2%}   during held trades")
        print(f"  Sharpe (ann):  {sharpe:>+7.2f}")
        print(f"  Best pick:     {best:>6}   ({actuals.loc[actuals['execution_excess_return'].idxmax(), 'execution_excess_return']:>+7.2%})" if total > 0 else "")
        print(f"  Worst pick:    {worst:>6}   ({actuals.loc[actuals['execution_excess_return'].idxmin(), 'execution_excess_return']:>+7.2%})" if total > 0 else "")
        print()
        print("  Top 5 by avg excess return:")
        top5 = actuals.groupby("ticker")["execution_excess_return"].agg(["mean", "count", "sum"]).sort_values("mean", ascending=False).head(5)
        for t, r in top5.iterrows():
            print(f"    {t:<6} avg {r['mean']:>+7.2%}  ({r['count']:.0f} signals)")
        print()

    elif cmd == "signal":
        df = _load_signal_history(limit=10)
        if df.empty:
            print("No signals yet. Run 'pipeline' first.")
        else:
            print(df.to_string(index=False))

    elif cmd == "history":
        days = 30
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            days = int(sys.argv[2])
        df = _load_signal_history(limit=days * 10)
        if df.empty:
            print("No history yet. Run 'pipeline' first.")
            return
        from src.actuals import add_execution_excess_column
        df = add_execution_excess_column(df)

        df = df.sort_values(["signal_date", "rank"])
        dates = df["signal_date"].nunique()
        total = len(df)

        print(f"\n{'='*70}")
        print(f"  SIGNAL HISTORY (last {dates} trading day(s), {total} picks)")
        print(f"{'='*70}")
        for d in sorted(df["signal_date"].unique(), reverse=True):
            day = df[df["signal_date"] == d]
            realized = day["execution_excess_return"].notna().any()
            print(f"\n  {d} {'(actuals available)' if realized else '(pending...)'}")
            print(f"  {'Rank':<4} {'Ticker':<6} {'Score':<8} {'ExRet':<9} {'Outperform':<10}")
            print(f"  {'-'*37}")
            for _, r in day.iterrows():
                er = f"{r['execution_excess_return']:>+7.2%}" if pd.notna(r.get("execution_excess_return")) else "     N/A"
                op = "WIN " if r.get("actual_outperform") == 1 else ("LOSS" if r.get("actual_outperform") == 0 else "N/A  ")
                print(f"  {int(r['rank']):<4} {r['ticker']:<6} {r['score']:<8.4f} {er:<9} {op:<10}")

        print(f"\n{'='*70}")
        print("  TRACKER SUMMARY")
        print(f"{'='*70}")
        realized = df[df["execution_excess_return"].notna()]
        if not realized.empty:
            wr = realized["actual_outperform"].mean()
            avg_ret = realized["execution_excess_return"].mean()
            best = realized.loc[realized["execution_excess_return"].idxmax()]
            worst = realized.loc[realized["execution_excess_return"].idxmin()]
            print(f"  Overall Win Rate:  {wr:>7.1%} ({realized['actual_outperform'].sum():.0f}/{len(realized)} picks)")
            print(f"  Avg Excess Return: {avg_ret:>+7.2%}")
            print(f"  Best pick:         {best['ticker']} on {best['signal_date']} ({best['execution_excess_return']:>+7.2%})")
            print(f"  Worst pick:        {worst['ticker']} on {worst['signal_date']} ({worst['execution_excess_return']:>+7.2%})")

            print("\n  Top tickers by avg excess return:")
            top = realized.groupby("ticker").agg(
                avg_ret=("execution_excess_return", "mean"),
                count=("actual_outperform", "count"),
                wins=("actual_outperform", "sum"),
            ).sort_values("avg_ret", ascending=False).head(10)
            print(f"  {'Ticker':<6} {'Signals':<8} {'Wins':<6} {'WinRate':<9} {'AvgRet':<9}")
            print(f"  {'-'*38}")
            for t, r in top.iterrows():
                print(f"  {t:<6} {r['count']:<8.0f} {r['wins']:<6.0f} {r['wins']/r['count']:>7.1%}  {r['avg_ret']:>+7.2%}")
        else:
            print("  No realized actuals yet. Run pipeline daily to accumulate history.")

    elif cmd == "ceiling":
        from src.filters.ceiling_context import report_ceiling_context
        tickers = sys.argv[2:] if len(sys.argv) > 2 else ["PNJ", "VIC", "FRT", "VHM", "SJS", "DIG"]
        results = report_ceiling_context(tickers)
        if not results:
            print("No results — check ticker symbols")
            return
        print(f"\n{'='*75}")
        print("  CEILING/FLOOR CONTEXT ANALYSIS")
        print(f"{'='*75}")
        print(f"  {'Ticker':<6} {'Context':<20} {'Drawdown':<10} {'Floors':<8} {'Ceilings':<10} {'Adj':<6}")
        print(f"  {'-'*60}")
        for r in results:
            ctx = r.get("context_label", "?")
            dd = f"{r['drawdown_60d']*100:.1f}%"
            fl = f"{r['consecutive_floors']} ({r['recent_floor_count']})"
            cl = f"{r['consecutive_ceilings']} ({r['recent_ceiling_count']})"
            adj = f"{r['score_adjustment']:+.3f}"
            print(f"  {r['ticker']:<6} {ctx:<20} {dd:<10} {fl:<8} {cl:<10} {adj:<6}")
        print()

    elif cmd == "telegram-test":
        from src.notification.telegram import send_message
        ok = send_message("Edge Engine: Telegram test OK \u2705")
        print("Telegram test:", "sent" if ok else "failed (not configured)")
        if not ok:
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")

    elif cmd == "backtest-score":
        from src.backtest import backtest_score_validation
        backtest_score_validation()

    elif cmd == "accumulation":
        from src.accumulation import backtest_multi, backtest_tich_san, print_report
        if len(sys.argv) > 2 and sys.argv[2] == "all":
            df = backtest_multi(top_n=10)
            print("\nTop 10 by CAGR:")
            print(df.head(10).to_string(index=False))
        else:
            ticker = sys.argv[2] if len(sys.argv) > 2 else "HPG"
            result = backtest_tich_san(ticker)
            print_report(result)

    elif cmd == "sync-cloud":
        from src.supabase_client import sync_all
        result = sync_all()
        if result is None:
            print("Supabase not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in .env")
        else:
            total = sum(result.values())
            print(f"Synced {total} rows to Supabase:")
            for table, n in result.items():
                if n > 0:
                    print(f"  {table}: {n}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
