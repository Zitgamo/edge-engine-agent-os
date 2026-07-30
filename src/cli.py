from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli <command>")
        print("Commands:")
        print("  pipeline          Run full pipeline")
        print("  backfill          Backfill actual T+20 performance")
        print("  summary           Show performance summary")
        print("  signal            Show latest signal")
        print("  backtest-score    Walk-forward: top 3 vs bottom 3")
        print("  history           Show full history + tracker")
        print("  ceiling           Ceiling/floor context analysis for tickers")
        print("  telegram-test     Test Telegram notification")
        print("  sync-cloud        Sync local SQLite data to Supabase")
        return

    cmd = sys.argv[1]

    if cmd == "pipeline":
        from src.pipeline import run_pipeline
        run_pipeline()

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
        print(f"\nEnsemble weights (no hard switch):")
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {name:<20} weight={w:.2f}")

    elif cmd == "summary":
        from src.database import get_conn
        conn = get_conn()
        actuals = pd.read_sql_query(
            """SELECT a.signal_date, a.ticker, a.actual_excess_return_5d, a.actual_outperform,
                      s.score, s.ensemble_score, s.rank
               FROM actuals a
               LEFT JOIN signals s ON a.signal_date = s.signal_date AND a.ticker = s.ticker
               ORDER BY a.signal_date DESC""",
            conn,
        )
        conn.close()
        if actuals.empty:
            print("No performance data yet. Run 'pipeline' first (backfill is automatic now).")
            return
        total = len(actuals)
        win_rate = actuals["actual_outperform"].mean()
        avg_ret = actuals["actual_excess_return_5d"].mean()
        sharpe = (avg_ret / actuals["actual_excess_return_5d"].std() * np.sqrt(252 / 20)) if actuals["actual_excess_return_5d"].std() > 0 else 0.0
        best = actuals.loc[actuals["actual_excess_return_5d"].idxmax(), "ticker"] if total > 0 else "N/A"
        worst = actuals.loc[actuals["actual_excess_return_5d"].idxmin(), "ticker"] if total > 0 else "N/A"
        dates = actuals["signal_date"].nunique()
        print(f"\n{'='*55}")
        print(f"  PERFORMANCE SUMMARY (from {total} actuals across {dates} trading days)")
        print(f"{'='*55}")
        print(f"  Win Rate:      {win_rate:>7.1%}   ({actuals['actual_outperform'].sum():.0f}/{total} wins)")
        print(f"  Avg Excess Ret:{avg_ret:>+7.2%}   per T+20 trade")
        print(f"  Sharpe (ann):  {sharpe:>+7.2f}")
        print(f"  Best pick:     {best:>6}   ({actuals.loc[actuals['actual_excess_return_5d'].idxmax(), 'actual_excess_return_5d']:>+7.2%})" if total > 0 else "")
        print(f"  Worst pick:    {worst:>6}   ({actuals.loc[actuals['actual_excess_return_5d'].idxmin(), 'actual_excess_return_5d']:>+7.2%})" if total > 0 else "")
        print()
        print(f"  Top 5 by avg excess return:")
        top5 = actuals.groupby("ticker")["actual_excess_return_5d"].agg(["mean", "count", "sum"]).sort_values("mean", ascending=False).head(5)
        for t, r in top5.iterrows():
            print(f"    {t:<6} avg {r['mean']:>+7.2%}  ({r['count']:.0f} signals)")
        print()

    elif cmd == "signal":
        from src.database import get_signals
        df = get_signals(limit=10)
        if df.empty:
            print("No signals yet. Run 'pipeline' first.")
        else:
            print(df.to_string(index=False))

    elif cmd == "history":
        days = 30
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            days = int(sys.argv[2])
        from src.database import get_signals
        df = get_signals(limit=days * 10)
        if df.empty:
            print("No history yet. Run 'pipeline' first.")
            return

        df = df.sort_values(["signal_date", "rank"])
        dates = df["signal_date"].nunique()
        total = len(df)

        print(f"\n{'='*70}")
        print(f"  SIGNAL HISTORY (last {dates} trading day(s), {total} picks)")
        print(f"{'='*70}")
        for d in sorted(df["signal_date"].unique(), reverse=True):
            day = df[df["signal_date"] == d]
            realized = day["actual_excess_return_5d"].notna().any()
            print(f"\n  {d} {'(actuals available)' if realized else '(pending...)'}")
            print(f"  {'Rank':<4} {'Ticker':<6} {'Score':<8} {'ExRet':<9} {'Outperform':<10}")
            print(f"  {'-'*37}")
            for _, r in day.iterrows():
                er = f"{r['actual_excess_return_5d']:>+7.2%}" if pd.notna(r.get("actual_excess_return_5d")) else "     N/A"
                op = "WIN " if r.get("actual_outperform") == 1 else ("LOSS" if r.get("actual_outperform") == 0 else "N/A  ")
                print(f"  {int(r['rank']):<4} {r['ticker']:<6} {r['score']:<8.4f} {er:<9} {op:<10}")

        print(f"\n{'='*70}")
        print(f"  TRACKER SUMMARY")
        print(f"{'='*70}")
        realized = df[df["actual_excess_return_5d"].notna()]
        if not realized.empty:
            wr = realized["actual_outperform"].mean()
            avg_ret = realized["actual_excess_return_5d"].mean()
            best = realized.loc[realized["actual_excess_return_5d"].idxmax()]
            worst = realized.loc[realized["actual_excess_return_5d"].idxmin()]
            print(f"  Overall Win Rate:  {wr:>7.1%} ({realized['actual_outperform'].sum():.0f}/{len(realized)} picks)")
            print(f"  Avg Excess Return: {avg_ret:>+7.2%}")
            print(f"  Best pick:         {best['ticker']} on {best['signal_date']} ({best['actual_excess_return_5d']:>+7.2%})")
            print(f"  Worst pick:        {worst['ticker']} on {worst['signal_date']} ({worst['actual_excess_return_5d']:>+7.2%})")

            print(f"\n  Top tickers by avg excess return:")
            top = realized.groupby("ticker").agg(
                avg_ret=("actual_excess_return_5d", "mean"),
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
        print(f"  CEILING/FLOOR CONTEXT ANALYSIS")
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
