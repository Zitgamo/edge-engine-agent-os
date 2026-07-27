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
        print("  backfill          Backfill actual T+5 performance")
        print("  summary           Show performance summary")
        print("  signal            Show latest signal")
        return

    cmd = sys.argv[1]

    if cmd == "pipeline":
        from src.pipeline import run_pipeline
        run_pipeline()

    elif cmd == "backfill":
        from src.database import backfill_actuals
        count = backfill_actuals()
        log.info("Backfilled %d actuals", count)

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
        sharpe = (avg_ret / actuals["actual_excess_return_5d"].std() * np.sqrt(252 / 5)) if actuals["actual_excess_return_5d"].std() > 0 else 0.0
        best = actuals.loc[actuals["actual_excess_return_5d"].idxmax(), "ticker"] if total > 0 else "N/A"
        worst = actuals.loc[actuals["actual_excess_return_5d"].idxmin(), "ticker"] if total > 0 else "N/A"
        dates = actuals["signal_date"].nunique()
        print(f"\n{'='*55}")
        print(f"  PERFORMANCE SUMMARY (from {total} actuals across {dates} trading days)")
        print(f"{'='*55}")
        print(f"  Win Rate:      {win_rate:>7.1%}   ({actuals['actual_outperform'].sum():.0f}/{total} wins)")
        print(f"  Avg Excess Ret:{avg_ret:>+7.2%}   per T+5 trade")
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

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
