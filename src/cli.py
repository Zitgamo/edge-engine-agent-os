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


def _parse_ticker_exit_args(args: list[str], config) -> dict[str, object]:
    """Parse the small, dependency-free option set for exit research."""
    research_dir: str | None = None
    universe = config.ticker_exit_universe
    folds = config.ticker_exit_walk_forward_folds
    consensus = config.ticker_exit_minimum_consensus_folds
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in {"-h", "--help"}:
            return {"help": True}
        if argument.startswith("--universe="):
            universe = argument.split("=", 1)[1]
        elif argument == "--universe":
            index += 1
            if index >= len(args):
                raise ValueError("--universe requires a value")
            universe = args[index]
        elif argument.startswith("--folds="):
            folds = int(argument.split("=", 1)[1])
        elif argument == "--folds":
            index += 1
            if index >= len(args):
                raise ValueError("--folds requires an integer")
            folds = int(args[index])
        elif argument.startswith("--consensus="):
            consensus = int(argument.split("=", 1)[1])
        elif argument == "--consensus":
            index += 1
            if index >= len(args):
                raise ValueError("--consensus requires an integer")
            consensus = int(args[index])
        elif argument.startswith("-"):
            raise ValueError(f"Unknown research-ticker-exits option: {argument}")
        elif research_dir is None:
            research_dir = argument
        else:
            raise ValueError(f"Unexpected research-ticker-exits argument: {argument}")
        index += 1
    return {
        "help": False,
        "research_dir": research_dir or "data/research_kbs_5y",
        "universe": universe,
        "folds": folds,
        "consensus": None if int(consensus) <= 0 else int(consensus),
    }


def _parse_bottom_now_args(args: list[str]) -> dict[str, object]:
    """Parse the dependency-free options for the current bottom diagnostic."""
    research_dir: str | None = None
    values: dict[str, object] = {
        "universe": "all",
        "lookback_bars": 252,
        "pivot_left_bars": 5,
        "pivot_right_bars": 5,
        "minimum_rebound": 0.05,
        "target_overshoot_margin": 0.05,
    }
    option_map = {
        "universe": str,
        "lookback": int,
        "lookback-bars": int,
        "pivot-left": int,
        "pivot-right": int,
        "rebound": float,
        "rebound-pct": float,
        "overshoot": float,
        "overshoot-pct": float,
    }
    key_aliases = {
        "lookback": "lookback_bars",
        "lookback-bars": "lookback_bars",
        "pivot-left": "pivot_left_bars",
        "pivot-right": "pivot_right_bars",
        "rebound": "minimum_rebound",
        "rebound-pct": "minimum_rebound",
        "overshoot": "target_overshoot_margin",
        "overshoot-pct": "target_overshoot_margin",
    }
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in {"-h", "--help"}:
            return {"help": True}
        if not argument.startswith("--"):
            if research_dir is None:
                research_dir = argument
            else:
                raise ValueError(f"Unexpected research-bottom-now argument: {argument}")
            index += 1
            continue
        raw_key, separator, inline_value = argument[2:].partition("=")
        key = raw_key.lower()
        if key not in option_map:
            raise ValueError(f"Unknown research-bottom-now option: {argument}")
        if separator:
            raw_value = inline_value
        else:
            index += 1
            if index >= len(args):
                raise ValueError(f"--{raw_key} requires a value")
            raw_value = args[index]
        values[key_aliases.get(key, key)] = option_map[key](raw_value)
        index += 1
    return {
        "help": False,
        "research_dir": research_dir or "data/research_kbs_5y",
        **values,
    }


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
    if len(sys.argv) < 2 or sys.argv[1].lower() in {"help", "-h", "--help"}:
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
        print("  paper-update      Refresh KBS paper data, write candidate, and backfill T+20")
        print("  research-build [dir]  Build KBS research features and execution outcomes")
        print("  research-refresh [dir]  Refresh KBS research OHLCV cache incrementally")
        print("  research-candidate [dir]  Reproduce candidate walk-forward and locked holdout")
        print("  research-ticker-exits [dir]  Fit per-ticker ATR/TP profiles with rolling validation")
        print("                       options: --universe vn30|all|TICK1,TICK2 --folds N --consensus N")
        print("  research-bottom-now [dir]  Diagnose bottom-to-now TP/SL and hold vs scalp")
        print("                       options: --universe all|vn30|TICK1,TICK2 --lookback N")
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

        attribution = load_realized_strategy_attribution(
            prefer_cloud=True,
            include_version=True,
        )
        if attribution.empty:
            print("No realized strategy trades yet.")
        else:
            print(attribution.to_string(index=False))

    elif cmd == "paper-test":
        from src.config import Config
        from src.research.paper_test import load_paper_test_readiness

        config = Config()
        report = load_paper_test_readiness(
            raw_data_dir=config.raw_data_dir,
            paper_raw_data_dir=config.paper_raw_data_dir,
            min_baskets=config.paper_min_baskets,
            target_baskets=config.paper_target_baskets,
            min_trades=config.paper_min_trades,
            target_trades=config.paper_target_trades,
            prefer_cloud=True,
            include_version=True,
        )
        print(
            f"Paper-test cost={config.round_trip_cost:.2%}; "
            f"a basket requires {config.paper_top_n} executable picks; "
            f"minimum={config.paper_min_trades}, target={config.paper_target_trades} trades"
        )
        if report.empty:
            print("No strategy signals or complete executable baskets yet.")
        else:
            print(report.to_string(index=False))

    elif cmd == "paper-update":
        from src.config import Config
        from src.research.paper_candidate import (
            paper_execution_config,
            run_paper_update,
        )

        config = Config()
        force_full = any(arg.lower() in {"full", "--full"} for arg in sys.argv[2:])
        result = run_paper_update(config, force_full=force_full)
        snapshot = result["snapshot"]
        print(
            f"Paper data refreshed in {result['paper_data_dir']}; "
            f"saved={result['signals_saved']}, backfilled={result['actuals_backfilled']}"
        )
        print(f"Candidate snapshot: {snapshot}")

        try:
            from src.supabase_client import get_client

            client = get_client()
            if client is not None:
                paper_schema_ready = all(
                    client._remote_column_available(
                        "strategy_performance",
                        column,
                    )
                    for column in ("stop_loss", "take_profit")
                )
                if not paper_schema_ready:
                    print(
                        "Cloud paper history skipped: apply supabase_setup.sql "
                        "once before syncing dynamic candidate stops"
                    )
                else:
                    synced = client.sync_strategy_performance()
                    remote_backfilled = client.backfill_remote_strategy_actuals(
                        config=config,
                        paper_config=paper_execution_config(config),
                        strategy_names=config.paper_strategy_names(),
                    )
                    print(
                        f"Cloud paper history: synced={synced}, "
                        f"remote_backfilled={remote_backfilled}"
                    )
        except Exception as exc:
            log.warning("Cloud paper history update failed: %s", exc)
            print(f"Cloud paper history unavailable: {exc}")

    elif cmd == "research-build":
        from src.config import Config
        from src.research.kbs_dataset import build_research_dataset

        research_dir = sys.argv[2] if len(sys.argv) > 2 else "data/research_kbs_10y"
        result = build_research_dataset(
            research_dir,
            round_trip_cost=Config().round_trip_cost,
        )
        outcomes = result["outcomes"]
        print(
            f"Built research dataset through {result['closed_date']}: "
            f"features={len(result['features'])}, "
            f"outcomes="
            f"{', '.join(f'{name}={len(frame)}' for name, frame in outcomes.items())}"
        )

    elif cmd == "research-refresh":
        from src.config import Config
        from src.research.kbs_dataset import refresh_research_data
        from src.research.ticker_exit_optimizer import _resolve_universe

        config = Config()
        force_full = any(arg.lower() in {"full", "--full"} for arg in sys.argv[2:])
        research_args = [arg for arg in sys.argv[2:] if arg.lower() not in {"full", "--full"}]
        parsed = _parse_ticker_exit_args(research_args, config)
        if parsed["help"]:
            print(
                "Usage: python -m src.cli research-refresh [dir] "
                "[--universe all|vn30|TICK1,TICK2] [--full]"
            )
            print("  Existing files are merged; --full requests the full lookback again")
            return
        research_dir = parsed["research_dir"]
        tickers = _resolve_universe(
            str(research_dir) + "/raw",
            str(parsed["universe"]),
        )
        report = refresh_research_data(
            research_dir,
            config=config,
            tickers=tickers,
            lookback_days=config.ticker_exit_research_lookback_days,
            refresh_days=config.ticker_exit_research_refresh_days,
            force_full=force_full,
        )
        status_counts = report["status"].value_counts().to_dict() if not report.empty else {}
        print(
            f"Research cache refreshed in {research_dir}; "
            f"tickers={len(report)}, statuses={status_counts}"
        )

    elif cmd == "research-candidate":
        from src.research.candidate_backtest import run_candidate_research

        research_dir = sys.argv[2] if len(sys.argv) > 2 else "data/research_kbs_5y"
        report = run_candidate_research(research_dir=research_dir)
        if report.empty:
            print("No candidate research rows were produced.")
        else:
            print(report.to_string(index=False))

    elif cmd == "research-ticker-exits":
        from src.config import Config
        from src.research.ticker_exit_optimizer import run_ticker_exit_research

        config = Config()
        parsed = _parse_ticker_exit_args(sys.argv[2:], config)
        if parsed["help"]:
            print(
                "Usage: python -m src.cli research-ticker-exits [dir] "
                "[--universe vn30|all|TICK1,TICK2] [--folds N] [--consensus N]"
            )
            print("  --universe all       Use all tickers represented in <dir>/raw")
            print("  --folds N            Number of expanding walk-forward validation folds")
            print("  --consensus N        Required passing folds; 0 uses automatic two-thirds")
            return
        result = run_ticker_exit_research(
            research_dir=parsed["research_dir"],
            round_trip_cost=config.round_trip_cost,
            baseline_atr_multiple=config.ticker_exit_baseline_atr_multiple,
            baseline_take_profit=config.ticker_exit_baseline_take_profit,
            universe=parsed["universe"],
            walk_forward_folds=int(parsed["folds"]),
            minimum_consensus_folds=parsed["consensus"],
        )
        print(f"Ticker exit profile: {result.get('profile_path', 'not saved')}")
        metadata = result.get("metadata", {})
        print(
            "Walk-forward: "
            f"universe={metadata.get('universe')}, "
            f"folds={metadata.get('walk_forward_folds')}, "
            f"consensus={metadata.get('minimum_consensus_folds')}; "
            f"locked holdout {metadata.get('holdout_start')}..{metadata.get('holdout_end')}"
        )
        print(
            "Deployment recommendation: "
            f"{metadata.get('deployment_recommendation', 'research_only')}"
        )
        table = result.get("profile_table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            print(table.to_string(index=False))
        summary = result.get("holdout_summary")
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            print("\nLocked holdout:")
            print(summary.to_string(index=False))

    elif cmd == "research-bottom-now":
        from src.config import Config
        from src.research.bottom_to_now import run_bottom_to_now_analysis

        config = Config()
        parsed = _parse_bottom_now_args(sys.argv[2:])
        if parsed["help"]:
            print(
                "Usage: python -m src.cli research-bottom-now [dir] "
                "[--universe all|vn30|TICK1,TICK2] [--lookback N]"
            )
            print("  Bottom = latest confirmed swing low in the last N bars")
            print("  Entry reference = first open after the 5-bar confirmation window")
            print("  --rebound and --overshoot use decimal values, e.g. 0.05 = 5%")
            return
        result = run_bottom_to_now_analysis(
            research_dir=parsed["research_dir"],
            universe=parsed["universe"],
            fixed_stop_loss=config.stop_loss,
            baseline_atr_multiple=config.ticker_exit_baseline_atr_multiple,
            baseline_take_profit=config.ticker_exit_baseline_take_profit,
            lookback_bars=int(parsed["lookback_bars"]),
            pivot_left_bars=int(parsed["pivot_left_bars"]),
            pivot_right_bars=int(parsed["pivot_right_bars"]),
            minimum_rebound=float(parsed["minimum_rebound"]),
            target_overshoot_margin=float(parsed["target_overshoot_margin"]),
        )
        summary = result["summary"]
        counts = summary["counts"]
        rates = summary["rates"]
        print(
            f"Bottom-to-now: {result.get('report_path', 'not saved')} | "
            f"latest={summary['metadata']['market_latest_date']} | "
            f"universe={summary['metadata']['universe']}"
        )
        print(
            f"Mã={counts['tickers_analyzed']} (pending entry={counts['tickers_pending_entry']}), "
            f"HOLD={counts['hold']}, SCALP={counts['scalp']}, WAIT={counts['wait']}, "
            f"stale={counts['stale']}"
        )
        print(
            f"Fixed SL {config.stop_loss:.2%}: stop-first={rates['fixed_sl_stop_first']:.1%}, "
            f"stop rồi vẫn đạt TP={rates['fixed_sl_then_tp10']:.1%}; "
            f"ATR×{config.ticker_exit_baseline_atr_multiple:g}: "
            f"stop-first={rates['atr2_sl_stop_first']:.1%}, "
            f"stop rồi vẫn đạt TP={rates['atr2_sl_then_tp10']:.1%}"
        )
        print(
            f"TP10 hit={rates['tp10_hit']:.1%}; "
            f"non theo giá hiện tại={rates['tp10_non_current']:.1%}; "
            f"non theo đỉnh sau TP={rates['tp10_non_peak']:.1%}"
        )
        report = result["report"]
        candidates = report[
            report["fixed_sl_then_tp10"].fillna(False).astype(bool)
            | report["tp10_non_current_flag"].fillna(False).astype(bool)
        ].copy()
        if not candidates.empty:
            print("\nMã có dấu hiệu SL/TP hiện tại bị non:")
            print(candidates[[
                "ticker",
                "bottom_date",
                "entry_to_now_return_pct",
                "max_favorable_excursion_pct",
                "peak_to_current_drawdown_pct",
                "fixed_sl_then_tp10",
                "tp10_non_current_flag",
                "management_mode",
            ]].head(25).to_string(index=False))

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
