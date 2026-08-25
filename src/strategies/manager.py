from __future__ import annotations

import logging

import pandas as pd

from src.config import Config
from src.database import _ensure_column, get_conn
from src.strategies.accumulation import AccumulationStrategy
from src.strategies.base import Strategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.breakout_volatility import BreakoutVolatilityStrategy
from src.strategies.defensive import DefensiveStrategy
from src.strategies.fundamental_value import FundamentalValueStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.outperform import OutperformStrategy
from src.strategies.rs_momentum import RSMomentumStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.time_utils import today_vn

log = logging.getLogger(__name__)

CORE_ENSEMBLE_STRATEGIES: list[Strategy] = [
    OutperformStrategy(),
    RSMomentumStrategy(),
    MeanReversionStrategy(),
    MomentumStrategy(),
    BreakoutStrategy(),
    RSIStrategy(),
    DefensiveStrategy(),
]

RESEARCH_ONLY_STRATEGIES: list[Strategy] = [
    FundamentalValueStrategy(),
    TrendFollowingStrategy(),
    BreakoutVolatilityStrategy(),
    AccumulationStrategy(),
]

# Backwards-compatible export: this is the default production ensemble.  A
# caller that explicitly opts into research modules is handled by the
# StrategyManager constructor below.
ENSEMBLE_STRATEGIES: list[Strategy] = CORE_ENSEMBLE_STRATEGIES
ALL_STRATEGIES: list[Strategy] = CORE_ENSEMBLE_STRATEGIES + RESEARCH_ONLY_STRATEGIES


class StrategyManager:
    def __init__(
        self,
        holding_period: int = 20,
        *,
        include_research: bool | None = None,
    ) -> None:
        if include_research is None:
            include_research = Config().enable_research_strategies
        self.strategies = list(CORE_ENSEMBLE_STRATEGIES)
        if include_research:
            self.strategies.extend(RESEARCH_ONLY_STRATEGIES)
        self.holding_period = holding_period
        self._init_db()

    def _init_db(self) -> None:
        conn = get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                signal_date DATE NOT NULL,
                ticker TEXT NOT NULL,
                rank INTEGER NOT NULL,
                score REAL NOT NULL,
                actual_excess_return_5d REAL,
                actual_excess_return_20d REAL,
                actual_outperform INTEGER,
                realized BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_strat_perf_name ON strategy_performance(strategy_name);
            CREATE INDEX IF NOT EXISTS idx_strat_perf_date ON strategy_performance(signal_date);
        """)
        _ensure_column(conn, "strategy_performance", "actual_excess_return_20d", "REAL")
        conn.execute(
            """UPDATE strategy_performance
               SET actual_excess_return_20d = actual_excess_return_5d
               WHERE actual_excess_return_20d IS NULL
                 AND actual_excess_return_5d IS NOT NULL"""
        )
        conn.commit()
        conn.close()

    def run_all(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        results = {}
        for strategy in self.strategies:
            try:
                ranking = strategy.rank(df)
                results[strategy.name] = ranking
                log.info("Strategy '%s' ranked %d stocks", strategy.name, len(ranking))
            except Exception as e:
                log.error("Strategy '%s' failed: %s", strategy.name, e)

        ensemble = self.get_ensemble_ranking(results)
        results["_ensemble"] = ensemble
        return results

    def save_signals(
        self,
        rankings: dict[str, pd.DataFrame],
        n: int = 5,
        signal_date: str | None = None,
    ) -> None:
        conn = get_conn()
        sig_date = signal_date or today_vn().isoformat()
        conn.execute("DELETE FROM strategy_performance WHERE signal_date = ?", (sig_date,))
        rows = []
        for strat_name, ranking in rankings.items():
            top = ranking.head(n)
            for _, r in top.iterrows():
                rows.append((
                    strat_name, sig_date, r["ticker"], int(r["rank"]),
                    float(r["score"]),
                ))
        conn.executemany(
            """INSERT INTO strategy_performance
               (strategy_name, signal_date, ticker, rank, score)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        conn.close()
        log.info("Saved %d strategy signals for %s", len(rows), sig_date)

    def get_strategy_weights(self, min_signals: int = 10) -> dict[str, float]:
        """Get performance-based weights for each strategy.

        No hard switching: all strategies contribute proportionally.  The
        local SQLite history is merged with Supabase history so a fresh
        GitHub runner does not silently reset adaptive weights to equal
        weights on every run.
        """
        conn = get_conn()
        local_df = pd.read_sql_query(
            """SELECT strategy_name, signal_date, ticker,
                      actual_outperform, realized,
                      COALESCE(actual_excess_return_20d, actual_excess_return_5d)
                      AS actual_excess_return_20d
               FROM strategy_performance
               WHERE realized = 1 AND actual_outperform IS NOT NULL""",
            conn,
        )
        conn.close()

        frames = [local_df] if not local_df.empty else []
        try:
            from src.supabase_client import get_client

            client = get_client()
            remote_rows = client.get_strategy_performance() if client else []
            remote_df = pd.DataFrame(remote_rows)
            if not remote_df.empty:
                if "actual_excess_return_20d" not in remote_df.columns:
                    remote_df["actual_excess_return_20d"] = pd.Series(
                        index=remote_df.index, dtype="float64"
                    )
                remote_df["actual_excess_return_20d"] = remote_df[
                    "actual_excess_return_20d"
                ].combine_first(
                    remote_df.get(
                        "actual_excess_return_5d",
                        pd.Series(index=remote_df.index, dtype="float64"),
                    )
                )
                if "realized" not in remote_df.columns:
                    remote_df["realized"] = 1
                frames.append(remote_df[[
                    "strategy_name",
                    "signal_date",
                    "ticker",
                    "actual_outperform",
                    "actual_excess_return_20d",
                    "realized",
                ]].copy())
        except Exception as exc:  # pragma: no cover - network/config dependent
            log.warning("Could not load strategy history from Supabase: %s", exc)

        if frames:
            combined = pd.concat(frames, ignore_index=True, sort=False)

            combined["realized"] = pd.to_numeric(
                combined["realized"], errors="coerce"
            ).fillna(0)
            combined["actual_outperform"] = pd.to_numeric(
                combined["actual_outperform"], errors="coerce"
            )
            combined["actual_excess_return_20d"] = pd.to_numeric(
                combined["actual_excess_return_20d"], errors="coerce"
            )
            # If the same row exists locally and remotely, keep the realized
            # copy and prefer the row that contains the executable return.
            key_columns = ["strategy_name", "signal_date", "ticker"]
            if set(key_columns).issubset(combined.columns):
                combined["_has_return"] = combined[
                    "actual_excess_return_20d"
                ].notna().astype(int)
                combined = combined.sort_values(
                    key_columns + ["realized", "_has_return"],
                    ascending=[True, True, True, False, False],
                    kind="stable",
                ).drop_duplicates(key_columns, keep="first")
            df = combined[
                combined["realized"].eq(1)
                & combined["actual_outperform"].notna()
            ][[
                "strategy_name",
                "actual_outperform",
                "actual_excess_return_20d",
            ]].copy()
        else:
            df = local_df

        strat_names = [s.name for s in self.strategies]
        weights = {name: 1.0 for name in strat_names}

        if df.empty:
            log.info("No realized signals — equal weights for all strategies")
            return weights

        grouped = df.groupby("strategy_name").agg(
            count=("actual_outperform", "count"),
            win_rate=("actual_outperform", "mean"),
            avg_return=("actual_excess_return_20d", "mean"),
        ).reset_index()

        for _, row in grouped.iterrows():
            name = row["strategy_name"]
            if name not in weights:
                continue
            if row["count"] < min_signals:
                continue
            wr = row["win_rate"] if pd.notna(row["win_rate"]) else 0.5
            ar = row["avg_return"] if pd.notna(row["avg_return"]) else 0.0
            ar_norm = max(-0.1, min(0.1, ar)) / 0.1
            weights[name] = max(0.1, wr * 0.6 + ar_norm * 0.4)

        log.info("Strategy weights: %s", {k: f"{v:.2f}" for k, v in sorted(weights.items())})
        return weights

    def get_ensemble_ranking(self, rankings: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Blend all strategy rankings into one ensemble ranking by weighted rank."""
        weights = self.get_strategy_weights()
        latest_date = None
        all_ranked: dict[str, float] = {}

        for name, ranking in rankings.items():
            w = weights.get(name, 1.0)
            if ranking.empty:
                continue
            if latest_date is None:
                latest_date = ranking["date"].max()
            latest = ranking[ranking["date"] == ranking["date"].max()]
            for _, r in latest.iterrows():
                ticker = r["ticker"]
                rank = int(r["rank"])
                score_inv = 1.0 / rank
                all_ranked[ticker] = all_ranked.get(ticker, 0.0) + score_inv * w

        if not all_ranked:
            return pd.DataFrame()

        ensemble = pd.DataFrame([
            {"ticker": t, "date": latest_date, "score": s / sum(weights.values())}
            for t, s in all_ranked.items()
        ])
        ensemble = ensemble.sort_values("score", ascending=False).reset_index(drop=True)
        ensemble["rank"] = range(1, len(ensemble) + 1)
        ensemble["ensemble_score"] = ensemble["score"]
        log.info("Ensemble ranking: top 3 = %s", list(ensemble.head(3)["ticker"]))
        return ensemble

    def backfill_strategy_actuals(self, config: Config | None = None) -> int:
        conn = get_conn()
        pending = pd.read_sql_query(
            """SELECT sp.id, sp.signal_date, sp.ticker
               FROM strategy_performance sp
               WHERE sp.realized = 0""",
            conn,
        )
        conn.close()

        if pending.empty:
            return 0

        # Strategy weights must be based on executable outcomes, not the
        # feature label computed from signal-day closes.  Calculate each
        # signal once, then apply the same realized result to every strategy
        # that selected it.
        from src.actuals import calculate_actuals

        unique_signals = pending[["signal_date", "ticker"]].drop_duplicates()
        actuals = calculate_actuals(
            unique_signals,
            holding_period=self.holding_period,
            config=config,
        )
        if actuals.empty:
            return 0
        actual_map = {
            (str(row["signal_date"])[:10], str(row["ticker"])): row
            for row in actuals.to_dict("records")
        }

        updates = []
        for row in pending.to_dict("records"):
            actual = actual_map.get((str(row["signal_date"])[:10], str(row["ticker"])))
            if actual is None:
                continue
            updates.append((
                float(actual["actual_excess_return"]),
                float(actual["actual_excess_return"]),
                int(actual["actual_outperform"]),
                int(row["id"]),
            ))

        if not updates:
            return 0
        conn = get_conn()
        conn.executemany(
            """UPDATE strategy_performance
               SET actual_excess_return_5d = ?, actual_excess_return_20d = ?,
                   actual_outperform = ?, realized = 1
               WHERE id = ?""",
            updates,
        )
        conn.commit()
        conn.close()
        updated = len(updates)

        log.info("Backfilled %d strategy actuals (T+%d)", updated, self.holding_period)
        return updated

    def show_comparison(self) -> None:
        conn = get_conn()
        df = pd.read_sql_query(
            """SELECT strategy_name,
                      COUNT(*) as total,
                      SUM(CASE WHEN realized=1 THEN 1 ELSE 0 END) as realized_count,
                      AVG(CASE WHEN realized=1 THEN actual_outperform ELSE NULL END) as win_rate,
                      AVG(CASE WHEN realized=1
                          THEN COALESCE(actual_excess_return_20d, actual_excess_return_5d)
                          ELSE NULL END) as avg_return
               FROM strategy_performance
               GROUP BY strategy_name
               ORDER BY win_rate DESC NULLS LAST""",
            conn,
        )
        conn.close()

        print(f"\n{'STRATEGY':<20} | {'Total':>6} | {'Realized':>9} | {'WinRate':>8} | {'AvgRet':>9}")
        print("-" * 60)
        for _, r in df.iterrows():
            wr = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "N/A"
            ar = f"{r['avg_return']:+.2%}" if r["avg_return"] is not None else "N/A"
            print(f"{r['strategy_name']:<20} | {r['total']:>6.0f} | {r['realized_count']:>9.0f} | {wr:>8} | {ar:>9}")
        print()
