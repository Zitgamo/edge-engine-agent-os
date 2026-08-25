from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config

DB_PATH = Path("data/engine.db")
EXECUTION_METRIC_COLUMNS = (
    "execution_evaluation_dates",
    "execution_top3_win_rate",
    "execution_top3_excess_return",
    "execution_universe_excess_return",
    "execution_top3_spread",
)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add a schema column when upgrading an existing local database."""
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            ensemble_score REAL,
            stop_loss REAL,
            take_profit REAL,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date);
        CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);

        CREATE TABLE IF NOT EXISTS actuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            actual_excess_return_5d REAL,
            actual_excess_return_20d REAL,
            actual_stock_return REAL,
            benchmark_return REAL,
            gross_stock_return REAL,
            transaction_cost REAL,
            actual_outperform INTEGER,
            realized_date DATE NOT NULL,
            entry_date DATE,
            entry_price REAL,
            exit_price REAL,
            execution_status TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_actuals_signal ON actuals(signal_date, ticker);

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            roc_auc REAL,
            execution_evaluation_dates REAL,
            execution_top3_win_rate REAL,
            execution_top3_excess_return REAL,
            execution_universe_excess_return REAL,
            execution_top3_spread REAL,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES pipeline_runs(id),
            metric_name TEXT,
            metric_value REAL
        );

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
    for column, definition in [
        ("actual_excess_return_20d", "REAL"),
        ("actual_stock_return", "REAL"),
        ("benchmark_return", "REAL"),
        ("gross_stock_return", "REAL"),
        ("transaction_cost", "REAL"),
        ("entry_date", "DATE"),
        ("entry_price", "REAL"),
        ("exit_price", "REAL"),
        ("execution_status", "TEXT"),
    ]:
        _ensure_column(conn, "actuals", column, definition)
    _ensure_column(conn, "pipeline_runs", "run_key", "TEXT")
    for column in EXECUTION_METRIC_COLUMNS:
        _ensure_column(conn, "pipeline_runs", column, "REAL")
    _ensure_column(conn, "strategy_performance", "actual_excess_return_20d", "REAL")
    # The legacy column was populated with executable T+20 outcomes despite
    # its misleading T+5 name. Preserve those outcomes under the canonical
    # column during the local schema migration.
    conn.execute(
        """UPDATE actuals
           SET actual_excess_return_20d = actual_excess_return_5d
           WHERE actual_excess_return_20d IS NULL
             AND actual_excess_return_5d IS NOT NULL"""
    )
    conn.execute(
        """UPDATE strategy_performance
           SET actual_excess_return_20d = actual_excess_return_5d
           WHERE actual_excess_return_20d IS NULL
             AND actual_excess_return_5d IS NOT NULL"""
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_runs_key ON pipeline_runs(run_key)"
    )
    # Migrate older local databases that allowed duplicate realized rows.
    conn.execute(
        """DELETE FROM actuals
           WHERE id NOT IN (
               SELECT MAX(id) FROM actuals GROUP BY signal_date, ticker
           )"""
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_actuals_unique ON actuals(signal_date, ticker)"
    )
    conn.commit()
    conn.close()
    log.info("Database initialized at %s", DB_PATH)


def save_signals(
    signals: pd.DataFrame,
    model_version: str = "xgboost_technical_v5_execution_primary",
) -> int:
    init_db()
    conn = get_conn()
    raw_sig_date = (
        signals["signal_date"].iloc[0]
        if "signal_date" in signals.columns
        else date.today().isoformat()
    )
    sig_date = str(raw_sig_date)[:10]
    # A rerun replaces the complete publication for this date.  Remove
    # realized rows too, otherwise a later cloud sync can re-publish actuals
    # belonging to tickers that are no longer in the replacement ranking.
    conn.execute("DELETE FROM signals WHERE signal_date = ?", (sig_date,))
    conn.execute("DELETE FROM actuals WHERE signal_date = ?", (sig_date,))
    rows = []
    for _, row in signals.iterrows():
        rows.append((
            str(sig_date),
            row["ticker"],
            int(row["rank"]),
            float(row["score"]),
            float(row.get("ensemble_score", row["score"])),
            float(row.get("stop_loss", 0.0)),
            float(row.get("take_profit", 0.0)),
            model_version,
        ))
    conn.executemany(
        "INSERT INTO signals (signal_date, ticker, rank, score, ensemble_score, stop_loss, take_profit, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    count = len(rows)
    conn.close()
    log.info("Saved %d signals for %s", count, sig_date)
    return count


def clear_publication_for_date(signal_date: str | date) -> dict[str, int]:
    """Remove all publishable rows for a date before recording no-trade.

    Signals, realized actuals and strategy rows share the same publication
    date.  Clearing them together makes a forced rerun idempotent: an empty
    ranking cannot leave yesterday's same-date rows visible locally or ready
    to be uploaded again.
    """
    init_db()
    date_key = str(signal_date)[:10]
    conn = get_conn()
    counts: dict[str, int] = {}
    for table in ("signals", "actuals", "strategy_performance"):
        cursor = conn.execute(
            f"DELETE FROM {table} WHERE signal_date = ?",
            (date_key,),
        )
        counts[table] = max(cursor.rowcount, 0)
    conn.commit()
    conn.close()
    log.info("Cleared publication rows for %s: %s", date_key, counts)
    return counts


def save_pipeline_run(
    metrics: dict[str, float],
    status: str = "success",
    run_key: str | None = None,
) -> int:
    init_db()
    run_key = str(run_key or date.today().isoformat())
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO pipeline_runs
             (run_key, accuracy, precision, recall, f1, roc_auc,
              execution_evaluation_dates, execution_top3_win_rate,
              execution_top3_excess_return, execution_universe_excess_return,
              execution_top3_spread, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(run_key) DO UPDATE SET
             accuracy = excluded.accuracy,
             precision = excluded.precision,
             recall = excluded.recall,
             f1 = excluded.f1,
             roc_auc = excluded.roc_auc,
             execution_evaluation_dates = excluded.execution_evaluation_dates,
             execution_top3_win_rate = excluded.execution_top3_win_rate,
             execution_top3_excess_return = excluded.execution_top3_excess_return,
             execution_universe_excess_return = excluded.execution_universe_excess_return,
             execution_top3_spread = excluded.execution_top3_spread,
             status = excluded.status,
             run_date = CURRENT_TIMESTAMP""",
        (
            run_key,
            metrics.get("accuracy"),
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1"),
            metrics.get("roc_auc"),
            *(metrics.get(column) for column in EXECUTION_METRIC_COLUMNS),
            status,
        ),
    )
    run_id = cur.lastrowid
    if not run_id:
        run_id = conn.execute(
            "SELECT id FROM pipeline_runs WHERE run_key = ?",
            (run_key,),
        ).fetchone()[0]
    conn.commit()
    conn.close()
    log.info("Saved pipeline run #%d", run_id)
    return run_id


def get_signals(limit: int = 100) -> pd.DataFrame:
    init_db()
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT s.id, s.signal_date, s.ticker, s.rank, s.score,
                  s.stop_loss, s.take_profit, s.model_version, s.created_at,
                  a.actual_excess_return_5d, a.actual_excess_return_20d,
                  a.actual_stock_return, a.benchmark_return,
                  a.gross_stock_return, a.transaction_cost,
                  a.actual_outperform, a.realized_date, a.entry_date,
                  a.entry_price, a.exit_price, a.execution_status
           FROM signals s
           LEFT JOIN actuals a ON s.signal_date = a.signal_date AND s.ticker = a.ticker
           ORDER BY s.signal_date DESC, s.rank ASC
           LIMIT ?""",
        conn,
        params=(limit,),
    )
    conn.close()
    return df


def update_actuals(df: pd.DataFrame) -> int:
    init_db()
    conn = get_conn()
    rows = []
    for _, row in df.iterrows():
        excess = None
        for column in (
            "actual_excess_return_20d",
            "actual_excess_return",
            "actual_excess_return_5d",
        ):
            if column in row and pd.notna(row[column]):
                excess = float(row[column])
                break
        if excess is None:
            continue
        rows.append((
            excess,
            excess,
            float(row["actual_stock_return"])
            if "actual_stock_return" in row and pd.notna(row["actual_stock_return"])
            else None,
            float(row["benchmark_return"])
            if "benchmark_return" in row and pd.notna(row["benchmark_return"])
            else None,
            float(row["gross_stock_return"])
            if "gross_stock_return" in row and pd.notna(row["gross_stock_return"])
            else None,
            float(row["transaction_cost"])
            if "transaction_cost" in row and pd.notna(row["transaction_cost"])
            else None,
            int(row["actual_outperform"]),
            str(row["realized_date"]),
            str(row["entry_date"]) if "entry_date" in row and pd.notna(row["entry_date"]) else None,
            float(row["entry_price"]) if "entry_price" in row and pd.notna(row["entry_price"]) else None,
            float(row["exit_price"]) if "exit_price" in row and pd.notna(row["exit_price"]) else None,
            row.get("status", row.get("execution_status")),
            str(row["signal_date"]),
            row["ticker"],
        ))
    conn.executemany(
        """INSERT INTO actuals
           (actual_excess_return_5d, actual_excess_return_20d,
            actual_stock_return, benchmark_return, gross_stock_return, transaction_cost,
            actual_outperform,
            realized_date, entry_date, entry_price, exit_price, execution_status,
            signal_date, ticker)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(signal_date, ticker) DO UPDATE SET
             actual_excess_return_5d = excluded.actual_excess_return_5d,
             actual_excess_return_20d = excluded.actual_excess_return_20d,
             actual_stock_return = excluded.actual_stock_return,
             benchmark_return = excluded.benchmark_return,
             gross_stock_return = excluded.gross_stock_return,
             transaction_cost = excluded.transaction_cost,
             actual_outperform = excluded.actual_outperform,
             realized_date = excluded.realized_date,
             entry_date = excluded.entry_date,
             entry_price = excluded.entry_price,
             exit_price = excluded.exit_price,
             execution_status = excluded.execution_status,
             updated_at = CURRENT_TIMESTAMP""",
        rows,
    )
    conn.commit()
    count = len(rows)
    conn.close()
    log.info("Updated %d actuals", count)
    return count


def backfill_actuals(
    holding_period: int = 20,
    config: Config | None = None,
) -> int:
    init_db()
    conn = get_conn()
    pending = pd.read_sql_query(
        """SELECT s.signal_date, s.ticker, s.stop_loss, s.take_profit
           FROM signals s
           WHERE (s.signal_date, s.ticker) NOT IN (
               SELECT signal_date, ticker FROM actuals
           )""",
        conn,
    )
    conn.close()
    if pending.empty:
        log.info("No pending signals to backfill")
        return 0

    from src.actuals import calculate_actuals

    actuals = calculate_actuals(
        pending,
        holding_period=holding_period,
        config=config,
    )
    if actuals.empty:
        return 0
    update_actuals(actuals)
    return len(actuals)


def get_performance_summary() -> pd.DataFrame:
    init_db()
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT s.signal_date,
                  COUNT(*) as total_picks,
                  SUM(CASE WHEN a.actual_outperform = 1 THEN 1 ELSE 0 END) as wins,
                  AVG(COALESCE(a.actual_excess_return_20d, a.actual_excess_return_5d)) as avg_excess_return,
                  SUM(COALESCE(a.actual_excess_return_20d, a.actual_excess_return_5d)) as total_excess_return,
                  AVG(a.actual_stock_return) as avg_stock_return,
                  AVG(a.benchmark_return) as avg_benchmark_return
           FROM signals s
           JOIN actuals a ON s.signal_date = a.signal_date AND s.ticker = a.ticker
           GROUP BY s.signal_date
           ORDER BY s.signal_date DESC
           LIMIT 50""",
        conn,
    )
    conn.close()
    if not df.empty:
        df["win_rate"] = df["wins"] / df["total_picks"]
    return df
