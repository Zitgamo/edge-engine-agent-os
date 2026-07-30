from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DB_PATH = Path("data/engine.db")


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
            actual_outperform INTEGER,
            realized_date DATE NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_actuals_signal ON actuals(signal_date, ticker);

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            roc_auc REAL,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES pipeline_runs(id),
            metric_name TEXT,
            metric_value REAL
        );
    """)
    conn.commit()
    conn.close()
    log.info("Database initialized at %s", DB_PATH)


def save_signals(signals: pd.DataFrame) -> int:
    conn = get_conn()
    sig_date = signals["signal_date"].iloc[0] if "signal_date" in signals.columns else date.today().isoformat()
    # Delete existing signals for same date to avoid duplicates
    conn.execute("DELETE FROM signals WHERE signal_date = ?", (sig_date,))
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
            "xgboost_v1",
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


def save_pipeline_run(metrics: dict[str, float], status: str = "success") -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO pipeline_runs (accuracy, precision, recall, f1, roc_auc, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            metrics.get("accuracy"),
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1"),
            metrics.get("roc_auc"),
            status,
        ),
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    log.info("Saved pipeline run #%d", run_id)
    return run_id


def get_signals(limit: int = 100) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT s.id, s.signal_date, s.ticker, s.rank, s.score, s.model_version, s.created_at,
                  a.actual_excess_return_5d, a.actual_outperform, a.realized_date
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
    conn = get_conn()
    rows = []
    for _, row in df.iterrows():
        rows.append((
            float(row["actual_excess_return_5d"]),
            int(row["actual_outperform"]),
            str(row["realized_date"]),
            str(row["signal_date"]),
            row["ticker"],
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO actuals
           (actual_excess_return_5d, actual_outperform, realized_date, signal_date, ticker)
           VALUES (?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    count = len(rows)
    conn.close()
    log.info("Updated %d actuals", count)
    return count


def backfill_actuals(holding_period: int = 20) -> int:
    from src.config import Config
    from src.data.collector import OHLCVCollector

    conn = get_conn()
    pending = pd.read_sql_query(
        "SELECT signal_date, ticker FROM signals WHERE (signal_date, ticker) NOT IN (SELECT signal_date, ticker FROM actuals)",
        conn,
    )
    conn.close()
    if pending.empty:
        log.info("No pending signals to backfill")
        return 0

    config = Config()
    collector = OHLCVCollector(config)
    bm = collector.fetch("VNINDEX", days=365)
    bm_sorted = bm.sort_values("date")[["date", "close"]].rename(columns={"close": "bm_close"})
    bm_dict = dict(zip(bm_sorted["date"].dt.strftime("%Y-%m-%d"), bm_sorted["bm_close"]))

    # Cache per-ticker fetches (avoid O(N²) HTTP calls for repeat tickers)
    price_cache: dict[str, pd.DataFrame] = {}
    def _get_prices(tk: str) -> pd.DataFrame:
        if tk not in price_cache:
            price_cache[tk] = collector.fetch(tk, days=365).sort_values("date")
        return price_cache[tk]

    actuals: list[dict] = []
    for _, row in pending.iterrows():
        sd = row["signal_date"]
        tk = row["ticker"]
        try:
            df_sorted = _get_prices(tk)
            dates = df_sorted["date"].dt.strftime("%Y-%m-%d").tolist()
            if sd not in dates:
                continue
            idx = dates.index(sd)
            if idx + holding_period >= len(dates):
                continue
            close_now = df_sorted.iloc[idx]["close"]
            close_future = df_sorted.iloc[idx + holding_period]["close"]
            bm_now = bm_dict.get(sd)
            bm_future = bm_dict.get(dates[idx + holding_period])
            if bm_now is None or bm_future is None:
                continue
            stock_ret = (close_future - close_now) / close_now
            bm_ret = (bm_future - bm_now) / bm_now
            excess = stock_ret - bm_ret
            actuals.append({
                "signal_date": sd,
                "ticker": tk,
                # Keep _5d name for backwards compat with dashboard SQL/dashboard
                "actual_excess_return_5d": excess,
                "actual_excess_return": excess,
                "actual_outperform": 1 if excess > 0 else 0,
                "realized_date": dates[idx + holding_period],
            })
        except Exception as e:
            log.warning("Backfill error for %s %s: %s", sd, tk, e)

    if not actuals:
        return 0
    update_actuals(pd.DataFrame(actuals))
    return len(actuals)


def get_performance_summary() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT s.signal_date,
                  COUNT(*) as total_picks,
                  SUM(CASE WHEN a.actual_outperform = 1 THEN 1 ELSE 0 END) as wins,
                  AVG(a.actual_excess_return_5d) as avg_excess_return,
                  SUM(a.actual_excess_return_5d) as total_excess_return
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
