from __future__ import annotations

import sys

import pandas as pd

from src import database


def test_save_pipeline_run_is_idempotent_by_run_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")

    database.save_pipeline_run({"accuracy": 0.51}, run_key="2026-08-18")
    database.save_pipeline_run({"accuracy": 0.62}, run_key="2026-08-18")

    conn = database.get_conn()
    rows = conn.execute(
        "SELECT run_key, accuracy FROM pipeline_runs WHERE run_key = ?",
        ("2026-08-18",),
    ).fetchall()
    conn.close()

    assert rows == [("2026-08-18", 0.62)]


def test_pipeline_run_persists_execution_quality_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")

    database.save_pipeline_run(
        {
            "accuracy": 0.62,
            "execution_evaluation_dates": 40,
            "execution_top3_excess_return": 0.012,
            "execution_top3_spread": 0.008,
        },
        run_key="2026-08-18",
    )

    conn = database.get_conn()
    row = conn.execute(
        """SELECT execution_evaluation_dates,
                         execution_top3_excess_return,
                         execution_top3_spread
           FROM pipeline_runs WHERE run_key = '2026-08-18'"""
    ).fetchone()
    conn.close()

    assert row == (40, 0.012, 0.008)


def test_init_db_migrates_legacy_schema_before_summary_queries(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    conn = database.get_conn()
    conn.executescript(
        """
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL
        );
        CREATE TABLE actuals (
            id INTEGER PRIMARY KEY,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            actual_excess_return_5d REAL,
            actual_outperform INTEGER,
            realized_date DATE NOT NULL
        );
        INSERT INTO signals VALUES (1, '2026-08-18', 'AAA', 1, 0.9);
        INSERT INTO actuals VALUES (1, '2026-08-18', 'AAA', 0.03, 1, '2026-09-15');
        """
    )
    conn.commit()
    conn.close()

    database.init_db()

    conn = database.get_conn()
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(actuals)").fetchall()
    }
    result = conn.execute(
        "SELECT actual_excess_return_20d FROM actuals WHERE ticker = 'AAA'"
    ).fetchone()[0]
    conn.close()

    assert "actual_excess_return_20d" in columns
    assert result == 0.03


def test_cli_summary_handles_legacy_database(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    conn = database.get_conn()
    conn.executescript(
        """
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            ensemble_score REAL
        );
        CREATE TABLE actuals (
            id INTEGER PRIMARY KEY,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            actual_excess_return_5d REAL,
            actual_outperform INTEGER,
            realized_date DATE NOT NULL
        );
        INSERT INTO signals VALUES (1, '2026-08-18', 'AAA', 1, 0.9, 0.9);
        INSERT INTO actuals VALUES (1, '2026-08-18', 'AAA', 0.03, 1, '2026-09-15');
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(sys, "argv", ["src.cli", "summary"])
    from src.cli import main

    main()

    assert "PERFORMANCE SUMMARY" in capsys.readouterr().out


def test_actual_return_columns_are_stored(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.update_actuals(
        pd.DataFrame([{
            "signal_date": "2026-08-18",
            "ticker": "AAA",
            "actual_excess_return_20d": 0.02,
            "actual_outperform": 1,
            "realized_date": "2026-09-15",
            "actual_stock_return": 0.05,
            "benchmark_return": 0.03,
            "gross_stock_return": 0.053,
            "transaction_cost": 0.003,
        }])
    )

    conn = database.get_conn()
    row = conn.execute(
        "SELECT actual_stock_return, benchmark_return, gross_stock_return, transaction_cost "
        "FROM actuals WHERE ticker = 'AAA'"
    ).fetchone()
    conn.close()

    assert row == (0.05, 0.03, 0.053, 0.003)


def test_rerunning_signals_removes_actuals_for_replaced_date(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.save_signals(pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "rank": 1,
        "score": 0.9,
    }]))
    database.update_actuals(pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "actual_excess_return_20d": 0.02,
        "actual_outperform": 1,
        "realized_date": "2026-09-15",
    }]))

    database.save_signals(pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "BBB",
        "rank": 1,
        "score": 0.8,
    }]))

    conn = database.get_conn()
    signals = conn.execute(
        "SELECT ticker FROM signals WHERE signal_date = '2026-08-18'"
    ).fetchall()
    actuals = conn.execute(
        "SELECT ticker FROM actuals WHERE signal_date = '2026-08-18'"
    ).fetchall()
    conn.close()

    assert signals == [("BBB",)]
    assert actuals == []


def test_clear_publication_for_date_removes_all_local_publishables(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.save_signals(pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "rank": 1,
        "score": 0.9,
    }]))
    database.update_actuals(pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "actual_excess_return_20d": 0.02,
        "actual_outperform": 1,
        "realized_date": "2026-09-15",
    }]))
    conn = database.get_conn()
    conn.execute(
        """INSERT INTO strategy_performance
           (strategy_name, signal_date, ticker, rank, score)
           VALUES ('outperform', '2026-08-18', 'AAA', 1, 0.9)"""
    )
    conn.commit()
    conn.close()

    counts = database.clear_publication_for_date("2026-08-18")

    assert counts == {"signals": 1, "actuals": 1, "strategy_performance": 1}
    conn = database.get_conn()
    remaining = [
        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("signals", "actuals", "strategy_performance")
    ]
    conn.close()
    assert remaining == [0, 0, 0]


def test_backfill_actuals_uses_supplied_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.save_signals(pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "rank": 1,
        "score": 0.9,
    }]))
    config = object()
    captured = {}

    def calculate(signals, holding_period, config=None):
        captured["config"] = config
        return pd.DataFrame()

    monkeypatch.setattr("src.actuals.calculate_actuals", calculate)

    assert database.backfill_actuals(config=config) == 0
    assert captured["config"] is config
