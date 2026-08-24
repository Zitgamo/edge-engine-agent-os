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
