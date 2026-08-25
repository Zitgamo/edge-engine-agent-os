from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src import actuals, database, supabase_client
from src.supabase_client import SupabaseClient, SupabaseConfig


def test_cloud_performance_summary_matches_dashboard_contract(monkeypatch) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    monkeypatch.setattr(
        client,
        "_query",
        lambda table, params: [
            {
                "signal_date": "2026-07-01",
                "ticker": "AAA",
                "actual_excess_return_5d": 0.10,
                "actual_outperform": 1,
            },
            {
                "signal_date": "2026-07-01",
                "ticker": "BBB",
                "actual_excess_return_5d": -0.02,
                "actual_outperform": 0,
            },
        ],
    )

    result = client.get_performance_summary()

    assert result == [{
        "signal_date": "2026-07-01",
        "total_picks": 2,
        "wins": 1.0,
        "avg_excess_return": 0.04,
        "total_excess_return": 0.08,
        "win_rate": 0.5,
    }]


def test_remote_backfill_upserts_only_unrealized_signals(monkeypatch) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    monkeypatch.setattr(client, "_remote_column_available", lambda table, column: False)

    def query(table, params):
        if table == "signals":
            return [{"signal_date": "2026-07-01", "ticker": "AAA"}]
        return []

    captured = {}
    captured_config = {}
    monkeypatch.setattr(client, "_query", query)
    def calculate(pending, holding_period, config=None):
        captured_config["value"] = config
        return pd.DataFrame([{
            "signal_date": "2026-07-01",
            "ticker": "AAA",
            "actual_excess_return_5d": 0.05,
            "actual_outperform": 1,
            "realized_date": "2026-07-29",
        }])

    monkeypatch.setattr(actuals, "calculate_actuals", calculate)
    monkeypatch.setattr(
        client,
        "_upsert",
        lambda table, rows, on_conflict=None: captured.update(
            {"table": table, "rows": rows, "on_conflict": on_conflict}
        ) or len(rows),
    )

    config = object()
    result = client.backfill_remote_actuals(config=config)

    assert result == 1
    assert captured["table"] == "actuals"
    assert captured["on_conflict"] == "signal_date,ticker"
    assert captured["rows"][0]["actual_outperform"] == 1
    assert "actual_excess_return_20d" not in captured["rows"][0]
    assert captured_config["value"] is config


def test_sync_all_initializes_empty_local_database(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    monkeypatch.setattr(client, "init_tables", lambda: True)
    monkeypatch.setattr(client, "_query", lambda table, params: [])

    result = client.sync_all()

    assert result == {
        "signals": 0,
        "remote_actuals": 0,
        "actuals": 0,
        "pipeline_runs": 0,
        "strategy_performance": 0,
    }


def test_sync_pipeline_runs_upserts_by_stable_run_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.save_pipeline_run(
        {
            "accuracy": 0.62,
            "execution_evaluation_dates": 40,
            "execution_top3_excess_return": 0.012,
        },
        run_key="2026-08-18",
    )

    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    monkeypatch.setattr(client, "_remote_column_available", lambda table, column: True)
    captured = {}

    def capture(table, rows, on_conflict=None):
        captured.update({"table": table, "rows": rows, "on_conflict": on_conflict})
        return len(rows)

    monkeypatch.setattr(client, "_upsert", capture)

    assert client.sync_pipeline_runs() == 1
    assert captured["table"] == "pipeline_runs"
    assert captured["on_conflict"] == "run_key"
    assert captured["rows"][0]["run_key"] == "2026-08-18"
    assert captured["rows"][0]["execution_evaluation_dates"] == 40.0
    assert captured["rows"][0]["execution_top3_excess_return"] == 0.012


def test_sync_actuals_includes_absolute_return_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.update_actuals(pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "actual_excess_return_20d": 0.02,
        "actual_outperform": 1,
        "realized_date": "2026-09-15",
        "actual_stock_return": 0.05,
        "benchmark_return": 0.03,
        "gross_stock_return": 0.053,
        "transaction_cost": 0.003,
    }]))
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    monkeypatch.setattr(client, "_remote_column_available", lambda table, column: True)
    captured = {}
    monkeypatch.setattr(
        client,
        "_upsert",
        lambda table, rows, on_conflict=None: captured.update(
            {"table": table, "rows": rows, "on_conflict": on_conflict}
        ) or len(rows),
    )

    assert client.sync_actuals() == 1
    row = captured["rows"][0]
    assert row["actual_stock_return"] == 0.05
    assert row["benchmark_return"] == 0.03
    assert row["gross_stock_return"] == 0.053
    assert row["transaction_cost"] == 0.003


def test_clear_publication_for_date_deletes_all_cloud_publishables(monkeypatch) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    calls = []

    def delete(url, headers, params, timeout):
        calls.append((url, params))
        return SimpleNamespace(status_code=204, content=b"", text="")

    monkeypatch.setattr(supabase_client.requests, "delete", delete)

    result = client.clear_publication_for_date("2026-08-18")

    assert result == {"signals": 0, "actuals": 0, "strategy_performance": 0}
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == [
        "signals",
        "actuals",
        "strategy_performance",
    ]
    assert all(params == {"signal_date": "eq.2026-08-18"} for _, params in calls)
