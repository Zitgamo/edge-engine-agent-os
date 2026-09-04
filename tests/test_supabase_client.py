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


def test_remote_strategy_backfill_updates_each_strategy_selection(monkeypatch) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    remote_rows = [
        {
            "strategy_name": "outperform",
            "signal_date": "2026-07-01",
            "ticker": "AAA",
            "rank": 1,
            "score": 0.9,
            "realized": 0,
        },
        {
            "strategy_name": "rs_momentum",
            "signal_date": "2026-07-01",
            "ticker": "AAA",
            "rank": 2,
            "score": 0.8,
            "realized": 0,
        },
    ]
    monkeypatch.setattr(client, "get_strategy_performance", lambda: remote_rows)
    monkeypatch.setattr(client, "_remote_column_available", lambda table, column: True)
    captured = {}
    captured_calculation = {}

    def calculate(signals, holding_period, config=None):
        captured_calculation.update({
            "signals": signals,
            "holding_period": holding_period,
            "config": config,
        })
        return pd.DataFrame([{
            "signal_date": "2026-07-01",
            "ticker": "AAA",
            "actual_excess_return_5d": 0.05,
            "actual_excess_return_20d": 0.07,
            "actual_outperform": 1,
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
    result = client.backfill_remote_strategy_actuals(config=config)

    assert result == 2
    assert len(captured_calculation["signals"]) == 1
    assert captured_calculation["holding_period"] == 20
    assert captured_calculation["config"] is config
    assert captured["table"] == "strategy_performance"
    assert captured["on_conflict"] == "strategy_name,signal_date,ticker"
    assert {row["strategy_name"] for row in captured["rows"]} == {
        "outperform",
        "rs_momentum",
    }
    assert all(row["actual_excess_return_20d"] == 0.07 for row in captured["rows"])


def test_remote_strategy_backfill_uses_separate_paper_and_production_configs(
    monkeypatch,
) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    remote_rows = [
        {
            "strategy_name": "outperform",
            "signal_date": "2026-07-01",
            "ticker": "AAA",
            "rank": 1,
            "score": 0.9,
            "stop_loss": None,
            "take_profit": None,
            "realized": 0,
        },
        {
            "strategy_name": "vn30_rs_atr2_tp10",
            "signal_date": "2026-07-01",
            "ticker": "AAA",
            "rank": 1,
            "score": 0.8,
            "stop_loss": -0.04,
            "take_profit": 0.10,
            "realized": 0,
        },
    ]
    monkeypatch.setattr(client, "get_strategy_performance", lambda: remote_rows)
    monkeypatch.setattr(client, "_remote_column_available", lambda table, column: True)
    captured = {}
    calculations = []

    class RuntimeConfig:
        stop_loss = -0.03
        take_profit = 0.08
        raw_data_dir = "production"

        @staticmethod
        def paper_strategy_names() -> set[str]:
            return {"vn30_rs_atr2_tp10"}

    class PaperConfig(RuntimeConfig):
        stop_loss = -0.02
        take_profit = 0.10
        raw_data_dir = "paper"

    def calculate(signals, holding_period, config=None):
        calculations.append((signals.copy(), config))
        result = signals.copy()
        result["actual_excess_return_5d"] = 0.05
        result["actual_excess_return_20d"] = 0.07
        result["actual_outperform"] = 1
        return result

    monkeypatch.setattr(actuals, "calculate_actuals", calculate)
    monkeypatch.setattr(
        client,
        "_upsert",
        lambda table, rows, on_conflict=None: captured.update(
            {"table": table, "rows": rows, "on_conflict": on_conflict}
        ) or len(rows),
    )

    result = client.backfill_remote_strategy_actuals(
        config=RuntimeConfig(),
        paper_config=PaperConfig(),
    )

    assert result == 2
    assert len(calculations) == 2
    assert {config.raw_data_dir for _, config in calculations} == {
        "production",
        "paper",
    }
    exits = {
        row["strategy_name"]: (row["stop_loss"], row["take_profit"])
        for row in captured["rows"]
    }
    assert exits == {
        "outperform": (-0.03, 0.08),
        "vn30_rs_atr2_tp10": (-0.04, 0.10),
    }


def test_query_all_reads_beyond_one_page(monkeypatch) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    calls = []

    def query(table, params):
        calls.append(params)
        offset = int(params["offset"])
        pages = {
            0: [{"id": 1}, {"id": 2}],
            2: [{"id": 3}],
        }
        return pages.get(offset, [])

    monkeypatch.setattr(client, "_query", query)

    rows = client._query_all("strategy_performance", {"select": "id"}, page_size=2)

    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert [call["offset"] for call in calls] == ["0", "2"]


def test_sync_signals_resets_same_key_actuals_without_deleting_strategy_history(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.save_signals(pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "rank": 1,
        "score": 0.9,
    }]))
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    calls = []

    def delete(url, headers, params, timeout):
        calls.append((url.rsplit("/", 1)[-1], params))
        return SimpleNamespace(status_code=204, content=b"", text="")

    monkeypatch.setattr(supabase_client.requests, "delete", delete)
    monkeypatch.setattr(client, "_upsert", lambda table, rows, on_conflict=None: len(rows))

    assert client.sync_signals() == 1
    assert calls == [
        (
            "signals",
            {"signal_date": "eq.2026-08-18", "ticker": "not.in.(AAA)"},
        ),
        ("actuals", {"signal_date": "eq.2026-08-18"}),
    ]


def test_strategy_sync_prunes_only_an_explicit_complete_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.init_db()
    conn = database.get_conn()
    conn.execute(
        """INSERT INTO strategy_performance
           (strategy_name, signal_date, ticker, rank, score)
           VALUES ('outperform', '2026-08-18', 'AAA', 1, 0.9)"""
    )
    conn.commit()
    conn.close()

    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    monkeypatch.setattr(client, "_remote_column_available", lambda table, column: True)
    monkeypatch.setattr(
        client,
        "_upsert",
        lambda table, rows, on_conflict=None: len(rows),
    )
    monkeypatch.setattr(client, "_query", lambda table, params: [
        {
            "strategy_name": "outperform",
            "signal_date": "2026-08-18",
            "ticker": "AAA",
        },
        {
            "strategy_name": "outperform",
            "signal_date": "2026-08-18",
            "ticker": "BBB",
        },
        {
            "strategy_name": "vn30_rs_atr2_tp10",
            "signal_date": "2026-08-18",
            "ticker": "PAPER",
        },
    ])
    calls = []

    def delete(url, headers, params, timeout):
        calls.append((url.rsplit("/", 1)[-1], params))
        return SimpleNamespace(status_code=204, content=b"", text="")

    monkeypatch.setattr(supabase_client.requests, "delete", delete)

    assert client.sync_strategy_performance() == 1
    assert calls == []
    assert client.sync_strategy_performance(complete_dates={"2026-08-18"}) == 1
    assert calls == [
        (
            "strategy_performance",
            {
                "strategy_name": "eq.outperform",
                "signal_date": "eq.2026-08-18",
                "ticker": "eq.BBB",
            },
        )
    ]


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
        "remote_strategy_actuals": 0,
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
    assert [
        (url.rsplit("/", 1)[-1], params)
        for url, params in calls[:2]
    ] == [
        ("signals", {"signal_date": "eq.2026-08-18"}),
        ("actuals", {"signal_date": "eq.2026-08-18"}),
    ]
    assert calls[2][0].rsplit("/", 1)[-1] == "strategy_performance"
    assert calls[2][1]["signal_date"] == "eq.2026-08-18"
    assert calls[2][1]["strategy_name"].startswith("not.in.(")
