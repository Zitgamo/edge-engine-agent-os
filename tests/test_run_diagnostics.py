from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import database, pipeline, supabase_client
from src.dashboard.run_status import parse_diagnostics, run_state
from src.data.health import assess_prices
from src.data.collector import OHLCVCollector
from src.supabase_client import SupabaseClient, SupabaseConfig


def prices(dates=("2026-09-03", "2026-09-04")):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(list(dates)),
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000,
        }
    )


def test_stale_data_is_relative_to_benchmark_not_weekend():
    assert assess_prices("AAA", prices(), "2026-09-04")["status"] == "ok"
    report = assess_prices("TCL", prices(("2026-08-05",)), "2026-09-04")
    assert report["status"] == "stale"
    assert report["lag_calendar_days"] == 30


@pytest.mark.parametrize("cloud_fails", [False, True])
def test_collection_failure_saves_and_attempts_sync_before_raising(monkeypatch, tmp_path, cloud_fails):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    monkeypatch.setattr(pipeline, "setup_logging", lambda: None)
    monkeypatch.setattr(pipeline, "now_vn", lambda: pd.Timestamp("2026-09-04T16:00:00+07:00"))
    monkeypatch.setattr(pipeline, "get_ticker_universe", lambda: ["AAA", "BBB"])
    config = SimpleNamespace(
        data_source="kbs", data_lookback_days=365,
        raw_data_dir=tmp_path,
        ensure_dirs=lambda: None,
        model_registry_path=tmp_path / "registry.json",
        model_artifact_dir=tmp_path / "artifacts",
        model_path_for_horizon=lambda horizon: tmp_path / f"model_{horizon}.json",
    )

    class Collector:
        last_benchmark_source = "kbs"
        last_invalid_count = 0

        def fetch(self, ticker, days):
            if ticker == "BBB":
                raise RuntimeError("fetch unavailable")
            dates = ("2026-09-04",) if ticker == "VNINDEX" else ("2026-09-03",)
            return prices(dates).assign(ticker=ticker)

    monkeypatch.setattr(pipeline, "OHLCVCollector", lambda config: Collector())
    synced = []

    def sync():
        with database.get_conn() as conn:
            row = conn.execute("SELECT status, run_key, diagnostics FROM pipeline_runs").fetchone()
        synced.append(row)
        if cloud_fails:
            raise RuntimeError("cloud unavailable")

    monkeypatch.setattr(supabase_client, "get_client", lambda: SimpleNamespace(sync_pipeline_runs=sync))
    with pytest.raises(RuntimeError, match="Only collected 0/2"):
        pipeline.run_pipeline(config, force=True)
    assert len(synced) == 1
    status, run_key, encoded = synced[0]
    report = json.loads(encoded)
    assert (status, run_key) == ("data_failed", "2026-09-04")
    assert report["collection"] == {"collected": 0, "universe_count": 2, "minimum_required": 30}
    assert [r["status"] for r in report["data_health"]] == ["stale", "fetch_failed"]
    assert run_state({"status": status, "run_key": run_key}, "2026-09-03")["blocked"]


def test_kbs_rejected_rows_survive_health_report_and_are_visible(monkeypatch):
    rows = [
        {"t": "2026-09-03", "o": 100, "h": 90, "l": 95, "c": 105, "v": 1000},
        {"t": "2026-09-04", "o": 100, "h": 110, "l": 90, "c": 105, "v": 1000},
    ]
    monkeypatch.setattr("src.data.collector.requests.get", lambda *a, **kw: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"data_day": rows},
    ))
    collector = OHLCVCollector(SimpleNamespace(
        data_source="kbs", kbs_base_url="https://example.test", kbs_timeout_seconds=1,
    ))
    frame = collector.fetch("AAA")
    report = assess_prices("AAA", frame, "2026-09-04", source_invalid_rows=collector.last_invalid_count)
    assert len(frame) == 1
    assert report["source_invalid_rows"] == report["invalid_rows"] == 1
    # The cleaned data remains eligible, but source rejections must be visible.
    assert report["status"] == "ok"
    run = {"run_key": "2026-09-04", "status": "data_failed", "diagnostics": {
        "data_health": [report],
        "collection": {"collected": 1, "universe_count": 30, "minimum_required": 30},
    }}
    app = AppTest.from_string(
        "from src.dashboard.run_status import render_run_status\n"
        f"render_run_status({run!r}, '2026-09-03')"
    ).run()
    assert not app.exception
    assert any("DATA FAILED" in e.value for e in app.error)
    assert any("Không đủ dữ liệu hợp lệ" in w.value for w in app.warning)
    assert any("dòng giá lỗi bị loại" in w.value for w in app.warning)
    assert app.dataframe[0].value["Dòng lỗi bị loại tại nguồn"].iloc[0] == 1


def test_source_rejections_count_even_when_cleaned_frame_is_empty():
    report = assess_prices("AAA", pd.DataFrame(), "2026-09-04", source_invalid_rows=3)
    assert report["status"] == "invalid"
    assert report["source_invalid_rows"] == report["invalid_rows"] == 3


def test_detects_large_history_revision_without_misclassifying_as_invalid():
    old = prices()
    new = old.copy()
    new[["open", "high", "low", "close"]] *= 0.8
    report = assess_prices("MSB", new, "2026-09-04", old)
    assert report["status"] == "ok"
    assert report["large_revision"]
    assert report["revised_rows"] == 2
    assert assess_prices("AAA", old, "2026-09-04", old)["revised_rows"] == 0


def test_rejects_nonfinite_prices_negative_volume_and_future_dates():
    for column, value in [("close", float("inf")), ("volume", -1), ("high", 1)]:
        frame = prices()
        frame.loc[0, column] = value
        assert assess_prices("AAA", frame, "2026-09-04")["status"] == "invalid"
    assert assess_prices("AAA", prices(), "2026-09-03")["status"] == "invalid"


def test_run_metadata_roundtrips_and_clears_on_legacy_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    report = {
        "version": 1,
        "data_date": "2026-09-04",
        "entry_filters": {"reason": "score margin below threshold"},
    }
    database.save_pipeline_run({}, status="no_trade", run_key="2026-09-04", diagnostics=report)
    with database.get_conn() as conn:
        assert (
            json.loads(conn.execute("select diagnostics from pipeline_runs").fetchone()[0])
            == report
        )
    database.save_pipeline_run({}, run_key="2026-09-04")
    with database.get_conn() as conn:
        assert conn.execute("select diagnostics from pipeline_runs").fetchone()[0] is None


def test_cloud_sync_supports_new_and_legacy_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    report = {"data_date": "2026-09-04"}
    database.save_pipeline_run({}, run_key="2026-09-04", diagnostics=report)
    for supports in (True, False):
        client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
        monkeypatch.setattr(
            client,
            "_remote_column_available",
            lambda table, column: column != "diagnostics" or supports,
        )
        rows = []
        monkeypatch.setattr(
            client, "_upsert", lambda table, data, on_conflict=None: rows.extend(data) or len(data)
        )
        assert client.sync_pipeline_runs() == 1
        assert (rows[0].get("diagnostics") == report) is supports


def test_failed_collection_preserves_successful_publication(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    original = {"data_date": "2026-09-04", "entry_filters": {"status": "passed"}}
    run_id = database.save_pipeline_run(
        {"accuracy": 0.8}, run_key="2026-09-04", diagnostics=original,
    )
    conn = database.get_conn()
    conn.execute("UPDATE pipeline_runs SET run_date = '2026-09-04 10:00:00'")
    conn.execute("INSERT INTO signals(signal_date,ticker,rank,score) VALUES ('2026-09-04','AAA',1,0.9)")
    conn.commit()
    conn.close()
    failure = {"collection": {"collected": 0, "minimum_required": 30}}
    assert database.save_pipeline_run(
        {}, status="data_failed", run_key="2026-09-04", diagnostics=failure,
    ) == run_id
    conn = database.get_conn()
    row = conn.execute("SELECT status, accuracy, run_date, diagnostics FROM pipeline_runs").fetchone()
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 1
    conn.close()
    assert row[:3] == ("success", 0.8, "2026-09-04 10:00:00")
    saved = json.loads(row[3])
    assert saved["entry_filters"] == original["entry_filters"]
    assert saved["last_failed_attempt"]["diagnostics"] == failure
    run = {"run_key": "2026-09-04", "status": "success", "diagnostics": saved}
    app = AppTest.from_string(
        "from src.dashboard.run_status import render_run_status\n"
        f"render_run_status({run!r}, '2026-09-04')"
    ).run()
    assert not app.exception
    assert not app.error
    assert any("Lần chạy lại bị lỗi" in w.value for w in app.warning)
    database.save_pipeline_run({}, run_key="2026-09-04", diagnostics=original)
    conn = database.get_conn()
    assert "last_failed_attempt" not in json.loads(conn.execute("SELECT diagnostics FROM pipeline_runs").fetchone()[0])
    conn.close()


def test_old_no_trade_remains_visible_with_previous_signal():
    state = run_state(
        {"run_key": "2026-09-04", "run_date": "2026-09-04T15:01:50Z", "status": "no_trade"},
        "2026-09-03",
    )
    assert state["no_trade"] and state["historical"]
    assert state["data_day"] is None
    assert state["executed"] == "04/09/2026 22:01"
    assert not run_state({"run_key": "2026-08-01", "status": "no_trade"}, "2026-09-03")["no_trade"]
    for value in (None, float("nan"), "invalid", "[]"):
        assert parse_diagnostics(value) == {}


def test_ui_shows_old_signal_and_no_trade_explanation():
    app = AppTest.from_string("""
from src.dashboard.run_status import render_run_status
render_run_status({"run_key": "2026-09-04", "run_date": "2026-09-04T15:01:50Z", "status": "no_trade",
 "diagnostics": {"data_date": "2026-09-04", "entry_filters": {
 "reason": "score margin below threshold", "input_count": 97, "published_candidate_count": 0,
 "cutoff_score_margin": .017, "min_entry_score_margin": .02,
 "stages": [{"gate": "score_margin", "before": 78, "after": 0, "removed": 78}]},
 "data_health": [{"ticker": "TCL", "status": "stale", "latest_date": "2026-08-05"}]}}, "2026-09-03")
""").run()
    assert not app.exception
    assert [m.value for m in app.metric] == ["2026-09-04", "04/09/2026 22:01", "2026-09-03"]
    assert any("NO TRADE" in warning.value for warning in app.warning)
    assert any("Chênh lệch" in m.value for m in app.markdown)
    assert len(app.dataframe) == 3


def test_stale_success_is_distinct_from_fresh_no_trade():
    stale = run_state(
        {"run_key": "2026-08-01", "status": "success"}, "2026-08-01",
        as_of="2026-09-08T10:00:00+07:00",
    )
    assert stale["stale"]
    fresh = run_state(
        {"run_key": "2026-09-07", "status": "no_trade"}, "2026-09-03",
        as_of="2026-09-08T10:00:00+07:00",
    )
    assert not fresh["stale"] and fresh["no_trade"] and fresh["historical"]
    rerun = run_state(
        {"run_key": "2026-08-01", "status": "no_trade"}, "2026-09-07",
        as_of="2026-09-08T10:00:00+07:00",
    )
    assert not rerun["stale"] and not rerun["no_trade"]


def test_weekends_and_eod_retry_window_do_not_mark_friday_stale():
    run = {"run_key": "2026-09-04", "status": "success"}
    for as_of in ("2026-09-05T22:00:00+07:00", "2026-09-06T22:00:00+07:00",
                  "2026-09-07T18:59:00+07:00"):
        assert not run_state(run, "2026-09-04", as_of=as_of)["stale"]
    assert run_state(run, "2026-09-04", as_of="2026-09-07T19:00:00+07:00")["stale"]


def test_legacy_run_uses_vietnam_execution_day_without_inventing_data_day():
    for status in ("no_trade", "quality_failed", "artifact_failed", "registry_failed"):
        state = run_state(
            {"run_date": "2026-09-06T18:00:00Z", "status": status}, "2026-09-04",
            as_of="2026-09-07T10:00:00+07:00",
        )
        assert state["run_day"] == "2026-09-07"
        assert state["legacy_run_day"] and state["data_day"] is None
        assert state["no_trade"] or state["blocked"]


def test_ui_warns_about_stale_success():
    app = AppTest.from_string('''
from src.dashboard.run_status import render_run_status
render_run_status({"run_key": "2000-01-03", "status": "success"}, "2000-01-03")
''').run()
    assert not app.exception
    assert any("Cần kiểm tra độ mới" in warning.value for warning in app.warning)


def test_ui_legacy_run_does_not_invent_filter_report():
    app = AppTest.from_string("""
from src.dashboard.run_status import render_run_status
render_run_status({"run_key": "2026-09-04", "status": "no_trade"}, "2026-09-03")
""").run()
    assert not app.exception
    assert app.metric[0].value == "Chưa ghi nhận"
    assert any("chưa lưu báo cáo" in m.value for m in app.info)


def test_full_dashboard_and_history_keep_no_trade_visible(monkeypatch):
    from pathlib import Path
    from src import supabase_client
    from src.tracking import realtime
    import streamlit as st

    class FakeCloud:
        def get_signals(self, limit=200):
            return [
                {
                    "signal_date": "2026-09-03",
                    "ticker": "AAA",
                    "rank": 1,
                    "score": 0.6,
                    "stop_loss": -0.005,
                    "take_profit": 0.1,
                    "actual_outperform": None,
                }
            ]

        def get_performance_summary(self):
            return []

        def get_pipeline_summary(self):
            return [
                {"run_key": "2026-09-04", "run_date": "2026-09-04T15:01:50Z", "status": "no_trade"}
            ]

        def get_strategy_signals(self, **kwargs):
            return []

    monkeypatch.setattr(supabase_client, "get_client", lambda: FakeCloud())
    monkeypatch.setattr(realtime, "track_signals", lambda *args, **kwargs: [])
    st.cache_data.clear()
    dashboard = Path(__file__).resolve().parents[1] / "src" / "dashboard"
    for path in (dashboard / "app.py", next((dashboard / "pages").glob("1_*.py"))):
        app = AppTest.from_file(str(path)).run(timeout=30)
        assert not app.exception, [(e.message) for e in app.exception]
        assert any("NO TRADE" in w.value for w in app.warning)
        assert app.metric[2].value == "2026-09-03"
    st.cache_data.clear()
