"""Run presentation shared by the dashboard and offline UI regression tests."""

from __future__ import annotations

import json

import pandas as pd

from src.time_utils import now_vn


REASONS = {
    "empty ranking": "Không có mã trong bảng xếp hạng đầu vào.",
    "market breadth below threshold": "Tỷ lệ mã tăng trong 20 phiên thấp hơn ngưỡng.",
    "fewer than minimum eligible picks": "Không đủ số mã vượt bộ lọc.",
    "top score below minimum": "Điểm cao nhất chưa đạt ngưỡng.",
    "score margin below threshold": "Chênh lệch điểm tại ranh giới chọn mã chưa đạt ngưỡng.",
    "all entry filters passed": "Đã vượt các bộ lọc vào lệnh.",
    "feature flag disabled": "Bộ lọc vào lệnh đang tắt.",
}
GATES = {
    "market_breadth": "Độ rộng thị trường",
    "atr": "Biến động ATR",
    "trend": "Xu hướng",
    "minimum_picks": "Số mã tối thiểu",
    "top_score": "Điểm cao nhất",
    "score_margin": "Chênh lệch điểm",
}


def parse_diagnostics(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def day(value) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date().isoformat() if pd.notna(parsed) else None


def session_day(run: dict) -> str | None:
    """Return the market session for a run, falling back for legacy rows."""
    keyed_day = day(run.get("run_key"))
    if keyed_day:
        return keyed_day
    executed = pd.to_datetime(run.get("run_date"), errors="coerce", utc=True)
    if pd.notna(executed):
        return executed.tz_convert("Asia/Ho_Chi_Minh").date().isoformat()
    return None


def latest_publication(rows) -> dict:
    """Choose status by market session, then by execution time within a session."""
    candidates = [dict(row) for row in rows if isinstance(row, dict)]
    if not candidates:
        return {}

    def key(run: dict) -> tuple[str, int]:
        executed = pd.to_datetime(run.get("run_date"), errors="coerce", utc=True)
        return (
            session_day(run) or "",
            int(executed.value) if pd.notna(executed) else -1,
        )

    return max(candidates, key=key)


def latest_execution(rows) -> dict:
    """Keep the newest actual execution available as separate UI context."""
    candidates = [dict(row) for row in rows if isinstance(row, dict)]
    if not candidates:
        return {}

    def key(run: dict) -> tuple[int, str]:
        executed = pd.to_datetime(run.get("run_date"), errors="coerce", utc=True)
        return (
            int(executed.value) if pd.notna(executed) else -1,
            session_day(run) or "",
        )

    return max(candidates, key=key)


def expected_weekday(as_of=None) -> str:
    """Estimate the completed weekday, allowing the scheduled EOD retries.

    This is a freshness hint, not an exchange holiday calendar.
    """
    current = pd.Timestamp(as_of if as_of is not None else now_vn())
    if current.tzinfo is None:
        current = current.tz_localize("Asia/Ho_Chi_Minh")
    else:
        current = current.tz_convert("Asia/Ho_Chi_Minh")
    target = current.normalize()
    if current.hour < 19:
        target -= pd.Timedelta(days=1)
    while target.weekday() >= 5:
        target -= pd.Timedelta(days=1)
    return target.date().isoformat()


def run_state(run: dict, signal_date=None, *, as_of=None) -> dict:
    diagnostics = parse_diagnostics(run.get("diagnostics"))
    run_day = session_day(run)
    executed = pd.to_datetime(run.get("run_date"), errors="coerce", utc=True)
    legacy_run_day = not day(run.get("run_key")) and pd.notna(executed)
    signal_day = day(signal_date)
    status = str(run.get("status") or "").strip().lower()
    # A late rerun of an older session must not supersede a newer signal.
    relevant = bool(run_day and (not signal_day or run_day >= signal_day))
    label = status.upper().replace("_", " ") if relevant else "CHƯA RÕ"
    if not relevant and signal_day:
        label = "CÓ TÍN HIỆU"
    expected = expected_weekday(as_of)
    freshness_day = day(diagnostics.get("data_date")) or run_day or signal_day
    if not relevant and signal_day:
        freshness_day = signal_day
    stale = bool(freshness_day and freshness_day < expected)
    return {
        "label": label,
        "stale": stale,
        "expected_day": expected,
        "legacy_run_day": bool(legacy_run_day),
        "run_day": run_day,
        "signal_day": signal_day,
        "data_day": day(diagnostics.get("data_date")),
        "executed": executed.tz_convert("Asia/Ho_Chi_Minh").strftime("%d/%m/%Y %H:%M")
        if pd.notna(executed)
        else "Chưa rõ",
        "historical": bool(signal_day and run_day and signal_day < run_day),
        "no_trade": relevant and status == "no_trade",
        "blocked": relevant
        and status
        in {"data_failed", "quality_failed", "registry_failed", "artifact_failed", "challenger_rejected"},
        "diagnostics": diagnostics,
    }


def render_run_status(run: dict, signal_date=None, *, latest_execution_run: dict | None = None) -> None:
    import streamlit as st

    state = run_state(run, signal_date)
    cols = st.columns(3)
    cols[0].metric("Dữ liệu dùng trong pipeline", state["data_day"] or "Chưa ghi nhận")
    execution_state = run_state(latest_execution_run or run, signal_date)
    cols[1].metric("Pipeline chạy gần nhất (giờ VN)", execution_state["executed"])
    cols[2].metric("Tín hiệu gần nhất", state["signal_day"] or "Chưa có")
    st.caption(f"Phiên pipeline: {state['run_day'] or 'Chưa rõ'} · Kết quả: {state['label']}")
    if state["legacy_run_day"]:
        st.caption("Thiếu mã phiên; ngày pipeline tạm lấy từ thời điểm chạy theo giờ VN.")
    if state["stale"]:
        st.warning(
            f"Cần kiểm tra độ mới: chưa ghi nhận cập nhật cho ngày làm việc "
            f"{state['expected_day']}. Kiểm tra lịch nghỉ giao dịch và lịch chạy pipeline. "
            "Mốc này bỏ qua cuối tuần, cho phép cập nhật cuối ngày đến 19:00 giờ VN."
        )
    if state["no_trade"]:
        st.warning(
            f"Phiên {state['run_day']}: NO TRADE — không phát tín hiệu mới. "
            "Các tín hiệu phiên trước được giữ lại để theo dõi, không phải khuyến nghị mới."
        )
    elif state["blocked"]:
        st.error(f"Phiên {state['run_day']}: {state['label']} — không phát tín hiệu mới.")
    elif state["historical"]:
        st.info("Tín hiệu hiển thị thuộc phiên trước; xem ngày tín hiệu trước khi sử dụng.")
    if not state["data_day"]:
        st.caption(
            "Lần chạy cũ chưa ghi ngày dữ liệu; ngày tín hiệu không đại diện cho ngày cập nhật giá."
        )

    diagnostics = state["diagnostics"]
    failed_attempt = diagnostics.get("last_failed_attempt")
    if isinstance(failed_attempt, dict):
        st.warning(
            "Lần chạy lại bị lỗi thu thập dữ liệu. Kết quả thành công và tín hiệu "
            "đã phát trước đó được giữ nguyên."
        )
        with st.expander("Chi tiết lần chạy lại thất bại"):
            st.json(failed_attempt)
    filters = diagnostics.get("entry_filters")
    with st.expander("Vì sao có / không có tín hiệu?", expanded=state["no_trade"]):
        collection = diagnostics.get("collection")
        if isinstance(collection, dict):
            st.warning(
                f"Không đủ dữ liệu hợp lệ: {collection.get('collected')} / "
                f"{collection.get('universe_count')} mã; cần tối thiểu "
                f"{collection.get('minimum_required')} mã. Pipeline đã dừng."
            )
        elif not isinstance(filters, dict):
            st.info(
                "Lần chạy này chưa lưu báo cáo bộ lọc. Chi tiết sẽ có từ lần chạy pipeline mới; không suy đoán từ log khác."
            )
        else:
            reason = str(filters.get("reason") or "Chưa ghi nhận lý do")
            st.write(REASONS.get(reason, reason))
            published = filters.get("published_candidate_count")
            st.caption(
                f"Đầu vào: {filters.get('input_count', '—')} mã · "
                f"Còn lại sau toàn bộ bộ lọc: {published if published is not None else '—'} mã"
            )
            stages = filters.get("stages")
            if isinstance(stages, list) and stages:
                rows = [
                    {
                        "Bộ lọc": GATES.get(s.get("gate"), s.get("gate")),
                        "Trước": s.get("before"),
                        "Còn lại": s.get("after"),
                        "Bị loại": s.get("removed"),
                    }
                    for s in stages
                    if isinstance(s, dict)
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Chỉ liệt kê các bước đã chạy. Pipeline dừng khi một bộ lọc chặn toàn bộ.")
            comparisons = []
            for label, observed, threshold in (
                ("Độ rộng thị trường 20 phiên", "market_breadth_20d", "min_market_breadth_20d"),
                ("Điểm cao nhất", "top_score", "min_entry_score"),
                ("Chênh lệch điểm", "cutoff_score_margin", "min_entry_score_margin"),
            ):
                if observed in filters:
                    comparisons.append(
                        {
                            "Chỉ tiêu": label,
                            "Thực tế": filters[observed],
                            "Ngưỡng tối thiểu": filters.get(threshold),
                        }
                    )
            if comparisons:
                st.dataframe(pd.DataFrame(comparisons), hide_index=True)
    with st.expander("Chất lượng dữ liệu của phiên pipeline"):
        health = diagnostics.get("data_health")
        if not isinstance(health, list) or not health:
            st.info("Chưa có báo cáo chất lượng dữ liệu cho lần chạy này.")
        else:
            rows = [r for r in health if isinstance(r, dict)]
            flagged = [r for r in rows if r.get("status") != "ok"
                       or r.get("large_revision") or r.get("source_invalid_rows", 0)]
            st.caption(
                f"Đã kiểm tra {len(rows)} mã · {len(flagged)} mã cần chú ý. "
                "So sánh độ mới với phiên VNINDEX, không với ngày cuối tuần."
            )
            if flagged:
                st.warning(
                    "Có dữ liệu chậm, dòng giá lỗi bị loại hoặc giá lịch sử điều chỉnh lớn. Điều chỉnh giá có thể do sự kiện doanh nghiệp."
                )
            st.dataframe(
                pd.DataFrame(flagged or rows).rename(
                    columns={
                        "ticker": "Mã",
                        "latest_date": "Ngày giá",
                        "status": "Trạng thái",
                        "lag_calendar_days": "Chậm (ngày lịch)",
                        "revised_rows": "Số dòng giá điều chỉnh",
                        "max_price_revision_pct": "Mức điều chỉnh tối đa (tỷ lệ)",
                        "large_revision": "Điều chỉnh từ 5%",
                        "rows": "Số dòng",
                        "source_invalid_rows": "Dòng lỗi bị loại tại nguồn",
                        "invalid_rows": "Tổng số dòng lỗi",
                    }
                ),
                hide_index=True,
            )
