from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

log = logging.getLogger(__name__)

import pandas as pd
import streamlit as st

from src.dashboard.style import CUSTOM_CSS
from src.config import Config
from src.time_utils import today_vn

st.set_page_config(
    page_title="Edge Engine — VN Stock Alpha OS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=120, max_entries=4)
def load_overview():
    try:
        from src.supabase_client import get_client
        client = get_client()
        if client:
            sigs_raw = client.get_signals(limit=200)
            perf_raw = client.get_performance_summary()
            runs_raw = client.get_pipeline_summary()
            sigs = pd.DataFrame(sigs_raw) if sigs_raw else pd.DataFrame()
            perf = pd.DataFrame(perf_raw) if perf_raw else pd.DataFrame()
            runs = pd.DataFrame(runs_raw) if runs_raw else pd.DataFrame()
        else:
            from src.database import get_performance_summary, get_signals
            sigs = get_signals(limit=200)
            perf = get_performance_summary()
            from src.database import get_conn
            conn = get_conn()
            runs = pd.read_sql_query(
                "SELECT * FROM pipeline_runs ORDER BY run_date DESC LIMIT 50",
                conn,
            )
            conn.close()
            return sigs, perf, runs
        return sigs, perf, runs
    except Exception as e:
        log.exception("load_overview failed: %s", e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def _format_price(value, fallback=None) -> str:
    """Format optional tracker prices without breaking the whole detail table."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        number = pd.to_numeric(fallback, errors="coerce")
    return f"{number:,.0f}" if pd.notna(number) else "—"


@st.cache_data(ttl=300, max_entries=2)
def load_bottom_to_now_snapshot(
    report_path: str,
    summary_path: str,
) -> tuple[pd.DataFrame, dict]:
    """Load the latest local all-ticker exit diagnostic for the dashboard."""
    report_file = Path(report_path)
    summary_file = Path(summary_path)
    if not report_file.exists() or not summary_file.exists():
        return pd.DataFrame(), {}
    try:
        report = pd.read_csv(report_file)
        with summary_file.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.warning("bottom-to-now snapshot unavailable: %s", exc)
        return pd.DataFrame(), {}
    return report, summary


sigs, perf, runs = load_overview()
run_count = len(runs)
if not sigs.empty and "signal_date" in sigs.columns:
    sigs["signal_date"] = pd.to_datetime(
        sigs["signal_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    sigs = sigs.dropna(subset=["signal_date"])
latest_signal_date = (
    sigs["signal_date"].max()
    if not sigs.empty and "signal_date" in sigs.columns
    else None
)
latest_signal_day = (
    pd.Timestamp(latest_signal_date).date()
    if latest_signal_date is not None
    else None
)
current_market_date = today_vn()
latest_run = runs.iloc[0].to_dict() if not runs.empty else {}
raw_run_key = latest_run.get("run_key")
latest_run_key = (
    str(raw_run_key).strip()[:10]
    if raw_run_key is not None and not pd.isna(raw_run_key)
    else None
)
latest_run_day = None
if latest_run_key:
    parsed_run_key = pd.to_datetime(latest_run_key, errors="coerce")
    if pd.notna(parsed_run_key):
        latest_run_day = parsed_run_key.date()
else:
    raw_run_date = latest_run.get("run_date")
    parsed_run_date = pd.to_datetime(raw_run_date, errors="coerce", utc=True)
    if pd.notna(parsed_run_date):
        latest_run_day = parsed_run_date.tz_convert("Asia/Ho_Chi_Minh").date()
latest_run_status = str(latest_run.get("status") or "").strip().lower()
if latest_run_day == current_market_date and latest_run_status == "quality_failed":
    signal_status = "QUALITY BLOCKED"
    signal_status_color = "#FF5252"
elif latest_run_day == current_market_date and latest_run_status == "no_trade":
    signal_status = "NO TRADE"
    signal_status_color = "#FFB74D"
elif latest_signal_day == current_market_date:
    signal_status = "SIGNAL UPDATED"
    signal_status_color = "#00E676"
elif latest_signal_day is not None:
    signal_status = "SIGNAL LAGGING"
    signal_status_color = "#FFB74D"
else:
    signal_status = "NO SIGNAL"
    signal_status_color = "#FFB74D"
runtime_config = Config()
production_exit_label = (
    f"{runtime_config.stop_loss:+.1%} / {runtime_config.take_profit:+.1%}"
)
paper_exit_label = (
    f"ATR×{runtime_config.ticker_exit_baseline_atr_multiple:g} / "
    f"TP {runtime_config.ticker_exit_baseline_take_profit:+.1%}"
)
try:
    from src.actuals import add_execution_excess_column
    sigs = add_execution_excess_column(sigs)
except Exception:
    pass


# === HEADER ===
cols = st.columns([3, 1])
with cols[0]:
    st.markdown(
        '<div class="main-header"><h1>Edge Engine Pro</h1>'
        '<div class="subtitle">AI Quantitative Stock Ranking & Alpha OS &mdash; T+20 Outperformance</div></div>',
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        f'<div style="text-align:right;padding-top:1rem">'
        f'<span class="live-dot"></span>'
        f'<span style="color:{signal_status_color};font-weight:600;font-size:0.85rem">'
        f'{signal_status}</span> &nbsp;'
        f'<span style="color:#888;font-size:0.8rem">'
        f'Market {current_market_date.isoformat()} · '
        f'Run {latest_run_key or "—"} · '
        f'Signal {latest_signal_date or "—"}</span></div>',
        unsafe_allow_html=True,
    )

# === NAVIGATION TABS ===
tab_signals, tab_leaderboard, tab_deepdive, tab_system = st.tabs([
    "📡 Tín Hiệu & P&L Real-time",
    "🏆 Bảng Xếp Hạng Chiến Lược",
    "🔍 Chi Tiết Cổ Phiếu",
    "⚙️ Hệ Thống & Trạng Thái",
])

with tab_signals:
    # === LATEST TOP SIGNALS ===
    latest_sigs = (
        sigs[sigs["signal_date"] == latest_signal_date]
        if latest_signal_date is not None
        else pd.DataFrame()
    )
    if not latest_sigs.empty:
        st.markdown(
            f'<div class="section-title">TOP TÍN HIỆU KHUYẾN NGHỊ · PHIÊN {latest_signal_date}</div>',
            unsafe_allow_html=True,
        )

        tickers = []
        scores = []
        sls = []
        tps = []
        for _, r in latest_sigs.iterrows():
            tickers.append(r.get("ticker", "?"))
            scores.append(r.get("score", 0))
            sls.append(r.get("stop_loss", runtime_config.stop_loss))
            tps.append(r.get("take_profit", runtime_config.take_profit))

        pairs = sorted(zip(tickers, scores, sls, tps), key=lambda x: x[1], reverse=True)

        cards_html = '<div class="signal-grid">'
        medals = ["#1", "#2", "#3", "#4", "#5"]
        for i, (ticker, score, sl, tp) in enumerate(pairs[:min(5, len(pairs))]):
            medal = medals[i] if i < len(medals) else f"#{i+1}"
            cards_html += f"""
            <div class="signal-card">
                <div class="rank-badge">{medal}</div>
                <div class="ticker">{ticker}</div>
                <div class="score">Alpha Score {score:.2%}</div>
                <div class="meta">
                    <span class="sl">SL {sl:+.0%}</span>
                    <span class="tp">TP {tp:+.0%}</span>
                </div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        if latest_run_day == current_market_date and latest_run_status == "no_trade":
            st.info(
                f"Pipeline phiên {latest_run_key} đã chạy xong nhưng không có mã vượt "
                "entry gate, nên hệ thống không phát tín hiệu mới. "
                f"Tín hiệu gần nhất: {latest_signal_date or '—'}."
            )
        elif latest_run_day == current_market_date and latest_run_status == "quality_failed":
            st.error(
                f"Pipeline phiên {latest_run_key} bị chặn bởi quality gate; "
                "không phát tín hiệu để bảo toàn dữ liệu production."
            )
        else:
            st.info(
                "Chưa có signal phiên hôm nay. Pipeline chạy tự động theo lịch, "
                "có retry sau giờ đóng cửa (giờ Việt Nam)."
            )

    # === REALTIME P&L TRACKING ===
    st.markdown('<div class="section-title">THEO DÕI VỊ THẾ & HIỆU SUẤT REAL-TIME</div>', unsafe_allow_html=True)

    try:
        from src.tracking.realtime import get_signal_summary, track_signals

        past_signals = sigs.copy() if not sigs.empty else pd.DataFrame()
        if not past_signals.empty:
            past_signals = past_signals[past_signals["signal_date"] != str(today_vn())]
            tracked_signals = past_signals.head(100).to_dict("records")
            tracking_results = track_signals(tracked_signals)

            if tracking_results:
                summary = get_signal_summary(tracking_results)
                kpi_html = '<div class="kpi-row">'
                kpi_data = [
                    ("🟢", str(summary["hit_tp"]), f"Hit TP ({runtime_config.take_profit:+.1%})", "kpi-green"),
                    ("🔴", str(summary["hit_sl"]), f"Hit SL ({runtime_config.stop_loss:+.1%})", "kpi-red"),
                    ("🟡", str(summary["settling"]), "T+2 Chờ Về", "kpi-blue"),
                    ("🔵", str(summary["active"]), "Đang Giữ (Active)", "kpi-blue"),
                    ("🎯", f"{summary['win_rate']:.0f}%", "Tỷ Lệ Thắng (Win Rate)", "kpi-green" if summary['win_rate'] >= 50 else "kpi-red"),
                    ("💰", f"{summary['total_pnl']:+.2%}", "Lợi Nhuận Danh Mục", "kpi-green" if summary['total_pnl'] > 0 else "kpi-red"),
                ]
                for icon, val, label, cls in kpi_data:
                    kpi_html += f"""
                    <div class="kpi-card">
                        <div style="font-size:1.2rem;margin-bottom:0.2rem">{icon}</div>
                        <div class="kpi-value {cls}">{val}</div>
                        <div class="kpi-label">{label}</div>
                    </div>"""
                kpi_html += "</div>"
                st.markdown(kpi_html, unsafe_allow_html=True)

                # Detail table
                rows_html = ""
                for r in tracking_results[:25]:
                    pnl_cls = "kpi-green" if r["pnl"] > 0 else ("kpi-red" if r["pnl"] < 0 else "")
                    if r["status"] == "SETTLING":
                        sd = r.get("settlement_delay", 2)
                        status_badge = f'<span class="badge badge-pending">T+2 ({r["days_held"]}/{sd})</span>'
                    else:
                        status_badge = {
                            "ACTIVE": '<span class="badge badge-pending">ACTIVE</span>',
                            "HIT_TP": '<span class="badge badge-win">HIT TP</span>',
                            "HIT_SL": '<span class="badge badge-loss">HIT SL</span>',
                            "EXPIRED": '<span class="badge badge-pending">EXPIRED</span>',
                            "PENDING": '<span class="badge badge-pending">PENDING</span>',
                            "NO_DATA": '<span class="badge badge-pending">NO DATA</span>',
                        }.get(r["status"], r["status"])
                    entry_p = r.get('entry_price')
                    exit_p = r.get('exit_price')
                    rows_html += f"""<tr>
                        <td><strong>{r['ticker']}</strong></td>
                        <td>{r['signal_date'][:10]}</td>
                        <td>{status_badge}</td>
                        <td class="{pnl_cls}"><strong>{r['pnl']:+.2%}</strong></td>
                        <td>{r['days_held']} phiên</td>
                        <td>{_format_price(entry_p)}</td>
                        <td>{_format_price(exit_p, fallback=entry_p)}</td>
                    </tr>"""
                if rows_html:
                    st.markdown(f"""<table class="dataframe" style="width:100%">
                    <thead><tr><th>Mã CK</th><th>Ngày Tín Hiệu</th><th>Trạng Thái</th><th>P&L Hiện Tại</th><th>Thời Gian Giữ</th><th>Giá Vào ($T_0$)</th><th>Giá Hiện Tại / Thoát</th></tr></thead>
                    <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"Theo dõi P&L realtime chưa khả dụng: {e}")

    # === RECENT SIGNALS HISTORY ===
    st.markdown('<div class="section-title">LỊCH SỬ TÍN HIỆU ĐÃ PHÁT</div>', unsafe_allow_html=True)
    if not sigs.empty:
        display = sigs.sort_values(["signal_date", "rank"], ascending=[False, True]).head(30).copy()
        display["signal_date"] = pd.to_datetime(display["signal_date"]).dt.strftime("%d/%m/%Y")
        display["score"] = display["score"].apply(lambda x: f"{x:.2%}")
        has_excess = "execution_excess_return" in display.columns
        has_outperform = "actual_outperform" in display.columns
        display["excess"] = display["execution_excess_return"].apply(
            lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
        ) if has_excess else "—"
        display["result"] = display["actual_outperform"].apply(
            lambda x: '<span class="badge badge-win">WIN</span>' if x == 1
            else ('<span class="badge badge-loss">LOSS</span>' if x == 0
                  else '<span class="badge badge-pending">PENDING</span>')
        ) if has_outperform else '<span class="badge badge-pending">PENDING</span>'
        cols_show = ["signal_date", "ticker", "score"]
        if has_excess:
            cols_show.append("excess")
        cols_show.append("result")
        table = display[cols_show].rename(columns={
            "signal_date": "Ngày", "ticker": "Mã CK", "score": "Điểm Alpha",
            "excess": "Vượt Trội (Excess)", "result": "Kết Quả",
        })
        st.markdown(table.to_html(escape=False, index=False, classes="dataframe"), unsafe_allow_html=True)

with tab_leaderboard:
    st.markdown('<div class="section-title">SO SÁNH HIỆU SUẤT CÁC CHIẾN LƯỢC ALPHA</div>', unsafe_allow_html=True)

    strategies_info = [
        {"Chiến Lược": "Ensemble (Tổng Hợp)", "Mã": "ensemble", "Mô Tả": "XGBoost 4 horizons + weighted rank; quality gate theo execution T+20", "Khung TG": "T+20", "SL/TP": production_exit_label, "Trạng thái": "Active"},
        {"Chiến Lược": "Trend Following", "Mã": "trend_following", "Mô Tả": "EMA20/EMA60 alignment, momentum/RS, volume và ATR penalty", "Khung TG": "T+20", "SL/TP": "-3% / +8%", "Trạng thái": "Research gate (OFF)"},
        {"Chiến Lược": "Breakout Volatility", "Mã": "breakout_volatility", "Mô Tả": "Vượt prior high 20 phiên + volume ≥1.8x + close position ≥70%", "Khung TG": "T+20", "SL/TP": "-3% / +8%", "Trạng thái": "Research gate (OFF)"},
        {"Chiến Lược": "RS Momentum", "Mã": "rs_momentum", "Mô Tả": "Sức mạnh giá tương đối vượt trội so với chỉ số VN-INDEX", "Khung TG": "T+20", "SL/TP": production_exit_label, "Trạng thái": "Active"},
        {"Chiến Lược": "Accumulation (Gom Hàng)", "Mã": "accumulation", "Mô Tả": "Module nghiên cứu riêng; chưa nằm trong daily ensemble", "Khung TG": "T+30", "SL/TP": "-4% / +12%", "Trạng thái": "Research only"},
        {"Chiến Lược": "Mean Reversion", "Mã": "mean_reversion", "Mô Tả": "Hồi phục kỹ thuật khi giá rớt sâu vào vùng quá bán RSI < 30", "Khung TG": "T+5", "SL/TP": "-3% / +6%", "Trạng thái": "Active"},
        {"Chiến Lược": "Fundamental Value", "Mã": "fundamental_value", "Mô Tả": "P/E, P/B, ROE, margin; research-only và tự khóa khi snapshot coverage không đủ", "Khung TG": "T+60", "SL/TP": "-5% / +15%", "Trạng thái": "Guarded / OFF"},
        {"Chiến Lược": "Defensive (Phòng Thủ)", "Mã": "defensive", "Mô Tả": "Cổ phiếu biến động thấp (Low Beta) bảo toàn vốn khi thị trường rủi ro", "Khung TG": "T+20", "SL/TP": "-2.5% / +6%", "Trạng thái": "Active"},
        {"Chiến Lược": "Paper baseline", "Mã": "vn30_rs_atr2_tp10", "Mô Tả": "Candidate độc lập: VN30 + RS/momentum; baseline ATR×2 và TP cấu hình", "Khung TG": "T+20", "SL/TP": paper_exit_label, "Trạng thái": "Paper test"},
        {"Chiến Lược": "Paper per-ticker exits", "Mã": "vn30_rs_ticker_exit_v1", "Mô Tả": "Candidate paper tách cohort; chỉ dùng profile đã qua rolling validation và opt-in", "Khung TG": "T+20", "SL/TP": "Theo mã / opt-in", "Trạng thái": "Paper gate (OFF)"},
    ]
    df_strats = pd.DataFrame(strategies_info)
    st.dataframe(df_strats, width="stretch", hide_index=True)

    try:
        from src.config import Config
        from src.research.paper_test import load_paper_test_readiness
        from src.research.strategy_attribution import load_realized_strategy_attribution

        attribution = load_realized_strategy_attribution(
            prefer_cloud=True,
            include_version=True,
        )
        if not attribution.empty:
            st.markdown('<div class="section-title">ATTRIBUTION THEO TRADE ĐÃ REALIZE</div>', unsafe_allow_html=True)
            display_attr = attribution.copy()
            display_attr["win_rate"] = display_attr["win_rate"].map(lambda value: f"{value:.1%}")
            display_attr["positive_basket_rate"] = display_attr["positive_basket_rate"].map(lambda value: f"{value:.1%}")
            display_attr["avg_return_net"] = display_attr["avg_return_net"].map(lambda value: f"{value:+.2%}")
            display_attr["avg_basket_return_net"] = display_attr["avg_basket_return_net"].map(lambda value: f"{value:+.2%}")
            st.dataframe(display_attr, width="stretch", hide_index=True)
        else:
            st.caption("Chưa đủ realized trades để attribution theo strategy.")

        config = Config()
        readiness = load_paper_test_readiness(
            raw_data_dir=config.raw_data_dir,
            paper_raw_data_dir=config.paper_raw_data_dir,
            min_baskets=config.paper_min_baskets,
            target_baskets=config.paper_target_baskets,
            min_trades=config.paper_min_trades,
            target_trades=config.paper_target_trades,
            prefer_cloud=True,
            include_version=True,
        )
        if not readiness.empty:
            st.markdown('<div class="section-title">PAPER-TEST READINESS</div>', unsafe_allow_html=True)
            display_readiness = readiness[[
                "strategy_name", "basket_count", "trade_count",
                "avg_basket_return_net", "positive_basket_rate",
                "baskets_to_minimum", "progress_to_target",
                "trades_to_minimum", "progress_to_trade_target", "readiness",
            ]].copy()
            display_readiness["avg_basket_return_net"] = display_readiness[
                "avg_basket_return_net"
            ].map(lambda value: f"{value:+.2%}" if pd.notna(value) else "N/A")
            display_readiness["positive_basket_rate"] = display_readiness[
                "positive_basket_rate"
            ].map(lambda value: f"{value:.1%}" if pd.notna(value) else "N/A")
            display_readiness["progress_to_target"] = display_readiness[
                "progress_to_target"
            ].map(lambda value: f"{value:.1%}")
            display_readiness["progress_to_trade_target"] = display_readiness[
                "progress_to_trade_target"
            ].map(lambda value: f"{value:.1%}")
            st.dataframe(display_readiness, width="stretch", hide_index=True)
    except Exception as exc:
        st.caption(f"Attribution chưa khả dụng: {exc}")

with tab_deepdive:
    st.markdown('<div class="section-title">TRA CỨU & PHÂN TÍCH TỪNG MÃ CỔ PHIẾU</div>', unsafe_allow_html=True)
    if not sigs.empty and "ticker" in sigs.columns:
        available_tickers = sorted(sigs["ticker"].unique())
        selected_ticker = st.selectbox("Chọn mã cổ phiếu cần phân tích:", available_tickers)
        if selected_ticker:
            ticker_sigs = sigs[sigs["ticker"] == selected_ticker].sort_values("signal_date", ascending=False)
            st.write(f"### Lịch sử khuyến nghị mã **{selected_ticker}** ({len(ticker_sigs)} lần xuất hiện)")
            st.dataframe(
                ticker_sigs[["signal_date", "rank", "score", "stop_loss", "take_profit"]],
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("Chưa có dữ liệu cổ phiếu để tra cứu.")

with tab_system:
    st.markdown('<div class="section-title">TRẠNG THÁI HỆ THỐNG & ĐỒNG BỘ CLOUD</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng số lần Pipeline đã chạy", f"{run_count} phiên")
    with col2:
        st.metric("Cơ sở dữ liệu", "SQLite + Supabase Cloud Sync")
    with col3:
        st.metric("Mô hình AI", "XGBoost + execution quality gate")
    run_status_label = latest_run_status.replace("_", " ").upper() if latest_run_status else "—"
    st.caption(
        f"Pipeline gần nhất: {latest_run_key or '—'} · "
        f"Trạng thái: {run_status_label} · "
        f"Tín hiệu gần nhất: {latest_signal_date or '—'}"
    )

    st.markdown(
        '<div class="section-title">SL/TP THEO ĐÁY GẦN NHẤT · HOLD / SCALP</div>',
        unsafe_allow_html=True,
    )
    diagnostic_dir = runtime_config.ticker_exit_profile_path.parent
    diagnostic_report, diagnostic_summary = load_bottom_to_now_snapshot(
        str(diagnostic_dir / "bottom_to_now_analysis.csv"),
        str(diagnostic_dir / "bottom_to_now_summary.json"),
    )
    if diagnostic_report.empty:
        st.info(
            "Chưa có snapshot bottom-to-now. Chạy "
            "python -m src.cli research-bottom-now data/research_kbs_5y --universe all"
        )
    else:
        analyzed = diagnostic_report[
            (diagnostic_report["analysis_status"] == "analyzed")
            & (diagnostic_report["data_status"] == "fresh")
        ].copy()
        counts = diagnostic_summary.get("counts", {})
        fixed_rate = (
            (analyzed["fixed_sl_first_event"] == "stop").mean()
            if not analyzed.empty
            else float("nan")
        )
        atr_rate = (
            (analyzed["atr2_sl_first_event"] == "stop").mean()
            if not analyzed.empty
            else float("nan")
        )
        metric_cols = st.columns(5)
        metric_cols[0].metric("Entry-ready", f"{len(analyzed)} mã")
        metric_cols[1].metric(
            "HOLD",
            f"{int((analyzed['management_mode'] == 'HOLD').sum())} mã",
        )
        metric_cols[2].metric(
            "SCALP",
            f"{int((analyzed['management_mode'] == 'SCALP').sum())} mã",
        )
        metric_cols[3].metric(
            "SL -0,5% chạm trước",
            f"{fixed_rate:.1%}" if pd.notna(fixed_rate) else "—",
        )
        metric_cols[4].metric(
            "ATR×2 chạm trước",
            f"{atr_rate:.1%}" if pd.notna(atr_rate) else "—",
        )

        mode_filter = st.selectbox(
            "Lọc trạng thái quản trị",
            ["Tất cả", "HOLD", "SCALP", "WAIT", "STALE_DATA", "pending_entry"],
            key="bottom_now_mode_filter",
        )
        display = diagnostic_report.copy()
        if mode_filter != "Tất cả":
            if mode_filter == "pending_entry":
                display = display[display["analysis_status"] == mode_filter]
            else:
                display = display[display["management_mode"] == mode_filter]
        display = display[
            [
                "ticker",
                "data_status",
                "analysis_status",
                "management_mode",
                "bottom_date",
                "entry_date",
                "entry_to_now_return_pct",
                "peak_to_current_drawdown_pct",
                "atr2_stop_loss_pct",
                "tp10_state",
                "management_reason",
            ]
        ].copy()
        display = display.rename(
            columns={
                "ticker": "Mã",
                "data_status": "Dữ liệu",
                "analysis_status": "Phân tích",
                "management_mode": "Chế độ",
                "bottom_date": "Đáy gần nhất",
                "entry_date": "Entry tham chiếu",
                "entry_to_now_return_pct": "Entry → hiện tại",
                "peak_to_current_drawdown_pct": "Drawdown từ đỉnh",
                "atr2_stop_loss_pct": "SL ATR×2",
                "tp10_state": "Trạng thái TP10",
                "management_reason": "Lý do",
            }
        )
        for column in ["Entry → hiện tại", "Drawdown từ đỉnh", "SL ATR×2"]:
            display[column] = display[column].map(
                lambda value: f"{value:+.2%}" if pd.notna(value) else "—"
            )
        st.dataframe(display, width="stretch", hide_index=True)
        metadata = diagnostic_summary.get("metadata", {})
        st.caption(
            f"Snapshot đóng cửa {metadata.get('closed_date', '—')} · "
            f"{counts.get('tickers_requested', len(diagnostic_report))} mã · "
            "đáy/entry là chẩn đoán ex-post, không phải tín hiệu mua tự động."
        )

# === FOOTER ===
st.markdown(
    '<div style="text-align:center;color:#555;font-size:0.75rem;margin-top:3rem;padding:1rem;'
    'border-top:1px solid #1A1D29">'
    "Edge Engine Agent OS &mdash; Powered by Vnstock 4.x & Streamlit Pro</div>",
    unsafe_allow_html=True,
    )
