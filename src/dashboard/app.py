from __future__ import annotations

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
            run_count = len(runs_raw) if runs_raw else 0
        else:
            from src.database import get_signals, get_performance_summary
            sigs = get_signals(limit=200)
            perf = get_performance_summary()
            from src.database import get_conn
            conn = get_conn()
            run_count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
            conn.close()
        return sigs, perf, run_count
    except Exception as e:
        log.exception("load_overview failed: %s", e)
        return pd.DataFrame(), pd.DataFrame(), 0


def _format_price(value, fallback=None) -> str:
    """Format optional tracker prices without breaking the whole detail table."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        number = pd.to_numeric(fallback, errors="coerce")
    return f"{number:,.0f}" if pd.notna(number) else "—"


sigs, perf, run_count = load_overview()
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
        f'<span style="color:#00E676;font-weight:600;font-size:0.85rem">MARKET LIVE</span> &nbsp;'
        f'<span style="color:#888;font-size:0.8rem">{today_vn().isoformat()}</span></div>',
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
    latest_signal_date = sigs["signal_date"].max() if not sigs.empty else None
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
            sls.append(r.get("stop_loss", -0.03))
            tps.append(r.get("take_profit", 0.08))

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
        st.info("Chưa có signal phiên hôm nay. Pipeline chạy tự động trước giờ mở cửa (8:00 VN).")

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
                    ("🟢", str(summary["hit_tp"]), "Hit TP (+8%)", "kpi-green"),
                    ("🔴", str(summary["hit_sl"]), "Hit SL (-3%)", "kpi-red"),
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
        {"Chiến Lược": "Ensemble (Tổng Hợp)", "Mã": "ensemble", "Mô Tả": "XGBoost 4 horizons + weighted rank; quality gate theo execution T+20", "Khung TG": "T+20", "SL/TP": "-3% / +8%", "Trạng thái": "Active"},
        {"Chiến Lược": "Trend Following", "Mã": "trend_following", "Mô Tả": "EMA20/EMA60 alignment, momentum/RS, volume và ATR penalty", "Khung TG": "T+20", "SL/TP": "-3% / +8%", "Trạng thái": "Research gate (OFF)"},
        {"Chiến Lược": "Breakout Volatility", "Mã": "breakout_volatility", "Mô Tả": "Vượt prior high 20 phiên + volume ≥1.8x + close position ≥70%", "Khung TG": "T+20", "SL/TP": "-3% / +8%", "Trạng thái": "Research gate (OFF)"},
        {"Chiến Lược": "RS Momentum", "Mã": "rs_momentum", "Mô Tả": "Sức mạnh giá tương đối vượt trội so với chỉ số VN-INDEX", "Khung TG": "T+20", "SL/TP": "-3% / +8%", "Trạng thái": "Active"},
        {"Chiến Lược": "Accumulation (Gom Hàng)", "Mã": "accumulation", "Mô Tả": "Module nghiên cứu riêng; chưa nằm trong daily ensemble", "Khung TG": "T+30", "SL/TP": "-4% / +12%", "Trạng thái": "Research only"},
        {"Chiến Lược": "Mean Reversion", "Mã": "mean_reversion", "Mô Tả": "Hồi phục kỹ thuật khi giá rớt sâu vào vùng quá bán RSI < 30", "Khung TG": "T+5", "SL/TP": "-3% / +6%", "Trạng thái": "Active"},
        {"Chiến Lược": "Fundamental Value", "Mã": "fundamental_value", "Mô Tả": "P/E, P/B, ROE, margin; research-only và tự khóa khi snapshot coverage không đủ", "Khung TG": "T+60", "SL/TP": "-5% / +15%", "Trạng thái": "Guarded / OFF"},
        {"Chiến Lược": "Defensive (Phòng Thủ)", "Mã": "defensive", "Mô Tả": "Cổ phiếu biến động thấp (Low Beta) bảo toàn vốn khi thị trường rủi ro", "Khung TG": "T+20", "SL/TP": "-2.5% / +6%", "Trạng thái": "Active"},
    ]
    df_strats = pd.DataFrame(strategies_info)
    st.dataframe(df_strats, width="stretch", hide_index=True)

    try:
        from src.research.strategy_attribution import load_realized_strategy_attribution
        from src.research.paper_test import load_paper_test_readiness

        attribution = load_realized_strategy_attribution(
            round_trip_cost=Config().round_trip_cost,
            prefer_cloud=True,
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

        readiness = load_paper_test_readiness(prefer_cloud=True)
        if not readiness.empty:
            st.markdown('<div class="section-title">PAPER-TEST READINESS</div>', unsafe_allow_html=True)
            display_readiness = readiness[[
                "strategy_name", "basket_count", "trade_count",
                "avg_basket_return_net", "positive_basket_rate",
                "baskets_to_minimum", "progress_to_target", "readiness",
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

# === FOOTER ===
st.markdown(
    '<div style="text-align:center;color:#555;font-size:0.75rem;margin-top:3rem;padding:1rem;'
    'border-top:1px solid #1A1D29">'
    "Edge Engine Agent OS &mdash; Powered by Vnstock 4.x & Streamlit Pro</div>",
    unsafe_allow_html=True,
    )
