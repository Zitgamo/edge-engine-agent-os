from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st
from datetime import date, datetime

from src.dashboard.style import CUSTOM_CSS

st.set_page_config(
    page_title="Edge Engine — Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_overview():
    use_cloud = os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY")
    try:
        if use_cloud:
            from src.supabase_client import get_client
            client = get_client()
            if not client:
                return None, None, 0
            sigs_raw = client.get_signals(limit=100)
            perf_raw = client.get_performance_summary()
            runs_raw = client.get_pipeline_summary()
            sigs = pd.DataFrame(sigs_raw) if sigs_raw else pd.DataFrame()
            perf = pd.DataFrame(perf_raw) if perf_raw else pd.DataFrame()
            run_count = len(runs_raw) if runs_raw else 0
        else:
            from src.database import get_signals, get_performance_summary
            sigs = get_signals(limit=100)
            perf = get_performance_summary()
            from src.database import get_conn
            conn = get_conn()
            run_count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
            conn.close()
        return sigs, perf, run_count
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), 0


sigs, perf, run_count = load_overview()


# === HEADER ===
cols = st.columns([3, 1])
with cols[0]:
    st.markdown(
        '<div class="main-header"><h1>Edge Engine</h1>'
        '<div class="subtitle">VN Stock Ranking — T+20 Outperformance</div></div>',
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        f'<div style="text-align:right;padding-top:1rem">'
        f'<span class="live-dot"></span>'
        f'<span style="color:#666;font-size:0.8rem">{date.today().isoformat()}</span></div>',
        unsafe_allow_html=True,
    )

# === TODAY'S SIGNALS ===
today_sigs = sigs[sigs["signal_date"] == str(date.today())] if not sigs.empty else pd.DataFrame()
if not today_sigs.empty:
    st.markdown('<div class="section-title">TODAY\'S TOP 3 PICKS</div>', unsafe_allow_html=True)

    tickers = []
    scores = []
    sls = []
    tps = []
    for _, r in today_sigs.iterrows():
        tickers.append(r.get("ticker", "?"))
        scores.append(r.get("score", 0))
        sls.append(r.get("stop_loss", -0.03))
        tps.append(r.get("take_profit", 0.08))

    # Sort by rank
    pairs = sorted(zip(tickers, scores, sls, tps), key=lambda x: x[1], reverse=True)

    cards_html = '<div class="signal-grid">'
    medals = ["#1", "#2", "#3"]
    for i, (ticker, score, sl, tp) in enumerate(pairs[:3]):
        cards_html += f"""
        <div class="signal-card">
            <div class="rank-badge">{medals[i]}</div>
            <div class="ticker">{ticker}</div>
            <div class="score">Score {score:.2%}</div>
            <div class="meta">
                <span class="sl">SL {sl:+.0%}</span>
                <span class="tp">TP {tp:+.0%}</span>
            </div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
else:
    st.info("No signal for today yet. Pipeline runs every trading day at 9 AM VN time.")


# === KPI ROW ===
st.markdown('<div class="section-title">PERFORMANCE OVERVIEW</div>', unsafe_allow_html=True)

if perf is not None and not perf.empty:
    total_signals = int(perf["total_picks"].sum()) if "total_picks" in perf else len(perf)
    win_rate = perf["win_rate"].mean() if "win_rate" in perf else 0
    avg_ret = perf["avg_excess_return"].mean() if "avg_excess_return" in perf else 0
    trading_days = len(perf)
    wins = int(perf["wins"].sum()) if "wins" in perf else 0

    kpi_html = '<div class="kpi-row">'
    kpi_data = [
        ("📊", f"{total_signals}", "Total Signals", ""),
        ("🎯", f"{win_rate:.1%}", "Win Rate", "kpi-green"),
        ("💰", f"{avg_ret:+.2%}", "Avg Excess Return", "kpi-green" if avg_ret > 0 else "kpi-red"),
        ("📅", f"{trading_days}", "Trading Days", "kpi-blue"),
        ("🏆", f"{wins}/{total_signals}", "Wins", "kpi-green"),
    ]
    for icon, val, label, cls in kpi_data:
        kpi_html += f"""
        <div class="kpi-card">
            <div style="font-size:1.3rem;margin-bottom:0.3rem">{icon}</div>
            <div class="kpi-value {cls}">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>"""
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)
else:
    st.info("No actuals data yet. Pipeline needs ~20 days to accumulate results.")


# === REALTIME P&L TRACKING ===
st.markdown('<div class="section-title">REALTIME P&L TRACKING</div>', unsafe_allow_html=True)

try:
    from src.tracking.realtime import track_signal, get_signal_summary

    past_signals = sigs.copy() if not sigs.empty else pd.DataFrame()
    if not past_signals.empty:
        past_signals = past_signals[past_signals["signal_date"] != str(date.today())]
        tracking_results = []
        for _, r in past_signals.head(15).iterrows():
            result = track_signal(
                r["ticker"],
                str(r["signal_date"])[:10],
                stop_loss=float(r.get("stop_loss", -0.03)),
                take_profit=float(r.get("take_profit", 0.08)),
            )
            tracking_results.append(result)

        if tracking_results:
            summary = get_signal_summary(tracking_results)
            kpi_html = '<div class="kpi-row">'
            kpi_data = [
                ("🟢", str(summary["hit_tp"]), "Hit TP", "kpi-green"),
                ("🔴", str(summary["hit_sl"]), "Hit SL", "kpi-red"),
                ("🟡", str(summary["active"]), "Active", "kpi-blue"),
                ("📊", f"{summary['win_rate']:.0f}%", "Win Rate", "kpi-green" if summary['win_rate'] >= 50 else "kpi-red"),
                ("💰", f"{summary['total_pnl']:+.2%}", "Total P&L", "kpi-green" if summary['total_pnl'] > 0 else "kpi-red"),
            ]
            for icon, val, label, cls in kpi_data:
                kpi_html += f"""
                <div class="kpi-card">
                    <div style="font-size:1.3rem;margin-bottom:0.3rem">{icon}</div>
                    <div class="kpi-value {cls}">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>"""
            kpi_html += "</div>"
            st.markdown(kpi_html, unsafe_allow_html=True)

            # Detail table
            rows_html = ""
            for r in tracking_results:
                pnl_cls = "kpi-green" if r["pnl"] > 0 else "kpi-red"
                status_badge = {
                    "SETTLING": '<span class="badge badge-pending">T+2...</span>',
                    "ACTIVE": '<span class="badge badge-pending">ACTIVE</span>',
                    "HIT_TP": '<span class="badge badge-win">HIT TP</span>',
                    "HIT_SL": '<span class="badge badge-loss">HIT SL</span>',
                    "EXPIRED": '<span class="badge badge-pending">EXPIRED</span>',
                    "NO_DATA": "—",
                }.get(r["status"], r["status"])
                rows_html += f"""<tr>
                    <td>{r['ticker']}</td>
                    <td>{r['signal_date'][:10]}</td>
                    <td>{status_badge}</td>
                    <td class="{pnl_cls}">{r['pnl']:+.2%}</td>
                    <td>{r['days_held']}d</td>
                    <td>{r.get('entry_price', 0):,.0f}</td>
                </tr>"""
            if rows_html:
                st.markdown(f"""<table class="dataframe" style="width:100%">
                <thead><tr><th>Ticker</th><th>Date</th><th>Status</th><th>P&L</th><th>Held</th><th>Entry</th></tr></thead>
                <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
except Exception as e:
    st.caption(f"P&L tracking unavailable: {e}")

# === HISTORY TABLE ===
st.markdown('<div class="section-title">RECENT SIGNALS</div>', unsafe_allow_html=True)

if not sigs.empty:
    display = sigs.sort_values(["signal_date", "rank"], ascending=[False, True]).head(30).copy()
    display["signal_date"] = pd.to_datetime(display["signal_date"]).dt.strftime("%d/%m/%Y")
    display["score"] = display["score"].apply(lambda x: f"{x:.2%}")
    has_excess = "actual_excess_return_5d" in display.columns
    has_outperform = "actual_outperform" in display.columns
    display["excess"] = display["actual_excess_return_5d"].apply(
        lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
    ) if has_excess else "—"
    display["result"] = display["actual_outperform"].apply(
        lambda x: '<span class="badge badge-win">WIN</span>' if x == 1
        else ('<span class="badge badge-loss">LOSS</span>' if x == 0
              else '<span class="badge badge-pending">PENDING</span>')
    ) if has_outperform else '<span class="badge badge-pending">PENDING</span>'
    cols = ["signal_date", "ticker", "score"]
    if has_excess:
        cols.append("excess")
    cols.append("result")
    table = display[cols].rename(columns={
        "signal_date": "Date", "ticker": "Ticker", "score": "Score",
        "excess": "Excess Return", "result": "Result",
    })
    st.markdown(table.to_html(escape=False, index=False, classes="dataframe"), unsafe_allow_html=True)
else:
    st.caption("No signal history yet.")


# === FOOTER ===
st.markdown(
    '<div style="text-align:center;color:#333;font-size:0.7rem;margin-top:3rem;padding:1rem;'
    'border-top:1px solid #1A1D29">'
    "Edge Engine Agent OS &mdash; Data: Yahoo Finance &mdash; Model: XGBoost Ensemble</div>",
    unsafe_allow_html=True,
)
