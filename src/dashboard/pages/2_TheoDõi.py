from __future__ import annotations

import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.style import CUSTOM_CSS
from src.time_utils import today_vn

log = logging.getLogger(__name__)

st.set_page_config(page_title="Theo Dõi", page_icon="📊", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, max_entries=2)
def load_data():
    try:
        from src.supabase_client import get_client
        client = get_client()
        if client:
            raw = client.get_performance_summary()
            return pd.DataFrame(raw) if raw else pd.DataFrame()
        else:
            from src.database import get_performance_summary
            return get_performance_summary()
    except Exception as e:
        log.exception("load_data failed: %s", e)
        return pd.DataFrame()


st.markdown(
    '<div class="main-header"><h1>Phân Tích Hiệu Suất</h1>'
    '<div class="subtitle">Kết quả thực tế theo dõi tại T+20</div></div>',
    unsafe_allow_html=True,
)

perf = load_data()

if perf.empty:
    st.info("No performance data yet. Pipeline needs ~20 days to accumulate actuals.")
    st.stop()

perf["signal_date"] = pd.to_datetime(perf["signal_date"])
perf = perf.sort_values("signal_date")

total = int(perf["total_picks"].sum()) if "total_picks" in perf else 0
wins = int(perf["wins"].sum()) if "wins" in perf else 0
win_rate = perf["win_rate"].mean() if "win_rate" in perf else 0
avg_ret = perf["avg_excess_return"].mean() if "avg_excess_return" in perf else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Picks", total)
col2.metric("Win Rate", f"{win_rate:.1%}")
col3.metric("Avg Excess Return", f"{avg_ret:+.2%}")
col4.metric("Trading Days", len(perf))

# Win Rate Chart
st.markdown('<div class="section-title">WIN RATE OVER TIME</div>', unsafe_allow_html=True)
fig1 = px.bar(
    perf,
    x="signal_date",
    y="win_rate",
    title=None,
    labels={"signal_date": "", "win_rate": "Win Rate"},
    color="win_rate",
    color_continuous_scale=["#FF5252", "#FFA726", "#00C853"],
    range_color=[0, 1],
    height=300,
)
fig1.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#888",
    margin=dict(l=0, r=0, t=0, b=0),
    yaxis_tickformat=".0%",
    coloraxis_showscale=False,
)
fig1.update_xaxes(gridcolor="#1A1D29")
fig1.update_yaxes(gridcolor="#1A1D29", range=[0, 1])
st.plotly_chart(fig1, width="stretch")

# Cumulative Excess Return
st.markdown('<div class="section-title">CUMULATIVE EXCESS RETURN</div>', unsafe_allow_html=True)
perf["cum_excess"] = (1 + perf["avg_excess_return"].fillna(0)).cumprod() - 1

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=perf["signal_date"],
    y=perf["cum_excess"],
    mode="lines",
    name="Cumulative Excess Return",
    line=dict(color="#00C853", width=2),
    fill="tozeroy",
    fillcolor="rgba(0,200,83,0.08)",
))
fig2.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#888",
    margin=dict(l=0, r=0, t=0, b=0),
    yaxis_tickformat="+.1%",
    hovermode="x unified",
    height=300,
)
fig2.update_xaxes(gridcolor="#1A1D29")
fig2.update_yaxes(gridcolor="#1A1D29")
st.plotly_chart(fig2, width="stretch")

# Daily breakdown
st.markdown('<div class="section-title">DAILY BREAKDOWN</div>', unsafe_allow_html=True)
daily = perf[["signal_date", "total_picks", "wins", "win_rate", "avg_excess_return"]].copy()
daily["signal_date"] = daily["signal_date"].dt.strftime("%d/%m/%Y")
daily["win_rate"] = daily["win_rate"].apply(lambda x: f"{x:.0%}")
daily["avg_excess_return"] = daily["avg_excess_return"].apply(lambda x: f"{x:+.2%}")
daily = daily.rename(columns={
    "signal_date": "Date", "total_picks": "Picks", "wins": "Wins",
    "win_rate": "Win Rate", "avg_excess_return": "Avg Return",
})
st.dataframe(daily, width="stretch", hide_index=True)

# === REALTIME TRACKING ===
st.markdown('<div class="section-title">REALTIME SIGNAL TRACKING</div>', unsafe_allow_html=True)

try:
    # Load signals from either source
    from src.supabase_client import get_client
    from src.tracking.realtime import get_signal_summary, track_signals
    client = get_client()
    if client:
        sigs_raw = client.get_signals(limit=100) if client else []
        sigs = pd.DataFrame(sigs_raw) if sigs_raw else pd.DataFrame()
    else:
        from src.database import get_signals
        sigs = get_signals(limit=100)

    if not sigs.empty:
        past = sigs[sigs["signal_date"] != str(today_vn())].head(100)
        results = track_signals(past.to_dict("records"))

        if results:
            summary = get_signal_summary(results)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("T+2...", summary["settling"])
            c2.metric("Hit TP", summary["hit_tp"])
            c3.metric("Hit SL", summary["hit_sl"])
            c4.metric("Active", summary["active"])
            c5.metric("Win Rate", f"{summary['win_rate']:.0f}%")
            c6.metric("Weighted basket P&L", f"{summary['total_pnl']:+.2%}")

            # Table
            rows = []
            for r in results:
                status = r["status"]
                if status == "SETTLING":
                    sd = r.get("settlement_delay", 2)
                    status = f"T+2 ({r['days_held']}/{sd})"
                elif status == "PENDING":
                    status = "PENDING (chờ T+1)"
                rows.append({
                    "Ticker": r["ticker"],
                    "Date": r["signal_date"][:10],
                    "Status": status,
                    "P&L": r["pnl"],
                    "Days": r["days_held"],
                    "Entry": r.get("entry_price", 0),
                })
            df_rt = pd.DataFrame(rows)
            df_rt["P&L"] = df_rt["P&L"].apply(lambda x: f"{x:+.2%}")
            df_rt["Entry"] = df_rt["Entry"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(df_rt, width="stretch", hide_index=True)
except Exception as e:
    st.caption(f"Realtime tracking unavailable: {e}")

st.markdown(
    '<div style="text-align:center;color:#333;font-size:0.7rem;margin-top:3rem;'
    'border-top:1px solid #1A1D29;padding:1rem">'
    "Results are excess returns vs VNINDEX over T+20 holding period.</div>",
    unsafe_allow_html=True,
)
