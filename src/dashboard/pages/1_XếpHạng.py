from __future__ import annotations

import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from src.dashboard.style import CUSTOM_CSS

log = logging.getLogger(__name__)

st.set_page_config(page_title="Xếp Hạng", page_icon="📡", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, max_entries=2)
def load_data():
    try:
        from src.supabase_client import get_client
        client = get_client()
        if client:
            raw = client.get_signals(limit=500)
            return pd.DataFrame(raw) if raw else pd.DataFrame()
        else:
            from src.database import get_signals
            return get_signals(limit=500)
    except Exception as e:
        log.exception("load_data failed: %s", e)
        return pd.DataFrame()


st.markdown(
    '<div class="main-header"><h1>Lịch Sử Tín Hiệu</h1>'
    '<div class="subtitle">Các tín hiệu quá khứ với lợi nhuận vượt trội T+20</div></div>',
    unsafe_allow_html=True,
)

df = load_data()

if df.empty:
    st.info("No signals yet. Pipeline runs daily at 9 AM VN time.")
    st.stop()

df["signal_date"] = pd.to_datetime(df["signal_date"])
df["score_pct"] = df["score"].apply(lambda x: f"{x:.2%}")

# Filters
dates = sorted(df["signal_date"].dt.date.unique(), reverse=True)
col1, col2 = st.columns([1, 3])
with col1:
    selected_date = st.selectbox("Filter by date", ["All"] + [str(d) for d in dates])

filtered = df if selected_date == "All" else df[df["signal_date"].dt.date == pd.Timestamp(selected_date).date()]

st.markdown(f"<div style='color:#888;margin-bottom:0.5rem'>{len(filtered)} signals</div>", unsafe_allow_html=True)

# Display as table
display = filtered.sort_values(["signal_date", "rank"], ascending=[False, True]).copy()
display["date_str"] = display["signal_date"].dt.strftime("%d/%m/%Y")
display["result"] = display["actual_outperform"].apply(
    lambda x: "WIN" if x == 1 else ("LOSS" if x == 0 else "PENDING")
)
display["excess"] = display["actual_excess_return_5d"].apply(
    lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
)

table = display[["date_str", "ticker", "score_pct", "excess", "result"]].rename(columns={
    "date_str": "Date", "ticker": "Ticker", "score_pct": "Score",
    "excess": "Excess Return", "result": "Result",
})

st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    column_config={
        "Result": st.column_config.TextColumn("Result", help="WIN / LOSS / PENDING"),
    },
)

# Summary stats
st.markdown('<div class="section-title">SUMMARY</div>', unsafe_allow_html=True)
if "actual_outperform" in df.columns:
    realized = df[df["actual_outperform"].notna()]
    if not realized.empty:
        wr = realized["actual_outperform"].mean()
        avg_ret = realized["actual_excess_return_5d"].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Win Rate", f"{wr:.1%}")
        col2.metric("Avg Excess Return", f"{avg_ret:+.2%}")
        col3.metric("Total Realized", f"{len(realized)} signals")

        # Win rate by ticker
        st.markdown('<div class="section-title">TOP TICKERS BY WIN RATE</div>', unsafe_allow_html=True)
        top = realized.groupby("ticker").agg(
            signals=("actual_outperform", "count"),
            wins=("actual_outperform", "sum"),
            avg_ret=("actual_excess_return_5d", "mean"),
        ).reset_index()
        top["win_rate"] = top["wins"] / top["signals"]
        top = top.sort_values("avg_ret", ascending=False).head(10)
        top["avg_ret"] = top["avg_ret"].apply(lambda x: f"{x:+.2%}")
        top["win_rate"] = top["win_rate"].apply(lambda x: f"{x:.0%}")
        st.dataframe(top, width="stretch", hide_index=True)
