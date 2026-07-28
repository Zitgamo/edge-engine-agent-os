from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tracking", layout="wide")
st.title("Performance Tracking")

use_cloud = os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY")


def load_local():
    from src.database import get_performance_summary, get_signals
    perf = get_performance_summary()
    history = get_signals(limit=50)
    return perf, history


def load_cloud():
    from src.supabase_client import get_client
    client = get_client()
    if client is None:
        return pd.DataFrame(), pd.DataFrame()
    perf_raw = client.get_performance_summary()
    sigs_raw = client.get_signals(limit=50)
    perf = pd.DataFrame(perf_raw)
    history = pd.DataFrame(sigs_raw)
    return perf, history


try:
    if use_cloud:
        perf, history = load_cloud()
    else:
        perf, history = load_local()

    if perf.empty:
        st.info("Chưa có dữ liệu performance. Chạy pipeline để tích lũy actuals.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng tín hiệu", int(perf["total_picks"].sum()) if "total_picks" in perf else int(len(perf)))
        col2.metric("Win Rate TB", f"{perf['win_rate'].mean():.1%}" if "win_rate" in perf else "N/A")
        col3.metric("Avg Excess Return", f"{perf['avg_excess_return'].mean():+.3%}" if "avg_excess_return" in perf else "N/A")
        col4.metric("Số ngày giao dịch", len(perf))

        if "win_rate" in perf.columns:
            st.subheader("Win Rate Theo Ngày")
            chart_df = perf[["signal_date", "win_rate"]].copy()
            chart_df["signal_date"] = pd.to_datetime(chart_df["signal_date"])
            chart_df = chart_df.sort_values("signal_date")
            st.line_chart(chart_df.set_index("signal_date")["win_rate"])

        if "avg_excess_return" in perf.columns:
            st.subheader("Excess Return Tích Lũy")
            perf_sorted = perf.sort_values("signal_date")
            perf_sorted["cum_excess"] = (1 + perf_sorted["avg_excess_return"].fillna(0)).cumprod() - 1
            cum_chart = perf_sorted[["signal_date", "cum_excess"]].copy()
            cum_chart["signal_date"] = pd.to_datetime(cum_chart["signal_date"])
            st.line_chart(cum_chart.set_index("signal_date")["cum_excess"])

            st.subheader("Chi Tiết Từng Ngày")
            st.dataframe(perf_sorted, use_container_width=True)

    if not history.empty:
        st.divider()
        st.subheader("Lịch Sử Tín Hiệu")
        history["signal_date"] = pd.to_datetime(history["signal_date"])
        st.dataframe(history.sort_values(["signal_date", "rank"], ascending=[False, True]), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
