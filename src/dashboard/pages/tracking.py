from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tracking", layout="wide")
st.title("Performance Tracking")

try:
    from src.database import get_performance_summary, get_signals

    perf = get_performance_summary()
    if perf.empty:
        st.info("Chưa có dữ liệu performance. Chạy `backfill` sau 5 ngày để có kết quả.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng tín hiệu", int(perf["total_picks"].sum()))
        col2.metric("Win Rate TB", f"{perf['win_rate'].mean():.1%}")
        col3.metric("Avg Excess Return", f"{perf['avg_excess_return'].mean():+.3%}")
        col4.metric("Số ngày giao dịch", len(perf))

        st.subheader("Win Rate Theo Ngày")
        chart_df = perf[["signal_date", "win_rate"]].copy()
        chart_df["signal_date"] = pd.to_datetime(chart_df["signal_date"])
        chart_df = chart_df.sort_values("signal_date")
        st.line_chart(chart_df.set_index("signal_date")["win_rate"])

        st.subheader("Excess Return Tích Lũy")
        perf_sorted = perf.sort_values("signal_date")
        perf_sorted["cum_excess"] = (1 + perf_sorted["avg_excess_return"].fillna(0)).cumprod() - 1
        cum_chart = perf_sorted[["signal_date", "cum_excess"]].copy()
        cum_chart["signal_date"] = pd.to_datetime(cum_chart["signal_date"])
        st.line_chart(cum_chart.set_index("signal_date")["cum_excess"])

        st.subheader("Chi Tiết Từng Ngày")
        st.dataframe(perf_sorted, use_container_width=True)

except ImportError:
    st.warning("Database module not available")

st.divider()
st.subheader("Lịch Sử Tín Hiệu")

try:
    history = get_signals(limit=50)
    if not history.empty:
        history["signal_date"] = pd.to_datetime(history["signal_date"])
        st.dataframe(history.sort_values(["signal_date", "rank"], ascending=[False, True]), use_container_width=True)
except Exception as e:
    st.error(f"Error: {e}")
