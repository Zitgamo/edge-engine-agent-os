from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Edge Engine Agent OS", layout="wide")

st.title("Edge Engine Agent OS")
st.subheader("VN Stock Ranking — T+5 Outperformance vs VNINDEX")

try:

    from src.database import get_conn, get_performance_summary, get_signals

    perf = get_performance_summary()
    sigs = get_signals(limit=3)

    if not perf.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng tín hiệu", int(perf["total_picks"].sum()))
        col2.metric("Win Rate TB", f"{perf['win_rate'].mean():.1%}")
        col3.metric("Avg Excess Return", f"{perf['avg_excess_return'].mean():+.3%}")
        col4.metric("Số ngày", len(perf))

    if not sigs.empty:
        st.subheader("Top 3 Mới Nhất")
        cols = st.columns(3)
        for i, (_, row) in enumerate(sigs.iterrows()):
            with cols[i]:
                st.metric(f"#{row['rank']} — {row['ticker']}", f"{row['score']:.2%}", delta="BUY")

    pipeline_count = get_conn().execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    st.caption(f"Pipeline runs: {pipeline_count} | SQLite: data/engine.db")

except Exception:
    pass

st.sidebar.markdown("### Navigation")
st.sidebar.page_link("app.py", label="Overview")
st.sidebar.page_link("pages/ranking.py", label="Ranking")
st.sidebar.page_link("pages/tracking.py", label="Tracking")

st.markdown(
    """
### Commands
| `python -m src.cli pipeline` | Chạy pipeline + lưu tín hiệu |
| `python -m src.cli backfill` | Cập nhật kết quả thực tế T+5 |
| `python -m src.cli summary`  | Xem performance history |
| `python -m src.cli signal`   | Xem tín hiệu mới nhất |

### Pipeline
**Data** Yahoo Finance (free) → **Features** Returns/RS/ATR/Volume → **XGBoost** → **Ranking** Top20 → **Signal** Top3

### DB Schema
`signals`(date, ticker, rank, score) + `actuals`(excess_return, outperform) + `pipeline_runs`(metrics)
"""
)
