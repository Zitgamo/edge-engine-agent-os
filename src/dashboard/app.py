from __future__ import annotations

import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Edge Engine Agent OS", layout="wide")

st.title("Edge Engine Agent OS")
st.subheader("VN Stock Ranking — T+20 Outperformance vs VNINDEX")


def load_from_supabase():
    from src.supabase_client import get_client
    client = get_client()
    if client is None:
        return None, None, 0
    sigs_raw = client.get_signals(limit=3)
    perf_raw = client.get_performance_summary()
    runs_raw = client.get_pipeline_summary()
    sigs = pd.DataFrame(sigs_raw)
    perf = pd.DataFrame(perf_raw)
    run_count = len(runs_raw)
    return sigs, perf, run_count


def load_from_sqlite():
    from src.database import get_conn, get_performance_summary, get_signals
    sigs = get_signals(limit=3)
    perf = get_performance_summary()
    conn = get_conn()
    run_count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    conn.close()
    return sigs, perf, run_count


use_cloud = os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY")
data_source = "cloud" if use_cloud else "local"

try:
    if use_cloud:
        sigs, perf, run_count = load_from_supabase()
    else:
        sigs, perf, run_count = load_from_sqlite()

    if perf is not None and not perf.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng tín hiệu", int(perf["total_picks"].sum()) if "total_picks" in perf else int(len(perf)))
        col2.metric("Win Rate TB", f"{perf['win_rate'].mean():.1%}" if "win_rate" in perf else "N/A")
        col3.metric("Avg Excess Return", f"{perf['avg_excess_return'].mean():+.3%}" if "avg_excess_return" in perf else "N/A")
        col4.metric("Số ngày", len(perf))

    if sigs is not None and not sigs.empty:
        st.subheader("Top 3 Mới Nhất")
        cols = st.columns(3)
        for i, (_, row) in enumerate(sigs.iterrows()):
            if i >= 3:
                break
            with cols[i]:
                score = row.get("score", 0)
                st.metric(f"#{row.get('rank', i+1)} — {row.get('ticker', '?')}", f"{float(score):.2%}", delta="BUY")

    st.caption(f"Pipeline runs: {run_count} | Data source: {data_source}")

except Exception as e:
    st.warning(f"Không thể tải dữ liệu ({e})")

st.sidebar.markdown("### Navigation")
st.sidebar.page_link("app.py", label="Overview", disabled=True)
st.sidebar.page_link("pages/ranking.py", label="Ranking")
st.sidebar.page_link("pages/tracking.py", label="Tracking")

st.markdown(
    """
### Pipeline
**Data** Yahoo Finance → **Features** Returns/RS/ATR/Volume → **XGBoost Ensemble** → **Ranking** → **Signal** Top3

### Commands
| `python -m src.cli pipeline` | Chạy pipeline + sync lên cloud |
| `python -m src.cli backfill` | Cập nhật kết quả thực tế T+20 |
| `python -m src.cli summary`  | Xem performance history |
| `python -m src.cli signal`   | Xem tín hiệu mới nhất |
| `python -m src.cli history`  | Full history + tracker |

### Deployment
Code: [Edge Engine Agent OS](https://github.com/Zitgamo/edge-engine-agent-os)
"""
)
