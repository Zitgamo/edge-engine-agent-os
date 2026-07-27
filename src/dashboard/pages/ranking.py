from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ranking", layout="wide")
st.title("Top 20 Ranking")

try:
    signal = pd.read_parquet("data/processed/signal.parquet")
    st.subheader("Today's Top 3 Picks")
    cols = st.columns(3)
    for i, (_, row) in enumerate(signal.iterrows()):
        with cols[i]:
            st.metric(f"#{row['rank']} — {row['ticker']}", f"{row['score']:.2%}", delta="BUY")
    st.divider()
except FileNotFoundError:
    st.info("No signal yet.")

try:
    df = pd.read_parquet("data/processed/ranking.parquet")
    latest = df[df["date"] == df["date"].max()]
    st.subheader(f"Full Ranking — {latest['date'].iloc[0].date()}")
    st.dataframe(latest[["rank", "ticker", "score"]].set_index("rank"), use_container_width=True)

    if "score" in df.columns:
        st.subheader("Score Distribution (All Time)")
        st.bar_chart(df.set_index("ticker")["score"])

except FileNotFoundError:
    st.warning("No ranking data found. Run the pipeline first.")
except (ValueError, OSError) as e:
    st.error(f"Error loading ranking: {e}")
