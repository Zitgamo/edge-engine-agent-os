from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import logging

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.accumulation import backtest_tich_san, backtest_multi, backtest_compare_frequencies, INVESTMENT_DEFAULTS
from src.config import Config
from src.data.collector import OHLCVCollector
from src.data.universe import VN30_TICKERS, filter_quality, get_ticker_universe
from src.data.validator import DataValidator
from src.dashboard.style import CUSTOM_CSS
from src.features.returns import ReturnFeatures
from src.features.rs import RelativeStrength
from src.features.volatility import ATR
from src.features.volume import VolumeSurge
from src.features.fundamental import add_fundamental_features
from src.features.macro import add_macro_features

log = logging.getLogger(__name__)


def _generate_features_minimal() -> None:
    try:
        config = Config()
        collector = OHLCVCollector(config)
        validator = DataValidator()
        universe = get_ticker_universe()
        bm = collector.fetch("VNINDEX", days=365)
        all_dfs = []
        for ticker in universe:
            df = collector.fetch(ticker, days=365)
            df = filter_quality(df, ticker)
            if df is not None:
                errors = validator.validate(df)
                if not errors:
                    all_dfs.append(df)
        ret = ReturnFeatures()
        rs = RelativeStrength()
        atr = ATR()
        vol = VolumeSurge()
        feature_dfs = []
        for df in all_dfs:
            df = ret.compute(df)
            df = rs.compute(df, bm)
            df = atr.compute(df)
            df = vol.compute(df)
            feature_dfs.append(df)
        features = pd.concat(feature_dfs, ignore_index=True)
        features = add_macro_features(features)
        features = add_fundamental_features(features)
        feat_path = _root / "data" / "processed" / "features.parquet"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(feat_path)
        log.info("Generated features.parquet (%d rows)", len(features))
    except Exception as e:
        log.warning("Feature generation failed: %s", e)

st.set_page_config(page_title="Tích Sản", page_icon="🏦", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="main-header"><h1>Tích Sản — DCA Backtest</h1>'
    '<div class="subtitle">Long-term accumulation simulation: monthly DCA into VN stocks</div></div>',
    unsafe_allow_html=True,
)

# Sidebar params
with st.sidebar:
    st.markdown("### Thông số")
    monthly = st.number_input("Số tiền mỗi tháng (VND)", min_value=1_000_000, value=10_000_000, step=1_000_000, format="%d")
    freq = st.selectbox("Tần suất", ["monthly", "quarterly"], index=0)
    start = st.date_input("Ngày bắt đầu", value=pd.to_datetime("2020-01-01"))
    ticker_input = st.text_input("Mã cổ phiếu (cách nhau bằng dấu phẩy)", value="HPG, FPT, VNM, ACB, VCB")
    run_btn = st.button("Chạy Backtest", type="primary")

if run_btn:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    tab1, tab2, tab3, tab4 = st.tabs(["So Sánh", "Chi Tiết", "Tần Suất", "Xếp Hạng"])

    with tab1:
        st.markdown("### So sánh nhiều mã")
        with st.spinner(f"Backtesting {len(tickers)} tickers..."):
            df = backtest_multi(
                tickers=tickers,
                monthly_amount=monthly,
                frequency=freq,
                start_date=start.isoformat(),
            )

        if not df.empty:
            df_display = df.copy()
            df_display["total_return"] = df_display["total_return"].apply(lambda x: f"{x:+.2%}")
            df_display["cagr"] = df_display["cagr"].apply(lambda x: f"{x:+.2%}")
            df_display["sharpe"] = df_display["sharpe"].apply(lambda x: f"{x:.2f}")
            df_display["max_dd"] = df_display["max_dd"].apply(lambda x: f"{x:.2%}")
            df_display["active_return"] = df_display["active_return"].apply(lambda x: f"{x:+.2%}")
            df_display["final_value"] = df_display["final_value"].apply(lambda x: f"{x:,.0f}")
            df_display["total_invested"] = df_display["total_invested"].apply(lambda x: f"{x:,.0f}")
            df_display = df_display.rename(columns={
                "ticker": "Ticker", "total_return": "Total Return", "cagr": "CAGR",
                "sharpe": "Sharpe", "max_dd": "Max DD", "final_value": "Final",
                "total_invested": "Invested", "years": "Years", "active_return": "Active Ret",
            })
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # CAGR bar chart
            df_plot = df.copy().head(10)
            fig = px.bar(
                df_plot, x="ticker", y="cagr",
                title="CAGR by ticker",
                labels={"ticker": "", "cagr": "CAGR"},
                color="cagr",
                color_continuous_scale=["#FF5252", "#FFA726", "#00C853"],
                height=350,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#888", margin=dict(l=0, r=0, t=30, b=0),
                yaxis_tickformat=".0%",
            )
            fig.update_xaxes(gridcolor="#1A1D29")
            fig.update_yaxes(gridcolor="#1A1D29")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Chi tiết từng mã")
        detail_ticker = st.selectbox("Chọn mã", tickers if tickers else ["HPG"])
        with st.spinner(f"Backtesting {detail_ticker}..."):
            result = backtest_tich_san(
                ticker=detail_ticker,
                monthly_amount=monthly,
                frequency=freq,
                start_date=start.isoformat(),
            )

        if "error" not in result:
            m = result["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return", f"{m['total_return']:+.2%}")
            c2.metric("CAGR", f"{m['cagr']:+.2%}")
            c3.metric("Sharpe", f"{m['sharpe']:.2f}")
            c4.metric("Max DD", f"{m['max_drawdown']:.2%}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Invested", f"{m['total_invested']:,.0f}")
            c6.metric("Final Value", f"{m['final_value']:,.0f}")
            c7.metric("Active Return", f"{m.get('active_return', 0):+.2%}")
            c8.metric("Price Change", f"{result['price_change']:+.2%}")

            # Portfolio value chart
            dca_df = result["dca_history"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dca_df["date"], y=dca_df["portfolio_value"],
                mode="lines", name="Portfolio Value",
                line=dict(color="#00C853", width=2),
                fill="tozeroy", fillcolor="rgba(0,200,83,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=dca_df["date"], y=dca_df["total_invested"],
                mode="lines", name="Total Invested",
                line=dict(color="#FF5252", width=2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=dca_df["date"], y=dca_df["price"],
                mode="lines", name="Stock Price",
                line=dict(color="#FFA726", width=1),
                yaxis="y2",
            ))
            fig.update_layout(
                title=f"{detail_ticker} DCA Portfolio",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#888", margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified", height=400,
                yaxis=dict(title="Portfolio Value", tickformat=",.0f"),
                yaxis2=dict(
                    title="Price", overlaying="y", side="right", tickformat=",.0f",
                    showgrid=False,
                ),
                legend=dict(orientation="h", y=1.1),
            )
            fig.update_xaxes(gridcolor="#1A1D29")
            fig.update_yaxes(gridcolor="#1A1D29")
            st.plotly_chart(fig, use_container_width=True)

            # DCA history table
            st.markdown("### Lịch sử DCA (24 tháng gần nhất)")
            dca_display = dca_df.tail(24).copy()
            dca_display["date"] = pd.to_datetime(dca_display["date"]).dt.strftime("%Y-%m-%d")
            dca_display["price"] = dca_display["price"].apply(lambda x: f"{x:,.0f}")
            dca_display["total_shares"] = dca_display["total_shares"].apply(lambda x: f"{x:.1f}")
            dca_display["total_invested"] = dca_display["total_invested"].apply(lambda x: f"{x:,.0f}")
            dca_display["portfolio_value"] = dca_display["portfolio_value"].apply(lambda x: f"{x:,.0f}")
            dca_display["pnl_pct"] = dca_display["pnl_pct"].apply(lambda x: f"{x:+.2%}")
            dca_display = dca_display.rename(columns={
                "date": "Date", "price": "Price", "total_shares": "Shares",
                "total_invested": "Invested", "portfolio_value": "Value",
                "pnl_pct": "P&L",
            })
            st.dataframe(dca_display[["Date", "Price", "Shares", "Invested", "Value", "P&L"]],
                        use_container_width=True, hide_index=True)
        else:
            st.error(f"No data for {detail_ticker}")

    with tab3:
        st.markdown("### So sánh tần suất DCA")
        st.caption("Cùng số tiền, cùng mã, cùng kỳ — khác tần suất mua")
        freq_ticker = st.selectbox("Chọn mã để so sánh tần suất", tickers if tickers else ["HPG"], key="freq_ticker")
        total_per_year = monthly * 12
        with st.spinner(f"So sánh tần suất cho {freq_ticker}..."):
            df_freq, histories = backtest_compare_frequencies(
                ticker=freq_ticker,
                total_per_year=total_per_year,
                start_date=start.isoformat(),
            )
        if not df_freq.empty:
            fig = go.Figure()
            colors = {"monthly": "#00C853", "quarterly": "#FFA726", "yearly": "#FF5252"}
            for freq, h in histories.items():
                fig.add_trace(go.Scatter(
                    x=h["date"], y=h["portfolio_value"],
                    mode="lines", name=freq,
                    line=dict(color=colors.get(freq, "#888"), width=2),
                ))
            fig.update_layout(
                title="Portfolio Value by Frequency",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#888", margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified", height=400,
                yaxis=dict(title="Portfolio Value", tickformat=",.0f"),
                legend=dict(orientation="h", y=1.1),
            )
            fig.update_xaxes(gridcolor="#1A1D29")
            fig.update_yaxes(gridcolor="#1A1D29")
            st.plotly_chart(fig, use_container_width=True)

            df_display = df_freq.copy()
            df_display["total_return"] = df_display["total_return"].apply(lambda x: f"{x:+.2%}")
            df_display["cagr"] = df_display["cagr"].apply(lambda x: f"{x:+.2%}")
            df_display["sharpe"] = df_display["sharpe"].apply(lambda x: f"{x:.2f}")
            df_display["max_dd"] = df_display["max_dd"].apply(lambda x: f"{x:.2%}")
            df_display["final_value"] = df_display["final_value"].apply(lambda x: f"{x:,.0f}")
            df_display["total_invested"] = df_display["total_invested"].apply(lambda x: f"{x:,.0f}")
            df_display["price_change"] = df_display["price_change"].apply(lambda x: f"{x:+.2%}")
            df_display = df_display.rename(columns={
                "frequency": "Tần suất", "total_invested": "Đã đầu tư", "final_value": "Giá trị cuối",
                "total_return": "Tổng lợi nhuận", "cagr": "CAGR", "sharpe": "Sharpe",
                "max_dd": "Sụt giảm tối đa", "price_change": "Biến động giá",
            })
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.error("Không có dữ liệu")

    with tab4:
        st.markdown("### Xếp Hạng Cổ Phiếu Tích Sản")
        st.caption("Cổ phiếu được xếp hạng theo tiêu chí tích sản: ROE cao, PE/PB thấp, tăng trưởng ổn định, biến động thấp")
        try:
            from src.strategies.accumulation import AccumulationStrategy
            feat_path = _root / "data" / "processed" / "features.parquet"
            if not feat_path.exists():
                with st.spinner("Generating features (first time, ~2 min)..."):
                    _generate_features_minimal()
            if feat_path.exists():
                df_feat = pd.read_parquet(feat_path)
                strat = AccumulationStrategy()
                ranking = strat.rank(df_feat)
                if not ranking.empty:
                    top = ranking.head(20)
                    top["score"] = top["score"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(top, use_container_width=True, hide_index=True)
                else:
                    st.info("No ranking available")
            else:
                st.info("Chưa có dữ liệu. Chạy pipeline trước để tạo features.")
        except Exception as e:
            st.info(f"Strategy ranking unavailable: {e}")

else:
    st.info("Điều chỉnh thông số bên sidebar và nhấn **Chạy Backtest**")
