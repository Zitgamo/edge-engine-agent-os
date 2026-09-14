"""Published research snapshot, kept separate from live trading signals."""
from pathlib import Path

import pandas as pd
import streamlit as st

REVIEW_DIR = Path(__file__).resolve().parents[2] / "docs/strategy_review_20260914"


def render_strategy_review() -> None:
    st.title("Đánh giá chiến thuật")
    st.caption("Bản đánh giá 14/09/2026 · VN đến 27/08 · ETF Mỹ đến 11/09")
    st.info(
        "Đã thử đổi chiến thuật và thị trường. Các ứng viên chưa đạt để thay live. "
        "Đây là kết quả nghiên cứu, không phải tín hiệu mua mới."
    )
    st.subheader("Vì sao chưa có tín hiệu?")
    st.markdown(
        "- **Phiên 10/09:** độ rộng thị trường 36,08%, dưới ngưỡng 50%.\n"
        "- **Phiên 11/09:** lợi nhuận vượt chỉ số trong kiểm định của mô hình "
        "là −0,03%, dưới ngưỡng 0%.\n"
        "- **Paper VN30 ngày 11/09:** chỉ 30% mã tăng trong 20 phiên, "
        "dưới ngưỡng 60%."
    )
    st.caption("Chẩn đoán lịch sử tại ngày review; trạng thái mới nhất nằm ở trang chính.")
    try:
        vn = pd.read_csv(REVIEW_DIR / "comparison.csv")
        etf = pd.read_csv(REVIEW_DIR / "etf_comparison.csv")
        report = (REVIEW_DIR / "REVIEW.md").read_text(encoding="utf-8")
    except (OSError, ValueError):
        st.warning("Chưa tải được dữ liệu đánh giá. Vui lòng thử lại sau.")
        return

    vn_tab, etf_tab = st.tabs(["Chiến thuật Việt Nam", "Thị trường ETF Mỹ"])
    with vn_tab:
        st.write("Lợi nhuận trung bình mỗi lệnh sau phí; không phải lợi nhuận danh mục.")
        names = {
            "momentum_breadth60": "Momentum · độ rộng ≥60%",
            "reversion_weak": "Mua hồi · thị trường yếu",
            "regime_switch": "Ghép momentum và mua hồi",
        }
        table = vn.assign(strategy=vn.strategy.map(names)).pivot(
            index="strategy", columns="period", values="avg_net_return"
        ).rename_axis("Chiến thuật").rename(columns={"2026_seen": "2026 đến 27/08"})
        st.dataframe(table.map(lambda value: f"{value:.2%}"))
        st.warning(
            "Mua hồi và ghép chiến thuật đều lỗ ở hai giai đoạn. "
            "Tăng số tín hiệu chưa cải thiện kết quả."
        )
        st.caption(
            "ATR×2 / TP10 / tối đa 20 phiên; phí 0,3% vòng mua-bán. "
            "Dữ liệu đã được xem, VN30 cố định và các lệnh chồng lấn; "
            "chưa phải kiểm định độc lập mới."
        )
    with etf_tab:
        st.write(
            "Luân chuyển SPY, QQQ, IWM, GLD, TLT mỗi 20 phiên: "
            "chọn momentum 126 phiên dương và giá trên SMA200, hoặc giữ tiền mặt."
        )
        cost = st.selectbox("Phí mỗi chiều của chiến thuật luân chuyển", ["0,10%", "0,30%"])
        policy = "rotation_10bps_side" if cost == "0,10%" else "rotation_30bps_side"
        selected = etf[etf.strategy.isin([policy, "SPY_buy_hold_10bps_side"])].copy()
        selected["strategy"] = selected.strategy.map({
            policy: f"Luân chuyển · phí {cost}/chiều",
            "SPY_buy_hold_10bps_side": "Mua-giữ SPY · phí 0,10%/chiều",
        })
        table = selected[["strategy", "period", "total_return", "max_drawdown", "position_changes"]].rename(
            columns={"strategy": "Phương án", "period": "Giai đoạn",
                     "total_return": "Lợi nhuận", "max_drawdown": "Sụt giảm lớn nhất",
                     "position_changes": "Lần đổi vị thế"}
        )
        for column in ["Lợi nhuận", "Sụt giảm lớn nhất"]:
            table[column] = table[column].map(lambda value: f"{value:.2%}")
        st.dataframe(table, hide_index=True)
        st.warning(
            "Chưa chọn để thay live: thua mua-giữ SPY trong giai đoạn dài; "
            "sụt giảm 2026 lớn hơn SPY."
        )
        st.caption(
            "Lợi nhuận danh mục mô phỏng; 2026 tính đến 11/09. "
            "Giá điều chỉnh từ Yahoo Finance; chưa tính thuế, FX và lãi tiền mặt. "
            "Không so trực tiếp với lợi nhuận mỗi lệnh của bảng Việt Nam."
        )

    st.subheader("Hướng tiếp theo")
    st.write(
        "Đánh giá mô hình cũ và mới trên cùng dữ liệu ngoài mẫu; "
        "tiếp tục nghiên cứu đa thị trường với benchmark, phí, sụt giảm và "
        "tần suất cơ hội. Chỉ thay live khi đủ bằng chứng kiểm định và paper."
    )
    st.download_button("Tải báo cáo đầy đủ", report, file_name="strategy-review-20260914.md", mime="text/markdown")
    st.markdown(
        "[Log kiểm chứng 11/09](https://github.com/Zitgamo/edge-engine-agent-os/actions/runs/34605321577)"
    )
