from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from src.dashboard.strategy_review import render_strategy_review
from src.dashboard.style import CUSTOM_CSS

st.set_page_config(page_title="Đánh giá chiến thuật", page_icon="📊", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_strategy_review()
