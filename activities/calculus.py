# activities/calculus.py
import sys
from pathlib import Path

import streamlit as st

# ✅ 현재 파일(calculus.py)이 있는 폴더를 모듈 탐색 경로에 추가
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- 개별 시뮬레이션 모듈 import ---
import calculus_geometric_sequence_limit as geom_seq_limit

SIMULATIONS = {
    geom_seq_limit.TITLE: geom_seq_limit,
}

def main():
    st.title("📘 미적분 시뮬레이션 페이지")
    st.header("수열의 극한")

    selected_title = st.selectbox(
        "실행할 시뮬레이션을 선택하세요",
        list(SIMULATIONS.keys()),
    )

    st.divider()
    SIMULATIONS[selected_title].render()

if __name__ == "__main__":
    main()
